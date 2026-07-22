# Prompt Stream KV Block 管理笔记

这份笔记记录当前流式 prefill 原型中，KV Cache Block 如何在源码里按 segment 管理。重点解释三个问题：

- `prefix`，也就是**通用模板 + Append chunk**，如何拥有自己的 KV blocks。
- `suffix`，也就是**Update chunk + 最后的用户问题**，如何拥有另一组 KV blocks。
- Append、Update、Question 三个计算阶段如何选择可见 KV、写入 segment、申请新块和释放失效块。

当前实现是研究原型版本。为了让 P/D 节点通过 NIXL 按 block table 直接传输，系统选择了**两段物理 KV 域**，而不是 token 级无缝拼接。

## 总体模型

逻辑 prompt 仍然是：

```text
通用模板 prefix + append chunks + update snapshot + final question
```

但物理 KV block 管理拆成两段：

```text
PREFIX_APPEND segment = 通用模板 prefix + append chunks + padding
UPDATE segment        = update snapshot + final question
```

其中：

- **PREFIX_APPEND**：保存稳定单调增长的部分，即通用模板和所有 append chunk。
- **UPDATE**：保存会被 update 快照替换的部分，以及最后的用户问题。
- **padding**：位于 PREFIX_APPEND 末尾，用来把 PREFIX_APPEND 补齐到 block 边界，让 UPDATE 可以从新的完整 block 开始。

当前代码里没有做 token 级 compaction，也没有为中间 padding 做特殊 attention mask。padding token 会参与 attention，这是当前近似的一部分。

## 核心数据结构

### `PromptStreamKVSegment`

位置：`vllm/v1/request.py`

```python
class PromptStreamKVSegment(enum.Enum):
    """流式 prompt 的两个物理 KV 域。"""

    PREFIX_APPEND = "prefix_append"
    UPDATE = "update"
```

这两个枚举不是逻辑 source 类型，而是**物理 KV block 域**：

- `PREFIX_APPEND` 对应原始 `request_id`。
- `UPDATE` 对应派生 key：`{request_id}::prompt_stream::update`。

这样做的直接目的，是让 update 快照失效时只截断 UPDATE 域，不影响 prefix 和 append 已经算好的 KV。

### `PromptStreamState`

位置：`vllm/v1/request.py`

重要字段：

```python
prefix_token_ids: list[int]
append_token_ids: list[int]
suffix_token_ids: list[int]
prefix_append_num_computed_tokens: int
update_num_computed_tokens: int
question_num_computed_tokens: int
block_size: int | None
padding_token_id: int
```

字段含义：

- `prefix_token_ids`：start 阶段的固定模板。
- `append_token_ids`：所有 append chunk 按到达顺序形成的请求级 token 日志。
- `suffix_token_ids`：`PROMPT_FINISHED` 携带的最终用户问题。
- `prefix_append_num_computed_tokens`：PREFIX_APPEND 物理域已经 prefill 的 token 数，包含 padding。
- `update_num_computed_tokens`：当前 update 快照已经 prefill 的 token 数。
- `question_num_computed_tokens`：最后用户问题已经 prefill 的 token 数。
- `padding_token_id`：PREFIX_APPEND 补 block 时使用的确定性 token id。

这里的 computed token 计数是 segment-aware 的：update 的 computed tokens 不包含 prefix 和 append。

## 物理 prompt 如何构造

逻辑 prompt 不包含 padding：

```python
prefix_token_ids + append_token_ids + update_token_ids + suffix_token_ids
```

但是物理 prompt 会包含 PREFIX_APPEND 尾部 padding：

```python
padded(prefix_token_ids + append_token_ids) + update_token_ids + suffix_token_ids
```

对应源码在 `Request._get_prefix_append_physical_token_ids()`：

```python
token_ids = list(self.prompt_stream.prefix_token_ids)
token_ids.extend(
    self._get_prompt_stream_segment_token_ids(PromptSourceType.APPEND)
)
if token_ids:
    num_padding_tokens = (-len(token_ids)) % self.prompt_stream.block_size
    token_ids.extend(
        [self.prompt_stream.padding_token_id] * num_padding_tokens
    )
```

这里的 padding 解决的是 block table 拼接问题：

- KV manager 只能按 block 管理和拼接。
- 如果 PREFIX_APPEND 最后一块没满，而 UPDATE 立刻接在它后面，就需要 token slot 级别的精细拼接。
- 当前为了让 P/D transfer 简单稳定，选择把 PREFIX_APPEND 补满，让 UPDATE 从新 block 开始。

代价是 padding token 会位于完整物理 prompt 的中间：

```text
[prefix + append + padding][update + question]
```

因此当前计算不是完全等价于无 padding 的真实 prompt，而是一个明确的近似。

## Append 阶段如何计算

Append 事件只修改 `PREFIX_APPEND` 域。

流程如下：

1. 新 append chunk 到达。
2. token ids 追加到 `prompt_stream.append_token_ids`。
3. 如果上一轮 PREFIX_APPEND 已经因为 block 对齐算过 padding，则把 computed 进度回退到旧逻辑长度。
4. scheduler 后续优先补算 PREFIX_APPEND，直到 `prefix + append + padding` 全部完成。

对应源码在 `Request._apply_prompt_stream_append()`：

```python
old_prefix_append_len = self.get_prompt_stream_prefix_append_len()
old_prefix_append_num_computed_tokens = (
    prompt_stream.prefix_append_num_computed_tokens
)

prompt_stream.append_token_ids.extend(event.token_ids or [])

prompt_stream.prefix_append_num_computed_tokens = min(
    old_prefix_append_num_computed_tokens,
    old_prefix_append_len,
)
```

为什么要回退？

假设上一轮：

```text
prefix_append_len = 19
block_size = 16
physical_prefix_append_len = 32
```

其中 20 到 32 是 padding。如果后来又 append 了 10 个 token，新 token 应该覆盖旧 padding 的位置。于是 computed 进度不能继续认为 32 都是真实 prefix_append，只能保留到旧逻辑长度 19，再让 scheduler 重新计算新增 append 以及新的 padding。

Append 阶段的执行视图是：

```text
读取 blocks: PREFIX_APPEND
写入 blocks: PREFIX_APPEND
计算 tokens: padded(prefix + append)
```

对应源码在 `Request._select_prompt_stream_execution_view()`：

```python
if self.prompt_stream.prefix_append_num_computed_tokens < prefix_append_physical_len:
    return PromptStreamExecutionView(
        phase=PromptStreamExecutionPhase.PREFIX_APPEND,
        write_segment=PromptStreamKVSegment.PREFIX_APPEND,
        required_segments=(PromptStreamKVSegment.PREFIX_APPEND,),
        token_ids=tuple(prefix_append_token_ids),
        num_computed_tokens=self.prompt_stream.prefix_append_num_computed_tokens,
        segment_offset=0,
    )
```

这里 `required_segments` 只有 `PREFIX_APPEND`，所以 append prefill 不会读取 UPDATE blocks。

## Update 阶段如何计算

Update 事件表示某个 update source 的**完整快照**，不是增量 diff。当前只支持一个 update source。

Update 到达后做的是**update 源内 LCP**：

```python
update_common_prefix_len = self._get_common_prefix_len(
    old_update_token_ids,
    new_update_token_ids,
)
```

注意：LCP 只比较旧 update 快照和新 update 快照，不包含 prefix/template，也不包含 append。

然后系统会：

- 把 `update_num_computed_tokens` 回退到 `min(old_computed, update_common_prefix_len)`。
- 只要求 KV manager 截断 UPDATE segment。
- 保留 PREFIX_APPEND segment 的 blocks 和 computed 进度。

对应源码在 `Request._apply_prompt_stream_update()`：

```python
prompt_stream.update_num_computed_tokens = min(
    old_update_num_computed_tokens,
    update_common_prefix_len,
)
mutation.kv_segment_to_truncate = PromptStreamKVSegment.UPDATE
mutation.kv_segment_keep_tokens = update_common_prefix_len
```

Update 阶段的执行视图是：

```text
读取 blocks: PREFIX_APPEND + UPDATE
写入 blocks: UPDATE
计算 tokens: padded(prefix + append) + update_snapshot
```

对应源码：

```python
execution_token_ids = prefix_append_token_ids + update_token_ids
return PromptStreamExecutionView(
    phase=PromptStreamExecutionPhase.UPDATE,
    write_segment=PromptStreamKVSegment.UPDATE,
    required_segments=(
        PromptStreamKVSegment.PREFIX_APPEND,
        PromptStreamKVSegment.UPDATE,
    ),
    token_ids=tuple(execution_token_ids),
    num_computed_tokens=(
        prefix_append_physical_len
        + self.prompt_stream.update_num_computed_tokens
    ),
    segment_offset=prefix_append_physical_len,
)
```

这里最关键的是 `required_segments` 和 `write_segment` 不一样：

- `required_segments` 包含 PREFIX_APPEND 和 UPDATE，表示 update token prefill 时可以 attend 到完整前缀 KV 和已经保留的 update LCP KV。
- `write_segment` 是 UPDATE，表示新计算出来的 KV 只写到 UPDATE 域。
- `segment_offset=prefix_append_physical_len`，表示 UPDATE 域内部 token 位置要从全局物理位置中减掉 PREFIX_APPEND 的物理长度。

## Question 阶段如何计算

`PROMPT_FINISHED` 到达后，事件文本被保存到 `suffix_token_ids`。这里的 suffix 是最后用户问题。

Question 阶段不会创建第三个 KV segment，而是和 update 共用 UPDATE 物理域：

```text
UPDATE segment = update snapshot + final question
```

这样做是为了避免出现第三个 partial-block 边界。如果 question 单独一个 segment，那么 update 末尾也可能需要 padding，物理误差会更多。

Question 阶段的执行视图是：

```text
读取 blocks: PREFIX_APPEND + UPDATE
写入 blocks: UPDATE
计算 tokens: padded(prefix + append) + update_snapshot + final_question
```

对应源码：

```python
execution_token_ids = (
    prefix_append_token_ids + update_token_ids + question_token_ids
)
return PromptStreamExecutionView(
    phase=PromptStreamExecutionPhase.QUESTION,
    write_segment=PromptStreamKVSegment.UPDATE,
    required_segments=(
        PromptStreamKVSegment.PREFIX_APPEND,
        PromptStreamKVSegment.UPDATE,
    ),
    token_ids=tuple(execution_token_ids),
    num_computed_tokens=(
        prefix_append_physical_len
        + len(update_token_ids)
        + self.prompt_stream.question_num_computed_tokens
    ),
    segment_offset=prefix_append_physical_len,
)
```

也就是说，最后用户问题的 KV 会直接接在 update 快照后面，仍然存入 UPDATE segment。

## KV block 申请逻辑

普通请求的 KV block table 是一条连续 request_id 维度的表。prompt stream 请求不同：它必须按当前 `write_segment` 使用 segment-local key 申请 blocks。

入口在 `KVCacheManager._allocate_prompt_stream_slots()`：

```python
write_segment = request.get_prompt_stream_write_segment()
segment_key = request.get_prompt_stream_segment_key(write_segment)
segment_num_computed_tokens = (
    request.get_prompt_stream_segment_num_computed_tokens()
)
segment_num_tokens = segment_num_computed_tokens + num_new_tokens
```

其中：

- Append 阶段：`write_segment = PREFIX_APPEND`，`segment_key = request_id`。
- Update 阶段：`write_segment = UPDATE`，`segment_key = request_id::prompt_stream::update`。
- Question 阶段：`write_segment = UPDATE`，继续追加到 update segment 后面。

block 申请只看当前写入 segment 的长度：

```python
num_blocks_to_allocate = self.coordinator.get_num_blocks_to_allocate(
    request_id=segment_key,
    num_tokens=segment_num_tokens,
    total_computed_tokens=segment_num_computed_tokens,
)
```

因此 update prefill 虽然读取 PREFIX_APPEND blocks，但不会给 PREFIX_APPEND 再申请新 blocks。

## worker 看到的 block table

worker 每轮执行需要完整 block table。这个 block table 由 `required_segments` 决定。

入口在 `KVCacheManager.get_prompt_stream_execution_blocks()`：

```python
required_segments = request.get_prompt_stream_required_segments()
segment_keys = [
    self._prompt_stream_segment_key(request, segment)
    for segment in required_segments
]
return self.create_kv_cache_blocks(
    self.coordinator.get_merged_blocks(segment_keys)
)
```

底层拼接在 `KVCacheCoordinator.get_merged_blocks()`：

```python
for request_id in request_ids:
    for group_id, manager in enumerate(self.single_type_managers):
        merged_blocks[group_id].extend(
            manager.req_to_blocks.get(request_id) or []
        )
```

这说明当前系统是**按 block list 直接拼接**，不是按 token slot 做 compaction。

三个阶段对应关系如下：

| 阶段 | `required_segments` | `write_segment` | 新 KV 写入位置 |
| --- | --- | --- | --- |
| Append | `PREFIX_APPEND` | `PREFIX_APPEND` | prefix+append 域 |
| Update | `PREFIX_APPEND, UPDATE` | `UPDATE` | update 域 |
| Question | `PREFIX_APPEND, UPDATE` | `UPDATE` | update 域中 update 后面 |

## 失效和释放逻辑

### Append 到来

Append 到来时不会释放 UPDATE blocks。它只会把 PREFIX_APPEND 的 computed 进度从旧 padding 处回退到旧逻辑长度。

原因是旧 padding 后续会被真实 append token 覆盖。下一轮 scheduler 会继续给 PREFIX_APPEND 补算新的 append token 和新的 padding。

### Update 到来

Update 到来时会只截断 UPDATE segment：

```python
mutation.kv_segment_to_truncate = PromptStreamKVSegment.UPDATE
mutation.kv_segment_keep_tokens = update_common_prefix_len
```

scheduler 收到 mutation 后调用：

```python
self.kv_cache_manager.truncate_prompt_stream_segment_blocks(
    existing,
    mutation.kv_segment_to_truncate,
    mutation.kv_segment_keep_tokens,
)
```

KV manager 再转成 segment key：

```python
segment_key = self._prompt_stream_segment_key(request, segment)
return self.coordinator.truncate_blocks_from_position(
    segment_key,
    keep_tokens,
)
```

底层释放是 block 粒度：

```python
num_blocks_to_keep = 0 if keep_tokens == 0 else cdiv(keep_tokens, self.block_size)
blocks_to_free = req_blocks[num_blocks_to_keep:]
del req_blocks[num_blocks_to_keep:]
```

所以：

- LCP 是 token 级。
- block 释放是 block 级。
- 如果 LCP 落在半个 block 里，会保留包含 LCP 的整块。
- 后续 forward 从 token 级 computed 位置继续写入 UPDATE 域。

## finish 和 P/D KV transfer

finish 后，P 节点需要把 block ids 暴露给 connector。当前顺序固定为：

```text
PREFIX_APPEND blocks + UPDATE blocks
```

入口在 `KVCacheManager.get_prompt_stream_blocks()`：

```python
segment_keys = self.get_prompt_stream_segment_keys(request)
return self.create_kv_cache_blocks(
    self.coordinator.get_merged_blocks(segment_keys)
)
```

scheduler 在 request finished 时会取这个合并 block table：

```python
block_ids = self.kv_cache_manager.get_prompt_stream_block_ids(request)
```

然后检查 block 数是否覆盖当前物理 computed tokens：

```python
expected_num_blocks = (
    request.num_computed_tokens + block_size - 1
) // block_size
if len(group) != expected_num_blocks:
    raise RuntimeError(...)
```

如果 connector 返回 `kv_transfer_params`，scheduler 会额外塞入物理 prompt token ids：

```python
physical_prompt_token_ids = (
    request.get_prompt_stream_physical_prompt_token_ids()
)
kv_transfer_params["prompt_stream_prompt_token_ids"] = physical_prompt_token_ids
kv_transfer_params["prompt_stream_physical_prompt_len"] = physical_prompt_len
```

这里必须传**物理 prompt**，不是逻辑 prompt。因为 D 节点加载远端 KV 时，看到的是带 padding 的 block 序列。如果 D 侧 prompt token ids 不包含同样 padding，位置会错位。

## 当前近似和限制

当前实现能保证：

- Append 只写 PREFIX_APPEND blocks。
- Update 可以读取 PREFIX_APPEND + UPDATE，但只写 UPDATE blocks。
- Update 失效只释放 UPDATE blocks，不影响稳定 prefix/append。
- Question 接在 UPDATE segment 后面，避免第三个 partial-block segment。
- P/D transfer 使用同一份物理 prompt token ids，避免 remote KV 和 D prompt 错位。

当前实现不能保证：

- padding token 对 attention 完全无影响。
- update 的位置编码等价于无 padding 逻辑 prompt。
- LCP 落在半块里时可以只释放半块后缀。
- finish 时把两个 segment 做 token 级 compaction。

更精细的后续方向包括：

- **显式 mask**：物理 slot 保留 padding，但 attention 跳过 padding。
- **逻辑 position ids**：让 update/question 的位置按无 padding 逻辑序列继续。
- **finish-time KV compaction**：P 侧最终把两段 KV 搬成紧凑连续布局再交给 D。
- **segment-aware attention**：worker/model runner 原生理解物理 slot 和逻辑 token 位置的映射。

这些方向都会触及 worker 输入构造、attention mask、position encoding 和 KV transfer 对齐，不适合混在当前最小 REST/P-D 链路里直接改。
