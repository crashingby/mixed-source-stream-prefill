基于三个信号的在线决策

对每个 Update 源 s，维护：

1. 更新波动率
λ
s
	​


表示该源近期的 Update 到达频率，可以用 EWMA 估计：

λ
^
s
	​

(t)=α⋅
t
i
	​

−t
i−1
	​

1
	​

+(1−α)
λ
^
s
	​

(t
i−1
	​

)

频率越高，当前版本越可能短命。

2. 当前版本的 Prefill 成本
C
s
	​


它不应该简单等于 token 数，而应该根据实际失效范围估计：

C
s
	​

=T
prefill
	​

(N
s
	​

−LCP(V
i−1
	​

,V
i
	​

),B,hardware)

这里：

V
i
	​

 是新的 Update 版本；
N
s
	​

−LCP 是需要重算的 token；
B 是当前 batch 状态；
T
prefill
	​

 可以来自离线 profiling 或在线性能模型。
3. 当前 GPU 资源压力
μ(t)

它可以理解成“Prefill GPU 时间的影子价格”。例如由以下量构造：

μ(t)=1+aQ
P
	​

(t)+bU
P
	​

(t)+cM
P
	​

(t)

其中：

Q
P
	​

：Prefill 等待队列长度；
U
P
	​

：Prefill GPU 利用率；
M
P
	​

：KV cache 内存压力。

负载越高，浪费一次 Prefill 的代价越大，因此越应该合并 Update。

一个可以真正放进论文的决策模型

假设当前 Update 到达后，考虑等待 w 时间。

定义：

S
s
	​

(w)=P(未来 w 时间内没有新 Update)

那么：

1−S
s
	​

(w)

就是当前版本在等待窗口内被覆盖的概率。

等待 w 可能避免的期望浪费为：

G
s
	​

(w)=μ(t)C
s
	​

[1−S
s
	​

(w)]

等待造成的延迟代价为：

L
s
	​

(w)=β
s
	​

w

于是选择：

w
s
∗
	​

=arg
0≤w≤W
s
max
	​

max
	​

{μ(t)C
s
	​

[1−S
s
	​

(w)]−β
s
	​

w}

这个式子的意义非常清楚：

Update 越频繁，等待越值得；
Prefill 重算越贵，等待越值得；
GPU 越繁忙，等待越值得；
TTFT 越敏感，等待越不值得；
最大等待时间保证不会无限挂起。
在 Poisson 近似下

假设 Update 到达率是 λ
s
	​

：

S
s
	​

(w)=e
−λ
s
	​

w

目标变成：

w
s
∗
	​

=arg
0≤w≤W
s
max
	​

max
	​

[μ(t)C
s
	​

(1−e
−λ
s
	​

w
)−β
s
	​

w]

求导后得到：

w
s
∗
	​

=[
λ
s
	​

1
	​

ln
β
s
	​

μ(t)C
s
	​

λ
s
	​

	​

]
0
W
s
max
	​

	​


其中截断符号表示限制在 [0,W
s
max
	​

] 内。

这个结果还很符合直觉：

当 μCλ≤β 时，w
∗
=0，立即执行；
当 Update 高频、计算昂贵、系统繁忙时，w
∗
>0；
当 TTFT deadline 接近时，通过增大 β 或减小 W
max
 强制执行。

这就不是为了“论文里有公式”，而是一个公式真正产生了调度决策。

运行时状态机

每个 Update 源只保留：

pending_version
last_processed_version
last_update_time
estimated_arrival_rate
estimated_recompute_cost
deadline

每次新事件到达时：

1. 用新版本覆盖 pending_version
2. 更新 λ̂ 和重算成本 Ĉ
3. 重新计算最优等待窗口 w*
4. 若 w*=0，立即进入 Prefill 队列
5. 否则挂起，等待新事件或定时器

以下情况强制 flush：

retrieval 明确发送 finished；
达到最大 staleness；
请求即将超过 TTFT SLO；
一段时间没有新 Update；
Prefill 队列突然空闲；
当前 Update 的信息增益特别高。

最后一点可以作为后续扩展，例如 Top-K 变化幅度很大时立即执行：

Δ(V
i−1
	​

,V
i
	​

)=1−
K
∣V
i−1
	​

∩V
i
	​

∣
	​


变化幅度越大，越可能值得提前 Prefill；只替换一个低排名文档时，可以继续等待。