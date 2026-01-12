# 批次的调度逻辑和攒批逻辑

## Scheduler

调度逻辑：vllm/v1/core/sched/request_queue.py

- FCFS
- PRIORITY

Scheduler 提供默认调度策略 _schedule_default，实现吞吐量最大化

_schedule_default 逻辑：

- Prefill First: 尽可能多地批处理 Prefill 请求
- Decode: 如果 batch 还有空间，没有新的 Prefill 调度，则调度 Decode 请求
- Swap In: 如果仍有空闲内存，则尝试将 Swapped 请求换入 GPU

_schedule_chunked_prefill 逻辑：

- chunked prefill 调度策略更加灵活，支持 prefill 和 decode 混合 batch 批处理，提升 GPU 利用率，避免 decode 请求被 prefill 抢占得不到调度，减少 decode 请求等待时间。满足 token 延迟和吞吐量之间的平衡。

### BlockSpaceManager

Scheduler 初始化时创建了 BlockSpaceManager，用于管理 GPU/CPU 内存块

Scheduler 负责管理多个状态队列

- waiting：用于存放所有还未开始做推理的 seq_group
- running：用于存放当前正在做推理的 seq_group。更准确地说，它存放的是上 1 个推理阶段被送去做推理的 seq_group 们
- swapped：被交换到 CPU 的请求；用于存放被抢占的 seq_group

### SequenceGroup

「1 个 prompt -> 多个 outputs」 这样的结构组成一个 SequenceGroup 实例

其中每组 「prompt -> output」 组成一个序列 (seq，属于 Sequence 实例)

每个 seq 下有若干状态 (属于 SequenceStatus 实例)
 
- FINISHED_STOPPED：正常执行完毕，例如碰到符号，该 seq 的推理正常结束了
- FINISHED_LENGTH_CAPPED：因为 seq 的长度达到最大长度限制，而结束推理
- FINISHED_ABORTED：因不正常状态，而被终止的推理。例如客户端断开连接，则服务器会终止相关 seq 的推理
- FINISHED_IGNORED：因 prompt 过长而被终止执行的推理。本质上也是受到长度限制
- WAITING：正在 waiting 队列中。waiting 队列中的序列都没有做过 prefill
- RUNNING：正在 running 队列中，即已经开始做推理
- SWAPPED：正在 swapped 队列中，表示此时 GPU 资源不足，相关的 seq_group 被抢占，导致其暂停推理，相关的 KV block 被置换到 CPU 上 (swap out)，等待 GPU 资源充足时再置换回来重新计算 (swap in)

### vLLM 对输入数据做的预处理

- 在 vLLM 内部计算逻辑中，1个 prompt 是1个 request
- 每个 prompt 将被包装成一个 SequenceGroup 实例提供给调度器做调度
- 1个 SequenceGroup 实例下维护着若干个 Sequence 实例，对应着 "1个 prompt -> 多个 outputs" 这种更一般性的解码场景。
- 1个 Sequence 实例下维护着属于自己的逻辑块列表，数据类型为 List[NullBlock]

调度逻辑：

- 如果当前 swapped 队列为空，那就去检查是否能从 waiting 队列中调度 seq_group，直到不满足调度条件为止 (gpu 空间不足，或 waiting 队列已为空等)。此时，1 个推理阶段中，所有的 seq_group 都处在 prefill 阶段。
- 如果当前 swapped 队列非空，或者无法从 waiting 队列中调度任何 seq_group 时： 
  - 检查是否能从 running 队列中调度 seq_group，直到不满足调度条件为止。 
  - 若本次无新的被抢占的 seq_group，且 swapped 队列非空，就检查是否能从 swapped 队列中调度 seq_group，直到不满足调度条件为止。
  - 此时，1 个推理阶段中，所有的 seq_group 要么全来自 running 队列，要么来自 running + swapped 队列，它们都处在 decode 阶段。

### Continuous Batching 核心逻辑

- 调度器优先调度 prefill 阶段请求 (waiting 队列中的请求)，其次会处理 running 队列的请求，如果调度 running 队列时没有发生抢占，则会调度 swap 队列的请求。
- 调度 prefill 阶段请求，会有一些准入条件，包括：
  - swap 队列为空
  - 距离上次调度 prefill 请求满足一定的时间间隔
- 调度器还支持了优先级调度的能力 (_schedule_priority_preemption 函数实现)，支持按照 seq_group 的优先级进行任务调度

