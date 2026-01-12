收到请求后，其基本处理步骤：

1. API server 响应请求，对请求的信息进行初步处理
2. 进行请求的前置处理，包括对 prompt 进行 token 转换获得 token id
3. engine core client 将请求发送给合适的 engine core，engine core 完成自回归运算
4. 进行后置处理，包括 token 转文本的过程
5. 返回请求结果给用户

## Scheduler 模块

- 持续批处理 (Continuous batching)：持续地往 GPU 中送入请求数据，而不是离散地进行推理，一个请求结束立即下发信的请求
- 分块预填充 (Chunked prefill)：将大的 prefill 分块成更小的块 (切分序列) 执行，也可以将它们与 decode 阶段的请求一起混合执行

整体逻辑：按照可用资源的数量和优先级构建调度输出，scheduler 里面有两个主要的队列 waiting 和 running，以及一些辅助队列。运行时，请求在不同的队列之间轮转。scheduler 通过 KV manager 为请求配备 KV cache。scheduler 的优先级默认是 FCFS (先到先服务)，也支持用户自定义

scheduler 处理的大致步骤：

1. 请求到达时，先进入 waiting 队列
2. 找 KV manager 申请 KV cache 块
3. 具备下发条件的请求转入 running 队列，组 batch 下发执行 (图示中有3个请求)；资源不足的请求会转回 waiting 队列

## KV Manager 模块

KV cache管理的整体架构示意图如下所示，分为了逻辑层和物理层。KV Manager负责逻辑层、Model Runner处理物理层；Scheduler（调度器）作为信息传递的桥梁，衔接了逻辑层与物理层。cache的管理元素包括：池（pool）表(table)、层(layer)、块(block)和槽(slot)。

- slot：为最小管理单元，每个token占一个slot；
- block：为请求分配的基本单位，一个block包含多个slot；
- pool：为逻辑层block的管理合集，通过链表将block数据组织起来；
- table：管理请求与数据的映射表，一个table可包含多个请求的信息。位于物理层；
- layer：一个整体的tensor，拆分成多个blocks使用。对应attenti

模块之间运行的关键步骤：

1. Scheduler 分配资源给请求，通过 KV Manager 申请逻辑 blocks
2. KV Manger 把 Pool 中空闲的 blocks 选中后给到对应请求
3. 分配好逻辑 blocks 后 Scheduler 构建 Scheduler.output 传递给 ModelRunner
4. ModelRunner 为每条请求创建 block table，并生成 slot_mapping
5. 计算时把 slot_mapping 传入 attention，就可以从物理 KV blocks 找到所需数据了

## Model Runner 模块

模型执行器 (model runner) 主要负责计算调度器发送过来的批请求，并返回执行结果。

从上面 engine core 的架构可知，executor 可以有多个 worker 模块，每个 worker 都会有自己的 model runner。model runner 的逻辑主要是模型运算、以及物理层的 kv cache 分配与管理。

执行基本步骤:

1. 根据映射表信息为每个待执行请求分配 KV blocks
2. 将请求组成序列 batch，并让模型处理该 batch 数据
3. 在 Attention 层运算阶段，每层拿取自己对应的 KV cache 数据，完成 MHA/GQA/MLA 运算

## Attention 模块

## Offline Inference example

![](offline_inference.jpg)

Step 1：tokenization

![](./step_1.jpg)

Step 2：KV Manager 分配逻辑块、计算 slot

![](./step_2.jpg)

Step 3：Model Runner 分配 KV cache

![](./step_3.jpg)

Step 4：Decoding 生成新 token，并更新 ids、positions、slot_mapping 数据。该过程需要迭代多次。

![](./step_4.jpg)

示例中KV manager的逻辑块是连续的，而物理块在model runner中不连续。

从模型输出的logits到token id，要经过采样(sampling)计算。

![](./step_4_2.jpg)

Step 5：De-Tokenization

![](./step_5.jpg)