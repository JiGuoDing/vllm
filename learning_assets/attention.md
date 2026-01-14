## 注意力机制
- Query: 当前输入的特征表示
- Key: 每个 token 的特征表示
- Value: 每个 token 的内容表示

### 步骤1：根据 Query 和 Key 计算两者相关性，即注意力分数

注意力分数 score(q, k) 衡量 Query 和 Key 之间的相关性

score(q, k) = softmax(a(q, k))

其中 a(q, k) 有很多变体：

- 加性 (拼接) 注意力：将 Query, Key 分别乘对应的可训练权重矩阵，然后相加
- 缩放点积注意力：直接将 Query, Key 相乘，再除以缩放因子

加性 (拼接) 注意力：

```python
import torch
from torch import nn
import torch.nn.functional as F

class Attention(nn.Module):
    def __init__(self, n_hidden_enc, n_hidden_dec):
        super().__init__()
        
        # 编码层隐藏层维度
        self.h_hidden_enc = n_hidden_enc
        # 解码层隐藏层维度
        self.h_hidden_dec = n_hidden_dec
        
        # 核心线性层：输入维度 = 2 * 编码器维度 + 解码器维度，输出维度 = 解码器维度，无偏置
        self.W = nn.Linear(2*n_hidden_enc + n_hidden_dec, n_hidden_dec, bias=False) 
        # 可学习参数 V：用于将变换后的特征投影为注意力分数，初始化为随机值
        self.V = nn.Parameter(torch.rand(n_hidden_dec))
        
    
    def forward(self, hidden_dec, last_layer_enc):
        # 步骤 1：获取输入张量的维度信息
        batch_size = last_layer_enc.size(0)
        # 编码器输入序列长度
        src_seq_len = last_layer_enc.size(1)

        # 步骤 2：调整解码器隐藏层状态维度，与编码器序列长度对齐
        hidden_dec = hidden_dec[:, -1, :].unsqueeze(1).repeat(1, src_seq_len, 1)       

        # 步骤 3：拼接 + 线性变换 + tanh 激活，计算注意力特征
        tanh_W_s_h = torch.tanh(self.W(torch.cat((hidden_dec, last_layer_enc), dim=2))) 
        # 步骤 4：调整维度，为后续矩阵乘法做准备
        tanh_W_s_h = tanh_W_s_h.permute(0, 2, 1)       #[b, n_hidde_dec, seq_len]
        
        # 步骤 5：调整参数 V 的维度，准备计算注意力分数
        V = self.V.repeat(batch_size, 1).unsqueeze(1)  #[b, 1, n_hidden_dec]
        # 步骤 6：计算注意力分数 e (bmm = batch matrix multiplication，批次矩阵乘法)，未归一化
        e = torch.bmm(V, tanh_W_s_h).squeeze(1)        #[b, seq_len]
        
        att_weights = F.softmax(e, dim=1)              #[b, src_seq_len]
        
        return att_weights
```

### 步骤2：根据注意力分数进行加权求和，得到带注意力分数的 Value

Output = score(Q, K) · V


## 自注意力机制

自注意力机制的基本思想是，在处理序列数据时，每个元素都可以与序列中的其他元素建立关联，而不仅仅是依赖于相邻位置的元素。它通过计算元素之间的相对重要性来自适应地捕捉元素之间的长程依赖关系。

具体而言，对于序列中的每个元素，自注意力机制计算其与其他元素之间的相似度，并将这些相似度归一化为注意力权重。然后，通过将每个元素与对应的注意力权重进行加权求和，可以得到自注意力机制的输出。

其中，q (Query) 的含义一般的解释是用来和其他单词进行匹配，更准确地说是用来计算当前单词或字与其他的单词或字之间的关联或者关系；k (Key) 的含义则是被用来和 q 进行匹配，也可理解为单词或者字的关键信息。

### 自注意力计算过程

每个 token 的嵌入向量 a 分别与三个矩阵 (共享的Wq, Wk, Wv) 相乘，得到对应的 q, k, v 向量

当要计算 token1 与其他 token 的注意力时，计算 token1 的 q 向量与所有 token 的 k 向量的点积，再除以 sqrt(dk) 得到注意力分数 (这个计算过程是可以优化创新的)

得到 token1,2、token1,3、token1,4 的注意力分数后，经过 softmax 归一化，得到注意力权重 w1,2、w1,3、w1,4

再将注意力权重分别与对应的 v 向量相乘，最后加和，得到 token1 的最终特征向量 (即 token1 自身以及相关 token 的信息做了加权融合后的新嵌入，会作为下一步网络层的输入，例如前馈神经网络)

![](./self-attention.jpg)

self-attention

```python
import torch
import torch.nn as nn
from math import sqrt

class SelfAttention(nn.Module):
    dim_in: int
    dim_k: int
    dim_v: int

    def __init__(self, dim_in, dim_k, dim_v):
        super(SelfAttention, self).__init__()
        self.dim_in = dim_in
        self.dim_k = dim_k
        self.dim_v = dim_v
        self.linear_q = nn.Linear(dim_in, dim_k, bias=False)
        self.linear_k = nn.Linear(dim_in, dim_k, bias=False)
        self.linear_v = nn.Linear(dim_in, dim_v, bias=False)
        self._norm_fact = 1 / sqrt(dim_k)

    def forward(self, x):
        # x: batch, n, dim_in
        batch, n, dim_in = x.shape
        assert dim_in == self.dim_in

        q = self.linear_q(x)  # batch, n, dim_k
        k = self.linear_k(x)  # batch, n, dim_k
        v = self.linear_v(x)  # batch, n, dim_v

        dist = torch.bmm(q, k.transpose(1, 2)) * self._norm_fact  # batch, n, n
        dist = torch.softmax(dist, dim=-1)  # batch, n, n

        att = torch.bmm(dist, v)
        return att
```

## 多头自注意力机制

多头注意力机制是在自注意力机制的基础上发展起来的，是自注意力机制的变体，旨在增强模型的表达能力和泛化能力。它通过使用多个独立的注意力头，分别计算注意力权重，并将它们的结果进行拼接或加权求和，从而获得更丰富的表示。

多头注意力机制就是在为一个 token 分配多个 q, k, v 矩阵，并行计算多个注意力权重。

![](./multi-head-self-attention.jpg)

multi-head self-attention

```python
import torch
import torch.nn as nn
from math import sqrt

class MultiHeadSelfAttention(nn.Module):
    dim_in: int  # input dimension
    dim_k: int   # key and query dimension
    dim_v: int   # value dimension
    num_heads: int  # number of heads, for each head, dim_* = dim_* // num_heads

    def __init__(self, dim_in, dim_k, dim_v, num_heads=8):
        super(MultiHeadSelfAttention, self).__init__()
        assert dim_k % num_heads == 0 and dim_v % num_heads == 0, "dim_k and dim_v must be multiple of num_heads"
        self.dim_in = dim_in
        self.dim_k = dim_k
        self.dim_v = dim_v
        self.num_heads = num_heads
        self.linear_q = nn.Linear(dim_in, dim_k, bias=False)
        self.linear_k = nn.Linear(dim_in, dim_k, bias=False)
        self.linear_v = nn.Linear(dim_in, dim_v, bias=False)
        self._norm_fact = 1 / sqrt(dim_k // num_heads)

    def forward(self, x):
        # x: tensor of shape (batch, n, dim_in)
        batch, n, dim_in = x.shape
        assert dim_in == self.dim_in

        nh = self.num_heads
        dk = self.dim_k // nh  # dim_k of each head
        dv = self.dim_v // nh  # dim_v of each head

        q = self.linear_q(x).reshape(batch, n, nh, dk).transpose(1, 2)  # (batch, nh, n, dk)
        k = self.linear_k(x).reshape(batch, n, nh, dk).transpose(1, 2)  # (batch, nh, n, dk)
        v = self.linear_v(x).reshape(batch, n, nh, dv).transpose(1, 2)  # (batch, nh, n, dv)

        dist = torch.matmul(q, k.transpose(2, 3)) * self._norm_fact  # batch, nh, n, n
        dist = torch.softmax(dist, dim=-1)  # batch, nh, n, n

        att = torch.matmul(dist, v)  # batch, nh, n, dv
        att = att.transpose(1, 2).reshape(batch, n, self.dim_v)  # batch, n, dim_v
        return att
```