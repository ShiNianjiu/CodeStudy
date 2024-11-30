# 导入并打印torch库的版本
from importlib.metadata import version
print("torch version:", version("torch"))

import torch

# 创建一个tensor，表示输入的嵌入向量
inputs = torch.tensor(
  [[0.43, 0.15, 0.89], # 输入数据，每个子列表代表一个单词的嵌入向量
   [0.55, 0.87, 0.66],
   [0.57, 0.85, 0.64],
   [0.22, 0.58, 0.33],
   [0.77, 0.25, 0.10],
   [0.05, 0.80, 0.55]]
)

# 查询向量，选择第二个输入作为查询向量
query = inputs[1]

# 计算每个输入向量与查询向量的点积
attn_scores_2 = torch.empty(inputs.shape[0])
for i, x_i in enumerate(inputs):
    attn_scores_2[i] = torch.dot(x_i, query) # 计算点积作为注意力分数

print(attn_scores_2)

# 手动计算第一个输入向量与查询向量的点积
res = 0.
for idx, element in enumerate(inputs[0]):
    res += inputs[0][idx] * query[idx]
print(res)
print(torch.dot(inputs[0], query)) # 使用torch.dot进行验证

# 对注意力分数进行归一化，得到注意力权重
attn_weights_2_tmp = attn_scores_2 / attn_scores_2.sum()
print("Attention weights:", attn_weights_2_tmp)
print("Sum:", attn_weights_2_tmp.sum())

# 定义一个简单的softmax函数
def softmax_naive(x):
    return torch.exp(x) / torch.exp(x).sum(dim=0) # 计算softmax

attn_weights_2_naive = softmax_naive(attn_scores_2)
print("Attention weights:", attn_weights_2_naive)
print("Sum:", attn_weights_2_naive.sum())

# 使用torch的softmax函数计算注意力权重
attn_weights_2 = torch.softmax(attn_scores_2, dim=0)
print("Attention weights:", attn_weights_2)
print("Sum:", attn_weights_2.sum())

# 根据注意力权重计算上下文向量
context_vec_2 = torch.zeros(query.shape)
for i, x_i in enumerate(inputs):
    context_vec_2 += attn_weights_2[i] * x_i # 加权求和得到上下文向量

print(context_vec_2)

# 计算所有输入向量之间的点积，得到注意力分数矩阵
attn_scores = torch.empty(6, 6)
for i, x_i in enumerate(inputs):
    for j, x_j in enumerate(inputs):
        attn_scores[i, j] = torch.dot(x_i, x_j)

print(attn_scores)

# 使用矩阵乘法计算注意力分数矩阵
attn_scores = inputs @ inputs.T
print(attn_scores)

# 对注意力分数矩阵应用softmax，得到注意力权重矩阵
attn_weights = torch.softmax(attn_scores, dim=-1)
print(attn_weights)

# 验证softmax后的一行元素的和
row_2_sum = sum([0.1385, 0.2379, 0.2333, 0.1240, 0.1082, 0.1581])
print("Row 2 sum:", row_2_sum)
print("All row sums:", attn_weights.sum(dim=-1))

# 使用注意力权重矩阵和输入向量计算所有上下文向量
all_context_vecs = attn_weights @ inputs
print(all_context_vecs)

# 打印之前计算的第二个上下文向量以进行比较
print("Previous 2nd context vector:", context_vec_2)

# 设置参数并计算查询、键和值的向量
x_2 = inputs[1] # 第二个输入元素
d_in = inputs.shape[1] # 输入嵌入大小
d_out = 2 # 输出嵌入大小

torch.manual_seed(123) # 设置随机种子

W_query = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False) # 查询矩阵
W_key = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False) # 键矩阵
W_value = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False) # 值矩阵

query_2 = x_2 @ W_query # 计算查询向量
key_2 = x_2 @ W_key # 计算键向量
value_2 = x_2 @ W_value # 计算值向量

print(query_2)

# 计算所有键和值
keys = inputs @ W_key
values = inputs @ W_value

print("keys.shape:", keys.shape)
print("values.shape:", values.shape)

# 计算第二个查询向量与第二个键向量的点积
keys_2 = keys[1]
attn_score_22 = query_2.dot(keys_2)
print(attn_score_22)

# 计算给定查询向量的所有注意力分数
attn_scores_2 = query_2 @ keys.T
print(attn_scores_2)

# 对注意力分数进行缩放并应用softmax
d_k = keys.shape[1]
attn_weights_2 = torch.softmax(attn_scores_2 / d_k**0.5, dim=-1)
print(attn_weights_2)

# 根据注意力权重计算上下文向量
context_vec_2 = attn_weights_2 @ values
print(context_vec_2)

import torch.nn as nn


# 自定义自注意力机制版本1，使用直接定义的权重矩阵
class SelfAttention_v1(nn.Module):

    def __init__(self, d_in, d_out):
        super().__init__()
        # 初始化查询、键、值的权重矩阵
        self.W_query = nn.Parameter(torch.rand(d_in, d_out))  # 查询权重矩阵
        self.W_key = nn.Parameter(torch.rand(d_in, d_out))  # 键权重矩阵
        self.W_value = nn.Parameter(torch.rand(d_in, d_out))  # 值权重矩阵

    def forward(self, x):
        # 计算键、查询和值
        keys = x @ self.W_key
        queries = x @ self.W_query
        values = x @ self.W_value

        # 计算注意力分数
        attn_scores = queries @ keys.T  # omega
        # 应用缩放后的softmax函数得到注意力权重
        attn_weights = torch.softmax(
            attn_scores / keys.shape[-1] ** 0.5, dim=-1
        )

        # 计算上下文向量
        context_vec = attn_weights @ values
        return context_vec


# 初始化随机种子
torch.manual_seed(123)
# 注意：这里d_in和d_out以及inputs变量未在代码片段中定义，
# 假设它们已经被适当地定义和初始化。
sa_v1 = SelfAttention_v1(d_in, d_out)
print(sa_v1(inputs))


# 自定义自注意力机制版本2，使用nn.Linear来定义权重矩阵
class SelfAttention_v2(nn.Module):

    def __init__(self, d_in, d_out, qkv_bias=False):
        super().__init__()
        # 使用nn.Linear定义查询、键、值的线性变换
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)

    def forward(self, x):
        # 计算键、查询和值
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        # 计算注意力分数
        attn_scores = queries @ keys.T
        # 应用缩放后的softmax函数得到注意力权重
        attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim=-1)

        # 计算上下文向量
        context_vec = attn_weights @ values
        return context_vec


torch.manual_seed(789)
sa_v2 = SelfAttention_v2(d_in, d_out)
print(sa_v2(inputs))

# 重复使用SelfAttention_v2对象的查询和键权重矩阵来计算注意力分数
queries = sa_v2.W_query(inputs)
keys = sa_v2.W_key(inputs)
attn_scores = queries @ keys.T

# 应用缩放后的softmax函数得到注意力权重
attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim=-1)
print(attn_weights)

# 创建一个简单的下三角矩阵掩码
context_length = attn_scores.shape[0]
mask_simple = torch.tril(torch.ones(context_length, context_length))
print(mask_simple)

# 应用简单掩码到注意力权重
masked_simple = attn_weights * mask_simple
print(masked_simple)

# 对掩码后的注意力权重进行归一化
row_sums = masked_simple.sum(dim=-1, keepdim=True)
masked_simple_norm = masked_simple / row_sums
print(masked_simple_norm)

# 创建一个上三角矩阵掩码，并将注意力分数中对应位置设置为负无穷
mask = torch.triu(torch.ones(context_length, context_length), diagonal=1)
masked = attn_scores.masked_fill(mask.bool(), -torch.inf)
print(masked)

# 对掩码后的注意力分数应用缩放后的softmax函数
attn_weights = torch.softmax(masked / keys.shape[-1] ** 0.5, dim=-1)
print(attn_weights)

# 使用Dropout层
torch.manual_seed(123)
dropout = torch.nn.Dropout(0.5)  # dropout率为50%
example = torch.ones(6, 6)  # 创建一个全1矩阵

print(dropout(example))

# 对注意力权重应用Dropout
print(dropout(attn_weights))

# 创建一个批次数据
batch = torch.stack((inputs, inputs), dim=0)
print(batch.shape)  # 2个输入，每个输入有6个token，每个token的嵌入维度为3
import torch
import torch.nn as nn

# 定义因果注意力机制类
class CausalAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, qkv_bias=False):
        super().__init__()
        self.d_out = d_out  # 输出维度
        # 定义查询、键、值的线性变换
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key   = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.dropout = nn.Dropout(dropout)  # Dropout层用于防止过拟合
        # 创建一个上三角矩阵作为掩码，用于因果注意力机制，确保只能关注到当前位置之前的元素
        self.register_buffer('mask', torch.triu(torch.ones(context_length, context_length), diagonal=1))

    def forward(self, x):
        b, num_tokens, d_in = x.shape  # 提取输入的形状，b是批次大小，num_tokens是序列长度，d_in是输入维度
        # 通过线性变换得到查询、键、值
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        # 计算注意力分数
        attn_scores = queries @ keys.transpose(1, 2)
        # 使用掩码将上三角区域（包括对角线）的分数设置为负无穷，实现因果注意力
        attn_scores.masked_fill_(self.mask.bool()[:num_tokens, :num_tokens], -torch.inf)
        # 应用缩放softmax得到注意力权重
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        # 应用dropout
        attn_weights = self.dropout(attn_weights)

        # 通过注意力权重加权求和得到上下文向量
        context_vec = attn_weights @ values
        return context_vec

# 定义多头注意力包装器类
class MultiHeadAttentionWrapper(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        # 初始化多个因果注意力头
        self.heads = nn.ModuleList([CausalAttention(d_in, d_out, context_length, dropout, qkv_bias) for _ in range(num_heads)])

    def forward(self, x):
        # 对每个头应用输入，并将结果沿最后一个维度拼接
        return torch.cat([head(x) for head in self.heads], dim=-1)

torch.manual_seed(123)  # 设置随机种子以确保结果可复现
d_in, d_out = 3, 2  # 输入和输出维度
context_length = 5  # 假设序列长度为5
batch = torch.randn(2, context_length, d_in)  # 创建一个随机张量作为示例输入

# 实例化因果注意力模型
ca = CausalAttention(d_in, d_out, context_length, 0.0)
context_vecs = ca(batch)
print(context_vecs)
print("context_vecs.shape:", context_vecs.shape)

# 实例化多头注意力包装器模型
mha = MultiHeadAttentionWrapper(d_in, d_out, context_length, 0.0, num_heads=2)
context_vecs = mha(batch)
print(context_vecs)
print("context_vecs.shape:", context_vecs.shape)

import torch
import torch.nn as nn


# 定义多头注意力机制类
class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        # 确保输出维度d_out可以被头数num_heads整除
        assert (d_out % num_heads == 0), "d_out must be divisible by num_heads"

        self.d_out = d_out  # 输出维度
        self.num_heads = num_heads  # 头数
        self.head_dim = d_out // num_heads  # 每个头的维度

        # 定义查询、键、值的线性变换
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        # 定义输出投影层，用于合并各个头的输出
        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)  # Dropout层用于防止过拟合
        # 创建一个上三角矩阵作为掩码，用于因果注意力机制，确保只能关注到当前位置之前的元素
        self.register_buffer(
            "mask",
            torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )

    def forward(self, x):
        b, num_tokens, d_in = x.shape  # 提取输入的形状，b是批次大小，num_tokens是序列长度，d_in是输入维度

        # 通过线性变换得到查询、键、值
        keys = self.W_key(x)  # 形状：(b, num_tokens, d_out)
        queries = self.W_query(x)
        values = self.W_value(x)

        # 隐式地通过增加`num_heads`维度来分割矩阵
        # 将最后一个维度展开：(b, num_tokens, d_out) -> (b, num_tokens, num_heads, head_dim)
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)

        # 转置：(b, num_tokens, num_heads, head_dim) -> (b, num_heads, num_tokens, head_dim)
        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        # 计算带有因果掩码的缩放点积注意力（也称为自注意力）
        attn_scores = queries @ keys.transpose(2, 3)  # 为每个头计算点积

        # 将原始掩码截断为与令牌数相同的大小，并转换为布尔类型
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]

        # 使用掩码填充注意力分数
        attn_scores.masked_fill_(mask_bool, -torch.inf)

        # 计算注意力权重，并应用Dropout
        attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # 形状：(b, num_heads, num_tokens, head_dim)
        context_vec = (attn_weights @ values).transpose(1, 2)  # 计算上下文向量

        # 合并头，其中self.d_out = self.num_heads * self.head_dim
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)
        # 可选投影层
        context_vec = self.out_proj(context_vec)

        return context_vec


# 以下是测试代码，用于验证MultiHeadAttention类的功能
torch.manual_seed(123)

# 假设batch是一个已经存在的张量，具有形状(batch_size, context_length, d_in)
batch_size, context_length, d_in = batch.shape
d_out = 2
mha = MultiHeadAttention(d_in, d_out, context_length, 0.0, num_heads=2)

context_vecs = mha(batch)

print(context_vecs)
print("context_vecs.shape:", context_vecs.shape)

# 示例张量a，用于演示矩阵乘法
a = torch.tensor([[[[0.2745, 0.6584, 0.2775, 0.8573],
                    [0.8993, 0.0390, 0.9268, 0.7388],
                    [0.7179, 0.7058, 0.9156, 0.4340]],

                   [[0.0772, 0.3565, 0.1479, 0.5331],
                    [0.4066, 0.2318, 0.4545, 0.9737],
                    [0.4606, 0.5159, 0.4220, 0.5786]]]])

# 演示矩阵a与其转置的矩阵乘法
print(a @ a.transpose(2, 3))

# 演示第一个头的矩阵乘法
first_head = a[0, 0, :, :]
first_res = first_head @ first_head.T
print("First head:\n", first_res)

# 演示第二个头的矩阵乘法
second_head = a[0, 1, :, :]
second_res = second_head @ second_head.T
print("\nSecond head:\n", second_res)