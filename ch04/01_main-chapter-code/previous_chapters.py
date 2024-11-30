import tiktoken  # 假设tiktoken是一个用于GPT-2文本编码的库
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader  # 导入PyTorch的数据集和数据加载器


# GPT-2数据集类，用于从文本中生成输入和目标token序列
class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        """
        初始化GPT-2数据集。

        参数:
            txt (str): 输入文本。
            tokenizer (callable): 用于编码文本的tokenizer函数。
            max_length (int): 输入序列的最大长度。
            stride (int): 滑动窗口的步长。
        """
        self.input_ids = []
        self.target_ids = []

        # 使用tokenizer将文本编码为token ID序列
        token_ids = tokenizer.encode(txt, allowed_special={"<|endoftext|>"})

        # 使用滑动窗口生成输入和目标序列
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i + max_length]  # 输入序列
            target_chunk = token_ids[i + 1: i + max_length + 1]  # 目标序列（输入序列右移一位）
            self.input_ids.append(torch.tensor(input_chunk))  # 将输入序列转换为Tensor并添加到列表中
            self.target_ids.append(torch.tensor(target_chunk))  # 将目标序列转换为Tensor并添加到列表中

    def __len__(self):
        """
        返回数据集中的样本数量。
        """
        return len(self.input_ids)

    def __getitem__(self, idx):
        """
        根据索引获取数据集中的样本。

        参数:
            idx (int): 样本索引。

        返回:
            tuple: 包含输入和目标Tensor的元组。
        """
        return self.input_ids[idx], self.target_ids[idx]


# 创建GPT-2数据加载器的函数
def create_dataloader_v1(txt, batch_size=4, max_length=256,
                         stride=128, shuffle=True, drop_last=True, num_workers=0):
    """
    创建用于GPT-2训练的数据加载器。

    参数:
        txt (str): 输入文本。
        batch_size (int): 每个批次中的样本数量。
        max_length (int): 输入序列的最大长度。
        stride (int): 滑动窗口的步长。
        shuffle (bool): 是否在每个epoch开始时打乱数据。
        drop_last (bool): 是否丢弃最后一个不完整的批次。
        num_workers (int): 加载数据时使用的进程数。

    返回:
        DataLoader: 用于加载数据的数据加载器。
    """
    tokenizer = tiktoken.get_encoding("gpt2")  # 获取GPT-2的tokenizer

    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)  # 创建GPT-2数据集

    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, num_workers=num_workers)  # 创建数据加载器

    return dataloader


# 多头注意力机制类
class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        """
        初始化多头注意力机制。

        参数:
            d_in (int): 输入特征的维度。
            d_out (int): 输出特征的维度。
            context_length (int): 上下文序列的长度。
            dropout (float): Dropout比率。
            num_heads (int): 注意力头的数量。
            qkv_bias (bool): 是否在查询、键、值投影中添加偏置项。
        """
        super().__init__()
        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"  # 确保输出维度可以被头的数量整除

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads  # 每个头的维度

        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)  # 查询投影矩阵
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)  # 键投影矩阵
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)  # 值投影矩阵
        self.out_proj = nn.Linear(d_out, d_out)  # 可选的输出投影矩阵
        self.dropout = nn.Dropout(dropout)  # Dropout层

        # 创建一个上三角矩阵作为自注意力机制的掩码，以避免注意到未来的token
        self.register_buffer('mask', torch.triu(torch.ones(context_length, context_length), diagonal=1))

    def forward(self, x):
        """
        前向传播函数。

        参数:
            x (Tensor): 输入特征张量，形状为(batch_size, num_tokens, d_in)。

        返回:
            Tensor: 输出特征张量，形状为(batch_size, num_tokens, d_out)。
        """
        b, num_tokens, d_in = x.shape

        # 将输入x分别投影到查询、键和值空间
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        # 重塑以准备多头注意力计算
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)

        # 转置以匹配维度进行矩阵乘法
        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        # 计算注意力得分
        attn_scores = queries @ keys.transpose(2, 3)

        # 应用掩码以避免注意到未来的token
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]
        attn_scores.masked_fill_(mask_bool, -torch.inf)

        # 应用softmax以获得注意力权重
        attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)  # 应用Dropout

        # 计算上下文向量
        context_vec = (attn_weights @ values).transpose(1, 2)

        # 重塑并可选地投影上下文向量
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)
        context_vec = self.out_proj(context_vec)  # 可选投影

        return context_vec