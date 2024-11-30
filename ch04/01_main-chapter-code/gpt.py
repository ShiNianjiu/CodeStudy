import tiktoken
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

#####################################
# Chapter 2
#####################################

# 定义GPT数据集类，用于处理文本数据并生成输入和目标的张量
class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []  # 存储输入块的ID列表
        self.target_ids = []  # 存储目标块的ID列表

        # 使用分词器对文本进行编码，并处理特殊标记
        token_ids = tokenizer.encode(txt, allowed_special={"<|endoftext|>"})

        # 遍历文本，按步长划分输入块和目标块
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i + max_length]  # 输入块
            target_chunk = token_ids[i + 1: i + max_length + 1]  # 目标块（输入块的下一个token）
            self.input_ids.append(torch.tensor(input_chunk))  # 将输入块转换为张量并添加到列表中
            self.target_ids.append(torch.tensor(target_chunk))  # 将目标块转换为张量并添加到列表中

    def __len__(self):
        """
        返回数据集中的样本数量。
        """
        return len(self.input_ids)

    def __getitem__(self, idx):
        """
        根据索引获取数据集中的样本。

        参数:
        idx (int): 样本的索引。

        返回:
        tuple: 包含输入块和目标块的张量。
        """
        return self.input_ids[idx], self.target_ids[idx]


# 创建GPT数据加载器的函数
def create_dataloader_v1(txt, batch_size=4, max_length=256,
                         stride=128, shuffle=True, drop_last=True, num_workers=0):
    # 获取GPT-2的分词器
    tokenizer = tiktoken.get_encoding("gpt2")  # 注意：这里假设tiktoken库有一个get_encoding函数来获取GPT-2的分词器

    # 创建GPT数据集
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)

    # 创建数据加载器
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, num_workers=num_workers)

    return dataloader

#####################################
# Chapter 3
#####################################

class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        """
        初始化多头注意力模块。

        参数:
        d_in (int): 输入特征的维度。
        d_out (int): 输出特征的维度，也是每个头的输出维度之和。
        context_length (int): 上下文序列的长度，用于生成注意力掩码。
        dropout (float): Dropout比率，用于防止过拟合。
        num_heads (int): 多头注意力的头数。
        qkv_bias (bool): 是否在查询、键和值的线性变换中添加偏置项。
        """
        super().__init__()
        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"  # 确保d_out能被num_heads整除

        self.d_out = d_out  # 输出特征的维度
        self.num_heads = num_heads  # 多头注意力的头数
        self.head_dim = d_out // num_heads  # 每个头的输出维度

        # 定义查询、键和值的线性变换
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        # 定义输出投影层
        self.out_proj = nn.Linear(d_out, d_out)
        # 定义Dropout层
        self.dropout = nn.Dropout(dropout)
        # 注册一个不可训练的缓冲区，用于存储注意力掩码（上三角矩阵，用于防止自注意力）
        self.register_buffer("mask", torch.triu(torch.ones(context_length, context_length), diagonal=1))

    def forward(self, x):
        """
        前向传播函数。

        参数:
        x (torch.Tensor): 输入张量，形状为(batch_size, num_tokens, d_in)。

        返回:
        torch.Tensor: 输出张量，形状为(batch_size, num_tokens, d_out)。
        """
        b, num_tokens, d_in = x.shape  # 获取输入张量的形状

        # 计算查询、键和值
        keys = self.W_key(x)  # 形状: (b, num_tokens, d_out)
        queries = self.W_query(x)  # 形状: (b, num_tokens, d_out)
        values = self.W_value(x)  # 形状: (b, num_tokens, d_out)

        # 重塑查询、键和值，以添加头维度
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)  # 形状: (b, num_tokens, num_heads, head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)  # 形状: (b, num_tokens, num_heads, head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)  # 形状: (b, num_tokens, num_heads, head_dim)

        # 转置查询、键和值，以将头维度移动到第二个维度
        keys = keys.transpose(1, 2)  # 形状: (b, num_heads, num_tokens, head_dim)
        queries = queries.transpose(1, 2)  # 形状: (b, num_heads, num_tokens, head_dim)
        values = values.transpose(1, 2)  # 形状: (b, num_heads, num_tokens, head_dim)

        # 计算注意力得分
        attn_scores = queries @ keys.transpose(2, 3)  # 形状: (b, num_heads, num_tokens, num_tokens)

        # 获取注意力掩码，并将其转换为布尔类型
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]  # 形状: (num_tokens, num_tokens)

        # 将掩码应用于注意力得分，将上三角区域的值设置为负无穷大
        attn_scores.masked_fill_(mask_bool, -torch.inf)

        # 计算注意力权重，并应用Dropout
        attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim=-1)  # 缩放因子为sqrt(head_dim)
        attn_weights = self.dropout(attn_weights)  # 应用Dropout

        # 计算上下文向量
        context_vec = (attn_weights @ values).transpose(1,2)  # 形状: (b, num_tokens, num_heads, head_dim) -> (b, num_tokens, d_out)

        # 重塑上下文向量，以去除头维度，并应用输出投影层
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)  # 形状: (b, num_tokens, d_out)
        context_vec = self.out_proj(context_vec)  # 可选的投影层

        return context_vec  # 返回输出张量

#####################################
# Chapter 4
#####################################

# 层归一化类
class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()  # 初始化父类
        self.eps = 1e-5  # 防止除零的小常数
        self.scale = nn.Parameter(torch.ones(emb_dim))  # 可学习的缩放参数
        self.shift = nn.Parameter(torch.zeros(emb_dim))  # 可学习的平移参数

    def forward(self, x):
        # 计算均值和方差
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        # 归一化
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        # 应用可学习的缩放和平移
        return self.scale * norm_x + self.shift

# GELU激活函数类
class GELU(nn.Module):
    def __init__(self):
        super().__init__()  # 初始化父类

    def forward(self, x):
        # GELU激活函数公式
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) *
            (x + 0.044715 * torch.pow(x, 3))
        ))

# 前馈神经网络类
class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()  # 初始化父类
        # 定义前馈网络层
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),  # 输入层到隐藏层
            GELU(),  # GELU激活函数
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"])  # 隐藏层到输出层
        )

    def forward(self, x):
        # 前向传播
        return self.layers(x)

# Transformer块类
class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()  # 初始化父类
        # 定义多头注意力机制
        self.att = MultiHeadAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            context_length=cfg["context_length"],
            num_heads=cfg["n_heads"],
            dropout=cfg["drop_rate"],
            qkv_bias=cfg["qkv_bias"]
        )
        # 定义前馈神经网络
        self.ff = FeedForward(cfg)
        # 定义两个层归一化
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])
        # 定义dropout层
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])

    def forward(self, x):
        # 多头注意力机制的前向传播，带有残差连接和层归一化
        shortcut = x
        x = self.norm1(x)
        x = self.att(x)
        x = self.drop_shortcut(x)
        x = x + shortcut

        # 前馈神经网络的前向传播，带有残差连接和层归一化
        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut

        return x

# GPT模型类
class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()  # 初始化父类
        # 定义词嵌入层
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        # 定义位置嵌入层
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        # 定义dropout层
        self.drop_emb = nn.Dropout(cfg["drop_rate"])

        # 定义多个Transformer块
        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])]
        )

        # 定义最终的层归一化
        self.final_norm = LayerNorm(cfg["emb_dim"])
        # 定义输出层
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

    def forward(self, in_idx):
        # 获取输入的形状
        batch_size, seq_len = in_idx.shape
        # 计算词嵌入和位置嵌入
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        # 将词嵌入和位置嵌入相加
        x = tok_embeds + pos_embeds
        # 应用dropout
        x = self.drop_emb(x)
        # 经过多个Transformer块
        x = self.trf_blocks(x)
        # 应用最终的层归一化
        x = self.final_norm(x)
        # 通过输出层得到logits
        logits = self.out_head(x)
        return logits

# 简单的文本生成函数
def generate_text_simple(model, idx, max_new_tokens, context_size):
    # 循环生成新的token，直到达到最大新token数
    for _ in range(max_new_tokens):
        # 取当前序列的最后context_size个token作为条件
        idx_cond = idx[:, -context_size:]

        # 在不计算梯度的情况下进行前向传播
        with torch.no_grad():
            logits = model(idx_cond)

        # 取最后一个位置的logits
        logits = logits[:, -1, :]

        # 选择具有最大logits值的token作为下一个token
        idx_next = torch.argmax(logits, dim=-1, keepdim=True)  # (batch, 1)

        # 将下一个token添加到序列中
        idx = torch.cat((idx, idx_next), dim=1)  # (batch, n_tokens+1)

    return idx

# GPT模型的配置字典
GPT_CONFIG_124M = {
    "vocab_size": 50257,  # GPT-2 124M模型的词汇表大小
    "context_length": 1024,  # GPT-2 124M模型的最大上下文长度
    "emb_dim": 768,  # 嵌入维度
    "n_heads": 12,  # 多头注意力机制中的头数
    "n_layers": 12,  # Transformer块的层数
    "drop_rate": 0.1,  # Dropout比率
    "qkv_bias": False  # 是否在查询、键、值投影中添加偏置项
}


def main():
    # 设置随机种子以确保结果的可重复性
    torch.manual_seed(123)

    # 初始化GPT模型
    model = GPTModel(GPT_CONFIG_124M)
    # 将模型设置为评估模式（即推理模式）
    model.eval()

    # 输入文本
    start_context = "Hello, I am"

    # 使用GPT-2的编码器将文本编码为索引
    # 注意：这里假设tiktoken.get_encoding("gpt2")返回一个GPT-2特定的编码器
    tokenizer = tiktoken.get_encoding("gpt2")
    encoded = tokenizer.encode(start_context)
    # 将编码后的索引转换为Tensor，并增加一个batch维度
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)

    # 打印输入信息
    print(f"\n{50 * '='}\n{22 * ' '}IN\n{50 * '='}")
    print("\nInput text:", start_context)
    print("Encoded input text:", encoded)
    print("encoded_tensor.shape:", encoded_tensor.shape)

    # 使用模型生成文本
    out = generate_text_simple(
        model=model,  # GPT模型
        idx=encoded_tensor,  # 编码后的输入文本Tensor
        max_new_tokens=10,  # 要生成的最大新token数
        context_size=GPT_CONFIG_124M["context_length"]
        # 上下文长度（这里实际上在generate_text_simple中未使用完整上下文长度，而是用了最后context_size个token作为条件）
    )
    # 使用解码器将生成的索引解码为文本
    decoded_text = tokenizer.decode(out.squeeze(0).tolist())

    # 打印输出信息
    print(f"\n\n{50 * '='}\n{22 * ' '}OUT\n{50 * '='}")
    print("\nOutput:", out)
    print("Output length:", len(out[0]))  # 注意：这里len(out[0])可能不准确，因为out是一个二维Tensor，通常应该使用out.shape[1]来获取长度
    print("Output text:", decoded_text)


# 程序入口
if __name__ == "__main__":
    main()