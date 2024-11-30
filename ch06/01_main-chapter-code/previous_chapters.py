import numpy as np
import tiktoken
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

#####################################
# Chapter 2
#####################################

# GPT-2数据集类，用于生成输入和目标token序列
class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []  # 存储输入token序列的列表
        self.target_ids = []  # 存储目标token序列的列表

        # 使用分词器对文本进行编码，并允许特殊标记"<|endoftext|>"
        token_ids = tokenizer.encode(txt, allowed_special={"<|endoftext|>"})

        # 通过滑动窗口生成输入和目标token序列
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i + max_length]  # 输入序列
            target_chunk = token_ids[i + 1: i + max_length + 1]  # 目标序列（通常是输入序列右移一位）
            self.input_ids.append(torch.tensor(input_chunk))  # 将输入序列转换为tensor并添加到列表中
            self.target_ids.append(torch.tensor(target_chunk))  # 将目标序列转换为tensor并添加到列表中

    def __len__(self):
        """
        返回数据集的大小（即样本数量）
        """
        return len(self.input_ids)

    def __getitem__(self, idx):
        """
        根据索引获取样本
        :param idx: 样本的索引
        :return: 输入token序列和目标token序列
        """
        return self.input_ids[idx], self.target_ids[idx]


# 创建GPT-2数据加载器
def create_dataloader_v1(txt, batch_size=4, max_length=256,
                         stride=128, shuffle=True, drop_last=True, num_workers=0):
    tokenizer = tiktoken.get_encoding("gpt2")  # 获取GPT-2模型的分词器

    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)  # 创建GPT-2数据集

    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, num_workers=num_workers)  # 创建数据加载器

    return dataloader

#####################################
# Chapter 3
#####################################

class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        # 确保输出维度可以被头的数量整除
        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads  # 每个头的维度

        # 定义查询、键、值的线性投影层
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        # 定义输出投影层（可选）
        self.out_proj = nn.Linear(d_out, d_out)
        # 定义Dropout层
        self.dropout = nn.Dropout(dropout)
        # 注册一个缓冲区，用于存储上三角矩阵掩码，以避免对后续位置进行注意力计算
        self.register_buffer('mask', torch.triu(torch.ones(context_length, context_length), diagonal=1))

    def forward(self, x):
        b, num_tokens, d_in = x.shape

        # 计算查询、键、值
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        # 调整形状以匹配头的数量
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)

        # 转置以匹配维度进行矩阵乘法
        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        # 计算注意力分数
        attn_scores = queries @ keys.transpose(2, 3)

        # 应用掩码以避免对后续位置进行注意力计算
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]
        attn_scores.masked_fill_(mask_bool, -torch.inf)

        # 计算注意力权重并应用Dropout
        attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # 计算上下文向量
        context_vec = (attn_weights @ values).transpose(1, 2)

        # 调整形状以匹配输出维度
        context_vec = context_vec.reshape(b, num_tokens, self.d_out)
        # 应用输出投影层（可选）
        context_vec = self.out_proj(context_vec)

        return context_vec


#####################################
# Chapter 4
#####################################

# 层归一化模块
class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5  # 防止除零的小常数
        self.scale = nn.Parameter(torch.ones(emb_dim))  # 归一化后的缩放因子
        self.shift = nn.Parameter(torch.zeros(emb_dim))  # 归一化后的平移因子

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)  # 计算均值
        var = x.var(dim=-1, keepdim=True, unbiased=False)  # 计算方差
        norm_x = (x - mean) / torch.sqrt(var + self.eps)  # 归一化
        return self.scale * norm_x + self.shift  # 应用缩放和平移


# GELU激活函数模块
class GELU(nn.Module):
    def __init__(self):
        """
        初始化GELU激活函数模块
        """
        super().__init__()

    def forward(self, x):
        """
        前向传播函数

        :param x: 输入张量
        :return: GELU激活后的张量
        """
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) *
            (x + 0.044715 * torch.pow(x, 3))
        ))


# 前馈神经网络模块
class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),  # 第一个线性层
            GELU(),  # GELU激活函数
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]),  # 第二个线性层
        )

    def forward(self, x):
        """
        前向传播函数

        :param x: 输入张量
        :return: 前馈神经网络输出张量
        """
        return self.layers(x)


# Transformer块模块
class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        """
        初始化Transformer块模块

        :param cfg: 配置字典，包含嵌入维度、头数等参数
        """
        super().__init__()
        self.att = MultiHeadAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            context_length=cfg["context_length"],
            num_heads=cfg["n_heads"],
            dropout=cfg["drop_rate"],
            qkv_bias=cfg["qkv_bias"]
        )  # 多头注意力模块
        self.ff = FeedForward(cfg)  # 前馈神经网络模块
        self.norm1 = LayerNorm(cfg["emb_dim"])  # 第一个层归一化模块
        self.norm2 = LayerNorm(cfg["emb_dim"])  # 第二个层归一化模块
        self.drop_resid = nn.Dropout(cfg["drop_rate"])  # Dropout模块

    def forward(self, x):
        shortcut = x  # 残差连接
        x = self.norm1(x)  # 第一个层归一化
        x = self.att(x)  # 多头注意力
        x = self.drop_resid(x)  # Dropout
        x = x + shortcut  # 残差连接

        shortcut = x  # 第二个残差连接
        x = self.norm2(x)  # 第二个层归一化
        x = self.ff(x)  # 前馈神经网络
        x = self.drop_resid(x)  # Dropout
        x = x + shortcut  # 残差连接

        return x


# GPT模型模块
class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])  # 词汇嵌入
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])  # 位置嵌入
        self.drop_emb = nn.Dropout(cfg["drop_rate"])  # Dropout模块

        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])]  # 多个Transformer块
        )

        self.final_norm = LayerNorm(cfg["emb_dim"])  # 最后的层归一化
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)  # 输出头

    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape  # 批量大小和序列长度
        tok_embeds = self.tok_emb(in_idx)  # 词汇嵌入
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))  # 位置嵌入
        x = tok_embeds + pos_embeds  # 词汇嵌入和位置嵌入相加
        x = self.drop_emb(x)  # Dropout
        x = self.trf_blocks(x)  # 通过多个Transformer块
        x = self.final_norm(x)  # 最后的层归一化
        logits = self.out_head(x)  # 输出头
        return logits


# 简单的文本生成函数
def generate_text_simple(model, idx, max_new_tokens, context_size):
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]  # 获取当前上下文

        with torch.no_grad():  # 不计算梯度
            logits = model(idx_cond)  # 获取对数几率

        logits = logits[:, -1, :]  # 取最后一个位置的对数几率

        idx_next = torch.argmax(logits, dim=-1, keepdim=True)  # 选择具有最高对数几率的令牌

        idx = torch.cat((idx, idx_next), dim=1)  # 将新令牌添加到输出中

    return idx


#####################################
# Chapter 5
#####################################

# 权重赋值函数
def assign(left, right):
    if left.shape != right.shape:
        raise ValueError(f"形状不匹配。左侧: {left.shape}, 右侧: {right.shape}")
    return torch.nn.Parameter(torch.tensor(right))


# 将预训练权重加载到GPT模型中
def load_weights_into_gpt(gpt, params):
    gpt.pos_emb.weight = assign(gpt.pos_emb.weight, params['wpe'])  # 位置嵌入权重
    gpt.tok_emb.weight = assign(gpt.tok_emb.weight, params['wte'])  # 令牌嵌入权重

    for b in range(len(params["blocks"])):  # 遍历所有Transformer块
        # 拆分注意力权重和偏置
        q_w, k_w, v_w = np.split(
            (params["blocks"][b]["attn"]["c_attn"])["w"], 3, axis=-1)
        gpt.trf_blocks[b].att.W_query.weight = assign(
            gpt.trf_blocks[b].att.W_query.weight, q_w.T)  # 查询权重
        gpt.trf_blocks[b].att.W_key.weight = assign(
            gpt.trf_blocks[b].att.W_key.weight, k_w.T)  # 键权重
        gpt.trf_blocks[b].att.W_value.weight = assign(
            gpt.trf_blocks[b].att.W_value.weight, v_w.T)  # 值权重

        q_b, k_b, v_b = np.split(
            (params["blocks"][b]["attn"]["c_attn"])["b"], 3, axis=-1)
        gpt.trf_blocks[b].att.W_query.bias = assign(
            gpt.trf_blocks[b].att.W_query.bias, q_b)  # 查询偏置
        gpt.trf_blocks[b].att.W_key.bias = assign(
            gpt.trf_blocks[b].att.W_key.bias, k_b)  # 键偏置
        gpt.trf_blocks[b].att.W_value.bias = assign(
            gpt.trf_blocks[b].att.W_value.bias, v_b)  # 值偏置

        gpt.trf_blocks[b].att.out_proj.weight = assign(
            gpt.trf_blocks[b].att.out_proj.weight,
            params["blocks"][b]["attn"]["c_proj"]["w"].T)  # 输出投影权重
        gpt.trf_blocks[b].att.out_proj.bias = assign(
            gpt.trf_blocks[b].att.out_proj.bias,
            params["blocks"][b]["attn"]["c_proj"]["b"])  # 输出投影偏置

        gpt.trf_blocks[b].ff.layers[0].weight = assign(
            gpt.trf_blocks[b].ff.layers[0].weight,
            params["blocks"][b]["mlp"]["c_fc"]["w"].T)  # 前馈网络第一层权重
        gpt.trf_blocks[b].ff.layers[0].bias = assign(
            gpt.trf_blocks[b].ff.layers[0].bias,
            params["blocks"][b]["mlp"]["c_fc"]["b"])  # 前馈网络第一层偏置
        gpt.trf_blocks[b].ff.layers[2].weight = assign(
            gpt.trf_blocks[b].ff.layers[2].weight,
            params["blocks"][b]["mlp"]["c_proj"]["w"].T)  # 前馈网络第二层权重（输出层）
        gpt.trf_blocks[b].ff.layers[2].bias = assign(
            gpt.trf_blocks[b].ff.layers[2].bias,
            params["blocks"][b]["mlp"]["c_proj"]["b"])  # 前馈网络第二层偏置（输出层）

        gpt.trf_blocks[b].norm1.scale = assign(
            gpt.trf_blocks[b].norm1.scale,
            params["blocks"][b]["ln_1"]["g"])  # 第一层LayerNorm的scale
        gpt.trf_blocks[b].norm1.shift = assign(
            gpt.trf_blocks[b].norm1.shift,
            params["blocks"][b]["ln_1"]["b"])  # 第一层LayerNorm的shift
        gpt.trf_blocks[b].norm2.scale = assign(
            gpt.trf_blocks[b].norm2.scale,
            params["blocks"][b]["ln_2"]["g"])  # 第二层LayerNorm的scale
        gpt.trf_blocks[b].norm2.shift = assign(
            gpt.trf_blocks[b].norm2.shift,
            params["blocks"][b]["ln_2"]["b"])  # 第二层LayerNorm的shift

    gpt.final_norm.scale = assign(gpt.final_norm.scale, params["g"])  # 最终LayerNorm的scale
    gpt.final_norm.shift = assign(gpt.final_norm.shift, params["b"])  # 最终LayerNorm的shift
    gpt.out_head.weight = assign(gpt.out_head.weight, params["wte"])  # 输出头的权重（通常与令牌嵌入共享）


# 文本转换为令牌ID
def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text, allowed_special={'<|endoftext|>'})  # 编码文本
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)  # 将编码转换为张量并增加批次维度
    return encoded_tensor


# 令牌ID转换为文本
def token_ids_to_text(token_ids, tokenizer):

    flat = token_ids.squeeze(0)  # 移除批次维度
    return tokenizer.decode(flat.tolist())  # 解码令牌ID为文本