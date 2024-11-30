import tiktoken
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt


#####################################
# Chapter 2
#####################################

# 定义一个用于GPT模型的数据集类
class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []  # 存储输入序列的ID
        self.target_ids = []  # 存储目标序列的ID（通常是输入序列右移一位）

        # 使用tokenizer对整个文本进行编码
        token_ids = tokenizer.encode(txt, allowed_special={"<|endoftext|>"})

        # 使用滑动窗口将文本分割成重叠的序列，每个序列的最大长度为max_length
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i + max_length]  # 输入序列
            target_chunk = token_ids[i + 1: i + max_length + 1]  # 目标序列（输入序列右移一位）
            self.input_ids.append(torch.tensor(input_chunk))  # 将输入序列转换为tensor并添加到列表中
            self.target_ids.append(torch.tensor(target_chunk))  # 将目标序列转换为tensor并添加到列表中

    # 返回数据集的大小
    def __len__(self):
        return len(self.input_ids)

    # 根据索引获取单个样本（输入序列和目标序列）
    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]


# 创建一个用于GPT模型的数据加载器函数
def create_dataloader_v1(txt, batch_size=4, max_length=256,
                         stride=128, shuffle=True, drop_last=True, num_workers=0):
    # 初始化GPT-2的tokenizer
    tokenizer = tiktoken.get_encoding("gpt2")

    # 创建GPT数据集
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)

    # 创建数据加载器
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, num_workers=num_workers)

    # 返回数据加载器
    return dataloader


#####################################
# Chapter 3
#####################################

class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"  # 确保d_out能被num_heads整除

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads  # 每个头的维度

        # 初始化查询、键、值投影层
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)

        # 初始化输出投影层，用于组合各个头的输出
        self.out_proj = nn.Linear(d_out, d_out)

        # 初始化Dropout层
        self.dropout = nn.Dropout(dropout)

        # 注册一个缓冲区，用于存储自注意力掩码（因果掩码），防止信息泄露到未来时间步
        self.register_buffer('mask', torch.triu(torch.ones(context_length, context_length), diagonal=1))

    def forward(self, x):
        b, num_tokens, d_in = x.shape

        # 计算查询、键、值矩阵
        keys = self.W_key(x)  # 形状: (b, num_tokens, d_out)
        queries = self.W_query(x)
        values = self.W_value(x)

        # 隐式地通过增加`num_heads`维度来分割矩阵
        # 将最后一个维度展开: (b, num_tokens, d_out) -> (b, num_tokens, num_heads, head_dim)
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)

        # 转置: (b, num_tokens, num_heads, head_dim) -> (b, num_heads, num_tokens, head_dim)
        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        # 计算带有因果掩码的缩放点积注意力（即自注意力）
        attn_scores = queries @ keys.transpose(2, 3)  # 对每个头进行点积

        # 将原始掩码截断为与令牌数量相匹配的大小，并转换为布尔类型
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]

        # 使用掩码填充注意力分数
        attn_scores.masked_fill_(mask_bool, -torch.inf)

        # 应用softmax函数计算注意力权重，并进行缩放（防止点积结果过大）
        attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # 计算上下文向量: (b, num_heads, num_tokens, head_dim) -> (b, num_tokens, num_heads, head_dim) -> 组合头输出
        context_vec = (attn_weights @ values).transpose(1, 2)

        # 将各个头的输出重新组合成一个单一的向量
        context_vec = context_vec.reshape(b, num_tokens, self.d_out)

        # 可选投影层，用于进一步处理上下文向量
        context_vec = self.out_proj(context_vec)

        return context_vec


#####################################
# Chapter 4
#####################################

# Layer Normalization 类
class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()  # 初始化父类
        self.eps = 1e-5  # 防止除以零的极小值
        self.scale = nn.Parameter(torch.ones(emb_dim))  # 可学习的缩放参数
        self.shift = nn.Parameter(torch.zeros(emb_dim))  # 可学习的平移参数

    def forward(self, x):
        # 计算均值和方差
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        # 归一化
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        # 应用缩放和平移
        return self.scale * norm_x + self.shift

# GELU 激活函数 类
class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        # GELU 激活函数的实现
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) *
            (x + 0.044715 * torch.pow(x, 3))
        ))

# 前馈神经网络 类
class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        # 定义一个包含两个线性层和GELU激活的前馈网络
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
            GELU(),
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]),
        )

    def forward(self, x):
        # 前向传播
        return self.layers(x)

# Transformer 块 类
class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        # 多头注意力机制
        self.att = MultiHeadAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            context_length=cfg["context_length"],
            num_heads=cfg["n_heads"],
            dropout=cfg["drop_rate"],
            qkv_bias=cfg["qkv_bias"]
        )
        # 前馈神经网络
        self.ff = FeedForward(cfg)
        # 层归一化
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])
        # Dropout 层
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])

    def forward(self, x):
        # 多头注意力机制的前向传播和残差连接
        shortcut = x
        x = self.norm1(x)
        x = self.att(x)
        x = self.drop_shortcut(x)
        x = x + shortcut

        # 前馈神经网络的前向传播和残差连接
        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut

        return x

# GPT 模型 类
class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        # 词嵌入
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        # 位置嵌入
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        # Dropout 层
        self.drop_emb = nn.Dropout(cfg["drop_rate"])

        # 多个Transformer块组成的序列
        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])]
        )

        # 最终的层归一化
        self.final_norm = LayerNorm(cfg["emb_dim"])
        # 输出头，用于生成词汇表的预测
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

    def forward(self, in_idx):
        # 输入索引的形状
        batch_size, seq_len = in_idx.shape
        # 获取词嵌入和位置嵌入
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        # 将词嵌入和位置嵌入相加
        x = tok_embeds + pos_embeds
        # 应用Dropout
        x = self.drop_emb(x)
        # 通过多个Transformer块
        x = self.trf_blocks(x)
        # 应用最终的层归一化
        x = self.final_norm(x)
        # 通过输出头生成logits
        logits = self.out_head(x)
        return logits

# 简单的文本生成函数
def generate_text_simple(model, idx, max_new_tokens, context_size):
    for _ in range(max_new_tokens):
        # 获取当前的输入条件（最后context_size个token）
        idx_cond = idx[:, -context_size:]

        # 在不计算梯度的情况下生成下一个token的logits
        with torch.no_grad():
            logits = model(idx_cond)

        # 获取最后一个token的logits
        logits = logits[:, -1, :]

        # 使用argmax选择最可能的下一个token
        idx_next = torch.argmax(logits, dim=-1, keepdim=True)  # (batch, 1)

        # 将新的token添加到序列中
        idx = torch.cat((idx, idx_next), dim=1)  # (batch, n_tokens+1)

    return idx


#####################################
# Chapter 5
####################################


# 计算单个批次数据的损失
def calc_loss_batch(input_batch, target_batch, model, device):
    # 将输入和目标数据移动到指定的设备上（如CPU或GPU）
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)
    # 通过模型前向传播得到预测结果（logits）
    logits = model(input_batch)
    # 计算交叉熵损失，注意这里将logits和目标数据都进行了展平操作
    loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), target_batch.flatten())
    return loss

# 计算数据加载器中所有（或部分）批次的平均损失
def calc_loss_loader(data_loader, model, device, num_batches=None):
    total_loss = 0.  # 初始化总损失为0
    # 如果数据加载器为空，则返回NaN
    if len(data_loader) == 0:
        return float("nan")
    # 如果没有指定计算损失的批次数量，则计算所有批次的损失
    elif num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))
    # 遍历数据加载器中的每个批次，计算损失并累加
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            total_loss += loss.item()
        else:
            break
    # 返回平均损失
    return total_loss / num_batches

# 评估模型在训练集和验证集上的性能
def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    # 将模型设置为评估模式
    model.eval()
    # 禁用梯度计算
    with torch.no_grad():
        # 计算训练集和验证集上的损失
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
    # 将模型设置回训练模式
    model.train()
    return train_loss, val_loss

# 根据给定的起始上下文生成文本并打印
def generate_and_print_sample(model, tokenizer, device, start_context):
    # 将模型设置为评估模式
    model.eval()
    # 获取位置嵌入的最大长度（即上下文大小）
    context_size = model.pos_emb.weight.shape[0]
    # 将起始上下文编码为token IDs并移动到设备上
    encoded = text_to_token_ids(start_context, tokenizer).to(device)
    # 禁用梯度计算，生成文本
    with torch.no_grad():
        # 使用模型生成文本
        token_ids = generate_text_simple(
            model=model, idx=encoded,
            max_new_tokens=50, context_size=context_size)
        # 将生成的token IDs解码为文本
        decoded_text = token_ids_to_text(token_ids, tokenizer)
        # 打印解码后的文本，将换行符替换为空格以紧凑显示
        print(decoded_text.replace("\n", " "))
    # 将模型设置回训练模式
    model.train()

# 绘制训练和验证损失图
def plot_losses(epochs_seen, tokens_seen, train_losses, val_losses):
    # 创建绘图
    fig, ax1 = plt.subplots(figsize=(5, 3))

    # 在主y轴上绘制训练和验证损失
    ax1.plot(epochs_seen, train_losses, label="Training loss")
    ax1.plot(epochs_seen, val_losses, linestyle="-.", label="Validation loss")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.legend(loc="upper right")

    # 创建共享x轴的第二个y轴，用于显示“看到的令牌数”对应的损失（虽然这里只绘制了框架，未实际显示）
    ax2 = ax1.twiny()
    ax2.plot(tokens_seen, train_losses, alpha=0)  # 这里alpha=0意味着这条线实际上是透明的
    ax2.set_xlabel("Tokens seen")

    # 调整布局以避免重叠
    fig.tight_layout()
    # 显示绘图（如果需要的话，可以取消注释plt.show()）
    # plt.show()

# 将文本编码为token IDs
def text_to_token_ids(text, tokenizer):
    # 使用分词器对文本进行编码
    encoded = tokenizer.encode(text)
    # 将编码结果转换为tensor，并增加一个批次维度
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)
    return encoded_tensor

# 将token IDs解码为文本
def token_ids_to_text(token_ids, tokenizer):
    # 移除批次维度（如果存在）
    flat = token_ids.squeeze(0)
    # 使用分词器对token IDs进行解码
    return tokenizer.decode(flat.tolist())