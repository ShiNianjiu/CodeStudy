import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import tiktoken
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


#####################################
# Chapter 2
#####################################

class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        # 初始化方法，接收文本、分词器、最大长度和步长作为参数
        self.tokenizer = tokenizer  # 存储分词器
        self.input_ids = []  # 初始化输入ID列表
        self.target_ids = []  # 初始化目标ID列表

        # 使用分词器对文本进行编码，允许特殊的<|endoftext|>标记
        token_ids = tokenizer.encode(txt, allowed_special={"<|endoftext|>"})

        # 遍历token_ids列表，根据步长划分输入和目标块
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i + max_length]  # 获取输入块（不包含目标字符的最后一个token）
            target_chunk = token_ids[i + 1: i + max_length + 1]  # 获取目标块（输入块的下一个token）
            # 将输入块和目标块转换为tensor并添加到列表中
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        # 返回数据集中样本的数量
        return len(self.input_ids)

    def __getitem__(self, idx):
        # 根据索引获取数据集中的样本（输入和目标ID）
        return self.input_ids[idx], self.target_ids[idx]


def create_dataloader_v1(txt, batch_size=4, max_length=256,
                         stride=128, shuffle=True, drop_last=True, num_workers=0):
    # 使用GPT2的分词器对文本进行编码
    tokenizer = tiktoken.get_encoding("gpt2")

    # 创建GPTDatasetV1数据集实例
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)

    # 创建DataLoader实例，用于加载数据集
    dataloader = DataLoader(
        dataset,  # 数据集实例
        batch_size=batch_size,  # 每个batch的样本数
        shuffle=shuffle,  # 是否在每个epoch开始时打乱数据
        drop_last=drop_last,  # 是否丢弃最后一个不完整的batch
        num_workers=num_workers  # 加载数据时使用的进程数
    )

    # 返回DataLoader实例
    return dataloader


#####################################
# Chapter 3
#####################################

class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        # 调用父类的构造函数
        super().__init__()

        # 确保输出维度d_out可以被头数num_heads整除
        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"

        # 初始化类的属性
        self.d_out = d_out  # 输出维度
        self.num_heads = num_heads  # 头数
        self.head_dim = d_out // num_heads  # 每个头的维度

        # 初始化查询（Q）、键（K）和值（V）的线性变换矩阵
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)

        # 初始化输出投影矩阵
        self.out_proj = nn.Linear(d_out, d_out)

        # 初始化Dropout层
        self.dropout = nn.Dropout(dropout)

        # 创建一个上三角矩阵作为自注意力机制的掩码，用于防止注意力机制关注到当前位置之前的元素
        # 这个掩码在训练时尤其重要，比如在解码器自注意力中
        self.register_buffer('mask', torch.triu(torch.ones(context_length, context_length), diagonal=1))

    def forward(self, x):
        # 获取输入x的形状：批次大小b，序列长度（或令牌数）num_tokens，输入维度d_in
        b, num_tokens, d_in = x.shape

        # 对输入x应用查询、键和值的线性变换
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        # 重塑变换后的张量，以分离出不同的头
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)

        # 转置键、查询和值的张量，以便头的维度在第一个位置
        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        # 计算注意力分数（即查询和键的点积）
        attn_scores = queries @ keys.transpose(2, 3)

        # 获取掩码，并将其转换为布尔类型，以便在后续步骤中使用
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]

        # 使用掩码将注意力分数中对应的位置设置为负无穷，这样在softmax后这些位置的权重将接近0
        attn_scores.masked_fill_(mask_bool, -torch.inf)

        # 应用缩放softmax来计算注意力权重
        attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim=-1)

        # 应用Dropout层来减少过拟合
        attn_weights = self.dropout(attn_weights)

        # 使用注意力权重来计算上下文向量（即值的加权和）
        context_vec = (attn_weights @ values).transpose(1, 2)

        # 重塑上下文向量，以恢复原始的批次大小和序列长度维度
        context_vec = context_vec.reshape(b, num_tokens, self.d_out)

        # 应用输出投影矩阵
        context_vec = self.out_proj(context_vec)

        # 返回最终的上下文向量
        return context_vec


#####################################
# Chapter 4
#####################################

# 层归一化类
class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5  # 防止除以零的小常数
        self.scale = nn.Parameter(torch.ones(emb_dim))  # 可学习的缩放参数
        self.shift = nn.Parameter(torch.zeros(emb_dim))  # 可学习的偏移参数

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)  # 计算最后一个维度的均值
        var = x.var(dim=-1, keepdim=True, unbiased=False)  # 计算最后一个维度的方差（无偏估计为False）
        norm_x = (x - mean) / torch.sqrt(var + self.eps)  # 归一化
        return self.scale * norm_x + self.shift  # 应用缩放和偏移

# GELU激活函数类
class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        # GELU激活函数的实现
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) *
            (x + 0.044715 * torch.pow(x, 3))
        ))

# 前馈网络类
class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),  # 输入维度到4倍输出维度的线性变换
            GELU(),  # GELU激活函数
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]),  # 4倍输出维度回到原始输出维度的线性变换
        )

    def forward(self, x):
        return self.layers(x)

# Transformer块类
class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att = MultiHeadAttention(  # 多头注意力机制
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            context_length=cfg["context_length"],
            num_heads=cfg["n_heads"],
            dropout=cfg["drop_rate"],
            qkv_bias=cfg["qkv_bias"]
        )
        self.ff = FeedForward(cfg)  # 前馈网络
        self.norm1 = LayerNorm(cfg["emb_dim"])  # 第一个层归一化
        self.norm2 = LayerNorm(cfg["emb_dim"])  # 第二个层归一化
        self.drop_resid = nn.Dropout(cfg["drop_rate"])  # 残差连接的Dropout

    def forward(self, x):
        shortcut = x  # 保存原始输入作为残差连接的短路
        x = self.norm1(x)  # 应用第一个层归一化
        x = self.att(x)  # 通过多头注意力机制
        x = self.drop_resid(x)  # 应用Dropout
        x = x + shortcut  # 残差连接

        shortcut = x  # 再次保存当前输入作为下一个残差连接的短路
        x = self.norm2(x)  # 应用第二个层归一化
        x = self.ff(x)  # 通过前馈网络
        x = self.drop_resid(x)  # 应用Dropout
        x = x + shortcut  # 残差连接

        return x

# GPT模型类
class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])  # 词汇嵌入
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])  # 位置嵌入
        self.drop_emb = nn.Dropout(cfg["drop_rate"])  # 嵌入层的Dropout

        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])])  # 多个Transformer块堆叠

        self.final_norm = LayerNorm(cfg["emb_dim"])  # 最后的层归一化
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)  # 输出头，无偏置项

    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape  # 获取批次大小和序列长度
        tok_embeds = self.tok_emb(in_idx)  # 获取词汇嵌入
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))  # 获取位置嵌入
        x = tok_embeds + pos_embeds  # 将词汇嵌入和位置嵌入相加
        x = self.drop_emb(x)  # 应用Dropout
        x = self.trf_blocks(x)  # 通过Transformer块
        x = self.final_norm(x)  # 应用最后的层归一化
        logits = self.out_head(x)  # 通过输出头得到logits
        return logits

# 简单的文本生成函数
def generate_text_simple(model, idx, max_new_tokens, context_size):
    for _ in range(max_new_tokens):  # 最多生成max_new_tokens个新令牌
        idx_cond = idx[:, -context_size:]  # 取最后context_size个令牌作为条件

        with torch.no_grad():  # 禁用梯度计算
            logits = model(idx_cond)  # 通过模型得到logits

        logits = logits[:, -1, :]  # 取最后一个位置的logits

        idx_next = torch.argmax(logits, dim=-1, keepdim=True)  # 选择logits最大的令牌作为下一个令牌

        idx = torch.cat((idx, idx_next), dim=1)  # 将下一个令牌添加到序列中

    return idx  # 返回生成的序列


#####################################
# Chapter 5
#####################################

# 生成文本的函数，基于给定的模型和上下文
def generate(model, idx, max_new_tokens, context_size, temperature=0.0, top_k=None, eos_id=None):
    # 循环直到生成了max_new_tokens数量的新token或遇到结束符(eos_id)
    for _ in range(max_new_tokens):
        # 提取最新的context_size个token作为条件输入
        idx_cond = idx[:, -context_size:]
        # 在不计算梯度的情况下，通过模型得到下一个token的logits
        with torch.no_grad():
            logits = model(idx_cond)
        # 只取最后一个位置的logits，因为模型输出可能包含多个位置的预测
        logits = logits[:, -1, :]

        # 如果指定了top_k，则只保留logits中最大的top_k个值，其余设置为负无穷
        if top_k is not None:
            top_logits, _ = torch.topk(logits, top_k)
            min_val = top_logits[:, -1]
            logits = torch.where(logits < min_val, torch.tensor(float('-inf')).to(logits.device), logits)

        # 如果指定了temperature，则调整logits的scale，影响概率分布
        if temperature > 0.0:
            logits = logits / temperature
            # 计算softmax概率
            probs = torch.softmax(logits, dim=-1)
            # 根据概率分布随机选择下一个token的索引
            idx_next = torch.multinomial(probs, num_samples=1)
        else:
            # 直接选择具有最大logits值的token索引
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)

        # 如果生成的token是结束符(eos_id)，则停止生成
        if idx_next == eos_id:
            break

        # 将新生成的token索引添加到已有的序列中
        idx = torch.cat((idx, idx_next), dim=1)

    return idx

# 简单训练模型的函数
def train_model_simple(model, train_loader, val_loader, optimizer, device, num_epochs,
                       eval_freq, eval_iter, start_context, tokenizer):
    # 初始化训练、验证损失和已见token数量的列表
    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen, global_step = 0, -1

    # 循环训练指定的epoch数
    for epoch in range(num_epochs):
        model.train()  # 设置模型为训练模式

        # 遍历训练数据加载器
        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()  # 清零梯度
            loss = calc_loss_batch(input_batch, target_batch, model, device)  # 计算损失
            loss.backward()  # 反向传播
            optimizer.step()  # 更新模型参数
            tokens_seen += input_batch.numel()  # 更新已见token数量
            global_step += 1  # 更新全局步数

            # 每隔eval_freq步评估一次模型
            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, device, eval_iter)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)
                print(f"Epoch {epoch+1} (Step {global_step:06d}): "
                      f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}")

        # 生成并打印样本
        generate_and_print_sample(
            model, tokenizer, device, start_context
        )

    return train_losses, val_losses, track_tokens_seen

# 评估模型的函数
def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    model.eval()  # 设置模型为评估模式
    with torch.no_grad():
        # 分别计算训练集和验证集的损失
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
    model.train()  # 恢复模型为训练模式
    return train_loss, val_loss

# 生成并打印样本的函数
def generate_and_print_sample(model, tokenizer, device, start_context):
    model.eval()  # 设置模型为评估模式
    # 获取模型位置嵌入的维度，通常作为上下文大小
    context_size = model.pos_emb.weight.shape[0]
    # 将开始上下文文本编码为token索引
    encoded = text_to_token_ids(start_context, tokenizer).to(device)
    with torch.no_grad():
        # 使用generate函数生成文本
        token_ids = generate_text_simple(
            model=model, idx=encoded,
            max_new_tokens=50, context_size=context_size
        )
        # 将token索引解码为文本
        decoded_text = token_ids_to_text(token_ids, tokenizer)
        # 打印解码后的文本，去除换行符以紧凑格式显示
        print(decoded_text.replace("\n", " "))
    model.train()  # 恢复模型为训练模式

# 辅助函数，用于将一个张量的值赋给另一个，确保两者形状相同
def assign(left, right):
    if left.shape != right.shape:
        raise ValueError(f"Shape mismatch. Left: {left.shape}, Right: {right.shape}")
    return torch.nn.Parameter(torch.tensor(right))


# 将权重加载到GPT模型中
def load_weights_into_gpt(gpt, params):
    # 将位置嵌入的权重赋值给GPT模型
    gpt.pos_emb.weight = assign(gpt.pos_emb.weight, params['wpe'])
    # 将词嵌入的权重赋值给GPT模型
    gpt.tok_emb.weight = assign(gpt.tok_emb.weight, params['wte'])

    # 遍历每个Transformer块
    for b in range(len(params["blocks"])):
        # 分割注意力机制的权重矩阵（查询、键、值）
        q_w, k_w, v_w = np.split(
            (params["blocks"][b]["attn"]["c_attn"])["w"], 3, axis=-1)
        # 为查询、键、值分别设置权重
        gpt.trf_blocks[b].att.W_query.weight = assign(
            gpt.trf_blocks[b].att.W_query.weight, q_w.T)
        gpt.trf_blocks[b].att.W_key.weight = assign(
            gpt.trf_blocks[b].att.W_key.weight, k_w.T)
        gpt.trf_blocks[b].att.W_value.weight = assign(
            gpt.trf_blocks[b].att.W_value.weight, v_w.T)

        # 分割注意力机制的偏置向量（查询、键、值）
        q_b, k_b, v_b = np.split(
            (params["blocks"][b]["attn"]["c_attn"])["b"], 3, axis=-1)
        # 为查询、键、值分别设置偏置
        gpt.trf_blocks[b].att.W_query.bias = assign(
            gpt.trf_blocks[b].att.W_query.bias, q_b)
        gpt.trf_blocks[b].att.W_key.bias = assign(
            gpt.trf_blocks[b].att.W_key.bias, k_b)
        gpt.trf_blocks[b].att.W_value.bias = assign(
            gpt.trf_blocks[b].att.W_value.bias, v_b)

        # 设置注意力输出投影的权重和偏置
        gpt.trf_blocks[b].att.out_proj.weight = assign(
            gpt.trf_blocks[b].att.out_proj.weight,
            params["blocks"][b]["attn"]["c_proj"]["w"].T)
        gpt.trf_blocks[b].att.out_proj.bias = assign(
            gpt.trf_blocks[b].att.out_proj.bias,
            params["blocks"][b]["attn"]["c_proj"]["b"])

        # 设置前馈网络的权重和偏置
        gpt.trf_blocks[b].ff.layers[0].weight = assign(
            gpt.trf_blocks[b].ff.layers[0].weight,
            params["blocks"][b]["mlp"]["c_fc"]["w"].T)
        gpt.trf_blocks[b].ff.layers[0].bias = assign(
            gpt.trf_blocks[b].ff.layers[0].bias,
            params["blocks"][b]["mlp"]["c_fc"]["b"])
        gpt.trf_blocks[b].ff.layers[2].weight = assign(
            gpt.trf_blocks[b].ff.layers[2].weight,
            params["blocks"][b]["mlp"]["c_proj"]["w"].T)
        gpt.trf_blocks[b].ff.layers[2].bias = assign(
            gpt.trf_blocks[b].ff.layers[2].bias,
            params["blocks"][b]["mlp"]["c_proj"]["b"])

        # 设置LayerNorm的缩放和平移参数
        gpt.trf_blocks[b].norm1.scale = assign(
            gpt.trf_blocks[b].norm1.scale,
            params["blocks"][b]["ln_1"]["g"])
        gpt.trf_blocks[b].norm1.shift = assign(
            gpt.trf_blocks[b].norm1.shift,
            params["blocks"][b]["ln_1"]["b"])
        gpt.trf_blocks[b].norm2.scale = assign(
            gpt.trf_blocks[b].norm2.scale,
            params["blocks"][b]["ln_2"]["g"])
        gpt.trf_blocks[b].norm2.shift = assign(
            gpt.trf_blocks[b].norm2.shift,
            params["blocks"][b]["ln_2"]["b"])

    # 设置最终LayerNorm的缩放和平移参数
    gpt.final_norm.scale = assign(gpt.final_norm.scale, params["g"])
    gpt.final_norm.shift = assign(gpt.final_norm.shift, params["b"])
    # 设置输出头的权重（与词嵌入共享）
    gpt.out_head.weight = assign(gpt.out_head.weight, params["wte"])

# 将文本转换为token ID
def text_to_token_ids(text, tokenizer):
    # 使用tokenizer对文本进行编码，并允许特定的特殊token
    encoded = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
    # 将编码转换为tensor，并增加一个维度以匹配模型输入
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)
    return encoded_tensor

# 将token ID转换为文本
def token_ids_to_text(token_ids, tokenizer):
    # 移除tensor的第一个维度
    flat = token_ids.squeeze(0)
    # 使用tokenizer对token ID进行解码
    return tokenizer.decode(flat.tolist())

# 计算一个批次的数据的损失
def calc_loss_batch(input_batch, target_batch, model, device):
    # 将输入和目标数据移动到指定的设备上（如GPU）
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)
    # 通过模型得到logits
    logits = model(input_batch)
    # 计算交叉熵损失
    loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), target_batch.flatten())
    return loss

# 计算数据加载器中所有批次的损失
def calc_loss_loader(data_loader, model, device, num_batches=None):
    total_loss = 0.
    # 如果数据加载器为空，返回NaN
    if len(data_loader) == 0:
        return float("nan")
    # 如果没有指定批次数量，则使用所有数据加载器的批次数量
    elif num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))
    # 遍历数据加载器中的每个批次
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            # 计算当前批次的损失
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            total_loss += loss.item()
        else:
            break
    # 返回平均损失
    return total_loss / num_batches

# 绘制损失曲线
def plot_losses(epochs_seen, tokens_seen, train_losses, val_losses):
    # 创建图形和轴
    fig, ax1 = plt.subplots(figsize=(5, 3))

    # 绘制训练损失和验证损失
    ax1.plot(epochs_seen, train_losses, label="Training loss")
    ax1.plot(epochs_seen, val_losses, linestyle="-.", label="Validation loss")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.legend(loc="upper right")
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))

    # 创建一个共享x轴的第二个y轴，用于显示tokens seen的信息（透明度设为0，不直接显示）
    ax2 = ax1.twiny()
    ax2.plot(tokens_seen, train_losses, alpha=0)
    ax2.set_xlabel("Tokens seen")

    # 调整布局并保存图像
    fig.tight_layout()
    plt.savefig("loss-plot.pdf")
    plt.show()