# 导入需要的库
import os
from pathlib import Path
import urllib
import zipfile

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tiktoken
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


#####################################
# Chapter 2：定义GPT数据集和数据加载器
#####################################


class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []  # 存储输入序列的token IDs
        self.target_ids = []  # 存储目标序列的token IDs（通常用于预测下一个token）

        # 使用tokenizer将整个文本编码为token IDs
        token_ids = tokenizer.encode(txt, allowed_special={"<|endoftext|>"})

        # 使用滑动窗口将文本分割成重叠的序列
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i + max_length]  # 输入序列
            target_chunk = token_ids[i + 1: i + max_length + 1]  # 目标序列（预测下一个token）
            self.input_ids.append(torch.tensor(input_chunk))  # 转换为tensor并添加到列表中
            self.target_ids.append(torch.tensor(target_chunk))  # 转换为tensor并添加到列表中

    def __len__(self):
        """
        返回数据集中的样本数量
        """
        return len(self.input_ids)

    def __getitem__(self, idx):
        """
        根据索引获取数据集中的样本
        :param idx: 样本的索引
        :return: 输入序列和目标序列的tensor对
        """
        return self.input_ids[idx], self.target_ids[idx]


def create_dataloader_v1(txt, batch_size=4, max_length=256,
                         stride=128, shuffle=True, drop_last=True):
    # 初始化tokenizer
    tokenizer = tiktoken.get_encoding("gpt2")  # 注意：这里可能是使用transformers库中的GPT2Tokenizer

    # 创建数据集
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)

    # 创建数据加载器
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)

    return dataloader

#####################################
# Chapter 3
#####################################

# 定义一个多头注意力（Multi-Head Attention）类，继承自nn.Module
class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()  # 初始化父类

        # 确保d_out可以被num_heads整除
        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"

        # 初始化输出维度、头数以及每个头的维度
        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads  # 将输出维度均分给每个头

        # 定义三个线性层，用于生成查询（Query）、键（Key）和值（Value）
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)

        # 定义一个线性层，用于将各个头的输出合并
        self.out_proj = nn.Linear(d_out, d_out)

        # 定义Dropout层，用于减少过拟合
        self.dropout = nn.Dropout(dropout)

        # 定义一个上三角矩阵作为因果掩码，用于确保注意力机制的自回归性质
        # 在序列建模中，这意味着当前位置只能关注到之前的位置
        self.register_buffer('mask', torch.triu(torch.ones(context_length, context_length), diagonal=1))

    def forward(self, x):
        # 获取输入的形状：批次大小（b）、序列长度（num_tokens）、输入维度（d_in）
        b, num_tokens, d_in = x.shape

        # 通过线性层生成查询、键和值
        keys = self.W_key(x)  # 形状：(b, num_tokens, d_out)
        queries = self.W_query(x)
        values = self.W_value(x)

        # 通过重塑和转置操作，将输入分割成多个头
        # 将最后一个维度展开，并添加一个新的头维度
        # 形状变换：(b, num_tokens, d_out) -> (b, num_tokens, num_heads, head_dim)
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)

        # 转置操作，将头维度移到序列维度之前
        # 形状变换：(b, num_tokens, num_heads, head_dim) -> (b, num_heads, num_tokens, head_dim)
        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        # 计算缩放点积注意力（scaled dot-product attention），并应用因果掩码
        # 计算每个头的注意力分数
        attn_scores = queries @ keys.transpose(2, 3)  # 点积操作

        # 将原始掩码截断为与序列长度相同，并转换为布尔类型
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]

        # 使用掩码将注意力分数中的对应位置填充为负无穷，以确保这些位置不会被选中
        attn_scores.masked_fill_(mask_bool, -torch.inf)

        # 计算注意力权重，并应用softmax函数和Dropout层
        attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # 计算上下文向量（context vector），并转置回原始形状
        # 形状变换：(b, num_heads, num_tokens, head_dim) -> (b, num_tokens, num_heads, head_dim)
        context_vec = (attn_weights @ values).transpose(1, 2)

        # 将各个头的输出合并回原始的维度
        # 形状变换：(b, num_tokens, num_heads, head_dim) -> (b, num_tokens, d_out)
        context_vec = context_vec.reshape(b, num_tokens, self.d_out)

        # 可选的线性投影层，用于进一步处理合并后的输出
        context_vec = self.out_proj(context_vec)

        # 返回最终的上下文向量
        return context_vec

#####################################
# Chapter 4
#####################################
import torch
import torch.nn as nn

# Layer Normalization 层，用于对输入张量进行归一化处理
class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5  # 防止除以零的小常数
        self.scale = nn.Parameter(torch.ones(emb_dim))  # 可学习的缩放参数
        self.shift = nn.Parameter(torch.zeros(emb_dim))  # 可学习的偏移参数

    def forward(self, x):
        # 计算输入张量x的均值和方差，进行归一化，并应用缩放和偏移
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift

# GELU（Gaussian Error Linear Unit）激活函数
class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        # 实现GELU激活函数
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) *
            (x + 0.044715 * torch.pow(x, 3))
        ))

# 前馈神经网络层，包含两个线性变换和一个GELU激活函数
class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),  # 第一个线性变换
            GELU(),  # GELU激活函数
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"])  # 第二个线性变换
        )

    def forward(self, x):
        # 前向传播，通过定义的层序列
        return self.layers(x)

# Transformer 块，包含多头注意力机制、前馈神经网络、归一化和残差连接
class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att = MultiHeadAttention(  # 假设MultiHeadAttention已经在其他地方定义
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            context_length=cfg["context_length"],
            num_heads=cfg["n_heads"],
            dropout=cfg["drop_rate"],
            qkv_bias=cfg["qkv_bias"]
        )
        self.ff = FeedForward(cfg)  # 前馈神经网络
        self.norm1 = LayerNorm(cfg["emb_dim"])  # 第一个归一化层
        self.norm2 = LayerNorm(cfg["emb_dim"])  # 第二个归一化层
        self.drop_resid = nn.Dropout(cfg["drop_rate"])  # 残差连接后的Dropout

    def forward(self, x):
        # 注意力块的残差连接
        shortcut = x
        x = self.norm1(x)
        x = self.att(x)  # 应用多头注意力
        x = self.drop_resid(x)
        x = x + shortcut  # 添加原始输入

        # 前馈块的残差连接
        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)  # 应用前馈神经网络
        x = self.drop_resid(x)
        x = x + shortcut  # 添加原始输入

        return x

# GPT 模型，包含嵌入层、多个Transformer块、归一化和输出头
class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])  # 词汇嵌入
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])  # 位置嵌入
        self.drop_emb = nn.Dropout(cfg["drop_rate"])  # 嵌入后的Dropout

        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])]  # 多个Transformer块
        )

        self.final_norm = LayerNorm(cfg["emb_dim"])  # 最终的归一化层
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)  # 输出头

    def forward(self, in_idx):
        # 输入索引的形状为 (batch_size, seq_len)
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)  # 词汇嵌入
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))  # 位置嵌入
        x = tok_embeds + pos_embeds  # 词汇嵌入和位置嵌入相加
        x = self.drop_emb(x)  # 应用Dropout
        x = self.trf_blocks(x)  # 通过Transformer块
        x = self.final_norm(x)  # 最终归一化
        logits = self.out_head(x)  # 通过输出头得到logits
        return logits

# 简单的文本生成函数，基于GPT模型
def generate_text_simple(model, idx, max_new_tokens, context_size):
    # idx 是当前上下文中的索引数组，形状为 (B, T)
    for _ in range(max_new_tokens):
        # 如果当前上下文超过支持的上下文大小，则进行裁剪
        idx_cond = idx[:, -context_size:]

        # 获取预测值
        with torch.no_grad():
            logits = model(idx_cond)

        # 只关注最后一个时间步的logits
        logits = logits[:, -1, :]

        # 获取具有最高logits值的词汇索引
        idx_next = torch.argmax(logits, dim=-1, keepdim=True)  # (batch, 1)

        # 将采样得到的索引追加到正在运行的序列中
        idx = torch.cat((idx, idx_next), dim=1)  # (batch, n_tokens+1)

    return idx


#####################################
# Chapter 5
#####################################
# 定义一个函数，用于将右边的tensor赋值给左边的参数，并确保它们的形状匹配
def assign(left, right):
    # 如果左右两边的形状不匹配，则抛出ValueError异常
    if left.shape != right.shape:
        raise ValueError(f"Shape mismatch. Left: {left.shape}, Right: {right.shape}")
    # 将右边的tensor转换为Parameter类型并返回
    return torch.nn.Parameter(torch.tensor(right))


# 定义一个函数，用于将预训练的权重加载到GPT模型中
def load_weights_into_gpt(gpt, params):
    # 为位置嵌入和词嵌入分配权重
    gpt.pos_emb.weight = assign(gpt.pos_emb.weight, params['wpe'])
    gpt.tok_emb.weight = assign(gpt.tok_emb.weight, params['wte'])

    # 遍历每个Transformer块，并为其分配权重
    for b in range(len(params["blocks"])):
        # 分割注意力机制的权重和偏置
        q_w, k_w, v_w = np.split(
            (params["blocks"][b]["attn"]["c_attn"])["w"], 3, axis=-1)
        gpt.trf_blocks[b].att.W_query.weight = assign(
            gpt.trf_blocks[b].att.W_query.weight, q_w.T)
        gpt.trf_blocks[b].att.W_key.weight = assign(
            gpt.trf_blocks[b].att.W_key.weight, k_w.T)
        gpt.trf_blocks[b].att.W_value.weight = assign(
            gpt.trf_blocks[b].att.W_value.weight, v_w.T)

        q_b, k_b, v_b = np.split(
            (params["blocks"][b]["attn"]["c_attn"])["b"], 3, axis=-1)
        gpt.trf_blocks[b].att.W_query.bias = assign(
            gpt.trf_blocks[b].att.W_query.bias, q_b)
        gpt.trf_blocks[b].att.W_key.bias = assign(
            gpt.trf_blocks[b].att.W_key.bias, k_b)
        gpt.trf_blocks[b].att.W_value.bias = assign(
            gpt.trf_blocks[b].att.W_value.bias, v_b)

        # 为注意力机制的输出投影分配权重和偏置
        gpt.trf_blocks[b].att.out_proj.weight = assign(
            gpt.trf_blocks[b].att.out_proj.weight,
            params["blocks"][b]["attn"]["c_proj"]["w"].T)
        gpt.trf_blocks[b].att.out_proj.bias = assign(
            gpt.trf_blocks[b].att.out_proj.bias,
            params["blocks"][b]["attn"]["c_proj"]["b"])

        # 为前馈网络的权重和偏置分配值
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

        # 为两个LayerNorm层分配scale和shift
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

    # 为最终的LayerNorm层和输出头分配权重
    gpt.final_norm.scale = assign(gpt.final_norm.scale, params["g"])
    gpt.final_norm.shift = assign(gpt.final_norm.shift, params["b"])
    gpt.out_head.weight = assign(gpt.out_head.weight, params["wte"])


# 定义一个函数，用于将文本转换为token IDs
def text_to_token_ids(text, tokenizer):
    # 使用tokenizer对文本进行编码，并允许特定的特殊标记
    encoded = tokenizer.encode(text, allowed_special={'<|endoftext|>'})
    # 将编码后的结果转换为tensor，并增加一个batch维度
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)
    return encoded_tensor


# 定义一个函数，用于将token IDs转换为文本
def token_ids_to_text(token_ids, tokenizer):
    # 移除batch维度
    flat = token_ids.squeeze(0)
    # 使用tokenizer对token IDs进行解码
    return tokenizer.decode(flat.tolist())


# 定义一个函数，用于计算数据加载器上的损失
def calc_loss_loader(data_loader, model, device, num_batches=None):
    total_loss = 0.
    # 如果数据加载器为空，则返回NaN
    if len(data_loader) == 0:
        return float("nan")
    # 如果没有指定num_batches，则使用数据加载器的长度
    elif num_batches is None:
        num_batches = len(data_loader)
    else:
        # 如果num_batches超过了数据加载器的长度，则进行调整
        num_batches = min(num_batches, len(data_loader))
    # 遍历数据加载器，计算损失
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            total_loss += loss.item()
        else:
            break
    # 返回平均损失
    return total_loss / num_batches


# 定义一个函数，用于评估模型在训练集和验证集上的性能
def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    # 将模型设置为评估模式
    model.eval()

    # 禁用梯度计算，因为评估阶段不需要进行反向传播
    with torch.no_grad():
        # 计算训练集上的损失
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
        # 计算验证集上的损失
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)

    # 将模型设置回训练模式
    model.train()

    # 返回训练集损失和验证集损失
    return train_loss, val_loss


#####################################
# Chapter 6
#####################################


# 下载并解压垃圾邮件数据集
def download_and_unzip_spam_data(url, zip_path, extracted_path, data_file_path):
    """
    下载并解压垃圾邮件数据集。

    参数:
    url (str): 数据集的下载链接。
    zip_path (str/Path): 压缩文件保存的路径。
    extracted_path (str/Path): 解压后文件保存的目录。
    data_file_path (str/Path): 解压并重命名后的数据文件路径。
    """
    if data_file_path.exists():
        print(f"{data_file_path} 已存在。跳过下载和解压。")
        return

    # 下载文件
    with urllib.request.urlopen(url) as response:
        with open(zip_path, "wb") as out_file:
            out_file.write(response.read())

    # 解压文件
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extracted_path)

    # 重命名文件，添加.tsv扩展名
    original_file_path = Path(extracted_path) / "SMSSpamCollection"
    os.rename(original_file_path, data_file_path)
    print(f"文件已下载并保存为 {data_file_path}")


# 创建平衡数据集
def create_balanced_dataset(df):
    """
    创建平衡的数据集，使“spam”和“ham”的数量相等。

    参数:
    df (pd.DataFrame): 包含标签和文本的DataFrame。

    返回:
    pd.DataFrame: 平衡后的DataFrame。
    """
    num_spam = df[df["Label"] == "spam"].shape[0]
    ham_subset = df[df["Label"] == "ham"].sample(num_spam, random_state=123)
    balanced_df = pd.concat([ham_subset, df[df["Label"] == "spam"]])
    return balanced_df


# 随机拆分数据集
def random_split(df, train_frac, validation_frac):
    """
    随机拆分数据集为训练集、验证集和测试集。

    参数:
    df (pd.DataFrame): 要拆分的DataFrame。
    train_frac (float): 训练集的比例。
    validation_frac (float): 验证集的比例（训练集之后的部分）。

    返回:
    train_df, validation_df, test_df (pd.DataFrame): 拆分后的训练集、验证集和测试集。
    """
    df = df.sample(frac=1, random_state=123).reset_index(drop=True)
    train_end = int(len(df) * train_frac)
    validation_end = train_end + int(len(df) * validation_frac)
    train_df = df[:train_end]
    validation_df = df[train_end:validation_end]
    test_df = df[validation_end:]
    return train_df, validation_df, test_df


# 自定义数据集类，用于加载垃圾邮件数据
class SpamDataset(Dataset):
    def __init__(self, csv_file, tokenizer, max_length=None, pad_token_id=50256):
        """
        初始化SpamDataset。

        参数:
        csv_file (str/Path): 包含文本和标签的CSV文件路径。
        tokenizer (transformers.PreTrainedTokenizer): 用于文本编码的分词器。
        max_length (int, optional): 编码文本的最大长度。如果为None，则使用最长编码文本的长度。
        pad_token_id (int): 用于填充序列的填充令牌ID。
        """
        self.data = pd.read_csv(csv_file)
        self.encoded_texts = [tokenizer.encode(text) for text in self.data["Text"]]
        if max_length is None:
            self.max_length = self._longest_encoded_length()
        else:
            self.max_length = max_length
            self.encoded_texts = [encoded_text[:self.max_length] for encoded_text in self.encoded_texts]
        self.encoded_texts = [encoded_text + [pad_token_id] * (self.max_length - len(encoded_text)) for encoded_text in
                              self.encoded_texts]

    def __getitem__(self, index):
        """
        获取单个样本。

        参数:
        index (int): 样本的索引。

        返回:
        encoded (torch.Tensor): 编码后的文本。
        label (torch.Tensor): 标签。
        """
        encoded = torch.tensor(self.encoded_texts[index], dtype=torch.long)
        label = torch.tensor(self.data.iloc[index]["Label"], dtype=torch.long)
        return encoded, label

    def __len__(self):
        """
        获取数据集中的样本数量。

        返回:
        int: 样本数量。
        """
        return len(self.data)

    def _longest_encoded_length(self):
        """
        计算最长编码文本的长度。

        返回:
        int: 最长编码文本的长度。
        """
        max_length = 0
        for encoded_text in self.encoded_texts:
            if len(encoded_text) > max_length:
                max_length = len(encoded_text)
        return max_length

# 使用@torch.no_grad()装饰器，表示在此函数内部不进行梯度计算，通常用于推理或评估阶段
@torch.no_grad()
def calc_accuracy_loader(data_loader, model, device, num_batches=None):
    model.eval()  # 将模型设置为评估模式
    correct_predictions, num_examples = 0, 0

    # 确定要评估的批次数量
    if num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))

    # 遍历数据加载器中的批次
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            # 将数据移至指定设备
            input_batch, target_batch = input_batch.to(device), target_batch.to(device)
            # 获取模型输出的最后一个token的logits
            logits = model(input_batch)[:, -1, :]
            # 预测标签为logits中最大值的索引
            predicted_labels = torch.argmax(logits, dim=-1)

            # 更新正确预测的数量和总样本数量
            num_examples += predicted_labels.shape[0]
            correct_predictions += (predicted_labels == target_batch).sum().item()
        else:
            break
    # 返回准确率
    return correct_predictions / num_examples


def calc_loss_batch(input_batch, target_batch, model, device):
    # 将数据移至指定设备
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)
    # 获取模型输出的最后一个token的logits
    logits = model(input_batch)[:, -1, :]
    # 计算交叉熵损失
    loss = torch.nn.functional.cross_entropy(logits, target_batch)
    return loss


def train_classifier_simple(model, train_loader, val_loader, optimizer, device, num_epochs,
                            eval_freq, eval_iter):
    # 初始化列表来存储训练过程中的损失和准确率
    train_losses, val_losses, train_accs, val_accs = [], [], [], []
    examples_seen, global_step = 0, -1

    # 遍历训练轮数
    for epoch in range(num_epochs):
        model.train()  # 将模型设置为训练模式

        # 遍历训练数据加载器中的批次
        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()  # 清空梯度
            loss = calc_loss_batch(input_batch, target_batch, model, device)  # 计算损失
            loss.backward()  # 反向传播
            optimizer.step()  # 更新模型参数
            examples_seen += input_batch.shape[0]  # 更新看到的样本数量
            global_step += 1  # 更新全局步数

            # 每隔eval_freq步进行一次评估
            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(  # 注意：evaluate_model函数未在代码中定义
                    model, train_loader, val_loader, device, eval_iter)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                print(f"Epoch {epoch+1} (Step {global_step:06d}): "
                      f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}")

        # 在每个训练轮结束时计算训练集和验证集上的准确率
        train_accuracy = calc_accuracy_loader(train_loader, model, device, num_batches=eval_iter)
        val_accuracy = calc_accuracy_loader(val_loader, model, device, num_batches=eval_iter)
        print(f"Training accuracy: {train_accuracy*100:.2f}% | ", end="")
        print(f"Validation accuracy: {val_accuracy*100:.2f}%")
        train_accs.append(train_accuracy)
        val_accs.append(val_accuracy)

    # 返回训练过程中的损失、验证损失、训练准确率和验证准确率列表，以及看到的总样本数量
    return train_losses, val_losses, train_accs, val_accs, examples_seen

# 注意：plot_values函数中的evaluate_model函数未在提供的代码段中定义，
# 因此以下注释基于plot_values函数本身的代码进行说明。
def plot_values(epochs_seen, examples_seen, train_values, val_values, label="loss"):

    # 创建图表和轴
    fig, ax1 = plt.subplots(figsize=(5, 3))

    # 绘制训练和验证过程中的值随轮数变化的曲线
    ax1.plot(epochs_seen, train_values, label=f"Training {label}")
    ax1.plot(epochs_seen, val_values, linestyle="-.", label=f"Validation {label}")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel(label.capitalize())
    ax1.legend()

    # 创建共享y轴但具有不同x轴的第二个轴，用于绘制随看到的样本数量变化的曲线（此曲线在此处为透明）
    ax2 = ax1.twiny()
    ax2.plot(examples_seen, train_values, alpha=0)
    ax2.set_xlabel("Examples seen")

    # 调整图表布局并保存为PDF文件，然后显示图表
    fig.tight_layout()
    plt.savefig(f"{label}-plot.pdf")
    plt.show()