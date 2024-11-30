# 导入必要的库和模块
from importlib.metadata import version  # 用于获取安装的包的版本
import torch  # PyTorch库，用于深度学习

# 打印matplotlib, torch, 和 tiktoken 的版本
print("matplotlib version:", version("matplotlib"))
print("torch version:", version("torch"))
# 注意：'tiktoken' 可能是一个拼写错误，通常我们使用的是 'tokenizers' 或者其他分词库
print("tiktoken version:", version("tiktoken"))

# GPT-2 124M 配置字典
GPT_CONFIG_124M = {
    "vocab_size": 50257,  # 词汇表大小
    "context_length": 1024,  # 上下文长度
    "emb_dim": 768,  # 嵌入维度
    "n_heads": 12,  # 注意力头的数量
    "n_layers": 12,  # 层的数量
    "drop_rate": 0.1,  # Dropout率
    "qkv_bias": False  # Query-Key-Value 是否有偏置
}

# 导入PyTorch的神经网络模块
import torch.nn as nn


# DummyGPTModel类，一个模拟的GPT模型
class DummyGPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])  # 词汇嵌入
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])  # 位置嵌入
        self.drop_emb = nn.Dropout(cfg["drop_rate"])  # Dropout层

        # 使用Transformer块的占位符
        self.trf_blocks = nn.Sequential(
            *[DummyTransformerBlock(cfg) for _ in range(cfg["n_layers"])])

        # 使用LayerNorm的占位符
        self.final_norm = DummyLayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(
            cfg["emb_dim"], cfg["vocab_size"], bias=False
        )  # 输出头

    def forward(self, in_idx):
        # 前向传播函数
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)  # 词汇嵌入
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))  # 位置嵌入
        x = tok_embeds + pos_embeds  # 词汇嵌入 + 位置嵌入
        x = self.drop_emb(x)  # Dropout
        x = self.trf_blocks(x)  # Transformer块
        x = self.final_norm(x)  # LayerNorm
        logits = self.out_head(x)  # 输出头
        return logits


# DummyTransformerBlock类，一个模拟的Transformer块
class DummyTransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        # 占位符，实际未实现任何功能

    def forward(self, x):
        # 前向传播函数，直接返回输入
        return x


# DummyLayerNorm类，一个模拟的LayerNorm层
class DummyLayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-5):
        super().__init__()
        # 参数仅用于模仿LayerNorm接口

    def forward(self, x):
        # 前向传播函数，直接返回输入
        return x


# 导入tiktoken库（可能是一个特定于项目的分词器）
import tiktoken

# 获取GPT-2的分词器
tokenizer = tiktoken.get_encoding("gpt2")

# 创建输入数据的batch
batch = []

txt1 = "Every effort moves you"
txt2 = "Every day holds a"

batch.append(torch.tensor(tokenizer.encode(txt1)))
batch.append(torch.tensor(tokenizer.encode(txt2)))
batch = torch.stack(batch, dim=0)
print(batch)

# 设置随机种子，确保结果可复现
torch.manual_seed(123)
# 初始化DummyGPTModel模型
model = DummyGPTModel(GPT_CONFIG_124M)

# 获取模型的输出
logits = model(batch)
print("Output shape:", logits.shape)
print(logits)

# 再次设置随机种子，确保结果可复现
torch.manual_seed(123)

# 创建一个包含2个训练样本，每个样本5个特征的batch
batch_example = torch.randn(2, 5)

# 创建一个包含线性层和ReLU激活函数的序列模型
layer = nn.Sequential(nn.Linear(5, 6), nn.ReLU())
# 获取模型的输出
out = layer(batch_example)
print(out)

# 计算输出的均值和方差
mean = out.mean(dim=-1, keepdim=True)
var = out.var(dim=-1, keepdim=True)

print("Mean:\n", mean)
print("Variance:\n", var)

# 对输出进行标准化
out_norm = (out - mean) / torch.sqrt(var)
print("Normalized layer outputs:\n", out_norm)

# 计算标准化后的输出的均值和方差
mean = out_norm.mean(dim=-1, keepdim=True)
var = out_norm.var(dim=-1, keepdim=True)
print("Mean:\n", mean)
print("Variance:\n", var)

# 设置打印选项，以非科学计数法打印
torch.set_printoptions(sci_mode=False)
print("Mean:\n", mean)
print("Variance:\n", var)

# 定义LayerNorm层，用于对输入进行层归一化
class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        # 定义一个很小的数，防止除以0的错误
        self.eps = 1e-5
        # 初始化scale参数，用于归一化后的缩放
        self.scale = nn.Parameter(torch.ones(emb_dim))
        # 初始化shift参数，用于归一化后的平移
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    # 定义前向传播
    def forward(self, x):
        # 计算输入x的均值
        mean = x.mean(dim=-1, keepdim=True)
        # 计算输入x的方差（无偏估计设置为False）
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        # 对输入x进行归一化
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        # 应用scale和shift参数进行缩放和平移
        return self.scale * norm_x + self.shift

# 实例化LayerNorm层，并传入嵌入维度
ln = LayerNorm(emb_dim=5)
# 假设batch_example是已经定义好的一个批次的数据
out_ln = ln(batch_example)

# 计算归一化后的数据的均值和方差
mean = out_ln.mean(dim=-1, keepdim=True)
var = out_ln.var(dim=-1, unbiased=False, keepdim=True)

# 打印均值和方差
print("Mean:\n", mean)
print("Variance:\n", var)

# 定义GELU激活函数
class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    # 定义前向传播
    def forward(self, x):
        # GELU激活函数的计算公式
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) *
            (x + 0.044715 * torch.pow(x, 3))
        ))

# 导入matplotlib.pyplot用于绘图
import matplotlib.pyplot as plt

# 实例化GELU和ReLU激活函数
gelu, relu = GELU(), nn.ReLU()

# 创建一个从-3到3的线性空间，用于绘制激活函数曲线
x = torch.linspace(-3, 3, 100)
# 计算GELU和ReLU激活函数的输出
y_gelu, y_relu = gelu(x), relu(x)

# 设置绘图的大小
plt.figure(figsize=(8, 3))
# 遍历GELU和ReLU激活函数及其标签
for i, (y, label) in enumerate(zip([y_gelu, y_relu], ["GELU", "ReLU"]), 1):
    # 绘制子图
    plt.subplot(1, 2, i)
    # 绘制激活函数曲线
    plt.plot(x, y)
    # 设置子图的标题
    plt.title(f"{label} activation function")
    # 设置x轴和y轴的标签
    plt.xlabel("x")
    plt.ylabel(f"{label}(x)")
    # 显示网格
    plt.grid(True)

# 调整子图布局，防止重叠
plt.tight_layout()
# 显示绘图
plt.show()
import torch
import torch.nn as nn

# 定义前馈神经网络类
class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        # 使用cfg字典中的配置信息初始化网络层
        # 首先是一个线性层，将输入维度扩展到4倍
        # 然后是一个GELU激活函数层
        # 最后是一个线性层，将维度还原到原始输入维度
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),  # 输入到4倍扩展
            GELU(),  # GELU激活函数
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]),  # 4倍扩展到原始输入维度
        )

    # 定义前向传播
    def forward(self, x):
        # 通过定义的层顺序传播输入x
        return self.layers(x)

# 假设GPT_CONFIG_124M是一个包含配置信息的字典，这里我们打印嵌入维度
print(GPT_CONFIG_124M["emb_dim"])

# 实例化前馈神经网络，传入GPT_CONFIG_124M作为配置
ffn = FeedForward(GPT_CONFIG_124M)

# 创建一个随机张量x作为输入，假设其形状为(batch_size, sequence_length, embedding_dim)
x = torch.rand(2, 3, 768)
# 通过前馈神经网络传播输入x，并打印输出形状
out = ffn(x)
print(out.shape)

# 定义带有可选捷径连接的深度神经网络类
class ExampleDeepNeuralNetwork(nn.Module):
    def __init__(self, layer_sizes, use_shortcut):
        super().__init__()
        # use_shortcut标志用于指示是否使用捷径连接
        self.use_shortcut = use_shortcut
        # 初始化一个ModuleList来存储网络层
        # 每一层都是一个线性层后接一个GELU激活函数
        self.layers = nn.ModuleList([
            nn.Sequential(nn.Linear(layer_sizes[i], layer_sizes[i+1]), GELU()) for i in range(len(layer_sizes)-1)
        ])

    # 定义前向传播
    def forward(self, x):
        for layer in self.layers:
            # 计算当前层的输出
            layer_output = layer(x)
            # 检查是否可以使用捷径连接
            # 如果use_shortcut为True且输入x和当前层输出layer_output的形状相同
            if self.use_shortcut and x.shape == layer_output.shape:
                x = x + layer_output  # 应用捷径连接
            else:
                x = layer_output  # 不使用捷径连接，直接更新x为当前层输出
        return x  # 返回最终输出
import torch
import torch.nn as nn

# 定义一个函数，用于打印模型的梯度
def print_gradients(model, x):
    # 通过模型前向传播输入x
    output = model(x)
    # 定义一个目标张量，用于计算损失
    target = torch.tensor([[0.]])

    # 初始化均方误差损失函数
    loss = nn.MSELoss()
    # 计算输出和目标之间的损失
    loss = loss(output, target)

    # 反向传播损失，计算梯度
    loss.backward()

    # 遍历模型的命名参数
    for name, param in model.named_parameters():
        # 如果参数名称中包含'weight'，则打印其梯度的绝对值均值
        if 'weight' in name:
            print(f"{name} has gradient mean of {param.grad.abs().mean().item()}")

# 定义层的大小
layer_sizes = [3, 3, 3, 3, 3, 1]

# 定义一个样本输入
sample_input = torch.tensor([[1., 0., -1.]])

# 设置随机种子以确保结果的可重复性
torch.manual_seed(123)
# 实例化一个没有捷径连接的深度神经网络模型
model_without_shortcut = ExampleDeepNeuralNetwork(
    layer_sizes, use_shortcut=False
)
# 打印没有捷径连接的模型的梯度
print_gradients(model_without_shortcut, sample_input)

# 再次设置随机种子以确保结果的可重复性
torch.manual_seed(123)
# 实例化一个有捷径连接的深度神经网络模型
model_with_shortcut = ExampleDeepNeuralNetwork(
    layer_sizes, use_shortcut=True
)
# 打印有捷径连接的模型的梯度
print_gradients(model_with_shortcut, sample_input)

# 从之前的章节中导入多头注意力类
from previous_chapters import MultiHeadAttention

# 定义Transformer块类
class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        # 初始化多头注意力机制
        self.att = MultiHeadAttention(
            d_in=cfg["emb_dim"],          # 输入维度
            d_out=cfg["emb_dim"],          # 输出维度
            context_length=cfg["context_length"],  # 上下文长度
            num_heads=cfg["n_heads"],      # 头数
            dropout=cfg["drop_rate"],      # Dropout率
            qkv_bias=cfg["qkv_bias"]       # 是否为qkv添加偏置
        )
        # 初始化前馈神经网络
        self.ff = FeedForward(cfg)
        # 初始化两个层归一化层
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])
        # 初始化Dropout层，用于捷径连接
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])

    # 定义前向传播
    def forward(self, x):
        # 保存原始输入作为第一个捷径连接的输入
        shortcut = x
        # 对输入进行第一个层归一化
        x = self.norm1(x)
        # 通过多头注意力机制传播归一化后的输入
        x = self.att(x)
        # 对注意力机制的输出应用Dropout
        x = self.drop_shortcut(x)
        # 将Dropout后的输出与原始输入相加，形成第一个捷径连接
        x = x + shortcut

        # 保存当前输出作为第二个捷径连接的输入
        shortcut = x
        # 对当前输出进行第二个层归一化
        x = self.norm2(x)
        # 通过前馈神经网络传播归一化后的输出
        x = self.ff(x)
        # 对前馈神经网络的输出应用Dropout
        x = self.drop_shortcut(x)
        # 将Dropout后的输出与之前的输出相加，形成第二个捷径连接
        x = x + shortcut

        # 返回最终的输出
        return x

# 设置随机种子以确保结果的可重复性
torch.manual_seed(123)

# 定义一个输入张量
x = torch.rand(2, 4, 768)
# 实例化一个Transformer块
block = TransformerBlock(GPT_CONFIG_124M)
# 通过Transformer块传播输入张量
output = block(x)

# 打印输入和输出的形状
print("Input shape:", x.shape)
print("Output shape:", output.shape)
import torch
import torch.nn as nn


# 定义GPT模型类
class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        # 初始化词嵌入层
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        # 初始化位置嵌入层
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        # 初始化嵌入层的Dropout
        self.drop_emb = nn.Dropout(cfg["drop_rate"])

        # 初始化Transformer块序列
        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])]  # 根据配置中的层数创建Transformer块
        )

        # 初始化最终的层归一化层
        self.final_norm = LayerNorm(cfg["emb_dim"])
        # 初始化输出头，用于将Transformer块的输出转换为词汇表大小的logits
        self.out_head = nn.Linear(
            cfg["emb_dim"], cfg["vocab_size"], bias=False  # 不使用偏置项
        )

    # 定义前向传播
    def forward(self, in_idx):
        # 获取批量大小和序列长度
        batch_size, seq_len = in_idx.shape
        # 获取词嵌入
        tok_embeds = self.tok_emb(in_idx)
        # 获取位置嵌入，注意位置嵌入是根据序列长度动态生成的
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        # 将词嵌入和位置嵌入相加
        x = tok_embeds + pos_embeds
        # 应用Dropout
        x = self.drop_emb(x)
        # 通过Transformer块序列传播
        x = self.trf_blocks(x)
        # 应用最终的层归一化
        x = self.final_norm(x)
        # 通过输出头获取logits
        logits = self.out_head(x)
        return logits


# 设置随机种子，确保结果的可重复性
torch.manual_seed(123)
# 实例化GPT模型
model = GPTModel(GPT_CONFIG_124M)

# 对一个批量的输入进行前向传播
out = model(batch)
# 打印输入批量和输出形状以及输出值
print("Input batch:\n", batch)
print("\nOutput shape:", out.shape)
print(out)

# 计算模型的总参数数量
total_params = sum(p.numel() for p in model.parameters())
print(f"Total number of parameters: {total_params:,}")

# 打印词嵌入层和输出层的形状
print("Token embedding layer shape:", model.tok_emb.weight.shape)
print("Output layer shape:", model.out_head.weight.shape)

# 如果考虑权重绑定（即词嵌入层和输出层共享权重），计算可训练参数数量
total_params_gpt2 = total_params - sum(p.numel() for p in model.out_head.parameters())
print(f"Number of trainable parameters considering weight tying: {total_params_gpt2:,}")

# 计算模型的总大小（以MB为单位）
total_size_bytes = total_params * 4
total_size_mb = total_size_bytes / (1024 * 1024)
print(f"Total size of the model: {total_size_mb:.2f} MB")


# 定义简单的文本生成函数
def generate_text_simple(model, idx, max_new_tokens, context_size):
    for _ in range(max_new_tokens):  # 循环生成新令牌直到达到最大数量
        # 获取当前上下文索引
        idx_cond = idx[:, -context_size:]
        # 在不计算梯度的情况下进行前向传播
        with torch.no_grad():
            logits = model(idx_cond)
        # 获取最后一个时间步的logits
        logits = logits[:, -1, :]
        # 应用softmax获取概率分布
        probas = torch.softmax(logits, dim=-1)
        # 获取概率最高的索引作为下一个令牌的预测
        idx_next = torch.argmax(probas, dim=-1, keepdim=True)
        # 将下一个令牌的索引拼接到当前上下文中
        idx = torch.cat((idx, idx_next), dim=1)
    return idx


# 定义起始上下文
start_context = "Hello, I am"
# 使用分词器对起始上下文进行编码
encoded = tokenizer.encode(start_context)
print("encoded:", encoded)
# 将编码转换为张量，并增加批次维度
encoded_tensor = torch.tensor(encoded).unsqueeze(0)
print("encoded_tensor.shape:", encoded_tensor.shape)

# 将模型设置为评估模式
model.eval()

# 使用简单的文本生成函数生成文本
out = generate_text_simple(
    model=model,
    idx=encoded_tensor,
    max_new_tokens=6,
    context_size=GPT_CONFIG_124M["context_length"]  # 使用配置中的上下文长度
)
# 打印生成的索引和长度
print("Output:", out)
print("Output length:", len(out[0]))

# 使用分词器对生成的索引进行解码，获取生成的文本
decoded_text = tokenizer.decode(out.squeeze(0).tolist())
print(decoded_text)