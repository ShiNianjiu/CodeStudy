# 从importlib.metadata导入version函数，用于获取安装的库版本
from importlib.metadata import version
import torch

# 打印torch库的版本
print("torch version:", version("torch"))

# 从previous_chapters模块导入GPTModel类
from previous_chapters import GPTModel

# GPT模型的配置字典
GPT_CONFIG_124M = {
    "vocab_size": 50257,  # 词汇表大小
    "context_length": 256,  # 上下文长度（缩短版，原版为1024）
    "emb_dim": 768,  # 嵌入维度
    "n_heads": 12,  # 注意力头数量
    "n_layers": 12,  # 层数
    "drop_rate": 0.1,  # Dropout率
    "qkv_bias": False  # Query-key-value是否有偏置项
}

# 设置设备为cuda（如果可用）或cpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(123)  # 设置随机种子以保证结果可复现

# 初始化GPT模型
model = GPTModel(GPT_CONFIG_124M)
model.eval();  # 禁用dropout以进行推理

import os
import urllib.request

# 文本数据文件的路径和URL
file_path = "the-verdict.txt"
url = "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/ch02/01_main-chapter-code/the-verdict.txt"

# 如果文件不存在，则下载；否则，读取文件
if not os.path.exists(file_path):
    with urllib.request.urlopen(url) as response:
        text_data = response.read().decode('utf-8')
    with open(file_path, "w", encoding="utf-8") as file:
        file.write(text_data)
else:
    with open(file_path, "r", encoding="utf-8") as file:
        text_data = file.read()

# 从previous_chapters模块导入create_dataloader_v1函数
from previous_chapters import create_dataloader_v1

# 训练/验证数据划分比例
train_ratio = 0.90
split_idx = int(train_ratio * len(text_data))

# 设置随机种子以保证数据划分可复现
torch.manual_seed(123)

# 创建训练和验证数据加载器
train_loader = create_dataloader_v1(
    text_data[:split_idx],  # 训练数据
    batch_size=2,  # 批量大小
    max_length=GPT_CONFIG_124M["context_length"],  # 最大长度
    stride=GPT_CONFIG_124M["context_length"],  # 步长
    drop_last=True,  # 是否丢弃最后一个不完整的批量
    shuffle=True,  # 是否打乱数据
    num_workers=0  # 加载数据时使用的进程数
)

val_loader = create_dataloader_v1(
    text_data[split_idx:],  # 验证数据
    batch_size=2,
    max_length=GPT_CONFIG_124M["context_length"],
    stride=GPT_CONFIG_124M["context_length"],
    drop_last=False,
    shuffle=False,
    num_workers=0
)

# 训练参数
n_epochs = 15  # 训练轮数
initial_lr = 0.0001  # 初始学习率
peak_lr = 0.01  # 峰值学习率

# 计算总步数和预热步数
total_steps = len(train_loader) * n_epochs
warmup_steps = int(0.2 * total_steps)  # 20%的预热步数
print(warmup_steps)

# 计算学习率增量
lr_increment = (peak_lr - initial_lr) / warmup_steps

# 用于跟踪学习率的列表
global_step = -1
track_lrs = []

# 初始化优化器
optimizer = torch.optim.AdamW(model.parameters(), weight_decay=0.1)

# 训练循环
for epoch in range(n_epochs):
    for input_batch, target_batch in train_loader:
        optimizer.zero_grad()  # 梯度清零
        global_step += 1  # 全局步数加1

        # 根据全局步数调整学习率
        if global_step < warmup_steps:
            lr = initial_lr + global_step * lr_increment
        else:
            lr = peak_lr

        # 将计算得到的学习率应用到优化器
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
        track_lrs.append(optimizer.param_groups[0]["lr"])  # 记录学习率

import matplotlib.pyplot as plt  # 导入matplotlib的pyplot模块，用于绘图

# 创建一个新的图形，并设置其大小为5x3英寸
plt.figure(figsize=(5, 3))
# 设置y轴的标签为"Learning rate"
plt.ylabel("Learning rate")
# 设置x轴的标签为"Step"
plt.xlabel("Step")
# 计算总的训练步数，这是通过训练数据加载器的长度（即批次数量）乘以训练的轮数（n_epochs）来得到的
total_training_steps = len(train_loader) * n_epochs
# 绘制学习率变化曲线，横坐标是训练步数（从0到total_training_steps-1），纵坐标是对应的学习率值（来自track_lrs列表）
plt.plot(range(total_training_steps), track_lrs)
# 自动调整子图参数, 使之填充整个图像区域，并避免标签和标题的重叠
plt.tight_layout()
# 将当前图形保存为PDF文件，文件名为"1.pdf"
plt.savefig("1.pdf")
# 显示图形
plt.show()

import math  # 导入math模块，用于数学运算，如计算余弦值

# 设置最小学习率，为初始学习率的0.1倍
min_lr = 0.1 * initial_lr
# 初始化一个空列表，用于记录每个训练步骤的学习率
track_lrs = []

# 计算学习率在预热阶段每一步的增量
lr_increment = (peak_lr - initial_lr) / warmup_steps
# 初始化全局步骤计数器为-1（通常用于表示还未开始训练）
global_step = -1

# 开始训练循环，遍历每个epoch
for epoch in range(n_epochs):
    # 在每个epoch内，遍历训练数据加载器中的每个批次
    for input_batch, target_batch in train_loader:
        # 将优化器的梯度清零，为下一个梯度的计算做准备
        optimizer.zero_grad()
        # 全局步骤计数器加1
        global_step += 1

        # 根据当前阶段（预热或余弦退火）调整学习率
        if global_step < warmup_steps:
            # 如果处于预热阶段，则学习率线性增加
            lr = initial_lr + global_step * lr_increment
        else:
            # 如果预热结束，则进入余弦退火阶段
            # 计算当前步骤在余弦退火阶段的进度（归一化到0-1之间）
            progress = ((global_step - warmup_steps) /
                        (total_training_steps - warmup_steps))
            # 根据余弦退火公式计算当前学习率
            lr = min_lr + (peak_lr - min_lr) * 0.5 * (
                    1 + math.cos(math.pi * progress))

        # 将计算出的学习率应用到优化器中
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
        # 记录当前步骤的学习率到列表中
        track_lrs.append(optimizer.param_groups[0]["lr"])

        # 以下是原本代码中省略的部分，通常包括：
        # 1. 通过模型计算损失
        # 2. 反向传播计算梯度
        # 3. 优化器更新模型权重
        # 例如:
        # output = model(input_batch)
        # loss = loss_fn(output, target_batch)
        # loss.backward()
        # optimizer.step()

# 绘制学习率变化曲线
plt.figure(figsize=(5, 3))
plt.ylabel("Learning rate")  # 设置y轴标签
plt.xlabel("Step")  # 设置x轴标签
# 绘制学习率曲线，x轴是训练步骤，y轴是学习率
plt.plot(range(total_training_steps), track_lrs)
plt.tight_layout()  # 自动调整子图参数，避免重叠
plt.savefig("2.pdf")  # 保存图形为PDF文件
plt.show()  # 显示图形
# 从之前的章节导入计算批量损失的函数
from previous_chapters import calc_loss_batch

# 设置随机种子以保证结果的可重复性
torch.manual_seed(123)
# 初始化GPT模型，使用GPT_CONFIG_124M配置
model = GPTModel(GPT_CONFIG_124M)
# 将模型移动到指定的设备上（如CPU或GPU）
model.to(device)

# 计算给定输入批次和目标批次的损失
loss = calc_loss_batch(input_batch, target_batch, model, device)
# 执行反向传播，计算梯度
loss.backward()

# 定义一个函数，用于找到模型中梯度最大的参数值
def find_highest_gradient(model):
    max_grad = None  # 初始化最大梯度值为None
    # 遍历模型中的所有参数
    for param in model.parameters():
        # 如果参数有梯度
        if param.grad is not None:
            # 将梯度数据展平为一维向量
            grad_values = param.grad.data.flatten()
            # 找到该参数梯度中的最大值
            max_grad_param = grad_values.max()
            # 如果当前最大梯度值为None，或者当前参数的梯度最大值大于当前最大梯度值
            if max_grad is None or max_grad_param > max_grad:
                # 更新最大梯度值
                max_grad = max_grad_param
    # 返回模型中的最大梯度值
    return max_grad

# 打印出模型中的最大梯度值
print(find_highest_gradient(model))

# 使用梯度裁剪，将模型所有参数的梯度的L2范数限制在1.0以内
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
# 再次打印出梯度裁剪后的模型中的最大梯度值
print(find_highest_gradient(model))

# 从之前的章节导入评估模型和生成样本的函数
from previous_chapters import evaluate_model, generate_and_print_sample

# 定义一个标志，用于控制是否使用书籍中的梯度裁剪版本
BOOK_VERSION = True


# 定义一个函数，用于训练模型
def train_model(model, train_loader, val_loader, optimizer, device,
                n_epochs, eval_freq, eval_iter, start_context, tokenizer,
                warmup_steps, initial_lr=3e-05, min_lr=1e-6):

    # 初始化损失列表和学习率、令牌数轨迹列表
    train_losses, val_losses, track_tokens_seen, track_lrs = [], [], [], []
    tokens_seen, global_step = 0, -1

    # 从优化器获取最大学习率
    peak_lr = optimizer.param_groups[0]["lr"]

    # 计算训练过程中的总迭代次数
    total_training_steps = len(train_loader) * n_epochs

    # 计算预热阶段的学习率增量
    lr_increment = (peak_lr - initial_lr) / warmup_steps

    # 开始训练循环
    for epoch in range(n_epochs):
        model.train()  # 设置模型为训练模式
        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()  # 清零梯度
            global_step += 1  # 更新全局步数

            # 根据当前阶段（预热或余弦退火）调整学习率
            if global_step < warmup_steps:
                # 预热阶段：线性增加学习率
                lr = initial_lr + global_step * lr_increment
            else:
                # 余弦退火阶段
                progress = ((global_step - warmup_steps) /
                            (total_training_steps - warmup_steps))
                lr = min_lr + (peak_lr - min_lr) * 0.5 * (1 + math.cos(math.pi * progress))

            # 将计算出的学习率应用到优化器
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr
            track_lrs.append(lr)  # 存储当前学习率

            # 计算损失并进行反向传播
            loss = calc_loss_batch(input_batch, target_batch, model, device)  # 注意：calc_loss_batch未在代码中定义，可能是外部函数
            loss.backward()

            # 预热阶段之后应用梯度裁剪以避免梯度爆炸
            if BOOK_VERSION:
                if global_step > warmup_steps:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            else:
                # 书籍原始版本可能存在的bug：使用>=会导致在warmup_steps后的一步跳过裁剪
                if global_step >= warmup_steps:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()  # 更新模型参数
            tokens_seen += input_batch.numel()  # 更新看到的令牌数

            # 每隔eval_freq步评估一次模型
            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader,
                    device, eval_iter
                )
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)
                # 打印当前损失
                print(f"Epoch {epoch + 1} (Iter {global_step:06d}): "
                      f"Train loss {train_loss:.3f}, "
                      f"Val loss {val_loss:.3f}"
                      )

        # 生成并打印一个样本以监控进度
        generate_and_print_sample(
            model, tokenizer, device, start_context
        )

    # 返回训练过程中的损失、验证损失、看到的令牌数和学习率轨迹
    return train_losses, val_losses, track_tokens_seen, track_lrs

import tiktoken  # 注意：这里可能是个笔误，正确的库名应该是 'tokenizers' 或者 'transformers' 的GPT-2 tokenizer，但保持原样注释

# 设置随机种子以保证结果的可重复性
torch.manual_seed(123)
# 初始化GPT模型，使用GPT_CONFIG_124M配置
model = GPTModel(GPT_CONFIG_124M)
# 将模型移动到指定的设备上（如GPU）
model.to(device)

# 设置学习率峰值，原书中误设为5e-4
peak_lr = 0.001
# 使用AdamW优化器，并指定学习率和权重衰减
optimizer = torch.optim.AdamW(model.parameters(), lr=peak_lr, weight_decay=0.1)
# 使用GPT-2的分词器
tokenizer = tiktoken.get_encoding("gpt2")  # 注意：这里可能是个笔误，正确的应该是GPT-2的分词器初始化

# 设置训练轮数
n_epochs = 15
# 训练模型，并返回训练损失、验证损失、看到的令牌数和学习率列表
train_losses, val_losses, tokens_seen, lrs = train_model(
    model, train_loader, val_loader, optimizer, device, n_epochs=n_epochs,
    eval_freq=5, eval_iter=1, start_context="Every effort moves you",  # 设置验证时的起始文本
    tokenizer=tokenizer, warmup_steps=warmup_steps,  # warmup_steps未在代码中定义，可能是全局变量
    initial_lr=1e-5, min_lr=1e-5  # 设置初始学习率和最小学习率
)

# 绘制学习率变化曲线
plt.figure(figsize=(5, 3))
plt.plot(range(len(lrs)), lrs)
plt.ylabel("Learning rate")
plt.xlabel("Steps")
plt.show()

# 从前面的章节导入plot_losses函数
from previous_chapters import plot_losses

# 生成与训练损失和验证损失对应的epoch张量
epochs_tensor = torch.linspace(1, n_epochs, len(train_losses))
# 绘制训练和验证损失曲线
plot_losses(epochs_tensor, tokens_seen, train_losses, val_losses)
plt.tight_layout(); plt.savefig("3.pdf")  # 调整布局并保存图像
plt.show()