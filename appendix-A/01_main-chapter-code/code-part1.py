# 从importlib.metadata模块中导入version函数，用于获取已安装包的版本信息
from importlib.metadata import version

# 导入PyTorch库
import torch

# 打印当前安装的torch库的版本信息
print("torch version:", version("torch"))

# 从previous_chapters模块中导入GPTModel类
from previous_chapters import GPTModel

# 定义一个字典，用于存储GPT-2 124M模型的配置参数
GPT_CONFIG_124M = {
    "vocab_size": 50257,  # 词汇表大小
    "context_length": 256,  # 上下文长度（已缩短，原始长度为1024）
    "emb_dim": 768,  # 嵌入维度
    "n_heads": 12,  # 注意力头的数量
    "n_layers": 12,  # Transformer层的数量
    "drop_rate": 0.1,  # Dropout比率
    "qkv_bias": False  # 查询（Query）、键（Key）、值（Value）是否添加偏置项
}

# 根据系统是否支持CUDA（NVIDIA GPU），选择运行设备（GPU或CPU）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 设置全局随机种子，确保结果的可重复性
torch.manual_seed(123)

# 初始化GPTModel模型，并传入配置参数
model = GPTModel(GPT_CONFIG_124M)

# 将模型设置为评估模式，禁用dropout
model.eval();

# 导入os和urllib.request模块，用于文件操作和网络请求
import os
import urllib.request

# 定义要下载的文件路径和URL
file_path = "../../01_main-chapter-code/the-verdict.txt"
url = "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/ch02/01_main-chapter-code/the-verdict.txt"

# 如果文件不存在，则从网络上下载；否则，从本地读取
if not os.path.exists(file_path):
    with urllib.request.urlopen(url) as response:
        text_data = response.read().decode('utf-8')  # 下载并解码文本数据
    with open(file_path, "w", encoding="utf-8") as file:
        file.write(text_data)  # 将文本数据写入文件
else:
    with open(file_path, "r", encoding="utf-8") as file:
        text_data = file.read()  # 从文件中读取文本数据

# 从previous_chapters模块中导入create_dataloader_v1函数
from previous_chapters import create_dataloader_v1

# 定义训练数据的比例
train_ratio = 0.90
# 根据训练比例计算分割索引
split_idx = int(train_ratio * len(text_data))

# 再次设置全局随机种子，确保数据划分的可重复性
torch.manual_seed(123)

# 创建训练数据加载器
train_loader = create_dataloader_v1(
    text_data[:split_idx],  # 训练数据
    batch_size=2,  # 批大小
    max_length=GPT_CONFIG_124M["context_length"],  # 最大长度
    stride=GPT_CONFIG_124M["context_length"],  # 步长
    drop_last=True,  # 是否丢弃最后一个不完整的批次
    shuffle=True,  # 是否打乱数据
    num_workers=0  # 加载数据时使用的进程数
)

# 创建验证数据加载器
val_loader = create_dataloader_v1(
    text_data[split_idx:],  # 验证数据
    batch_size=2,
    max_length=GPT_CONFIG_124M["context_length"],
    stride=GPT_CONFIG_124M["context_length"],
    drop_last=False,
    shuffle=False,
    num_workers=0
)

# 定义训练轮数和学习率参数
n_epochs = 15
initial_lr = 0.0001
peak_lr = 0.01

# 计算总训练步数
total_steps = len(train_loader) * n_epochs
# 计算预热步数（学习率逐渐增加到峰值前的步数）
warmup_steps = int(0.2 * total_steps)  # 20%的预热步数
print(warmup_steps)  # 打印预热步数

# 计算每步的学习率增量
lr_increment = (peak_lr - initial_lr) / warmup_steps

# 初始化全局步数和学习率跟踪列表
global_step = -1
track_lrs = []

# 使用AdamW优化器，并设置权重衰减
optimizer = torch.optim.AdamW(model.parameters(), weight_decay=0.1)

# 开始训练循环
for epoch in range(n_epochs):
    for input_batch, target_batch in train_loader:
        # 清零梯度
        optimizer.zero_grad()
        # 更新全局步数
        global_step += 1

        # 根据当前步数调整学习率
        if global_step < warmup_steps:
            lr = initial_lr + global_step * lr_increment
        else:
            lr = peak_lr

        # 将计算出的学习率应用到优化器上
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
        # 记录当前学习率
        track_lrs.append(optimizer.param_groups[0]["lr"])

# 导入matplotlib.pyplot模块，并简写为plt，用于绘图
import matplotlib.pyplot as plt

# 创建一个图形对象，设置大小为5x3英寸
plt.figure(figsize=(5, 3))
# 设置y轴标签为"Learning rate"
plt.ylabel("Learning rate")
# 设置x轴标签为"Step"
plt.xlabel("Step")
# 计算总训练步数，为训练集加载器长度乘以训练周期数
total_training_steps = len(train_loader) * n_epochs
# 绘制学习率随步数变化的曲线
plt.plot(range(total_training_steps), track_lrs)
# 调整布局以避免标签或标题被裁剪
plt.tight_layout();
plt.savefig("1.pdf")  # 保存图形为"1.pdf"
plt.show()  # 显示图形

# 导入math模块，用于数学计算
import math

# 计算最小学习率，为初始学习率的0.1倍
min_lr = 0.1 * initial_lr
# 初始化一个空列表，用于跟踪学习率
track_lrs = []

# 计算学习率增量，为峰值学习率与初始学习率之差除以预热步数
lr_increment = (peak_lr - initial_lr) / warmup_steps
# 初始化全局步数为-1
global_step = -1

# 训练循环，遍历每个训练周期
for epoch in range(n_epochs):
    # 遍历训练数据加载器中的每个批次
    for input_batch, target_batch in train_loader:
        # 将优化器的梯度清零
        optimizer.zero_grad()
        # 全局步数加1
        global_step += 1

        # 根据当前阶段（预热或余弦退火）调整学习率
        if global_step < warmup_steps:
            # 预热阶段，线性增加学习率
            lr = initial_lr + global_step * lr_increment
        else:
            # 余弦退火阶段
            progress = ((global_step - warmup_steps) /
                        (total_training_steps - warmup_steps))
            # 根据余弦函数计算当前学习率
            lr = min_lr + (peak_lr - min_lr) * 0.5 * (
                    1 + math.cos(math.pi * progress))

        # 将计算出的学习率应用到优化器
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
        # 跟踪当前学习率
        track_lrs.append(optimizer.param_groups[0]["lr"])

# 重复前面的绘图步骤，但保存为"2.pdf"
plt.figure(figsize=(5, 3))
plt.ylabel("Learning rate")
plt.xlabel("Step")
plt.plot(range(total_training_steps), track_lrs)
plt.tight_layout();
plt.savefig("2.pdf")
plt.show()

# 从前面的章节导入calc_loss_batch函数，用于计算批次损失
from previous_chapters import calc_loss_batch

# 设置随机种子以确保结果的可重复性
torch.manual_seed(123)
# 初始化GPT模型，并配置为GPT_CONFIG_124M
model = GPTModel(GPT_CONFIG_124M)
# 将模型移动到指定设备（如GPU）
model.to(device)

# 计算批次损失，并进行反向传播
loss = calc_loss_batch(input_batch, target_batch, model, device)
loss.backward()


# 定义find_highest_gradient函数，用于找到模型中梯度最大的参数
def find_highest_gradient(model):
    max_grad = None
    for param in model.parameters():
        if param.grad is not None:
            # 将梯度展平并找到最大值
            grad_values = param.grad.data.flatten()
            max_grad_param = grad_values.max()
            if max_grad is None or max_grad_param > max_grad:
                max_grad = max_grad_param
    return max_grad


# 打印模型中梯度最大的值
print(find_highest_gradient(model))

# 对模型的梯度进行裁剪，以避免梯度爆炸
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
# 再次打印梯度最大值，以观察裁剪效果
print(find_highest_gradient(model))

# 从前面的章节导入evaluate_model和generate_and_print_sample函数
from previous_chapters import evaluate_model, generate_and_print_sample

# 定义一个标志，用于区分书籍版本和实际代码实现中的细微差别
BOOK_VERSION = True


# 定义train_model函数，用于训练模型
def train_model(model, train_loader, val_loader, optimizer, device,
                n_epochs, eval_freq, eval_iter, start_context, tokenizer,
                warmup_steps, initial_lr=3e-05, min_lr=1e-6):
    # 初始化损失列表和跟踪变量
    train_losses, val_losses, track_tokens_seen, track_lrs = [], [], [], []
    tokens_seen, global_step = 0, -1

    # 从优化器中获取峰值学习率
    peak_lr = optimizer.param_groups[0]["lr"]

    # 计算总训练步数
    total_training_steps = len(train_loader) * n_epochs

    # 计算预热阶段的学习率增量
    lr_increment = (peak_lr - initial_lr) / warmup_steps

    # 训练循环
    for epoch in range(n_epochs):
        model.train()  # 设置模型为训练模式
        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()  # 清零梯度
            global_step += 1  # 步数加1

            # 根据当前阶段调整学习率
            if global_step < warmup_steps:
                lr = initial_lr + global_step * lr_increment  # 预热阶段
            else:
                progress = ((global_step - warmup_steps) /
                            (total_training_steps - warmup_steps))
                lr = min_lr + (peak_lr - min_lr) * 0.5 * (1 + math.cos(math.pi * progress))  # 余弦退火

            # 应用学习率
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr
            track_lrs.append(lr)  # 跟踪学习率

            # 计算损失并进行反向传播
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward()

            # 根据BOOK_VERSION标志决定是否裁剪梯度
            if BOOK_VERSION:
                if global_step > warmup_steps:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            else:
                if global_step >= warmup_steps:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()  # 更新参数
            tokens_seen += input_batch.numel()  # 更新处理的token数量

            # 定期评估模型
            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(model, train_loader, val_loader, device, eval_iter)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)
                print(f"Ep {epoch + 1} (Iter {global_step:06d}): Train loss {train_loss:.3f}, Val loss {val_loss:.3f}")

        # 生成并打印模型生成的样本
        generate_and_print_sample(model, tokenizer, device, start_context)

    return train_losses, val_losses, track_tokens_seen, track_lrs


# 导入tiktoken库，用于处理GPT模型的tokenizer
import tiktoken

# 初始化模型、优化器和tokenizer等
torch.manual_seed(123)
model = GPTModel(GPT_CONFIG_124M)
model.to(device)
peak_lr = 0.001
optimizer = torch.optim.AdamW(model.parameters(), lr=peak_lr, weight_decay=0.1)
tokenizer = tiktoken.get_encoding("gpt2")


# 设置训练的总轮数（epochs）
n_epochs = 15

# 调用train_model函数来训练模型，并收集训练损失、验证损失、观察到的token数量和学习率
# 参数包括模型、训练数据加载器、验证数据加载器、优化器、设备（CPU或GPU）、训练轮数、评估频率、评估迭代次数、
# 开始时的上下文文本、分词器、预热步数以及初始和最小学习率
train_losses, val_losses, tokens_seen, lrs = train_model(
    model, train_loader, val_loader, optimizer, device, n_epochs=n_epochs,
    eval_freq=5, eval_iter=1, start_context="Every effort moves you",
    tokenizer=tokenizer, warmup_steps=warmup_steps,
    initial_lr=1e-5, min_lr=1e-5
)

# 使用matplotlib创建一个新的图形，设置图形的大小
plt.figure(figsize=(5, 3))
# 绘制学习率随步骤变化的曲线
plt.plot(range(len(lrs)), lrs)
# 设置y轴的标签为"Learning rate"
plt.ylabel("Learning rate")
# 设置x轴的标签为"Steps"
plt.xlabel("Steps")
# 显示图形
plt.show()

# 从之前的章节导入plot_losses函数，用于绘制损失曲线
from previous_chapters import plot_losses

# 创建一个从1到n_epochs的等差数列，用于表示每个epoch的位置
# len(train_losses)确保了epoch的数量与训练损失的数量相匹配
epochs_tensor = torch.linspace(1, n_epochs, len(train_losses))
# 使用plot_losses函数绘制训练损失和验证损失随epoch变化的曲线
# 同时还绘制了观察到的token数量
plot_losses(epochs_tensor, tokens_seen, train_losses, val_losses)
# 调整图形的布局以避免重叠
plt.tight_layout();
# 保存图形为PDF文件
plt.savefig("3.pdf")
# 显示图形
plt.show()