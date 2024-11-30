# 导入version函数，用于获取已安装包的版本
from importlib.metadata import version

# 定义一个包含多个包名的列表，用于检查这些包的版本
pkgs = ["matplotlib",
        "numpy",
        "tiktoken",  # 注意：这里可能是个笔误，通常应为'transformers'或类似的库
        "torch",
        "tensorflow",
        "pandas"
       ]

# 遍历包名列表，打印每个包的版本
for p in pkgs:
    print(f"{p} version: {version(p)}")

# 导入必要的库和模块
from pathlib import Path
import pandas as pd
from previous_chapters import (
    download_and_unzip_spam_data,  # 从之前的章节导入的函数，用于下载和解压垃圾短信数据集
    create_balanced_dataset,       # 创建平衡数据集
    random_split                   # 随机分割数据集
)

# 定义数据集的URL、压缩文件路径、解压后的路径和数据文件路径
url = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
zip_path = "sms_spam_collection.zip"
extracted_path = "sms_spam_collection"
data_file_path = Path(extracted_path) / "SMSSpamCollection.tsv"

# 下载并解压垃圾短信数据集
download_and_unzip_spam_data(url, zip_path, extracted_path, data_file_path)

# 读取数据文件，设置分隔符、无表头和列名
df = pd.read_csv(data_file_path, sep="\t", header=None, names=["Label", "Text"])

# 创建平衡数据集，并将标签映射为数字（ham为0，spam为1）
balanced_df = create_balanced_dataset(df)
balanced_df["Label"] = balanced_df["Label"].map({"ham": 0, "spam": 1})

# 随机分割数据集为训练集、验证集和测试集
train_df, validation_df, test_df = random_split(balanced_df, 0.7, 0.1)

# 将分割后的数据集保存为CSV文件
train_df.to_csv("train.csv", index=None)
validation_df.to_csv("validation.csv", index=None)
test_df.to_csv("test.csv", index=None)

# 导入PyTorch库和自定义的Dataset类
import torch
from torch.utils.data import Dataset
import tiktoken  # 注意：这里可能是个笔误，通常应为'transformers'或类似的库
from previous_chapters import SpamDataset  # 从之前的章节导入的自定义Dataset类

# 使用GPT-2的tokenizer进行文本编码
tokenizer = tiktoken.get_encoding("gpt2")  # 注意：如果库名为'transformers'，则使用类似 transformers.GPT2Tokenizer.from_pretrained('gpt2')

# 创建训练集、验证集和测试集的Dataset对象
train_dataset = SpamDataset("train.csv", max_length=None, tokenizer=tokenizer)
val_dataset = SpamDataset("validation.csv", max_length=train_dataset.max_length, tokenizer=tokenizer)
test_dataset = SpamDataset("test.csv", max_length=train_dataset.max_length, tokenizer=tokenizer)

# 导入DataLoader类
from torch.utils.data import DataLoader

# 设置DataLoader的参数
num_workers = 0
batch_size = 8

# 设置随机种子以保证结果的可重复性
torch.manual_seed(123)

# 创建训练集、验证集和测试集的DataLoader对象
train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers,
    drop_last=True,
)

val_loader = DataLoader(
    dataset=val_dataset,
    batch_size=batch_size,
    num_workers=num_workers,
    drop_last=False,
)

test_loader = DataLoader(
    dataset=test_dataset,
    batch_size=batch_size,
    num_workers=num_workers,
    drop_last=False,
)

# 打印训练集DataLoader的一些信息，用于验证
print("Train loader:")
for input_batch, target_batch in train_loader:
    break  # 只需迭代一次即可获取批次维度信息

# 打印输入批次和标签批次的维度
print("Input batch dimensions:", input_batch.shape)
print("Label batch dimensions", target_batch.shape)

# 打印训练集、验证集和测试集的批次数量
print(f"{len(train_loader)} training batches")
print(f"{len(val_loader)} validation batches")
print(f"{len(test_loader)} test batches")

# 导入GPT-2下载和加载函数
from gpt_download import download_and_load_gpt2
# 导入自定义的GPT模型类和权重加载函数
from previous_chapters import GPTModel, load_weights_into_gpt

# 选择使用的GPT-2模型
CHOOSE_MODEL = "gpt2-small (124M)"
# 输入的文本提示
INPUT_PROMPT = "Every effort moves"

# 基础配置，包括词汇表大小、上下文长度、丢弃率和QKV偏差
BASE_CONFIG = {
    "vocab_size": 50257,     # 词汇表大小
    "context_length": 1024,  # 上下文长度
    "drop_rate": 0.0,        # 丢弃率
    "qkv_bias": True         # QKV（查询-键-值）偏差
}

# 不同GPT-2模型的配置信息
model_configs = {
    "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
    "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
    "gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
    "gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
}

# 更新基础配置以包含所选模型的特定配置
BASE_CONFIG.update(model_configs[CHOOSE_MODEL])

# 从模型名称中提取模型大小（去除括号）
model_size = CHOOSE_MODEL.split(" ")[-1].lstrip("(").rstrip(")")
# 下载并加载GPT-2模型及其参数
settings, params = download_and_load_gpt2(model_size=model_size, models_dir="gpt2")

# 初始化GPT模型
model = GPTModel(BASE_CONFIG)
# 将下载的参数加载到模型中
load_weights_into_gpt(model, params)
# 将模型设置为评估模式
model.eval();

# 导入文本生成、文本到令牌ID转换和令牌ID到文本转换的函数
from previous_chapters import (
    generate_text_simple,
    text_to_token_ids,
    token_ids_to_text
)

# 输入文本
text_1 = "Every effort moves you"

# 使用模型生成文本
# 注意：这里缺少了`tokenizer`的定义，假设它已经在其他地方被定义
token_ids = generate_text_simple(
    model=model,
    idx=text_to_token_ids(text_1, tokenizer),  # 将文本转换为令牌ID
    max_new_tokens=15,                         # 生成的最大新令牌数
    context_size=BASE_CONFIG["context_length"] # 上下文大小
)

# 将生成的令牌ID转换回文本并打印
print(token_ids_to_text(token_ids, tokenizer))

# 导入torch库（假设这里缺少import torch）
import torch
torch.manual_seed(123)  # 设置随机种子以保证结果可重复

# 定义输出层的类别数
num_classes = 2
# 为模型添加一个线性输出层
model.out_head = torch.nn.Linear(in_features=768, out_features=num_classes)

# 设置设备（如果可用则使用CUDA，否则使用CPU）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 将模型移动到指定的设备上
model.to(device);

# 从之前的章节导入计算加载器准确率的函数
from previous_chapters import calc_accuracy_loader

# 设置随机种子以确保结果的可重复性
torch.manual_seed(123)
# 计算训练集、验证集和测试集上的准确率
train_accuracy = calc_accuracy_loader(train_loader, model, device, num_batches=10)
val_accuracy = calc_accuracy_loader(val_loader, model, device, num_batches=10)
test_accuracy = calc_accuracy_loader(test_loader, model, device, num_batches=10)

# 打印准确率
print(f"Training accuracy: {train_accuracy*100:.2f}%")
print(f"Validation accuracy: {val_accuracy*100:.2f}%")
print(f"Test accuracy: {test_accuracy*100:.2f}%")

# 导入math模块用于数学计算
import math

# 定义LoRA层，这是一种参数高效的模型调整方法
class LoRALayer(torch.nn.Module):
    def __init__(self, in_dim, out_dim, rank, alpha):
        super().__init__()
        # 初始化A矩阵，用于低秩分解
        self.A = torch.nn.Parameter(torch.empty(in_dim, rank))
        torch.nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))  # 使用Kaiming初始化
        # 初始化B矩阵，用于低秩分解
        self.B = torch.nn.Parameter(torch.zeros(rank, out_dim))
        # alpha是缩放因子
        self.alpha = alpha

    def forward(self, x):
        # 计算LoRA调整后的输出
        x = self.alpha * (x @ self.A @ self.B)
        return x

# 定义包含LoRA层的线性层
class LinearWithLoRA(torch.nn.Module):
    def __init__(self, linear, rank, alpha):
        super().__init__()
        # 保存原始的线性层
        self.linear = linear
        # 创建LoRA层
        self.lora = LoRALayer(
            linear.in_features, linear.out_features, rank, alpha
        )

    def forward(self, x):
        # 计算包含LoRA调整的输出
        return self.linear(x) + self.lora(x)

# 递归地将模型中的所有线性层替换为包含LoRA层的线性层
def replace_linear_with_lora(model, rank, alpha):
    for name, module in model.named_children():
        if isinstance(module, torch.nn.Linear):
            # 替换为包含LoRA的线性层
            setattr(model, name, LinearWithLoRA(module, rank, alpha))
        else:
            # 递归处理子模块
            replace_linear_with_lora(module, rank, alpha)

# 计算并打印替换前的可训练参数数量
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total trainable parameters before: {total_params:,}")

# 冻结所有参数
for param in model.parameters():
    param.requires_grad = False

# 计算并打印替换后的可训练参数数量（此时应为0，因为所有参数都被冻结了）
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total trainable parameters after: {total_params:,}")

# 使用LoRA替换模型中的线性层
replace_linear_with_lora(model, rank=16, alpha=16)

# 计算并打印LoRA引入的可训练参数数量
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total trainable LoRA parameters: {total_params:,}")

# 设置设备（GPU或CPU）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 打印模型结构
print(model)

# 重新计算并打印准确率，以验证LoRA替换后的模型性能
torch.manual_seed(123)
train_accuracy = calc_accuracy_loader(train_loader, model, device, num_batches=10)
val_accuracy = calc_accuracy_loader(val_loader, model, device, num_batches=10)
test_accuracy = calc_accuracy_loader(test_loader, model, device, num_batches=10)

print(f"Training accuracy: {train_accuracy*100:.2f}%")
print(f"Validation accuracy: {val_accuracy*100:.2f}%")
print(f"Test accuracy: {test_accuracy*100:.2f}%")

# 导入时间模块和训练分类器的函数
import time
from previous_chapters import train_classifier_simple

# 记录开始时间
start_time = time.time()

# 设置随机种子
torch.manual_seed(123)

# 初始化优化器
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.1)

# 设置训练轮数
num_epochs = 5
# 训练模型并记录损失和准确率
train_losses, val_losses, train_accs, val_accs, examples_seen = train_classifier_simple(
    model, train_loader, val_loader, optimizer, device,
    num_epochs=num_epochs, eval_freq=50, eval_iter=5,
)

# 记录结束时间并计算总训练时间
end_time = time.time()
execution_time_minutes = (end_time - start_time) / 60
print(f"Training completed in {execution_time_minutes:.2f} minutes.")

# 从之前的章节导入绘图函数
from previous_chapters import plot_values

# 创建张量以记录训练轮数和看到的示例数
epochs_tensor = torch.linspace(0, num_epochs, len(train_losses))
examples_seen_tensor = torch.linspace(0, examples_seen, len(train_losses))

# 绘制损失图
plot_values(epochs_tensor, examples_seen_tensor, train_losses, val_losses, label="loss")

# 计算并打印最终准确率
train_accuracy = calc_accuracy_loader(train_loader, model, device)
val_accuracy = calc_accuracy_loader(val_loader, model, device)
test_accuracy = calc_accuracy_loader(test_loader, model, device)

print(f"Training accuracy: {train_accuracy*100:.2f}%")
print(f"Validation accuracy: {val_accuracy*100:.2f}%")
print(f"Test accuracy: {test_accuracy*100:.2f}%")