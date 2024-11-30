# 导入version函数，用于获取包的版本信息
from importlib.metadata import version

# 导入get_ipython函数，用于执行IPython魔术命令
from IPython import get_ipython

# 定义需要查询版本的包列表
pkgs = ["matplotlib",
        "numpy",
        "tiktoken",  # 注意：这里可能是个笔误，通常应为'tokenizers'或其他有效包名
        "torch",
        "tensorflow",
        "pandas"
       ]
# 遍历包列表，打印每个包的版本信息
for p in pkgs:
    print(f"{p} version: {version(p)}")


# 导入register_line_cell_magic装饰器，用于注册IPython魔术命令
from IPython.core.magic import register_line_cell_magic

# 用于记录已执行的单元格行标识的集合
executed_cells = set()

# 注册一个名为'run_once'的魔术命令，用于确保单元格内容只执行一次
@register_line_cell_magic
def run_once(line, cell):
    # 如果当前行标识不在已执行集合中
    if line not in executed_cells:
        # 执行当前单元格内容
        get_ipython().run_cell(cell)
        # 将当前行标识添加到已执行集合中
        executed_cells.add(line)
    else:
        # 如果当前行标识已在已执行集合中，则打印提示信息
        print(f"Cell '{line}' has already been executed.")


# 导入必要的库，用于下载和解压文件
import urllib.request
import zipfile
import os
from pathlib import Path

# 定义垃圾短信数据集的URL
url = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
# 定义压缩文件的保存路径
zip_path = "sms_spam_collection.zip"
# 定义解压后的文件夹路径
extracted_path = "sms_spam_collection"
# 定义数据文件的完整路径
data_file_path = Path(extracted_path) / "SMSSpamCollection.tsv"

# 定义下载并解压垃圾短信数据的函数
def download_and_unzip_spam_data(url, zip_path, extracted_path, data_file_path):
    # 如果数据文件已存在，则打印提示信息并返回
    if data_file_path.exists():
        print(f"{data_file_path} already exists. Skipping download and extraction.")
        return

    # 下载文件
    with urllib.request.urlopen(url) as response:
        with open(zip_path, "wb") as out_file:
            out_file.write(response.read())

    # 解压文件
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extracted_path)

    # 重命名解压后的文件，以匹配期望的数据文件路径
    original_file_path = Path(extracted_path) / "SMSSpamCollection"
    os.rename(original_file_path, data_file_path)
    # 打印文件下载和解压完成的提示信息
    print(f"File downloaded and saved as {data_file_path}")

# 调用函数下载并解压垃圾短信数据
download_and_unzip_spam_data(url, zip_path, extracted_path, data_file_path)

import pandas as pd

# 读取数据文件，创建DataFrame
df = pd.read_csv(data_file_path, sep="\t", header=None, names=["Label", "Text"])
df  # 显示DataFrame内容

# 打印标签的数量统计
print(df["Label"].value_counts())

# 使用'run_once'魔术命令执行下面的代码块，确保只执行一次
get_ipython().run_cell_magic('run_once', 'balance_df', '''
# 定义一个函数，用于创建平衡的数据集
def create_balanced_dataset(df):

    # 统计"spam"的数量
    num_spam = df[df["Label"] == "spam"].shape[0]

    # 随机采样与"spam"数量相等的"ham"实例
    ham_subset = df[df["Label"] == "ham"].sample(num_spam, random_state=123)

    # 将采样后的"ham"子集与"spam"合并，创建平衡的数据集
    balanced_df = pd.concat([ham_subset, df[df["Label"] == "spam"]])

    return balanced_df

# 创建平衡的数据集，并打印标签的数量统计
balanced_df = create_balanced_dataset(df)
print(balanced_df["Label"].value_counts())
''')

# 使用'run_once'魔术命令执行下面的代码，确保只执行一次
get_ipython().run_cell_magic('run_once', 'label_mapping',
                             'balanced_df["Label"] = balanced_df["Label"].map({"ham": 0, "spam": 1})')

# 显示平衡后的DataFrame内容
balanced_df


# 定义一个函数，用于将数据集随机分割为训练集、验证集和测试集
def random_split(df, train_frac, validation_frac):
    # 打乱整个DataFrame
    df = df.sample(frac=1, random_state=123).reset_index(drop=True)

    # 计算训练集和验证集的结束索引
    train_end = int(len(df) * train_frac)
    validation_end = train_end + int(len(df) * validation_frac)

    # 分割数据集
    train_df = df[:train_end]
    validation_df = df[train_end:validation_end]
    test_df = df[validation_end:]

    return train_df, validation_df, test_df


# 调用random_split函数，分割数据集，并保存到CSV文件
train_df, validation_df, test_df = random_split(balanced_df, 0.7, 0.1)

train_df.to_csv("train.csv", index=None)
validation_df.to_csv("validation.csv", index=None)
test_df.to_csv("test.csv", index=None)

# 导入tiktoken库（注意：这里可能是个特定环境或版本的库，标准库中无此库）
import tiktoken

# 获取GPT2的编码器
tokenizer = tiktoken.get_encoding("gpt2")
# 打印特定标记的编码
print(tokenizer.encode("<|endoftext|>", allowed_special={"<|endoftext|>"}))

import torch
from torch.utils.data import Dataset
import pandas as pd  # 导入pandas库，用于数据处理，但原代码中未明确导入，这里补充

# 定义一个用于加载垃圾邮件数据集的类，继承自torch.utils.data.Dataset
class SpamDataset(Dataset):
    def __init__(self, csv_file, tokenizer, max_length=None, pad_token_id=50256):
        # 初始化函数，读取CSV文件，并处理文本数据
        self.data = pd.read_csv(csv_file)  # 读取CSV文件

        # 使用tokenizer对文本进行编码
        self.encoded_texts = [
            tokenizer.encode(text) for text in self.data["Text"]
        ]

        # 如果未指定最大长度，则使用编码后文本的最长长度；否则，截断到指定长度
        if max_length is None:
            self.max_length = self._longest_encoded_length()
        else:
            self.max_length = max_length
            self.encoded_texts = [
                encoded_text[:self.max_length]
                for encoded_text in self.encoded_texts
            ]

        # 对不足最大长度的编码文本进行填充
        self.encoded_texts = [
            encoded_text + [pad_token_id] * (self.max_length - len(encoded_text))
            for encoded_text in self.encoded_texts
        ]

    def __getitem__(self, index):
        # 根据索引获取单个样本及其标签
        encoded = self.encoded_texts[index]
        label = self.data.iloc[index]["Label"]
        return (
            torch.tensor(encoded, dtype=torch.long),  # 返回编码后的文本张量
            torch.tensor(label, dtype=torch.long)  # 返回标签张量
        )

    def __len__(self):
        # 返回数据集的总样本数
        return len(self.data)

    def _longest_encoded_length(self):
        # 计算编码后文本的最长长度
        max_length = 0
        for encoded_text in self.encoded_texts:
            encoded_length = len(encoded_text)
            if encoded_length > max_length:
                max_length = encoded_length
        return max_length

# 创建训练集、验证集和测试集的数据加载器
train_dataset = SpamDataset(
    csv_file="train.csv",
    max_length=None,
    tokenizer=tokenizer  # 注意：tokenizer需要在外部定义并传入
)

print(train_dataset.max_length)  # 打印训练集的最大文本长度

val_dataset = SpamDataset(
    csv_file="validation.csv",
    max_length=train_dataset.max_length,
    tokenizer=tokenizer
)
test_dataset = SpamDataset(
    csv_file="test.csv",
    max_length=train_dataset.max_length,
    tokenizer=tokenizer
)

# 导入DataLoader类，用于创建数据加载器
from torch.utils.data import DataLoader

num_workers = 0  # 设置加载数据时使用的进程数
batch_size = 8  # 设置每个批次的样本数

torch.manual_seed(123)  # 设置随机种子以确保结果的可重复性

# 创建数据加载器
train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=batch_size,
    shuffle=True,  # 是否在每个epoch开始时打乱数据
    num_workers=num_workers,
    drop_last=True  # 是否丢弃最后一个不完整的批次
)

val_loader = DataLoader(
    dataset=val_dataset,
    batch_size=batch_size,
    num_workers=num_workers,
    drop_last=False  # 不丢弃最后一个批次（即使不完整）
)

test_loader = DataLoader(
    dataset=test_dataset,
    batch_size=batch_size,
    num_workers=num_workers,
    drop_last=False
)

# 打印训练加载器的信息
print("Train loader:")
for input_batch, target_batch in train_loader:
    pass  # 这里仅用于遍历加载器以获取最后一个批次的形状

print("Input batch dimensions:", input_batch.shape)  # 打印输入批次的形状
print("Label batch dimensions", target_batch.shape)  # 打印标签批次的形状

# 打印每个加载器的批次数
print(f"{len(train_loader)} training batches")
print(f"{len(val_loader)} validation batches")
print(f"{len(test_loader)} test batches")

# 配置模型参数
CHOOSE_MODEL = "gpt2-small (124M)"  # 选择模型
INPUT_PROMPT = "Every effort moves"  # 输入提示（虽然在此代码中未使用）

BASE_CONFIG = {
    "vocab_size": 50257,  # 词汇表大小
    "context_length": 1024,  # 上下文长度
    "drop_rate": 0.0,  # 丢弃率
    "qkv_bias": True  # 是否在查询、键、值投影中添加偏置项
}

# 定义不同模型的配置
model_configs = {
    "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
    "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
    "gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
    "gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
}

# 更新基础配置为所选模型的配置
BASE_CONFIG.update(model_configs[CHOOSE_MODEL])

# 断言数据集的最大长度不超过模型的上下文长度
assert train_dataset.max_length <= BASE_CONFIG["context_length"], (
    f"Dataset length {train_dataset.max_length} exceeds model's context "
    f"length {BASE_CONFIG['context_length']}. Reinitialize data sets with "
    f"`max_length={BASE_CONFIG['context_length']}`"
)

# 从gpt_download模块中导入download_and_load_gpt2函数，用于下载并加载GPT-2模型
from gpt_download import download_and_load_gpt2
# 从previous_chapters模块中导入GPTModel类和load_weights_into_gpt函数，以及文本生成和转换的相关函数
from previous_chapters import GPTModel, load_weights_into_gpt, generate_text_simple, text_to_token_ids, token_ids_to_text

# 根据选择的模型大小（如124M, 355M等），下载并加载GPT-2模型和参数
model_size = CHOOSE_MODEL.split(" ")[-1].lstrip("(").rstrip(")")
settings, params = download_and_load_gpt2(model_size=model_size, models_dir="gpt2")

# 初始化GPTModel模型，并将下载的参数加载到模型中
model = GPTModel(BASE_CONFIG)
load_weights_into_gpt(model, params)
# 将模型设置为评估模式
model.eval();

# 第一个示例：生成文本
text_1 = "Every effort moves you"
# 使用generate_text_simple函数生成文本，该函数需要模型、输入文本的token ids、最大新生成token数以及上下文大小作为参数
token_ids = generate_text_simple(
    model=model,
    idx=text_to_token_ids(text_1, tokenizer),  # 将输入文本转换为token ids
    max_new_tokens=15,  # 最大新生成token数
    context_size=BASE_CONFIG["context_length"]  # 上下文大小
)
# 将生成的token ids转换回文本并打印
print(token_ids_to_text(token_ids, tokenizer))

# 第二个示例：判断文本是否为垃圾邮件
text_2 = (
    "Is the following text 'spam'? Answer with 'yes' or 'no':"
    " 'You are a winner you have been specially selected to receive $1000 cash or a $2000 award.'"
)
# 同样使用generate_text_simple函数生成回答，这里假设模型被训练过以回答此类问题
token_ids = generate_text_simple(
    model=model,
    idx=text_to_token_ids(text_2, tokenizer),
    max_new_tokens=23,
    context_size=BASE_CONFIG["context_length"]
)
# 将生成的token ids转换回文本并打印
print(token_ids_to_text(token_ids, tokenizer))

# 打印模型结构
print(model)

# 设置模型中所有参数的requires_grad为False，意味着在训练过程中不会更新这些参数
for param in model.parameters():
    param.requires_grad = False

# 设置随机种子以保证结果的可复现性
torch.manual_seed(123)

# 定义分类任务的输出类别数
num_classes = 2
# 在模型最后添加一个线性层，用于分类任务
model.out_head = torch.nn.Linear(in_features=BASE_CONFIG["emb_dim"], out_features=num_classes)

# 设置模型中最后一个transformer块和最终归一化层的参数requires_grad为True，意味着在训练过程中会更新这些参数
for param in model.trf_blocks[-1].parameters():
    param.requires_grad = True

for param in model.final_norm.parameters():
    param.requires_grad = True

# 准备输入数据，将输入文本转换为token ids，并添加batch维度
inputs = tokenizer.encode("Do you have time")
inputs = torch.tensor(inputs).unsqueeze(0)
print("Inputs:", inputs)
print("Inputs dimensions:", inputs.shape)

# 在不计算梯度的情况下进行前向传播
with torch.no_grad():
    outputs = model(inputs)

# 打印输出和输出维度
print("Outputs:\n", outputs)
print("Outputs dimensions:", outputs.shape)

# 打印最后一个时间步的输出token的嵌入表示
print("Last output token:", outputs[:, -1, :])
# 注意：下面这一行是重复的，可以删除以避免重复输出
# print("Last output token:", outputs[:, -1, :])

# 对最后一个时间步的输出token的嵌入表示应用softmax函数，得到每个类别的概率
probas = torch.softmax(outputs[:, -1, :], dim=-1)
# 获取概率最高的类别的索引
label = torch.argmax(probas)
print("Class label:", label.item())

# 另一种获取类别标签的方式，直接对logits（即未经过softmax的输出）应用argmax函数
logits = outputs[:, -1, :]
label = torch.argmax(logits)
print("Class label:", label.item())


# 计算给定数据加载器（data_loader）在模型上的准确率
def calc_accuracy_loader(data_loader, model, device, num_batches=None):
    model.eval()  # 将模型设置为评估模式
    correct_predictions, num_examples = 0, 0  # 初始化正确预测数和总样本数

    # 如果未指定num_batches，则使用数据加载器的长度；否则使用两者中的较小值
    if num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))

    # 遍历数据加载器中的批次
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:  # 如果当前批次在指定的批次范围内
            input_batch, target_batch = input_batch.to(device), target_batch.to(device)  # 将数据移至指定设备

            with torch.no_grad():  # 禁用梯度计算
                logits = model(input_batch)[:, -1, :]  # 获取模型的输出（最后一个时间步的logits）
            predicted_labels = torch.argmax(logits, dim=-1)  # 获取预测标签

            num_examples += predicted_labels.shape[0]  # 更新总样本数
            correct_predictions += (predicted_labels == target_batch).sum().item()  # 更新正确预测数
        else:
            break  # 如果超出指定批次范围，则退出循环

    return correct_predictions / num_examples  # 返回准确率


# 设置设备（GPU或CPU）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 将模型移至指定设备
model.to(device)

# 设置随机种子
torch.manual_seed(123)

# 计算训练集、验证集和测试集的准确率
train_accuracy = calc_accuracy_loader(train_loader, model, device, num_batches=10)
val_accuracy = calc_accuracy_loader(val_loader, model, device, num_batches=10)
test_accuracy = calc_accuracy_loader(test_loader, model, device, num_batches=10)

# 打印准确率
print(f"Training accuracy: {train_accuracy * 100:.2f}%")
print(f"Validation accuracy: {val_accuracy * 100:.2f}%")
print(f"Test accuracy: {test_accuracy * 100:.2f}%")


# 计算单个批次的损失
def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)  # 将数据移至指定设备
    logits = model(input_batch)[:, -1, :]  # 获取模型的输出（最后一个时间步的logits）
    loss = torch.nn.functional.cross_entropy(logits, target_batch)  # 计算交叉熵损失
    return loss


# 计算给定数据加载器在模型上的平均损失
def calc_loss_loader(data_loader, model, device, num_batches=None):
    total_loss = 0.  # 初始化总损失
    if len(data_loader) == 0:  # 如果数据加载器为空
        return float("nan")  # 返回非数值
    elif num_batches is None:
        num_batches = len(data_loader)  # 如果未指定num_batches，则使用数据加载器的长度
    else:
        num_batches = min(num_batches, len(data_loader))  # 使用两者中的较小值

    # 遍历数据加载器中的批次
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:  # 如果当前批次在指定的批次范围内
            loss = calc_loss_batch(input_batch, target_batch, model, device)  # 计算损失
            total_loss += loss.item()  # 更新总损失
        else:
            break  # 如果超出指定批次范围，则退出循环

    return total_loss / num_batches  # 返回平均损失


# 计算训练集、验证集和测试集的损失
with torch.no_grad():
    train_loss = calc_loss_loader(train_loader, model, device, num_batches=5)
    val_loss = calc_loss_loader(val_loader, model, device, num_batches=5)
    test_loss = calc_loss_loader(test_loader, model, device, num_batches=5)

# 打印损失
print(f"Training loss: {train_loss:.3f}")
print(f"Validation loss: {val_loss:.3f}")
print(f"Test loss: {test_loss:.3f}")


# 训练分类器的简单函数
def train_classifier_simple(model, train_loader, val_loader, optimizer, device, num_epochs,
                            eval_freq, eval_iter):
    train_losses, val_losses, train_accs, val_accs = [], [], [], []  # 初始化损失和准确率列表
    examples_seen, global_step = 0, -1  # 初始化已见样本数和全局步数

    # 遍历指定的训练轮次
    for epoch in range(num_epochs):
        model.train()  # 将模型设置为训练模式

        # 遍历训练数据加载器中的批次
        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()  # 清空梯度
            loss = calc_loss_batch(input_batch, target_batch, model, device)  # 计算损失
            loss.backward()  # 反向传播
            optimizer.step()  # 更新模型参数
            examples_seen += input_batch.shape[0]  # 更新已见样本数
            global_step += 1  # 更新全局步数

            # 如果达到评估频率，则评估模型
            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, device, eval_iter)  # 调用evaluate_model函数（未在代码中定义）
                train_losses.append(train_loss)  # 更新训练损失列表
                val_losses.append(val_loss)  # 更新验证损失列表
                print(f"Ep {epoch + 1} (Step {global_step:06d}): "
                      f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}")  # 打印损失

        # 计算训练集和验证集的准确率
        train_accuracy = calc_accuracy_loader(train_loader, model, device, num_batches=eval_iter)
        val_accuracy = calc_accuracy_loader(val_loader, model, device, num_batches=eval_iter)
        print(f"Training accuracy: {train_accuracy * 100:.2f}% | ", end="")
        print(f"Validation accuracy: {val_accuracy * 100:.2f}%")  # 打印准确率
        train_accs.append(train_accuracy)  # 更新训练准确率列表
        val_accs.append(val_accuracy)  # 更新验证准确率列表

    return train_losses, val_losses, train_accs, val_accs, examples_seen  # 返回损失、准确率和已见样本数

# 评估模型函数，计算训练集和验证集上的损失
def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    model.eval()  # 设置模型为评估模式
    with torch.no_grad():  # 禁用梯度计算
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)  # 计算训练集损失
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)  # 计算验证集损失
    model.train()  # 设置模型为训练模式
    return train_loss, val_loss  # 返回训练集和验证集损失

# 初始化训练时间记录
import time
start_time = time.time()

# 设置随机种子以保证结果可复现
torch.manual_seed(123)

# 初始化优化器
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.1)

# 训练分类器，记录训练和验证的损失及准确率
num_epochs = 5
train_losses, val_losses, train_accs, val_accs, examples_seen = train_classifier_simple(
    model, train_loader, val_loader, optimizer, device,
    num_epochs=num_epochs, eval_freq=50, eval_iter=5,
)

# 计算训练完成所需时间
end_time = time.time()
execution_time_minutes = (end_time - start_time) / 60
print(f"Training completed in {execution_time_minutes:.2f} minutes.")

# 导入matplotlib用于绘图
import matplotlib.pyplot as plt

# 绘制训练和验证的损失或准确率变化曲线
def plot_values(epochs_seen, examples_seen, train_values, val_values, label="loss"):
    fig, ax1 = plt.subplots(figsize=(5, 3))

    ax1.plot(epochs_seen, train_values, label=f"Training {label}")  # 绘制训练曲线
    ax1.plot(epochs_seen, val_values, linestyle="-.", label=f"Validation {label}")  # 绘制验证曲线
    ax1.set_xlabel("Epochs")  # 设置x轴标签为Epochs
    ax1.set_ylabel(label.capitalize())  # 设置y轴标签
    ax1.legend()  # 显示图例

    ax2 = ax1.twiny()  # 创建共享y轴的第二个x轴
    ax2.plot(examples_seen, train_values, alpha=0)  # 绘制示例数量对应的训练曲线（透明度设为0，仅用于设置x轴范围）
    ax2.set_xlabel("Examples seen")  # 设置第二个x轴标签

    fig.tight_layout()  # 调整布局
    plt.savefig(f"{label}-plot.pdf")  # 保存图像
    plt.show()  # 显示图像

# 绘制训练和验证的损失曲线
epochs_tensor = torch.linspace(0, num_epochs, len(train_losses))
examples_seen_tensor = torch.linspace(0, examples_seen, len(train_losses))
plot_values(epochs_tensor, examples_seen_tensor, train_losses, val_losses)

# 绘制训练和验证的准确率曲线
epochs_tensor = torch.linspace(0, num_epochs, len(train_accs))
examples_seen_tensor = torch.linspace(0, examples_seen, len(train_accs))
plot_values(epochs_tensor, examples_seen_tensor, train_accs, val_accs, label="accuracy")

# 计算训练集、验证集和测试集上的准确率
train_accuracy = calc_accuracy_loader(train_loader, model, device)
val_accuracy = calc_accuracy_loader(val_loader, model, device)
test_accuracy = calc_accuracy_loader(test_loader, model, device)
print(f"Training accuracy: {train_accuracy*100:.2f}%")
print(f"Validation accuracy: {val_accuracy*100:.2f}%")
print(f"Test accuracy: {test_accuracy*100:.2f}%")

# 对给定的文本进行分类的函数
def classify_review(text, model, tokenizer, device, max_length=None, pad_token_id=50256):
    model.eval()  # 设置模型为评估模式

    # 将文本编码为输入ID
    input_ids = tokenizer.encode(text)
    # 获取模型支持的最大上下文长度
    supported_context_length = model.pos_emb.weight.shape[0]

    # 截取输入ID至模型支持的最大长度或给定的最大长度
    input_ids = input_ids[:min(max_length, supported_context_length)]

    # 用填充ID补齐至最大长度
    input_ids += [pad_token_id] * (max_length - len(input_ids))
    # 创建输入张量并移至指定设备
    input_tensor = torch.tensor(input_ids, device=device).unsqueeze(0)

    with torch.no_grad():  # 禁用梯度计算
        logits = model(input_tensor)[:, -1, :]  # 获取最后一个位置的logits
    predicted_label = torch.argmax(logits, dim=-1).item()  # 获取预测标签

    return "spam" if predicted_label == 1 else "not spam"  # 根据预测标签返回分类结果

# 对两条文本进行分类
text_1 = (
    "You are a winner you have been specially"
    " selected to receive $1000 cash or a $2000 award."
)
print(classify_review(
    text_1, model, tokenizer, device, max_length=train_dataset.max_length
))

text_2 = (
    "Hey, just wanted to check if we're still on"
    " for dinner tonight? Let me know!"
)
print(classify_review(
    text_2, model, tokenizer, device, max_length=train_dataset.max_length
))

# 保存模型状态字典
torch.save(model.state_dict(), "review_classifier.pth")

# 加载模型状态字典
model_state_dict = torch.load("review_classifier.pth", map_location=device, weights_only=True)
model.load_state_dict(model_state_dict)