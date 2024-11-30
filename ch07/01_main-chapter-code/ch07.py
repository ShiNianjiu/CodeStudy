# 导入所需的模块
import urllib.request  # 用于从URL下载文件

from importlib.metadata import version  # 用于获取已安装包的版本信息

# 定义一个包名列表
pkgs = [
    "matplotlib",
    "tiktoken",  # 注意：这里可能是个笔误，通常是'tokenizers'或其他库
    "torch",
    "tqdm",
    "tensorflow",
]
# 遍历包名列表，打印每个包的版本信息
for p in pkgs:
    print(f"{p} version: {version(p)}")

import json  # 用于处理JSON数据
import os  # 用于与操作系统交互，如检查文件是否存在
import urllib  # 用于处理URL

# 定义一个函数，用于从URL下载并加载JSON文件
def download_and_load_file(file_path, url):
    # 如果文件不存在，则从URL下载
    if not os.path.exists(file_path):
        with urllib.request.urlopen(url) as response:
            text_data = response.read().decode("utf-8")  # 读取并解码文件内容
        with open(file_path, "w", encoding="utf-8") as file:
            file.write(text_data)  # 将内容写入文件
    else:
        with open(file_path, "r", encoding="utf-8") as file:
            text_data = file.read()  # 如果文件已存在，则读取文件内容

    # 无论文件是否是新下载的，都重新打开并加载JSON数据
    with open(file_path, "r", encoding="utf-8") as file:
        data = json.load(file)  # 加载JSON数据

    return data  # 返回加载的数据

# 定义文件路径和URL
file_path = "instruction-data.json"
url = (
    "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch"
    "/main/ch07/01_main-chapter-code/instruction-data.json"
)

# 下载并加载数据
data = download_and_load_file(file_path, url)
print("Number of entries:", len(data))  # 打印数据条数
print("Example entry:\n", data[50])  # 打印第51条数据作为示例
print("Another example entry:\n", data[999])  # 打印第1000条数据作为另一个示例

# 定义一个函数，用于格式化输入文本
def format_input(entry):
    instruction_text = (
        f"Below is an instruction that describes a task. "
        f"Write a response that appropriately completes the request."
        f"\n\n### Instruction:\n{entry['instruction']}"
    )  # 格式化指令文本

    # 如果条目中包含输入文本，则添加输入文本部分
    input_text = f"\n\n### Input:\n{entry['input']}" if entry["input"] else ""

    return instruction_text + input_text  # 返回格式化后的文本

# 使用format_input函数格式化第51条和第1000条数据的输入文本
model_input = format_input(data[50])
desired_response = f"\n\n### Response:\n{data[50]['output']}"
print(model_input + desired_response)

model_input = format_input(data[999])
desired_response = f"\n\n### Response:\n{data[999]['output']}"
print(model_input + desired_response)

# 划分数据集为训练集、验证集和测试集
train_portion = int(len(data) * 0.85)  # 85%的数据作为训练集
test_portion = int(len(data) * 0.1)  # 10%的数据作为测试集
val_portion = len(data) - train_portion - test_portion  # 剩余5%的数据作为验证集

train_data = data[:train_portion]  # 划分训练数据
test_data = data[train_portion:train_portion + test_portion]  # 划分测试数据
val_data = data[train_portion + test_portion:]  # 划分验证数据

print("Training set length:", len(train_data))  # 打印训练集长度
print("Validation set length:", len(val_data))  # 打印验证集长度
print("Test set length:", len(test_data))  # 打印测试集长度

# 导入PyTorch模块
import torch
from torch.utils.data import Dataset  # 导入Dataset类，用于创建数据集

# 定义一个自定义的数据集类，用于处理指令数据
class InstructionDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data  # 存储数据

        # 初始化一个列表，用于存储编码后的文本
        self.encoded_texts = []
        for entry in data:
            instruction_plus_input = format_input(entry)  # 格式化指令和输入文本
            response_text = f"\n\n### Response:\n{entry['output']}"  # 获取期望的响应文本
            full_text = instruction_plus_input + response_text  # 拼接完整的文本
            self.encoded_texts.append(
                tokenizer.encode(full_text)  # 使用分词器编码完整的文本
            )

    def __getitem__(self, index):
        return self.encoded_texts[index]  # 根据索引获取编码后的文本

    def __len__(self):
        return len(self.data)  # 返回数据集的长度

# 导入tiktoken库，用于GPT-2的文本编码
import tiktoken
tokenizer = tiktoken.get_encoding("gpt2")

# 编码特殊标记"<|endoftext|>"，并允许该特殊标记
print(tokenizer.encode("<|endoftext|>", allowed_special={"<|endoftext|>"}))

# 自定义数据整理函数1：将一批数据填充到相同长度，并转换为张量
def custom_collate_draft_1(
    batch,  # 输入的一批数据，每个元素是一个列表
    pad_token_id=50256,  # 用于填充的token ID
    device="cpu"  # 目标设备
):
    # 计算批次中数据的最大长度（每个数据长度+1）
    batch_max_length = max(len(item)+1 for item in batch)

    inputs_lst = []  # 存储处理后的输入数据

    # 遍历每个数据项，进行填充并转换为张量
    for item in batch:
        new_item = item.copy()  # 复制数据项
        new_item += [pad_token_id]  # 添加填充token
        padded = (
            new_item + [pad_token_id] *  # 继续填充至最大长度
            (batch_max_length - len(new_item))
        )
        inputs = torch.tensor(padded[:-1])  # 转换为张量，不包含最后一个填充token
        inputs_lst.append(inputs)

    # 将所有输入数据张量堆叠起来，并发送到指定设备
    inputs_tensor = torch.stack(inputs_lst).to(device)
    return inputs_tensor

# 测试custom_collate_draft_1函数
inputs_1 = [0, 1, 2, 3, 4]
inputs_2 = [5, 6]
inputs_3 = [7, 8, 9]
batch = (inputs_1, inputs_2, inputs_3)
print(custom_collate_draft_1(batch))

# 自定义数据整理函数2：将一批数据及其目标数据填充到相同长度，并转换为张量
def custom_collate_draft_2(
    batch,  # 输入的一批数据，每个元素是一个列表，包含输入和目标数据对
    pad_token_id=50256,  # 用于填充的token ID
    device="cpu"  # 目标设备
):
    # 计算批次中数据的最大长度（每个数据长度+1）
    batch_max_length = max(len(item)+1 for item in batch)

    inputs_lst, targets_lst = [], []  # 存储处理后的输入和目标数据

    # 遍历每个数据项，进行填充并转换为张量
    for item in batch:
        new_item = item.copy()  # 复制数据项
        new_item += [pad_token_id]  # 添加填充token
        padded = (
            new_item + [pad_token_id] *  # 继续填充至最大长度
            (batch_max_length - len(new_item))
        )
        inputs = torch.tensor(padded[:-1])  # 转换为张量，不包含最后一个填充token
        targets = torch.tensor(padded[1:])  # 目标数据为填充后的数据，从第二个token开始
        inputs_lst.append(inputs)
        targets_lst.append(targets)

    # 将所有输入和目标数据张量堆叠起来，并发送到指定设备
    inputs_tensor = torch.stack(inputs_lst).to(device)
    targets_tensor = torch.stack(targets_lst).to(device)
    return inputs_tensor, targets_tensor

# 测试custom_collate_draft_2函数
inputs, targets = custom_collate_draft_2(batch)
print(inputs)
print(targets)

# 自定义数据整理函数：功能更强大的版本，支持忽略索引和最大长度限制
def custom_collate_fn(
    batch,  # 输入的一批数据，每个元素是一个列表，包含输入和目标数据对
    pad_token_id=50256,  # 用于填充的token ID
    ignore_index=-100,  # 在目标数据中用于忽略的索引值
    allowed_max_length=None,  # 允许的最大长度
    device="cpu"  # 目标设备
):
    # 计算批次中数据的最大长度（每个数据长度+1）
    batch_max_length = max(len(item)+1 for item in batch)

    inputs_lst, targets_lst = [], []  # 存储处理后的输入和目标数据

    # 遍历每个数据项，进行填充、忽略索引设置和长度限制
    for item in batch:
        new_item = item.copy()  # 复制数据项
        new_item += [pad_token_id]  # 添加填充token
        padded = (
            new_item + [pad_token_id] *  # 继续填充至最大长度
            (batch_max_length - len(new_item))
        )
        inputs = torch.tensor(padded[:-1])  # 转换为张量，不包含最后一个填充token
        targets = torch.tensor(padded[1:])  # 目标数据为填充后的数据，从第二个token开始

        # 设置忽略索引：将填充token之后的所有token设置为ignore_index
        mask = targets == pad_token_id
        indices = torch.nonzero(mask).squeeze()
        if indices.numel() > 1:
            targets[indices[1:]] = ignore_index

        # 应用最大长度限制
        if allowed_max_length is not None:
            inputs = inputs[:allowed_max_length]
            targets = targets[:allowed_max_length]

        inputs_lst.append(inputs)
        targets_lst.append(targets)

    # 将所有输入和目标数据张量堆叠起来，并发送到指定设备
    inputs_tensor = torch.stack(inputs_lst).to(device)
    targets_tensor = torch.stack(targets_lst).to(device)

    return inputs_tensor, targets_tensor

# 调用自定义的collate函数处理batch数据，返回inputs和targets
inputs, targets = custom_collate_fn(batch)
print(inputs)  # 打印处理后的输入数据
print(targets)  # 打印处理后的目标数据

# 定义两组logits（模型的原始输出）和对应的targets（真实标签）
logits_1 = torch.tensor(
    [[-1.0, 1.0],
     [-0.5, 1.5]]
)
targets_1 = torch.tensor([0, 1])

# 计算第一组logits和targets之间的交叉熵损失
loss_1 = torch.nn.functional.cross_entropy(logits_1, targets_1)
print(loss_1)  # 打印损失值

# 定义另一组logits和对应的targets
logits_2 = torch.tensor(
    [[-1.0, 1.0],
     [-0.5, 1.5],
     [-0.5, 1.5]]
)
targets_2 = torch.tensor([0, 1, 1])

# 计算第二组logits和targets之间的交叉熵损失
loss_2 = torch.nn.functional.cross_entropy(logits_2, targets_2)
print(loss_2)  # 打印损失值

# 定义一个包含无效标签（如-100）的targets，用于演示错误处理
targets_3 = torch.tensor([0, 1, -100])

# 尝试计算包含无效标签的targets和logits之间的交叉熵损失
# 注意：这通常会引发错误或进行特殊处理，具体取决于cross_entropy的实现
loss_3 = torch.nn.functional.cross_entropy(logits_2, targets_3)
print(loss_3)  # 打印损失值（如果处理得当）
print("loss_1 == loss_3:", loss_1 == loss_3)  # 比较loss_1和loss_3是否相等

# 设置设备为CUDA（如果可用）或CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Device:", device)  # 打印当前使用的设备

# 使用functools.partial创建一个具有预设参数的custom_collate_fn版本
from functools import partial

customized_collate_fn = partial(
    custom_collate_fn,
    device=device,
    allowed_max_length=1024
)

# 从torch.utils.data导入DataLoader，用于加载数据集
from torch.utils.data import DataLoader

# 设置数据加载参数
num_workers = 0  # 加载数据时使用的进程数
batch_size = 8  # 每个batch的样本数

torch.manual_seed(123)  # 设置随机种子以确保结果可重复

# 创建训练数据集和数据加载器
train_dataset = InstructionDataset(train_data, tokenizer)
train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    collate_fn=customized_collate_fn,  # 使用自定义的collate函数
    shuffle=True,  # 打乱数据顺序
    drop_last=True,  # 丢弃最后一个不完整的batch
    num_workers=num_workers
)

# 创建验证数据集和数据加载器
val_dataset = InstructionDataset(val_data, tokenizer)
val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    collate_fn=customized_collate_fn,
    shuffle=False,  # 不打乱数据顺序
    drop_last=False,  # 不丢弃最后一个batch
    num_workers=num_workers
)

# 创建测试数据集和数据加载器
test_dataset = InstructionDataset(test_data, tokenizer)
test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    collate_fn=customized_collate_fn,
    shuffle=False,
    drop_last=False,
    num_workers=num_workers
)

# 打印训练数据加载器的输出示例
print("Train loader:")
for inputs, targets in train_loader:
    print(inputs.shape, targets.shape)  # 打印输入和目标数据的形状

# 打印第一个batch的第一个输入样本
print(inputs[0])

# 打印第一个batch的第一个目标样本
print(targets[0])

# 导入GPT-2模型下载和加载的函数，以及自定义的GPT模型和权重加载函数
from gpt_download import download_and_load_gpt2
from previous_chapters import GPTModel, load_weights_into_gpt

# 定义基础配置，包括词汇表大小、上下文长度、丢弃率和qkv偏置
BASE_CONFIG = {
    "vocab_size": 50257,
    "context_length": 1024,
    "drop_rate": 0.0,
    "qkv_bias": True
}

# 定义不同GPT-2模型大小的配置
model_configs = {
    "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
    "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
    "gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
    "gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
}

# 选择要使用的GPT-2模型
CHOOSE_MODEL = "gpt2-medium (355M)"

# 更新基础配置以包含所选模型的特定配置
BASE_CONFIG.update(model_configs[CHOOSE_MODEL])

# 从模型名称中提取模型大小，用于下载
model_size = CHOOSE_MODEL.split(" ")[-1].lstrip("(").rstrip(")")
# 下载并加载GPT-2模型和权重
settings, params = download_and_load_gpt2(model_size=model_size, models_dir="gpt2")

# 初始化GPT模型
model = GPTModel(BASE_CONFIG)
# 将下载的权重加载到模型中
load_weights_into_gpt(model, params)
# 将模型设置为评估模式
model.eval();

# 设置随机种子以保证结果的可重复性
torch.manual_seed(123)

# 对验证数据进行格式化处理，并打印
input_text = format_input(val_data[0])
print(input_text)

# 导入文本生成、文本到token ID转换和token ID到文本转换的函数
from previous_chapters import (generate, text_to_token_ids, token_ids_to_text)

# 使用模型生成文本
token_ids = generate(
    model=model,  # GPT模型
    idx=text_to_token_ids(input_text, tokenizer),  # 输入文本的token ID
    max_new_tokens=35,  # 生成的最大新token数
    context_size=BASE_CONFIG["context_length"],  # 上下文长度
    eos_id=50256,  # 结束符号的ID
)
# 将生成的token ID转换为文本
generated_text = token_ids_to_text(token_ids, tokenizer)

# 提取并打印生成的响应文本
response_text = (
    generated_text[len(input_text):]  # 去除输入文本部分
    .replace("### Response:", "")  # 去除可能的响应标记
    .strip()  # 去除多余空白
)
print(response_text)

# 导入计算损失和简单训练模型的函数
from previous_chapters import calc_loss_loader, train_model_simple

# 将模型移动到指定设备（GPU或CPU）
model.to(device)

# 设置随机种子，计算训练和验证损失（不进行梯度计算）
with torch.no_grad():
    train_loss = calc_loss_loader(train_loader, model, device, num_batches=5)
    val_loss = calc_loss_loader(val_loader, model, device, num_batches=5)

print("Training loss:", train_loss)
print("Validation loss:", val_loss)

# 导入time模块以测量训练时间
import time

# 记录训练开始时间
start_time = time.time()

# 设置随机种子，优化器配置
torch.manual_seed(123)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.00005, weight_decay=0.1)

# 训练模型
num_epochs = 2
train_losses, val_losses, tokens_seen = train_model_simple(
    model, train_loader, val_loader, optimizer, device,
    num_epochs=num_epochs, eval_freq=5, eval_iter=5,
    start_context=format_input(val_data[0]), tokenizer=tokenizer
)

# 记录训练结束时间并计算总训练时间（分钟）
end_time = time.time()
execution_time_minutes = (end_time - start_time) / 60
print(f"Training completed in {execution_time_minutes:.2f} minutes.")

# 导入用于绘制损失图的函数
from previous_chapters import plot_losses

# 创建一个tensor，其值在0到num_epochs之间均匀分布，长度与train_losses相同
epochs_tensor = torch.linspace(0, num_epochs, len(train_losses))
# 调用plot_losses函数绘制训练和验证损失
plot_losses(epochs_tensor, tokens_seen, train_losses, val_losses)

# 设置随机种子以保证结果的可复现性
torch.manual_seed(123)

# 遍历测试数据的前三个条目
for entry in test_data[:3]:
    # 格式化输入文本
    input_text = format_input(entry)

    # 使用模型生成文本
    token_ids = generate(
        model=model,  # 使用的模型
        idx=text_to_token_ids(input_text, tokenizer).to(device),  # 输入文本转换为token ids并移动到设备
        max_new_tokens=256,  # 生成的最大新token数量
        context_size=BASE_CONFIG["context_length"],  # 上下文大小
        eos_id=50256  # 结束符号的token id
    )
    # 将生成的token ids转换为文本
    generated_text = token_ids_to_text(token_ids, tokenizer)
    # 处理生成的文本，移除不需要的部分
    response_text = (
        generated_text[len(input_text):]  # 移除输入文本部分
        .replace("### Response:", "")  # 移除"### Response:"标签
        .strip()  # 去除多余空白
    )

    # 打印输入文本、正确响应和模型响应
    print(input_text)
    print(f"\nCorrect response:\n>> {entry['output']}")
    print(f"\nModel response:\n>> {response_text.strip()}")
    print("-------------------------------------")

# 遍历整个测试数据集，并生成模型响应
from tqdm import tqdm

for i, entry in tqdm(enumerate(test_data), total=len(test_data)):
    input_text = format_input(entry)
    token_ids = generate(
        model=model,
        idx=text_to_token_ids(input_text, tokenizer).to(device),
        max_new_tokens=256,
        context_size=BASE_CONFIG["context_length"],
        eos_id=50256
    )
    generated_text = token_ids_to_text(token_ids, tokenizer)
    response_text = generated_text[len(input_text):].replace("### Response:", "").strip()
    test_data[i]["model_response"] = response_text  # 将模型响应添加到测试数据中

# 将测试数据保存到JSON文件中
with open("instruction-data-with-response.json", "w") as file:
    json.dump(test_data, file, indent=4)  # 使用缩进进行美化打印

# 打印测试数据的第一个条目以查看结果
print(test_data[0])

# 导入正则表达式模块
import re

# 根据CHOOSE_MODEL变量生成文件名，并移除其中的括号和空格
file_name = f"{re.sub(r'[ ()]', '', CHOOSE_MODEL)}-sft.pth"
# 保存模型状态字典到文件
torch.save(model.state_dict(), file_name)
print(f"Model saved as {file_name}")

# 导入psutil模块以检查进程是否正在运行
import psutil


def check_if_running(process_name):
    """检查指定名称的进程是否正在运行"""
    running = False
    for proc in psutil.process_iter(["name"]):
        if process_name in proc.info["name"]:
            running = True
            break
    return running


# 检查"ollama"进程是否正在运行
ollama_running = check_if_running("ollama")

# 如果"ollama"未运行，则抛出异常
if not ollama_running:
    raise RuntimeError("Ollama not running. Launch ollama before proceeding.")
# 打印"ollama"进程的运行状态
print("Ollama running:", check_if_running("ollama"))

# 导入json模块和tqdm模块
import json
from tqdm import tqdm

# 从文件中加载测试数据
file_path = "instruction-data-with-response.json"
with open(file_path, "r") as file:
    test_data = json.load(file)


def format_input(entry):
    """格式化输入文本，包括指令和（可选的）输入文本"""
    instruction_text = (
        f"Below is an instruction that describes a task. "
        f"Write a response that appropriately completes the request."
        f"\n\n### Instruction:\n{entry['instruction']}"
    )

    input_text = f"\n\n### Input:\n{entry['input']}" if entry["input"] else ""

    return instruction_text + input_text


import urllib.request  # 导入urllib.request模块用于发起HTTP请求

# 定义一个函数，用于向指定的模型服务器发送查询请求
def query_model(
    prompt,  # 用户输入的提示文本
    model="llama3",  # 要使用的模型名称，默认为"llama3"
    url="http://localhost:11434/api/chat"  # 模型服务器的URL地址
):
    # 构建请求数据，包括模型名称、消息和选项
    data = {
        "model": model,
        "messages": [
            {"role": "user", "content": prompt}  # 用户的消息，内容为输入的提示文本
        ],
        "options": {
            "seed": 123,  # 随机种子，用于保证结果的可复现性
            "temperature": 0,  # 温度参数，用于控制生成文本的随机性（0表示最不随机）
            "num_ctx": 2048  # 上下文大小
        }
    }

    # 将请求数据转换为JSON格式的字节串
    payload = json.dumps(data).encode("utf-8")

    # 创建HTTP POST请求
    request = urllib.request.Request(
        url,  # 请求的URL地址
        data=payload,  # 请求的数据
        method="POST"  # 请求的方法为POST
    )
    # 设置请求头，指定内容类型为application/json
    request.add_header("Content-Type", "application/json")

    # 发送请求并读取响应数据
    response_data = ""
    with urllib.request.urlopen(request) as response:
        while True:
            line = response.readline().decode("utf-8")  # 读取一行响应数据并解码
            if not line:
                break  # 如果读取到空行，则结束循环
            response_json = json.loads(line)  # 将读取到的JSON字符串转换为字典
            response_data += response_json["message"]["content"]  # 累加响应内容

    # 返回最终的响应数据
    return response_data

# 使用query_model函数查询模型对"What do Llamas eat?"的回答
model = "llama3"
result = query_model("What do Llamas eat?", model)
print(result)

# 遍历测试数据的前三个条目，并打印数据集响应、模型响应和模型得分
for entry in test_data[:3]:
    prompt = (
        f"Given the input `{format_input(entry)}` "
        f"and correct output `{entry['output']}`, "
        f"score the model response `{entry['model_response']}`"
        f" on a scale from 0 to 100, where 100 is the best score. "
    )
    print("\nDataset response:")
    print(">>", entry['output'])
    print("\nModel response:")
    print(">>", entry["model_response"])
    print("\nScore:")
    # 查询模型得分
    print(">>", query_model(prompt))
    print("\n-------------------------")

# 定义一个函数，用于生成模型得分
def generate_model_scores(json_data, json_key, model="llama3"):
    scores = []  # 初始化得分列表
    # 遍历json_data中的每个条目，并使用tqdm显示进度条
    for entry in tqdm(json_data, desc="Scoring entries"):
        prompt = (
            f"Given the input `{format_input(entry)}` "
            f"and correct output `{entry['output']}`, "
            f"score the model response `{entry[json_key]}`"
            f" on a scale from 0 to 100, where 100 is the best score. "
            f"Respond with the integer number only."
        )
        # 查询模型得分
        score = query_model(prompt, model)
        try:
            scores.append(int(score))  # 将得分转换为整数并添加到列表中
        except ValueError:
            print(f"Could not convert score: {score}")
            continue  # 如果得分无法转换为整数，则打印错误信息并继续下一个条目

    # 返回得分列表
    return scores

# 生成模型得分并打印相关信息
scores = generate_model_scores(test_data, "model_response")
print(f"Number of scores: {len(scores)} of {len(test_data)}")
print(f"Average score: {sum(scores)/len(scores):.2f}\n")