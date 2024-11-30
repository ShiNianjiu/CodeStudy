# 导入 version 函数从 importlib.metadata 模块，用于获取安装的包的版本
from importlib.metadata import version

# 打印 torch 和 tiktoken 包的版本信息
print("torch version:", version("torch"))
print("tiktoken version:", version("tiktoken"))

# 导入 tiktoken 库
import tiktoken

# 获取 GPT-2 模型的编码器
tokenizer = tiktoken.get_encoding("gpt2")

# 编码文本 "Akwirw ier" 并打印编码后的整数列表
integers = tokenizer.encode("Akwirw ier")
print(integers)

# 遍历编码后的整数列表，并打印每个整数及其对应的解码文本
for i in integers:
    print(f"{i} -> {tokenizer.decode([i])}")

# 分别对 "Ak", "w", "ir", "w", " ", "ier" 进行编码（这里只是调用函数，未使用返回值）
tokenizer.encode("Ak")
tokenizer.encode("w")
tokenizer.encode("ir")
tokenizer.encode("w")
tokenizer.encode(" ")
tokenizer.encode("ier")

# 解码给定的整数列表 [33901, 86, 343, 86, 220, 959] 并打印解码后的文本
tokenizer.decode([33901, 86, 343, 86, 220, 959])

# 再次导入 tiktoken 和 torch 库（实际代码中可以避免重复导入）
import tiktoken
import torch
from torch.utils.data import Dataset, DataLoader

# 定义一个用于处理 GPT 数据集的类 GPTDatasetV1
class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []

        # 使用 tokenizer 对整个文本进行编码
        token_ids = tokenizer.encode(txt, allowed_special={"<|endoftext|>"})

        # 使用滑动窗口将文本分割成重叠的序列，每个序列的长度为 max_length
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i + max_length]  # 输入序列
            target_chunk = token_ids[i + 1: i + max_length + 1]  # 目标序列（输入序列向右移动一个位置）
            self.input_ids.append(torch.tensor(input_chunk))  # 将输入序列添加到列表中
            self.target_ids.append(torch.tensor(target_chunk))  # 将目标序列添加到列表中

    def __len__(self):
        # 返回数据集的大小
        return len(self.input_ids)

    def __getitem__(self, idx):
        # 根据索引获取数据集中的输入序列和目标序列
        return self.input_ids[idx], self.target_ids[idx]

# 定义一个函数 create_dataloader，用于创建数据加载器
def create_dataloader(txt, batch_size=4, max_length=256, stride=128):
    # 初始化 tokenizer
    tokenizer = tiktoken.get_encoding("gpt2")

    # 创建数据集
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)

    # 创建数据加载器
    dataloader = DataLoader(dataset, batch_size=batch_size)

    return dataloader

# 读取文本文件并获取原始文本
with open("../../01_main-chapter-code/the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

# 再次获取 GPT-2 模型的编码器（实际代码中可以避免重复获取）
tokenizer = tiktoken.get_encoding("gpt2")
# 对原始文本进行编码
encoded_text = tokenizer.encode(raw_text)

# 定义一些参数
vocab_size = 50257  # 词汇表大小
output_dim = 256  # 输出维度
max_len = 4  # 最大长度（此变量后续未使用）
context_length = max_len  # 上下文长度（这里等于 max_len）

# 创建词嵌入层和位置嵌入层
token_embedding_layer = torch.nn.Embedding(context_length, output_dim)  # 注意：这里的 context_length 可能并不适合实际用途
pos_embedding_layer = torch.nn.Embedding(vocab_size, output_dim)

# 创建数据加载器，设置 batch_size=4, max_length=2, stride=2
dataloader = create_dataloader(raw_text, batch_size=4, max_length=2, stride=2)

# 遍历数据加载器，获取一个批次的数据并打印（然后跳出循环）
for batch in dataloader:
    x, y = batch
    break

# 再次创建数据加载器，设置 batch_size=4, max_length=8, stride=2
dataloader = create_dataloader(raw_text, batch_size=4, max_length=8, stride=2)

# 遍历数据加载器，获取一个批次的数据并打印（然后跳出循环）
for batch in dataloader:
    x, y = batch
    break