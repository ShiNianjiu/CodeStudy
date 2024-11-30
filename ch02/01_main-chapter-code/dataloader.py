# 从importlib.metadata模块导入version函数，用于获取安装包的版本
from importlib.metadata import version

# 打印torch库的版本
print("torch version:", version("torch"))
# 打印tiktoken库的版本
print("tiktoken version:", version("tiktoken"))

# 导入tiktoken库，用于文本分词
import tiktoken
# 导入torch库，用于深度学习
import torch
# 从torch.utils.data导入Dataset和DataLoader，用于创建数据集和数据加载器
from torch.utils.data import Dataset, DataLoader

# 定义一个名为GPTDatasetV1的类，继承自Dataset，用于处理GPT模型的数据集
class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []  # 存储输入序列的ID
        self.target_ids = []  # 存储目标序列的ID

        # 使用tokenizer对整个文本进行分词
        token_ids = tokenizer.encode(txt, allowed_special={"<|endoftext|>"})

        # 使用滑动窗口将文本分割成重叠的max_length长度的序列
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i + max_length]  # 输入序列
            target_chunk = token_ids[i + 1: i + max_length + 1]  # 目标序列（输入序列向右移动一位）
            self.input_ids.append(torch.tensor(input_chunk))  # 将输入序列转换为tensor并添加到列表中
            self.target_ids.append(torch.tensor(target_chunk))  # 将目标序列转换为tensor并添加到列表中

    def __len__(self):
        return len(self.input_ids)  # 返回数据集中样本的数量

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]  # 根据索引返回输入序列和目标序列

# 定义一个函数create_dataloader_v1，用于创建数据加载器
def create_dataloader_v1(txt, batch_size=4, max_length=256,
                         stride=128, shuffle=True, drop_last=True, num_workers=0):
    # 初始化tokenizer
    tokenizer = tiktoken.get_encoding("gpt2")

    # 创建数据集
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)

    # 创建数据加载器
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, num_workers=num_workers)

    return dataloader

# 读取文件"the-verdict.txt"的内容
with open("../../01_main-chapter-code/the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

# 初始化tokenizer
tokenizer = tiktoken.get_encoding("gpt2")
# 对原始文本进行分词
encoded_text = tokenizer.encode(raw_text)

# 定义词汇表大小、输出维度和上下文长度
vocab_size = 50257
output_dim = 256
context_length = 1024

# 创建词嵌入层和位置嵌入层
token_embedding_layer = torch.nn.Embedding(vocab_size, output_dim)
pos_embedding_layer = torch.nn.Embedding(context_length, output_dim)

# 设置max_length变量（这里为了演示设置为4，实际使用中可能不同）
max_length = 4
# 创建数据加载器
dataloader = create_dataloader_v1(raw_text, batch_size=8, max_length=max_length, stride=max_length)

# 遍历数据加载器中的批次
for batch in dataloader:
    x, y = batch  # x是输入序列，y是目标序列

    # 获取词嵌入
    token_embeddings = token_embedding_layer(x)
    # 获取位置嵌入（这里为了简单起见，使用0到max_length-1的整数作为位置索引）
    pos_embeddings = pos_embedding_layer(torch.arange(max_length))

    # 将词嵌入和位置嵌入相加得到输入嵌入
    input_embeddings = token_embeddings + pos_embeddings

    break  # 仅处理第一个批次作为示例

# 打印输入嵌入的形状
print(input_embeddings.shape)

