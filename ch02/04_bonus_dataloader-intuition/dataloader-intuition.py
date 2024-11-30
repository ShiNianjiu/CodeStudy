# 从importlib.metadata导入version函数，用于获取包的版本
from importlib.metadata import version
import torch

# 打印torch包的版本号
print("torch version:", version("torch"))

# 打开一个文件用于写入数字，编码为utf-8
with open("number-data.txt", "w", encoding="utf-8") as f:
    # 循环写入0到1000的数字，每个数字后面跟一个空格
    for number in range(1001):
        f.write(f"{number} ")

# 从torch.utils.data导入Dataset和DataLoader类
from torch.utils.data import Dataset, DataLoader

# 定义一个数据集类GPTDatasetV1，用于处理文本数据
class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        # 初始化输入和目标的id列表
        self.input_ids = []
        self.target_ids = []

        # 将文本转换为token id列表（这里假设txt已经是token id的字符串形式，实际使用中需要tokenizer.encode）
        # 注意：原代码中的tokenizer.encode部分被注释掉了，这里直接使用了txt.strip().split()转换
        token_ids = [int(i) for i in txt.strip().split()]

        # 使用滑动窗口将文本分割成重叠的序列
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i + max_length]  # 输入序列
            target_chunk = token_ids[i + 1: i + max_length + 1]  # 目标序列（输入序列向右移动一个token）
            self.input_ids.append(torch.tensor(input_chunk))  # 将序列转换为tensor并添加到列表中
            self.target_ids.append(torch.tensor(target_chunk))  # 同上

    # 返回数据集的大小
    def __len__(self):
        return len(self.input_ids)

    # 根据索引获取数据集中的项
    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]

# 定义一个函数create_dataloader_v1，用于创建数据加载器
def create_dataloader_v1(txt, batch_size=4, max_length=256, stride=128, shuffle=True, drop_last=True, num_workers=0):
    # 注意：这里tokenizer被设置为None，实际使用中需要初始化一个tokenizer
    tokenizer = None

    # 创建数据集
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)

    # 创建数据加载器
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers
    )

    return dataloader

# 读取之前写入的数字文本
with open("number-data.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

# 创建数据加载器，并获取第一批数据
dataloader = create_dataloader_v1(raw_text, batch_size=1, max_length=4, stride=1, shuffle=False)
data_iter = iter(dataloader)
first_batch = next(data_iter)
print(first_batch)

# 获取第二批数据
second_batch = next(data_iter)
print(second_batch)

# 获取第三批数据
third_batch = next(data_iter)
print(third_batch)

# 遍历数据加载器，但不处理数据，只为了获取最后一批数据（由于drop_last=True，最后一批可能不会被处理）
for batch in dataloader:
    pass
last_batch = batch  # 注意：这里的last_batch可能是未定义的，因为dataloader可能已经被完全遍历
print(last_batch)  # 这行代码可能会引发错误，因为last_batch可能没有在循环中被赋值

# 重新创建一个数据加载器，设置不同的参数，并遍历它但不处理数据

dataloader = create_dataloader_v1(raw_text, batch_size=2, max_length=4, stride=4, shuffle=False)
for inputs, targets in dataloader:
    pass
print("Inputs:\n", inputs)
print("\nTargets:\n", targets)

# 最后，创建一个带有打乱功能的数据加载器
torch.manual_seed(123)  # 设置随机种子以确保结果的可重复性
dataloader = create_dataloader_v1(raw_text, batch_size=2, max_length=4, stride=4, shuffle=True)
for inputs, targets in dataloader:
    pass
print("Inputs:\n", inputs)
print("\nTargets:\n", targets)