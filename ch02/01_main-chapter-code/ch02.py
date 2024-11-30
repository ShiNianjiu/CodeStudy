# 从importlib.metadata模块导入version函数，用于获取已安装包的版本
from importlib.metadata import version

# 打印torch库的版本
print("torch version:", version("torch"))
# 打印tiktoken库的版本（注意：tiktoken可能是一个假设的库名，实际中可能不存在）
print("tiktoken version:", version("tiktoken"))

# 导入os模块，用于处理文件和目录路径
import os
# 导入urllib.request模块，用于从URL下载文件
import urllib.request

# 检查文件"the-verdict.txt"是否存在
if not os.path.exists("../../01_main-chapter-code/the-verdict.txt"):
    # 如果不存在，设置要下载的文件的URL
    url = ("https://raw.githubusercontent.com/rasbt/"
           "LLMs-from-scratch/main/ch02/01_main-chapter-code/"
           "the-verdict.txt")
    # 设置文件保存路径
    file_path = "../../01_main-chapter-code/the-verdict.txt"
    # 使用urllib.request.urlretrieve函数下载文件
    urllib.request.urlretrieve(url, file_path)

# 使用with语句打开文件"the-verdict.txt"，确保文件使用完毕后自动关闭
with open("../../01_main-chapter-code/the-verdict.txt", "r", encoding="utf-8") as f:
    # 读取文件内容
    raw_text = f.read()

# 打印文件内容的字符总数
print("Total number of character:", len(raw_text))
# 打印文件内容的前99个字符
print(raw_text[:99])

# 导入re模块，用于正则表达式操作
import re

# 定义一个测试字符串
text = "Hello, world. This, is a test."
# 使用re.split函数按空白字符分割字符串，并保留分隔符
result = re.split(r'(\s)', text)
# 打印分割结果
print(result)

# 使用re.split函数按逗号、点或空白字符分割字符串，并保留分隔符
result = re.split(r'([,.]|\s)', text)
# 打印分割结果
print(result)

# 去除每个项目的前后空白字符，并过滤掉任何空字符串
result = [item for item in result if item.strip()]
# 打印处理后的结果
print(result)

# 定义另一个测试字符串，包含更多标点符号
text = "Hello, world. Is this-- a test?"
# 使用更复杂的正则表达式分割字符串，包括各种标点符号和空白字符
result = re.split(r'([,.:;?_!"()\']|--|\s)', text)
# 去除每个项目的前后空白字符，并过滤掉任何空字符串
result = [item.strip() for item in result if item.strip()]
# 打印处理后的结果
print(result)

# 对raw_text使用相同的分割和处理方法
preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
preprocessed = [item.strip() for item in preprocessed if item.strip()]
# 打印处理后的结果的前30项
print(preprocessed[:30])
# 打印处理后的结果的总数
print(len(preprocessed))

# 将处理后的结果转换为集合以去除重复项，然后排序
all_words = sorted(set(preprocessed))
# 计算词汇表的大小
vocab_size = len(all_words)
# 打印词汇表的大小
print(vocab_size)

# 创建一个词汇表字典，将每个词汇映射到一个唯一的整数
vocab = {token: integer for integer, token in enumerate(all_words)}

# 打印词汇表字典的前50项
for i, item in enumerate(vocab.items()):
    print(item)
    if i >= 50:
        break


# 定义一个简单的分词器类SimpleTokenizerV1
class SimpleTokenizerV1:
    def __init__(self, vocab):
        # 初始化词汇表映射
        self.str_to_int = vocab
        # 创建反向映射，将整数映射回词汇
        self.int_to_str = {i: s for s, i in vocab.items()}

    # 定义一个编码方法，将文本转换为整数列表
    def encode(self, text):
        # 对文本进行预处理
        preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)
        preprocessed = [item.strip() for item in preprocessed if item.strip()]
        # 将预处理后的词汇映射为整数
        ids = [self.str_to_int[s] for s in preprocessed]
        return ids

    # 定义一个解码方法，将整数列表转换回文本
    def decode(self, ids):
        # 将整数列表映射回词汇
        text = " ".join([self.int_to_str[i] for i in ids])
        # 替换指定标点符号前的空格
        text = re.sub(r'\s+([,.?!"()\'])', r'\1', text)
        return text


# 创建SimpleTokenizerV1的实例
tokenizer = SimpleTokenizerV1(vocab)

# 测试编码方法
text = """"It's the last he painted, you know," 
           Mrs. Gisburn said with pardonable pride."""
ids = tokenizer.encode(text)
print(ids)

# 测试解码方法
tokenizer.decode(ids)

# 测试编码后解码的完整性
tokenizer.decode(tokenizer.encode(text))

# 重新创建SimpleTokenizerV1的实例（此步骤在代码中是多余的，但可能是为了演示）
tokenizer = SimpleTokenizerV1(vocab)

# 测试另一个文本
text = "Hello, do you like tea. Is this-- a test?"
# 对文本进行编码
tokenizer.encode(text)

# 对预处理后的词汇进行去重、排序，并添加特殊标记<|endoftext|>和<|unk|>
all_tokens = sorted(list(set(preprocessed)))
all_tokens.extend(["<|endoftext|>", "<|unk|>"])

# 重新创建词汇表字典
vocab = {token: integer for integer, token in enumerate(all_tokens)}

# 打印词汇表字典的项数（这一步是多余的，因为结果没有赋值给变量或打印出来）
len(vocab.items())

# 打印词汇表字典的最后5项
for i, item in enumerate(list(vocab.items())[-5:]):
    print(item)


# 定义一个改进的分词器类SimpleTokenizerV2
class SimpleTokenizerV2:
    def __init__(self, vocab):
        # 初始化词汇表映射
        self.str_to_int = vocab
        # 创建反向映射，将整数映射回词汇
        self.int_to_str = {i: s for s, i in vocab.items()}

    # 定义一个编码方法，将文本转换为整数列表，对于不在词汇表中的词汇使用<|unk|>代替
    def encode(self, text):
        # 对文本进行预处理
        preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)
        preprocessed = [item.strip() for item in preprocessed if item.strip()]
        # 将预处理后的词汇映射为整数，不在词汇表中的词汇映射为<|unk|>
        preprocessed = [item if item in self.str_to_int else "<|unk|>" for item in preprocessed]
        ids = [self.str_to_int[s] for s in preprocessed]
        return ids

    # 解码方法与SimpleTokenizerV1相同
    def decode(self, ids):
        text = " ".join([self.int_to_str[i] for i in ids])
        text = re.sub(r'\s+([,.?!"()\'])', r'\1', text)
        return text


# 创建SimpleTokenizerV2的实例
tokenizer = SimpleTokenizerV2(vocab)

# 定义两个测试文本
text1 = "Hello, do you like tea?"
text2 = "In the sunlit terraces of the palace."
# 使用<|endoftext|>连接两个文本
text = " <|endoftext|> ".join((text1, text2))

# 对连接后的文本进行编码
tokenizer.encode(text)

# 对编码后的文本进行解码，验证编码解码的完整性
tokenizer.decode(tokenizer.encode(text))

# 导入importlib模块，用于动态导入模块和获取模块元数据
import importlib
# 导入tiktoken库，用于文本编码和解码（注意：这里可能是个笔误，正确的库名可能是transformers或类似的，但这里按照代码原样注释）
import tiktoken

# 打印tiktoken库的版本
print("tiktoken version:", importlib.metadata.version("tiktoken"))

# 获取GPT-2模型的编码器
tokenizer = tiktoken.get_encoding("gpt2")

# 定义要编码的文本，包含特殊标记<|endoftext|>
text = (
    "Hello, do you like tea? <|endoftext|> In the sunlit terraces "
    "of someunknownPlace."
)

# 使用tokenizer对文本进行编码，允许特殊标记<|endoftext|>
integers = tokenizer.encode(text, allowed_special={"<|endoftext|>"})

# 打印编码后的整数序列
print(integers)

# 将编码后的整数序列解码回原始文本（或接近原始文本的格式）
strings = tokenizer.decode(integers)

# 打印解码后的文本
print(strings)

# 打开并读取名为"the-verdict.txt"的文件内容
with open("../../01_main-chapter-code/the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

# 对读取的文本进行编码
enc_text = tokenizer.encode(raw_text)
# 打印编码后文本的长度
print(len(enc_text))

# 取编码后文本的一部分（从第50个元素开始到末尾）
enc_sample = enc_text[50:]

# 定义上下文大小
context_size = 4

# 定义x和y，x是enc_sample的前context_size个元素，y是x后移一个元素的结果
x = enc_sample[:context_size]
y = enc_sample[1:context_size + 1]

# 打印x和y
print(f"x: {x}")
print(f"y:      {y}")

# 遍历上下文大小范围内的每个位置，打印当前上下文和期望的下一个元素
for i in range(1, context_size + 1):
    context = enc_sample[:i]
    desired = enc_sample[i]
    print(context, "---->", desired)

# 类似于上面的循环，但这次打印的是解码后的文本
for i in range(1, context_size + 1):
    context = enc_sample[:i]
    desired = enc_sample[i]
    print(tokenizer.decode(context), "---->", tokenizer.decode([desired]))

# 导入PyTorch库，并打印PyTorch的版本
import torch

print("PyTorch version:", torch.__version__)

# 从PyTorch的utils.data模块导入Dataset和DataLoader类
from torch.utils.data import Dataset, DataLoader


# 定义一个用于GPT模型的数据集类GPTDatasetV1
class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        # 初始化输入和目标ID列表
        self.input_ids = []
        self.target_ids = []

        # 对整个文本进行编码
        token_ids = tokenizer.encode(txt, allowed_special={"<|endoftext|>"})

        # 使用滑动窗口将文本分割成重叠的序列
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i + 1: i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    # 实现__len__方法，返回数据集的大小
    def __len__(self):
        return len(self.input_ids)

    # 实现__getitem__方法，支持通过索引获取数据
    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]


# 定义一个函数，用于创建数据加载器
def create_dataloader_v1(txt, batch_size=4, max_length=256,
                         stride=128, shuffle=True, drop_last=True,
                         num_workers=0):
    # 初始化tokenizer
    tokenizer = tiktoken.get_encoding("gpt2")

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


# 再次打开并读取"the-verdict.txt"文件内容
with open("../../01_main-chapter-code/the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

# 创建数据加载器，并设置一些参数
dataloader = create_dataloader_v1(
    raw_text, batch_size=1, max_length=4, stride=1, shuffle=False
)

# 创建数据加载器的迭代器
data_iter = iter(dataloader)
# 获取第一个批次的数据并打印
first_batch = next(data_iter)
print(first_batch)

# 获取第二个批次的数据并打印
second_batch = next(data_iter)
print(second_batch)

# 创建另一个数据加载器，这次使用不同的参数
dataloader = create_dataloader_v1(raw_text, batch_size=8, max_length=4, stride=4, shuffle=False)

# 创建数据加载器的迭代器
data_iter = iter(dataloader)
# 获取并打印一个批次的数据
inputs, targets = next(data_iter)
print("Inputs:\n", inputs)
print("\nTargets:\n", targets)

# 定义一个张量，表示输入ID
input_ids = torch.tensor([2, 3, 5, 1])

# 定义词汇表大小和输出维度
vocab_size = 6
output_dim = 3

# 设置随机种子，并创建一个嵌入层
torch.manual_seed(123)
embedding_layer = torch.nn.Embedding(vocab_size, output_dim)

# 打印嵌入层的权重
print(embedding_layer.weight)

# 打印单个ID的嵌入表示
print(embedding_layer(torch.tensor([3])))

# 打印输入ID的嵌入表示
print(embedding_layer(input_ids))

# 定义新的词汇表大小和输出维度（通常用于大型模型）
vocab_size = 50257
output_dim = 256

# 创建一个新的嵌入层，用于处理更大的词汇表
token_embedding_layer = torch.nn.Embedding(vocab_size, output_dim)

# 定义上下文长度（即序列的最大长度）
max_length = 4
# 创建数据加载器，设置与上下文长度相同的步长
dataloader = create_dataloader_v1(
    raw_text, batch_size=8, max_length=max_length,
    stride=max_length, shuffle=False
)
data_iter = iter(dataloader)
# 获取并打印一个批次的数据
inputs, targets = next(data_iter)

# 打印输入的Token ID
print("Token IDs:\n", inputs)
# 打印输入的形状
print("\nInputs shape:\n", inputs.shape)

# 计算Token的嵌入表示
token_embeddings = token_embedding_layer(inputs)
# 打印Token嵌入的形状
print(token_embeddings.shape)

# 定义位置嵌入层，用于处理位置信息
context_length = max_length
pos_embedding_layer = torch.nn.Embedding(context_length, output_dim)

# 计算位置嵌入表示
pos_embeddings = pos_embedding_layer(torch.arange(max_length))
# 打印位置嵌入的形状
print(pos_embeddings.shape)

# 将Token嵌入和位置嵌入相加，得到最终的输入嵌入表示
input_embeddings = token_embeddings + pos_embeddings
# 打印最终输入嵌入的形状
print(input_embeddings.shape)