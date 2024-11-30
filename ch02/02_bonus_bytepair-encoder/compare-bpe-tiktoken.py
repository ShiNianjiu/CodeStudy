# 从importlib.metadata模块导入version函数，用于获取安装的包的版本
from importlib.metadata import version

from tqdm.autonotebook import get_ipython

# 打印tiktoken包的版本信息
print("tiktoken version:", version("tiktoken"))

# 导入tiktoken包，用于文本编码和解码
import tiktoken

# 使用tiktoken包的get_encoding函数获取GPT-2模型的编码器
tik_tokenizer = tiktoken.get_encoding("gpt2")

# 定义需要编码的文本
text = "Hello, world. Is this-- a test?"

# 使用tik_tokenizer的encode方法将文本编码为整数列表，允许特殊标记<|endoftext|>
integers = tik_tokenizer.encode(text, allowed_special={"<|endoftext|>"})

# 打印编码后的整数列表
print(integers)

# 使用tik_tokenizer的decode方法将整数列表解码回文本
strings = tik_tokenizer.decode(integers)

# 打印解码后的文本
print(strings)

# 打印编码器的词汇表大小
print(tik_tokenizer.n_vocab)

# 从bpe_openai_gpt2模块导入get_encoder和download_vocab函数
from bpe_openai_gpt2 import get_encoder, download_vocab

# 下载GPT-2模型的词汇表
download_vocab()

# 使用get_encoder函数获取GPT-2模型的原始编码器，指定模型名称和模型目录
orig_tokenizer = get_encoder(model_name="gpt2_model", models_dir=".")

# 使用原始编码器的encode方法将文本编码为整数列表
integers = orig_tokenizer.encode(text)

# 打印编码后的整数列表
print(integers)

# 使用原始编码器的decode方法将整数列表解码回文本
strings = orig_tokenizer.decode(integers)

# 打印解码后的文本
print(strings)

# 导入transformers包，用于处理自然语言任务
import transformers

# 获取transformers包的版本信息
transformers.__version__

# 从transformers包导入GPT2Tokenizer类
from transformers import GPT2Tokenizer

# 使用GPT2Tokenizer的from_pretrained方法加载预训练的GPT-2模型编码器
hf_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# 使用transformers的编码器处理字符串，获取输入ID（此行代码没有输出）
hf_tokenizer(strings)["input_ids"]

# 打开并读取文本文件的内容
with open('../01_main-chapter-code/the-verdict.txt', 'r', encoding='utf-8') as f:
    raw_text = f.read()

# 使用IPython的magic命令timeit测量原始编码器编码大文本的时间
get_ipython().run_line_magic('timeit', 'orig_tokenizer.encode(raw_text)')

# 使用IPython的magic命令timeit测量tiktoken编码器编码大文本的时间
get_ipython().run_line_magic('timeit', 'tik_tokenizer.encode(raw_text)')

# 使用IPython的magic命令timeit测量transformers的GPT2Tokenizer编码大文本的时间
get_ipython().run_line_magic('timeit', 'hf_tokenizer(raw_text)["input_ids"]')

# 使用IPython的magic命令timeit测量transformers的GPT2Tokenizer在限制最大长度和截断的情况下编码大文本的时间
get_ipython().run_line_magic('timeit', 'hf_tokenizer(raw_text, max_length=5145, truncation=True)["input_ids"]')