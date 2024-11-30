# 导入os模块，用于处理文件和目录路径
import os
# 导入json模块，用于处理JSON数据
import json
# 导入regex模块并重命名为re，用于进行复杂的正则表达式匹配
import regex as re
# 导入requests模块，用于发起HTTP请求
import requests
# 从tqdm模块导入tqdm，用于显示进度条
from tqdm import tqdm
# 从functools模块导入lru_cache，用于实现缓存功能，以减少函数调用的开销
from functools import lru_cache

# 使用lru_cache装饰器缓存bytes_to_unicode函数的返回值
@lru_cache()
def bytes_to_unicode():
    # 创建一个包含ASCII字符和一些扩展字符的列表
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    # 创建一个与bs相同的列表cs
    cs = bs[:]
    n = 0
    # 为不在bs中的字节值分配一个唯一的编码
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    # 将cs中的数值转换为字符
    cs = [chr(n) for n in cs]
    # 返回一个字典，将每个字节值映射到其唯一编码的字符
    return dict(zip(bs, cs))

# 定义一个函数，用于获取一个单词中所有相邻字符对
def get_pairs(word):
    pairs = set()
    prev_char = word[0]
    for char in word[1:]:
        pairs.add((prev_char, char))
        prev_char = char
    return pairs

# 定义一个Encoder类，用于文本编码
class Encoder:
    def __init__(self, encoder, bpe_merges, errors='replace'):
        # 初始化encoder字典
        self.encoder = encoder
        # 创建decoder字典，它是encoder字典的逆映射
        self.decoder = {v: k for k, v in self.encoder.items()}
        # 设置解码时遇到错误的处理方式
        self.errors = errors
        # 获取字节到唯一编码字符的映射
        self.byte_encoder = bytes_to_unicode()
        # 创建byte_decoder字典，它是byte_encoder字典的逆映射
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}
        # 创建一个字典，存储BPE合并规则的排名
        self.bpe_ranks = dict(zip(bpe_merges, range(len(bpe_merges))))
        # 创建一个缓存字典，用于存储已编码的token
        self.cache = {}
        # 编译一个正则表达式，用于匹配单词、数字和标点符号等
        self.pat = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")

    # 定义一个方法，用于应用BPE编码
    def bpe(self, token):
        # 如果token已经在缓存中，直接返回缓存的结果
        if token in self.cache:
            return self.cache[token]
        # 将token转换为字符元组
        word = tuple(token)
        # 获取token中所有相邻字符对
        pairs = get_pairs(word)

        # 如果没有字符对，直接返回原始token
        if not pairs:
            return token

        # 开始BPE编码过程
        while True:
            # 找到具有最低排名的字符对（即最优先合并的字符对）
            bigram = min(pairs, key=lambda pair: self.bpe_ranks.get(pair, float('inf')))
            # 如果该字符对不在BPE合并规则中，结束循环
            if bigram not in self.bpe_ranks:
                break
            first, second = bigram
            new_word = []
            i = 0
            # 遍历字符元组，应用BPE合并规则
            while i < len(word):
                try:
                    j = word.index(first, i)
                    new_word.extend(word[i:j])
                    i = j
                except ValueError:
                    new_word.extend(word[i:])
                    break

                # 检查是否可以将相邻的两个字符合并
                if word[i] == first and i < len(word) - 1 and word[i + 1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            # 更新字符元组
            new_word = tuple(new_word)
            word = new_word
            # 如果字符元组长度为1，结束循环
            if len(word) == 1:
                break
            else:
                # 更新字符对集合
                pairs = get_pairs(word)
        # 将编码后的字符元组转换回字符串
        word = ' '.join(word)
        # 将编码结果存入缓存
        self.cache[token] = word
        return word

    # 定义一个方法，用于将文本编码为BPE tokens
    def encode(self, text):
        bpe_tokens = []
        # 使用正则表达式匹配文本中的单词、数字和标点符号等
        for token in re.findall(self.pat, text):
            # 将token的字节值转换为唯一编码的字符
            token = ''.join(self.byte_encoder[b] for b in token.encode('utf-8'))
            # 对编码后的token应用BPE编码，并将结果转换为encoder中的tokens
            bpe_tokens.extend(self.encoder[bpe_token] for bpe_token in self.bpe(token).split(' '))
        return bpe_tokens

    # 定义一个方法，用于将BPE tokens解码为原始文本
    def decode(self, tokens):
        # 将tokens转换为字符串
        text = ''.join([self.decoder[token] for token in tokens])
        # 将字符串中的唯一编码字符转换回字节值
        text = bytearray([self.byte_decoder[c] for c in text]).decode('utf-8', errors=self.errors)
        return text

# 定义一个函数，用于加载指定模型的encoder和BPE合并规则
def get_encoder(model_name, models_dir):
    # 加载encoder.json文件
    with open(os.path.join(models_dir, model_name, 'encoder.json'), 'r') as f:
        encoder = json.load(f)
    # 加载vocab.bpe文件，并解析BPE合并规则
    with open(os.path.join(models_dir, model_name, 'vocab.bpe'), 'r', encoding="utf-8") as f:
        bpe_data = f.read()
    bpe_merges = [tuple(merge_str.split()) for merge_str in bpe_data.split('\n')[1:-1]]
    # 创建Encoder实例，并返回
    return Encoder(encoder=encoder, bpe_merges=bpe_merges)

# 定义一个函数，用于下载GPT-2模型的词汇表和编码器文件
def download_vocab():
    # 注释：以下是修改后的代码，原始来源未给出
    # 定义子目录名称，用于存放下载的GPT-2模型文件
    subdir = 'gpt2_model'
    # 检查该子目录是否存在，如果不存在则创建
    if not os.path.exists(subdir):
        os.makedirs(subdir)
    # 将子目录路径中的反斜杠替换为正斜杠，以确保在Windows系统上的路径兼容性
    subdir = subdir.replace('\\', '/')  # Windows系统需要这一步

    # 定义一个文件列表，包含需要下载的文件名
    for filename in ['encoder.json', 'vocab.bpe']:
        # 使用requests库发送GET请求，从指定的URL下载文件，并设置stream=True以流式传输数据
        r = requests.get("https://openaipublic.blob.core.windows.net/gpt-2/models/117M/" + filename, stream=True)

        # 打开（或创建）子目录下的文件，以二进制写入模式
        with open(os.path.join(subdir, filename), 'wb') as f:
            # 从响应头中获取文件大小
            file_size = int(r.headers["content-length"])
            # 设置每次读取的数据块大小（1000字节）
            chunk_size = 1000
            # 使用tqdm库创建一个进度条，用于显示下载进度
            # ncols=100设置进度条的宽度，desc设置进度条的描述，total设置总进度，unit_scale=True自动调整单位
            with tqdm(ncols=100, desc="Fetching " + filename, total=file_size, unit_scale=True) as pbar:
                # 由于以太网数据包大小约为1500字节，因此设置chunk_size为1000字节是合理的
                # 遍历响应内容的数据块
                for chunk in r.iter_content(chunk_size=chunk_size):
                    # 将数据块写入文件
                    f.write(chunk)
                    # 更新进度条的进度
                    pbar.update(chunk_size)