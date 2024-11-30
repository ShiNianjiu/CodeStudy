# 导入PyTorch库
import torch

# 打印当前PyTorch的版本号
print("PyTorch version:", torch.__version__)

# 创建一个包含索引的tensor，这些索引将用于从嵌入矩阵中检索嵌入向量
idx = torch.tensor([2, 3, 1])

# 计算嵌入矩阵的大小（即词汇表的大小），加1是因为索引是从0开始的
num_idx = max(idx)+1

# 嵌入向量的维度（一个超参数）
out_dim = 5

# 设置随机种子以确保结果的可重复性
torch.manual_seed(123)

# 创建一个嵌入层，其输入大小为词汇表大小，输出大小为嵌入向量的维度
embedding = torch.nn.Embedding(num_idx, out_dim)

# 打印嵌入层的权重（初始化为随机值）
embedding.weight

# 使用索引tensor检索单个嵌入向量（索引为1的嵌入向量）
embedding(torch.tensor([1]))

# 使用索引tensor检索单个嵌入向量（索引为2的嵌入向量）
embedding(torch.tensor([2]))

# 使用索引tensor批量检索嵌入向量
idx = torch.tensor([2, 3, 1])
embedding(idx)

# 将索引tensor转换为one-hot编码
onehot = torch.nn.functional.one_hot(idx)
# 打印one-hot编码
onehot

# 再次设置随机种子（虽然这里不影响结果，但为了保持一致性）
torch.manual_seed(123)
# 创建一个线性层，其输入特征数为词汇表大小，输出特征数为嵌入向量的维度，且不带偏置项
linear = torch.nn.Linear(num_idx, out_dim, bias=False)
# 打印线性层的权重（初始化为随机值）
linear.weight

# 将线性层的权重设置为嵌入层权重的转置，这样线性层就可以通过one-hot编码的输入得到相同的输出
linear.weight = torch.nn.Parameter(embedding.weight.T)

# 使用one-hot编码作为输入，通过线性层得到输出
linear(onehot.float())

# 再次使用索引tensor批量检索嵌入向量，以验证嵌入层的功能
embedding(idx)