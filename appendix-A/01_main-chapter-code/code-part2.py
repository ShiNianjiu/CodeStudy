# 导入PyTorch库
import torch

# 打印PyTorch库的版本信息
print(torch.__version__)

# 检查CUDA是否可用，即是否有支持CUDA的GPU可用
print(torch.cuda.is_available())

# 创建两个一维张量（tensor）
tensor_1 = torch.tensor([1., 2., 3.])
tensor_2 = torch.tensor([4., 5., 6.])

# 打印两个张量相加的结果
print(tensor_1 + tensor_2)

# 将两个张量移动到CUDA设备（如果有可用的GPU）
tensor_1 = tensor_1.to("cuda")
tensor_2 = tensor_2.to("cuda")

# 打印在CUDA设备上两个张量相加的结果
print(tensor_1 + tensor_2)

# 将两个张量移回CPU设备
tensor_1 = tensor_1.to("cpu")
# 由于tensor_2仍在CUDA设备上，这里需要确保tensor_2也被移回CPU，或者保持tensor_1在CUDA上，
# 但为了演示，我们将它们都移回CPU并打印相加结果（注意：实际运行时tensor_2也应被移回CPU）
# 假设tensor_2已经被移回CPU，打印相加结果
print(tensor_1 + tensor_2)

# 创建训练数据集和标签
X_train = torch.tensor([
    [-1.2, 3.1],
    [-0.9, 2.9],
    [-0.5, 2.6],
    [2.3, -1.1],
    [2.7, -1.5]
])
y_train = torch.tensor([0, 0, 0, 1, 1])

# 创建测试数据集和标签
X_test = torch.tensor([
    [-0.8, 2.8],
    [2.6, -1.6],
])
y_test = torch.tensor([0, 1])

# 从torch.utils.data导入Dataset类
from torch.utils.data import Dataset

# 定义一个自定义的数据集类
class ToyDataset(Dataset):
    def __init__(self, X, y):
        self.features = X  # 特征
        self.labels = y  # 标签

    def __getitem__(self, index):
        # 根据索引获取单个样本及其标签
        one_x = self.features[index]
        one_y = self.labels[index]
        return one_x, one_y

    def __len__(self):
        # 返回数据集中的样本总数
        return self.labels.shape[0]

# 创建训练和测试数据集实例
train_ds = ToyDataset(X_train, y_train)
test_ds = ToyDataset(X_test, y_test)

# 从torch.utils.data导入DataLoader类
from torch.utils.data import DataLoader

# 设置随机种子以确保结果的可重复性
torch.manual_seed(123)

# 创建训练数据加载器
train_loader = DataLoader(
    dataset=train_ds,  # 数据集
    batch_size=2,  # 每个批次的大小
    shuffle=True,  # 是否在每个epoch开始时打乱数据
    num_workers=1,  # 加载数据时使用的进程数
    drop_last=True  # 是否丢弃最后一个不完整的批次
)

# 创建测试数据加载器
test_loader = DataLoader(
    dataset=test_ds,
    batch_size=2,
    shuffle=False,
    num_workers=1
)

# 定义一个简单的神经网络模型
class NeuralNetwork(torch.nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super().__init__()

        self.layers = torch.nn.Sequential(
            # 第一个隐藏层
            torch.nn.Linear(num_inputs, 30),
            torch.nn.ReLU(),

            # 第二个隐藏层
            torch.nn.Linear(30, 20),
            torch.nn.ReLU(),

            # 输出层
            torch.nn.Linear(20, num_outputs),
        )

    def forward(self, x):
        # 前向传播
        logits = self.layers(x)
        return logits

# 导入torch.nn.functional模块，以便使用其中的函数（如cross_entropy）
import torch.nn.functional as F

# 设置随机种子，确保模型初始化的可重复性
torch.manual_seed(123)
# 实例化神经网络模型，指定输入和输出的大小
model = NeuralNetwork(num_inputs=2, num_outputs=2)

# 根据是否有可用的CUDA设备选择设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 将模型移动到所选设备
model = model.to(device)

# 实例化优化器，指定模型参数和学习率
optimizer = torch.optim.SGD(model.parameters(), lr=0.5)

# 设置训练的轮数（epochs）
num_epochs = 3

# 训练模型
for epoch in range(num_epochs):
    model.train()  # 设置模型为训练模式
    for batch_idx, (features, labels) in enumerate(train_loader):
        # 将数据和标签移动到所选设备
        features, labels = features.to(device), labels.to(device)
        # 前向传播
        logits = model(features)
        # 计算损失
        loss = F.cross_entropy(logits, labels)

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 打印训练日志
        print(f"Epoch: {epoch+1:03d}/{num_epochs:03d}"
              f" | Batch {batch_idx:03d}/{len(train_loader):03d}"
              f" | Train/Val Loss: {loss:.2f}")

    model.eval()  # 设置模型为评估模式（在测试时不需要梯度计算）

# 定义一个函数来计算模型在给定数据加载器上的准确率
def compute_accuracy(model, dataloader, device):
    model = model.eval()  # 确保模型在评估模式下
    correct = 0.0  # 正确预测的样本数
    total_examples = 0  # 总样本数

    for idx, (features, labels) in enumerate(dataloader):
        # 将数据和标签移动到所选设备
        features, labels = features.to(device), labels.to(device)

        with torch.no_grad():  # 禁用梯度计算
            logits = model(features)  # 前向传播

        # 获取预测结果（类别索引）
        predictions = torch.argmax(logits, dim=1)
        # 比较预测结果和真实标签
        compare = labels == predictions
        # 计算正确预测的样本数
        correct += torch.sum(compare)
        # 更新总样本数
        total_examples += len(compare)

    # 计算并返回准确率
    return (correct / total_examples).item()

# 计算并打印模型在训练数据上的准确率
compute_accuracy(model, train_loader, device=device)

# 计算并打印模型在测试数据上的准确率
compute_accuracy(model, test_loader, device=device)