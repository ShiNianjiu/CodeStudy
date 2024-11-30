# 导入PyTorch库，这是一个流行的深度学习框架
import torch
# 从tqdm库中导入get_ipython函数，用于在Jupyter Notebook中测量代码执行时间
from tqdm.autonotebook import get_ipython


# 定义一个名为NeuralNetwork的类，它继承自torch.nn.Module，这是所有神经网络模块的基类
class NeuralNetwork(torch.nn.Module):
    # 类的初始化方法，接收输入和输出节点的数量作为参数
    def __init__(self, num_inputs, num_outputs):
        super().__init__()  # 调用父类的初始化方法

        # 使用torch.nn.Sequential容器按顺序堆叠网络层
        self.layers = torch.nn.Sequential(

            # 定义第一层隐藏层：线性层，将输入维度从num_inputs转换为30
            torch.nn.Linear(num_inputs, 30),
            # 激活函数ReLU，增加非线性
            torch.nn.ReLU(),

            # 定义第二层隐藏层：线性层，将输入维度从30转换为20
            torch.nn.Linear(30, 20),
            # 激活函数ReLU，增加非线性
            torch.nn.ReLU(),

            # 定义输出层：线性层，将输入维度从20转换为num_outputs
            torch.nn.Linear(20, num_outputs),
        )

    # 定义前向传播方法，接收输入x并返回输出logits
    def forward(self, x):
        logits = self.layers(x)  # 通过定义的层传递输入x
        return logits


# 实例化NeuralNetwork类，输入节点数为2，输出节点数为2
model = NeuralNetwork(2, 2)

# 计算并打印模型中所有可训练参数的总数
num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("Total number of trainable model parameters:", num_params)

# 再次导入torch库（实际上这行代码是多余的，因为已经导入过了）
import torch

# 创建两个随机张量a和b，分别具有形状(100, 200)和(200, 300)
a = torch.rand(100, 200)
b = torch.rand(200, 300)

# 使用Jupyter Notebook的魔法命令%timeit测量a和b矩阵乘法的执行时间
get_ipython().run_line_magic('timeit', 'a @ b')

# 将张量a和b移动到GPU上（如果可用）
a, b = a.to("cuda"), b.to("cuda")

# 再次使用Jupyter Notebook的魔法命令%timeit测量在GPU上a和b矩阵乘法的执行时间
get_ipython().run_line_magic('timeit', 'a @ b')