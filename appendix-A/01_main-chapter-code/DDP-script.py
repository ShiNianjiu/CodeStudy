# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt).
# Source for "Build a Large Language Model From Scratch"
#   - https://www.manning.com/books/build-a-large-language-model-from-scratch
# Code: https://github.com/rasbt/LLMs-from-scratch

# This file collects all the relevant code that we covered thus far
# throughout Chapters 2-4.
# This file can be run as a standalone script.


# 导入所需的库
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import os
import platform
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group


# 分布式进程组初始化函数（每个GPU一个进程）
# 允许进程间通信
def ddp_setup(rank, world_size):
    # 设置主节点地址和端口
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12345"
    # Windows系统特殊设置
    if platform.system() == "Windows":
        os.environ["USE_LIBUV"] = "0"

    # 初始化进程组
    if platform.system() == "Windows":
        init_process_group(backend="gloo", rank=rank, world_size=world_size)
    else:
        init_process_group(backend="nccl", rank=rank, world_size=world_size)

    # 设置当前进程使用的GPU
    torch.cuda.set_device(rank)


# 自定义数据集类
class ToyDataset(Dataset):
    def __init__(self, X, y):
        self.features = X
        self.labels = y

    def __getitem__(self, index):
        return self.features[index], self.labels[index]

    def __len__(self):
        return len(self.labels)


# 神经网络模型类
class NeuralNetwork(torch.nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(num_inputs, 30),
            torch.nn.ReLU(),
            torch.nn.Linear(30, 20),
            torch.nn.ReLU(),
            torch.nn.Linear(20, num_outputs),
        )

    def forward(self, x):
        return self.layers(x)


# 准备数据集函数
def prepare_dataset():
    # 定义训练数据和测试数据
    X_train, y_train = ..., ...
    X_test, y_test = ..., ...

    # 创建数据集和数据加载器
    train_ds = ToyDataset(X_train, y_train)
    test_ds = ToyDataset(X_test, y_test)

    train_loader = DataLoader(
        dataset=train_ds,
        batch_size=2,
        shuffle=False,
        pin_memory=True,
        drop_last=True,
        sampler=DistributedSampler(train_ds)  # 分布式采样器
    )
    test_loader = DataLoader(
        dataset=test_ds,
        batch_size=2,
        shuffle=False,
    )
    return train_loader, test_loader


# 主函数（分布式训练入口）
def main(rank, world_size, num_epochs):
    ddp_setup(rank, world_size)  # 初始化分布式进程组

    train_loader, test_loader = prepare_dataset()
    model = NeuralNetwork(num_inputs=2, num_outputs=2)
    model.to(rank)  # 将模型移动到指定的GPU
    optimizer = torch.optim.SGD(model.parameters(), lr=0.5)

    model = DDP(model, device_ids=[rank])  # 使用DDP包装模型

    for epoch in range(num_epochs):
        train_loader.sampler.set_epoch(epoch)  # 设置采样器的epoch

        model.train()
        for features, labels in train_loader:
            features, labels = features.to(rank), labels.to(rank)
            logits = model(features)
            loss = F.cross_entropy(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 打印日志
            print(...)

    # 评估模型
    model.eval()
    train_acc = compute_accuracy(model, train_loader, device=rank)
    print(f"[GPU{rank}] Training accuracy", train_acc)
    test_acc = compute_accuracy(model, test_loader, device=rank)
    print(f"[GPU{rank}] Test accuracy", test_acc)

    destroy_process_group()  # 清理分布式环境


# 计算准确率的函数
def compute_accuracy(model, dataloader, device):
    model = model.eval()
    correct = 0.0
    total_examples = 0

    for features, labels in dataloader:
        features, labels = features.to(device), labels.to(device)

        with torch.no_grad():
            logits = model(features)
        predictions = torch.argmax(logits, dim=1)
        correct += torch.sum(labels == predictions)
        total_examples += len(labels)
    return (correct / total_examples).item()


# 程序入口
if __name__ == "__main__":
    # 打印PyTorch和CUDA信息
    print(...)

    # 设置随机种子
    torch.manual_seed(123)

    # 使用mp.spawn启动分布式训练
    num_epochs = 3
    world_size = torch.cuda.device_count()
    mp.spawn(main, args=(world_size, num_epochs), nprocs=world_size)

