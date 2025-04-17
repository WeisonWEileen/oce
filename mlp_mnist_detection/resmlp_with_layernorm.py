import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import random
from tqdm import tqdm

# ------------------ 设置全局随机种子 ------------------
SEED = 42  # 可修改为任意整数

def set_seed(seed):
    # Python 随机模块
    random.seed(seed)
    # NumPy
    np.random.seed(seed)
    # PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 多GPU时启用
    # CUDA 确定性模式（可能降低性能，但保证可复现）
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(SEED)

# ------------------ 其余代码保持不变 ------------------
# 超参数配置
BATCH_SIZE = 128
EPOCHS = 15
HIDDEN_DIM = 512
NUM_BLOCKS = 6
DROPOUT = 0.2
LR = 1e-3
WEIGHT_DECAY = 1e-4
LABEL_SMOOTHING = 0.1

# 设备配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 数据预处理（注意：DataLoader 需要指定 generator）
train_transform = transforms.Compose([
    transforms.RandomRotation(10),
    transforms.RandomAffine(0, translate=(0.1, 0.1)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# 加载数据集（关键：设置 worker_init_fn 和 generator）
def seed_worker(worker_id):
    worker_seed = SEED % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

g = torch.Generator()
g.manual_seed(SEED)

train_set = datasets.MNIST(root='./data', train=True, download=True, transform=train_transform)
test_set = datasets.MNIST(root='./data', train=False, download=True, transform=test_transform)

train_loader = DataLoader(
    train_set,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=2,
    worker_init_fn=seed_worker,
    generator=g
)

test_loader = DataLoader(
    test_set,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=2,
    worker_init_fn=seed_worker,
    generator=g
)

# 模型定义（与之前相同）
class ResidualMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_fc = nn.Linear(784, HIDDEN_DIM)
        
        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
                nn.BatchNorm1d(HIDDEN_DIM),
                nn.GELU(),
                nn.Dropout(DROPOUT),
                nn.Linear(HIDDEN_DIM, HIDDEN_DIM)
            ) for _ in range(NUM_BLOCKS)
        ])
        
        self.bn = nn.BatchNorm1d(HIDDEN_DIM)
        self.output_fc = nn.Linear(HIDDEN_DIM, 10)

    def forward(self, x):
        x = x.view(-1, 784)
        x = self.input_fc(x)
        
        for block in self.blocks:
            residual = x
            x = block(x)
            x = residual + x  # 残差连接
            
        x = self.bn(x)
        x = self.output_fc(x)
        return x

# 初始化模型
model = ResidualMLP().to(device)
optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)

# 训练和测试函数（与之前相同）
def train(epoch):
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{EPOCHS}')
    for inputs, targets in pbar:
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
        pbar.set_postfix({'Loss': f"{train_loss/(total/BATCH_SIZE):.3f}", 
                         'Acc': f"{100.*correct/total:.2f}%"})

def test():
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    acc = 100. * correct / total
    print(f"\nTest Results: Loss: {test_loss/len(test_loader):.4f} | Accuracy: {acc:.2f}%\n")
    return acc

# 主训练循环
best_acc = 0.0
for epoch in range(EPOCHS):
    train(epoch)
    current_acc = test()
    scheduler.step()
    
    if current_acc > best_acc:
        best_acc = current_acc
        torch.save(model.state_dict(), 'best_resmlp.pth')

print(f"Best Test Accuracy: {best_acc:.2f}%")