# Wei Pan, created at 4.16 2025 

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import random
from tqdm import tqdm
from model import ResMLP
# seed settings
SEED = 42  # to make it reproducible

def set_seed(seed):
    # Python 随机模块
    random.seed(seed)
    # NumPy
    np.random.seed(seed)
    # PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)  # if multi gpu, turn on this
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def seed_worker(worker_id):
    worker_seed = SEED % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# hyper params settings
BATCH_SIZE = 128
EPOCHS = 25
HIDDEN_DIM = 128
NUM_BLOCKS = 3
DROPOUT = 0.3
LR = 1e-3
WEIGHT_DECAY = 1e-4
LABEL_SMOOTHING = 0.1

if __name__ == "__main__":
    # set seed to reproduced
    set_seed(SEED)
    g = torch.Generator()
    g.manual_seed(SEED)

    # set cuda
    if not torch.cuda.is_available():
        print("CUDA is not available. Exiting...")
        exit(1)  # 使用非零状态码退出，表示异常退出

    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"using device {device}")
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


    g = torch.Generator()
    g.manual_seed(SEED)

    # data shape [60000, 28, 28]
    # target shape [60000]
    train_set = datasets.MNIST(root='./data', train=True, download=True, transform=train_transform)
    print(f"traing data shape {train_set.data.shape}")
    print(f"target data shape {train_set.targets.shape}")

    # shape [10000, 28, 28]
    test_set = datasets.MNIST(root='./data', train=False, download=True, transform=test_transform)
    print(f"testing data shape {test_set.data.shape}")

    model = ResMLP(depth=4, in_channels=1, dim=380, images_size=train_set.data.shape[1], patches_size=4, num_classes=10)
    model.to(device)

    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_acc = 0.0
    train_losses = []
    test_losses = []
    for epoch in range(EPOCHS):
        
        # training
        model.train()
        train_loss = 0
        correct = 0
        total = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{EPOCHS}")
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
        
        print(train_loss/len(train_loader))
        train_losses.append(train_loss)

        # testing
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
        test_loss = test_loss/len(test_loader)
        print(f"\nTest Results: Loss: {test_loss:.4f} | Accuracy: {acc:.2f}%\n")
        test_losses.append(test_loss)


        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), 'best_resmlp.pth')
    print(f"Best Test Accuracy: {best_acc:.2f}%")
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, EPOCHS + 1), train_losses, label='Training Loss', marker='o', color='blue')
    plt.plot(range(1, EPOCHS + 1), test_losses, label='Testing Loss', marker='o', color='orange')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Testing Loss')
    plt.legend()
    plt.grid(True)

    # 保存图像到文件
    plt.savefig('loss_curve.png')