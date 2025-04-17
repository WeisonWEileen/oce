# Wei Pan, created at 4.16 2025 

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import random
from tqdm import tqdm
from einops.layers.torch import Rearrange

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
    torch.cuda.manual_seed_all(seed)  # 多GPU时启用
    # CUDA 确定性模式（可能降低性能，但保证可复现）
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(SEED)

# hyper params settings
BATCH_SIZE = 128
EPOCHS = 15
HIDDEN_DIM = 512
NUM_BLOCKS = 6
DROPOUT = 0.2
LR = 1e-3
WEIGHT_DECAY = 1e-4
LABEL_SMOOTHING = 0.1




        # for cross-patch sublayer

        # for cross-channel sublayer
    


if __name__ == "__main__":
    # set cuda
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

    def seed_worker(worker_id):
        worker_seed = SEED % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    g = torch.Generator()
    g.manual_seed(SEED)

    # shape [60000, 28, 28]
    train_set = datasets.MNIST(root='./data', train=True, download=True, transform=train_transform)
    print(f"traing data shape {train_set.data.shape}")

    # shape [10000, 28, 28]
    test_set = datasets.MNIST(root='./data', train=False, download=True, transform=test_transform)
    print(f"testing data shape {test_set.data.shape}")
