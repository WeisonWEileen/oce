from einops.layers.torch import Rearrange
import torch
import torch.nn as nn

# note nn.Conv2d is used to generate image patch, not to implement a traditional convolutional nerual network
class Patch_rizer(nn.Module):
    def __init__(self, in_channels, images_size, patch_size, dim):
        super.__init__() 
        if images_size % patch_size != 0:
            raise ValueError("error: images_size must be divisible by patch_size!")
        self.patch = nn.Conv2d(in_channels, dim, patch_size, patch_size)
        self.reshape = Rearrange('b c h w -> b (h w) c')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch(x)       # [B, dim, H/p, W/p]
        x = self.reshape(x)    # flatten the patches [B, num_patches, dim]
        return x

class Affine(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones([1, 1, dim]))
        self.beta = nn.Parameter(torch.ones([1, 1, dim]))
    
    def forward(self, x):
        return x * self.alpha + self.beta 

class Mlp_black(nn.Module):
    def __init__(self, dim, dropout=0.):
        hidden_dim = 4 * dim
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        x = self.net(x)
        return x

class Block(nn.Module):
    def
        
class ResMLP(nn.Module):
    '''
    in_channels: channels of the input image
    dim: is the channel propagate inside the model
    '''
    def __init__(self, in_channels=3, dim=200, images_size=28, patch_size=4, num_classes=10):
        super().__init__()


