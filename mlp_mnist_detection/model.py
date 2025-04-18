from einops.layers.torch import Rearrange
import torch
import torch.nn as nn

# note nn.Conv2d is used to generate image patch, not to implement a traditional convolutional nerual network
class Patch_rizer(nn.Module):
    def __init__(self, in_channels, images_size, patches_size, dim):
        super().__init__() 
        if images_size % patches_size != 0:
            raise ValueError("error: images_size must be divisible by patches_size!")
        self.patch = nn.Conv2d(in_channels, dim, patches_size, patches_size)
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

class Mlp_block(nn.Module):
    def __init__(self, dim, dropout=0.2):
        super().__init__()
        hidden_dim = 4 * dim
        self.net = nn.Sequential(
            Affine(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
            Affine(dim)
        )
    def forward(self, x):
        x = self.net(x)
        return x

class ResMLP_block(nn.Module):
    def __init__(self, dim, images_size, patches_size, layerscale_init=1.0):
        super().__init__()
        num_patches = (images_size // patches_size) * (images_size // patches_size)
        self.cross_patch_sublayer = nn.Sequential(
            Affine(dim),
            Rearrange('b n d -> b d n'),
            nn.Linear(num_patches, num_patches),
            Rearrange('b d n -> b n d'),
            Affine(dim)
        )

        self.layerscale_1 = nn.Parameter(layerscale_init * torch.ones((dim))) # LayerScale
        self.layerscale_2 = nn.Parameter(layerscale_init * torch.ones((dim))) # parameters
        self.cross_channel_sublayer = Mlp_block(dim)

    def forward(self, x):
        res_1 = self.cross_patch_sublayer(x)
        x = x + self.layerscale_1 * res_1
        res_2 = self.cross_channel_sublayer(x)
        x = x + self.layerscale_2 * res_2
        return x

class ResMLP(nn.Module):
    '''
    the version of model checkpoint
    '''
    def __init__(self, depth=3, in_channels=3, dim=200, images_size=28, patches_size=4, num_classes=10):
        super().__init__()
        assert images_size % patches_size == 0
        self.patch_rizer = Patch_rizer(in_channels, images_size, patches_size, dim)
        self.blocks = nn.ModuleList(
            [
                ResMLP_block(dim, images_size, patches_size) for i in range(depth)               
            ] 
        )
        self.linear_head = nn.Linear(dim, num_classes)
    
    def forward(self, x: torch.tensor):
        B, C, H, W = x.shape
        x = self.patch_rizer(x)
        for blk in self.blocks:
            x = blk(x)
        x = x.mean(dim = 1).reshape(B, -1)
        output = self.linear_head(x)
        return output


