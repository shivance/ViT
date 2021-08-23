import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from torch import nn,Tensor
from PIL import Image
from torchvision.transforms import Compose,Resize,ToTensor
import einops.layers.torch as img_layers
from torchsummary import summary


patch_size = 16 # 16 patches required

class PatchEmbedding(nn.Module):

    def __init__(self,in_channels:int=3,patch_size:int=16,emb_size:int=768):
        self.patch_size = patch_size
        super().__init__()
        self.projection = nn.Sequential(
            img_layers.Rearrange('b c (h s1) (w s2) -> b (h w) (s1 s2 c)',
                s1=patch_size,
                s2=patch_size),
            nn.Linear(patch_size*patch_size*in_channels,emb_size) #s1*s2*c
        )

    def forward(self,x:Tensor) -> Tensor:
        x = self.projection(x)
        return x