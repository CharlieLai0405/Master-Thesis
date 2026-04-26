import torch
from torch import nn
import numpy as np
# from datas.dataset_3d import  *
from torch.nn import functional as F


class TextPromptAdapter(nn.Module):
    """Learnable residual adapter for ImageBind text embeddings.

    Applies a bottleneck MLP with a learnable scaling factor so that
    at initialisation the adapter is close to an identity function
    (scale starts small, up_proj is zero-initialised).
    """

    def __init__(self, embed_dim: int = 1024, hidden_dim: int = 128, scale_init: float = 0.1):
        super().__init__()
        self.down_proj = nn.Linear(embed_dim, hidden_dim)
        self.act = nn.GELU()
        self.up_proj = nn.Linear(hidden_dim, embed_dim)
        self.scale = nn.Parameter(torch.tensor(scale_init))
        nn.init.zeros_(self.up_proj.weight)
        nn.init.zeros_(self.up_proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.scale * self.up_proj(self.act(self.down_proj(x)))


class Normalize(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return torch.nn.functional.normalize(x, dim=self.dim, p=2)

    
class LinearLayer(nn.Module):
    def __init__(self, dim_in, dim_out, k):
        super(LinearLayer, self).__init__()
        self.fc = nn.ModuleList([nn.Linear(dim_in, dim_out) for i in range(k)])

    def forward(self, tokens):
        for i in range(len(tokens)):
            if len(tokens[i].shape) == 3:
                tokens[i] = tokens[i].transpose(0,1)
                tokens[i] = self.fc[i](tokens[i][:, 1:, :])
            else:
                B, C, H, W = tokens[i].shape
                tokens[i] = self.fc[i](tokens[i].view(B, C, -1).permute(0, 2, 1).contiguous())
        return tokens
    
class PromptLearner(nn.Module):
    def __init__(self, dim_in, dim_out) -> None:
        super().__init__()
        self.meta_net = nn.Sequential(
            nn.Conv2d(dim_in, dim_in * 4, kernel_size=3, padding=1),
            # nn.BatchNorm2d(dim_in * 4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2), # 112 * 112

            nn.Conv2d(dim_in * 4, dim_in * 16, kernel_size=3, padding=1),
            # nn.BatchNorm2d(dim_in * 16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2), # 56 * 56

            nn.Conv2d(dim_in * 16, dim_in * 64, kernel_size=3, padding=1),
            # nn.BatchNorm2d(dim_in * 64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2), # 28 * 28

            nn.Conv2d(dim_in * 64, dim_in * 256, kernel_size=3, padding=1),
            # nn.BatchNorm2d(dim_in * 256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2), # 14 * 14

            nn.Conv2d(dim_in * 256, dim_in * 1024, kernel_size=3, padding=1),
            # nn.BatchNorm2d(dim_in * 1024),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2), # 7 * 7

            nn.Conv2d(dim_in * 1024, dim_out, kernel_size=5, padding=0),
            # nn.BatchNorm2d(dim_out),
            # nn.ReLU(inplace=True),
        )
        self.base_prompts = nn.Parameter(torch.randn((9, dim_out)),requires_grad=True)

    def forward(self, input):
        B,C,H,W = input.shape
        img_prompts = self.meta_net(input)
        # print(input.shape, img_prompts.shape)
        img_prompts = img_prompts.reshape(B,4096,9).transpose(-2,-1)
        output = torch.cat([self.base_prompts.expand(B,-1,-1), img_prompts], dim=1)
        return output
