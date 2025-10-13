import torch
from torch import nn


class Logarithm(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.clamp(x, min=1e-9).log()
