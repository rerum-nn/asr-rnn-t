import random

import torch
from torch import nn


class GaussianNoise(nn.Module):
    def __init__(self, max_std=0.01, p=0.7):
        super().__init__()
        self.max_std = max_std
        self.p = p

    def __call__(self, x):
        if random.random() >= self.p:
            return x
        return x + torch.randn_like(x) * self.max_std
