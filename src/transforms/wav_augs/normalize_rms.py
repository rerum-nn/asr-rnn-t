import torch
from torch import nn


class NormalizeRMS(nn.Module):
    def __init__(self, target_rms=0.1, eps=1e-6):
        super().__init__()
        self.target_rms = target_rms
        self.eps = eps

    def __call__(self, x):
        return x * self.target_rms / (torch.sqrt(torch.mean(x**2)) + self.eps)
