import json
import torch
from torch import nn


class Normalize1D(nn.Module):
    def __init__(self, params_path, device):
        super().__init__()

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        params = json.load(open(params_path))
        self.mean = torch.tensor(params['mean'], device=device)
        self.std = torch.tensor(params['std'], device=device)

    def forward(self, x):
        return (x - self.mean) / self.std
