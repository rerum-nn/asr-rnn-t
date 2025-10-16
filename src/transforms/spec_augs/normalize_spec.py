import json

import torch
from torch import nn


class NormalizeSpec(nn.Module):
    def __init__(self, params_path, device="auto"):
        super().__init__()

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        params = json.load(open(params_path))
        self.mean = torch.tensor(params["mean"], device=self.device)
        self.std = torch.tensor(params["std"], device=self.device)

    def __call__(self, x):
        return (x - self.mean) / self.std
