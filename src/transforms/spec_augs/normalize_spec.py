import json

import torch
from torch import nn


class NormalizeSpec(nn.Module):
    def __init__(self, params_path):
        super().__init__()

        params = json.load(open(params_path))
        self.mean = torch.tensor(params["mean"])
        self.std = torch.tensor(params["std"])

    def __call__(self, x):
        return (x - self.mean[:, None]) / self.std[:, None]
