import random

import torchaudio
from torch import nn


class TimeMasking(nn.Module):
    def __init__(self, time_mask_param=100, p=0.7):
        super().__init__()
        self.time_mask_param = time_mask_param
        self.p = p
        self.time_masking = torchaudio.transforms.TimeMasking(self.time_mask_param)

    def __call__(self, x):
        if random.random() >= self.p:
            return x
        return self.time_masking(x)
