import random

import numpy as np
import torch
from librosa.effects import time_stretch
from torch import nn


class TimeStretch(nn.Module):
    def __init__(self, min_time_stretch=0.85, max_time_stretch=1.15, p=0.7):
        super().__init__()
        self.min_time_stretch = min_time_stretch
        self.max_time_stretch = max_time_stretch
        self.p = p

    def __call__(self, x):
        if random.random() >= self.p:
            return x
        return torch.from_numpy(
            time_stretch(
                x.numpy(),
                rate=np.random.uniform(self.min_time_stretch, self.max_time_stretch),
            )
        )
