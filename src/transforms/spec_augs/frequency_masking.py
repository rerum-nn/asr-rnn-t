import random

import torchaudio
from torch import nn


class FrequencyMasking(nn.Module):
    def __init__(self, frequency_mask_param=10, p=0.7):
        super().__init__()
        self.frequency_mask_param = frequency_mask_param
        self.p = p
        self.frequency_masking = torchaudio.transforms.FrequencyMasking(
            self.frequency_mask_param, iid_masks=True
        )

    def __call__(self, x):
        if random.random() >= self.p:
            return x
        return self.frequency_masking(x)
