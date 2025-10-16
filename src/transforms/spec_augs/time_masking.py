import random

import torchaudio
from torch import nn


class TimeMasking(nn.Module):
    def __init__(self, time_mask_param=100, p=0.7, masks_number=5):
        super().__init__()
        self.time_mask_param = time_mask_param
        self.time_masking = torchaudio.transforms.TimeMasking(
            self.time_mask_param, iid_masks=True, p=p
        )
        self.masks_number = masks_number

    def __call__(self, x):
        for _ in range(self.masks_number):
            x = self.time_masking(x)
        return x
