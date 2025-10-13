import random

import numpy as np
from torch import nn
from torchaudio.functional import pitch_shift


class PitchShift(nn.Module):
    def __init__(self, min_semitones=-2, max_semitones=2, sr=16000, p=0.7):
        super().__init__()
        self.min_semitones = min_semitones
        self.max_semitones = max_semitones
        self.sr = sr
        self.p = p

    def __call__(self, x):
        if random.random() >= self.p:
            return x
        return pitch_shift(
            x,
            sample_rate=self.sr,
            n_steps=np.random.uniform(self.min_semitones, self.max_semitones),
        )
