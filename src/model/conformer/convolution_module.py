from torch import nn
from torch.nn import functional as F

from ..utils import Transpose


class ConvolutionModule(nn.Module):
    def __init__(self, encoder_dim, kernel_size=31, dropout_rate=0.1):
        super().__init__()

        self.expansion_factor = 2
        self.encoder_dim = encoder_dim

        self.conv_module = nn.Sequential(
            nn.LayerNorm(encoder_dim),
            Transpose(-2, -1),
            nn.Conv1d(encoder_dim, encoder_dim * self.expansion_factor, kernel_size=1),
            nn.GLU(dim=1),
            nn.Conv1d(
                encoder_dim,
                encoder_dim,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
                groups=encoder_dim,
            ),
            nn.BatchNorm1d(encoder_dim),
            nn.SiLU(),
            nn.Conv1d(encoder_dim, encoder_dim, kernel_size=1),
            Transpose(-2, -1),
            nn.Dropout(dropout_rate),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.conv_module.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.conv_module(x)
