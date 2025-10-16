import torch.nn.functional as F
from torch import nn

from src.model.conformer.conformer import Conformer


class ConformerCTC(Conformer):
    def __init__(
        self,
        max_length,
        input_dim,
        n_tokens,
        encoder_dim=144,
        subsampling_dim=256,
        encoder_layers=16,
        attention_heads=4,
        conv_kernel_size=31,
        dropout_rate=0.1,
    ):
        super().__init__(
            max_length=max_length,
            input_dim=input_dim,
            output_dim=n_tokens,
            encoder_dim=encoder_dim,
            subsampling_dim=subsampling_dim,
            encoder_layers=encoder_layers,
            attention_heads=attention_heads,
            conv_kernel_size=conv_kernel_size,
            dropout_rate=dropout_rate,
        )

    def forward(self, x, spectrogram_length, **kwargs):
        logits, x_lengths = super().forward(x, spectrogram_length)
        log_probs = F.log_softmax(logits, dim=-1)
        return {"log_probs": log_probs, "log_probs_length": x_lengths}
