from torch import nn

from src.model.conformer.conformer import Conformer


class ConformerCTC(Conformer):
    def __init__(
        self,
        max_length,
        input_dim,
        n_tokens,
        encoding_dim=512,
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
            output_dim=encoding_dim,
            encoder_dim=encoder_dim,
            subsampling_dim=subsampling_dim,
            encoder_layers=encoder_layers,
            attention_heads=attention_heads,
            conv_kernel_size=conv_kernel_size,
            dropout_rate=dropout_rate,
        )

        self.out_projection = nn.Linear(encoding_dim, n_tokens)
        self.out_activation = nn.ReLU()

    def forward(self, x, spectrogram_length, **kwargs):
        x, x_lengths = super().forward(x, spectrogram_length)
        x = self.out_activation(x)
        x = self.out_projection(x)
        return {"log_probs": x, "log_probs_length": x_lengths}
