import torch
import torch.nn.functional as F
from torch import nn


class GRUWithBatchNorm(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout_rate=0.1):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.batch_norm = nn.BatchNorm1d(input_dim)
        self.relu = nn.ReLU()
        self.gru = nn.GRU(
            input_dim,
            hidden_dim,
            num_layers=1,
            bidirectional=True,
            dropout=dropout_rate,
            batch_first=True,
        )

    def forward(self, x):
        x = self.batch_norm(x.transpose(1, 2))
        x = self.relu(x)
        x, _ = self.gru(x.transpose(1, 2))

        x = x[:, :, : self.hidden_dim] + x[:, :, self.hidden_dim :]
        return x


class DeepSpeech2(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        n_tokens,
        subsampling_dim=32,
        gru_layers=5,
        dropout_rate=0.1,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.subsampling_dim = subsampling_dim
        self.hidden_dim = hidden_dim
        self.n_tokens = n_tokens
        self.gru_layers = gru_layers
        self.dropout_rate = dropout_rate

        self.subsampling = nn.Sequential(
            nn.Conv2d(
                1, subsampling_dim, kernel_size=(41, 11), stride=2, padding=(20, 5)
            ),
            nn.BatchNorm2d(subsampling_dim),
            nn.ReLU(),
            nn.Conv2d(
                subsampling_dim,
                subsampling_dim,
                kernel_size=(21, 11),
                stride=(2, 1),
                padding=(10, 5),
            ),
            nn.BatchNorm2d(subsampling_dim),
            nn.ReLU(),
        )

        self.rnn_layers = nn.ModuleList(
            [
                GRUWithBatchNorm(
                    subsampling_dim * ((input_dim + 3) // 4), hidden_dim, dropout_rate
                )
            ]
            + [
                GRUWithBatchNorm(hidden_dim, hidden_dim, dropout_rate)
                for _ in range(gru_layers - 1)
            ],
        )

        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, n_tokens)

    def forward(self, x, spectrogram_length, **kwargs):
        x = self.subsampling(x.unsqueeze(1).transpose(2, 3))
        x_lengths = (spectrogram_length + 3) // 4
        x = x.permute(0, 3, 1, 2).contiguous()
        x = x.view(x.size(0), x.size(1), -1).contiguous()

        for gru in self.rnn_layers:
            x = gru(x)

        x = self.layer_norm(x)
        logits = self.output_layer(x)
        log_probs = F.log_softmax(logits, dim=-1)
        return {"logits": logits, "log_probs": log_probs, "log_probs_length": x_lengths}

    def __str__(self):
        """
        Model prints with the number of parameters.
        """
        all_parameters = sum([p.numel() for p in self.parameters()])
        trainable_parameters = sum(
            [p.numel() for p in self.parameters() if p.requires_grad]
        )

        result_info = super().__str__()
        result_info = result_info + f"\nAll parameters: {all_parameters}"
        result_info = result_info + f"\nTrainable parameters: {trainable_parameters}"

        return result_info
