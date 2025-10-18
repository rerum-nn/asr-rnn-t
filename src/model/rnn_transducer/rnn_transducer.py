import torch
from torch import nn


class PredictionNetwork(nn.Module):
    def __init__(
        self,
        vocab_size,
        pad_idx,
        hidden_dim,
        output_dim,
        num_lstm_layers,
        dropout_rate=0.1,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_lstm_layers = num_lstm_layers

        self.embedding = nn.Embedding(vocab_size, hidden_dim, padding_idx=pad_idx)

        self.input_layer_norm = nn.LayerNorm(hidden_dim)
        self.hidden_activation = nn.LSTM(
            hidden_dim,
            hidden_dim,
            batch_first=True,
            dropout=dropout_rate,
            num_layers=num_lstm_layers,
        )
        self.hidden_output = nn.Linear(hidden_dim, output_dim)
        self.output_layer_norm = nn.LayerNorm(output_dim)

        self._init_weights()

    def _init_weights(self):
        for name, param in self.hidden_activation.named_parameters():
            if "weight_hh" in name:
                nn.init.orthogonal_(param)
            elif "weight_ih" in name:
                nn.init.kaiming_uniform_(param, nonlinearity="sigmoid")
            elif "bias" in name:
                nn.init.zeros_(param)

        nn.init.xavier_uniform_(self.hidden_output.weight)
        if self.hidden_output.bias is not None:
            nn.init.zeros_(self.hidden_output.bias)

    def forward(self, y, h=None, c=None):
        if h is None or c is None:
            c = torch.zeros(
                self.num_lstm_layers, y.size(0), self.hidden_dim, device=y.device
            )
            h = torch.zeros(
                self.num_lstm_layers, y.size(0), self.hidden_dim, device=y.device
            )

        y = self.embedding(y)
        y = self.input_layer_norm(y)
        output, (next_h, next_c) = self.hidden_activation(y, (h, c))
        g = self.hidden_output(output)
        g = self.output_layer_norm(g)

        return g, next_h, next_c


class JointNetwork(nn.Module):
    def __init__(self, hidden_dim, vocab_size):
        super().__init__()

        self.activation = nn.ReLU()
        self.linear = nn.Linear(hidden_dim, vocab_size)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.linear.weight)
        if self.linear.bias is not None:
            nn.init.zeros_(self.linear.bias)

    def forward(self, f, p):
        f = f.unsqueeze(2).contiguous()
        p = p.unsqueeze(1).contiguous()

        combined = f + p
        logits = self.linear(self.activation(combined))
        return logits

    def infer(self, f, p):
        combined = f + p
        logits = self.linear(self.activation(combined))
        return logits
