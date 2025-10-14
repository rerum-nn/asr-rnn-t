import torch.nn.functional as F
from torch import nn

from .conformer.conformer import Conformer
from .rnn_transducer.rnn_transducer import JointNetwork, PredictionNetwork


class ConformerRNNT(nn.Module):
    def __init__(
        self,
        max_length,
        input_dim,
        n_tokens,
        pad_idx,
        bos_idx,
        encoding_dim=512,
        conformer_encoder_dim=144,
        conformer_subsampling_dim=256,
        conformer_encoder_layers=16,
        conformer_attention_heads=4,
        conformer_conv_kernel_size=31,
        conformer_dropout_rate=0.1,
        num_lstm_layers=1,
        lstm_hidden_dim=256,
        lstm_dropout_rate=0.3,
        max_tokens=3,
    ):
        super().__init__()

        self.max_length = max_length
        self.vocab_size = n_tokens
        self.bos_idx = bos_idx
        self.pad_idx = pad_idx

        self.conformer = Conformer(
            max_length=max_length,
            input_dim=input_dim,
            output_dim=encoding_dim,
            encoder_dim=conformer_encoder_dim,
            subsampling_dim=conformer_subsampling_dim,
            encoder_layers=conformer_encoder_layers,
            attention_heads=conformer_attention_heads,
            conv_kernel_size=conformer_conv_kernel_size,
            dropout_rate=conformer_dropout_rate,
        )
        self.prediction_network = PredictionNetwork(
            hidden_dim=lstm_hidden_dim,
            output_dim=encoding_dim,
            num_lstm_layers=num_lstm_layers,
            vocab_size=n_tokens,
            pad_idx=pad_idx,
            dropout_rate=lstm_dropout_rate,
        )
        self.joint_network = JointNetwork(encoding_dim, n_tokens)

    def forward(self, x, text_encoded, spectrogram_length, **kwargs):
        f, x_lengths = self.conformer(x, spectrogram_length)
        g, _, _ = self.prediction_network(text_encoded)
        logits = self.joint_network(f, g)
        log_probs = F.log_softmax(logits, dim=-1)

        return {"log_probs": log_probs, "log_probs_length": x_lengths}

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
