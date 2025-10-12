import torch
from torch import nn
from .conformer.conformer import Conformer
from .rnn_transducer.rnn_transducer import PredictionNetwork, JointNetwork 
import torch.nn.functional as F


class ConformerRNNT(nn.Module):
    def __init__(
            self, 
            max_length,
            input_dim,
            n_tokens,
            pad_idx,
            bos_idx,
            encoder_dim=144,
            subsampling_dim=256,
            encoder_layers=16, 
            attention_heads=4,
            conv_kernel_size=31, 
            hidden_dim=128,
            max_tokens=3,
            dropout_rate=0.1
        ):
        super().__init__()

        self.max_length = max_length
        self.vocab_size = n_tokens
        self.bos_idx = bos_idx
        self.pad_idx = pad_idx

        self.conformer = Conformer(max_length, input_dim, hidden_dim, encoder_dim, subsampling_dim, encoder_layers, attention_heads, conv_kernel_size, dropout_rate)
        self.prediction_network = PredictionNetwork(encoder_dim, hidden_dim, hidden_dim, self.vocab_size, pad_idx)
        self.joint_network = JointNetwork(hidden_dim, self.vocab_size)

    def forward(self, spectrogram, text_encoded, spectrogram_length, **kwargs):
        f, x_lengths = self.conformer(spectrogram, spectrogram_length)
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
