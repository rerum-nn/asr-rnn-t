import torch
from torch import nn
from .conformer_block import ConformerBlock
from ..utils import Transpose

class Conformer(nn.Module):
    def __init__(
            self, 
            max_length,
            input_dim,
            output_dim,
            encoder_dim=144, 
            subsampling_dim=256,
            encoder_layers=16, 
            attention_heads=4, 
            conv_kernel_size=9, 
            dropout_rate=0.1
        ):
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.max_length = max_length
        self.encoder_dim = encoder_dim
        self.encoder_layers = encoder_layers
        self.attention_heads = attention_heads
        self.conv_kernel_size = conv_kernel_size
        self.dropout_rate = dropout_rate
        self.subsampling_dim = subsampling_dim

        self.conv_subsampling = nn.Sequential(
            nn.Conv2d(1, subsampling_dim, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(subsampling_dim, subsampling_dim, kernel_size=3, stride=2, groups=subsampling_dim, padding=1),
            nn.ReLU(),
            nn.Conv2d(subsampling_dim, subsampling_dim, kernel_size=3, stride=2, groups=subsampling_dim, padding=1),
            nn.ReLU(),
        )
        self.proj = nn.Sequential(
            nn.Linear(subsampling_dim * ((input_dim + 7) // 8), encoder_dim),
            nn.Dropout(dropout_rate)
        )
        
        self.conformer_blocks = nn.ModuleList(ConformerBlock(encoder_dim, attention_heads, conv_kernel_size, max_length, dropout_rate) for _ in range(encoder_layers))
        
        self.output_projection = nn.Linear(encoder_dim, output_dim)
        self.output_layer_norm = nn.LayerNorm(output_dim)

    def forward(self, x, input_lengths):
        x = self.conv_subsampling(x.unsqueeze(1))
        input_lengths = (input_lengths + 7) // 8

        x = x.permute(0, 2, 1, 3).contiguous().view(x.size(0), x.size(2), -1)
        x = self.proj(x)

        for conformer_block in self.conformer_blocks:
            x = conformer_block(x)

        x = self.output_projection(x)
        x = self.output_layer_norm(x)

        return x, input_lengths

    def __repr__(self):
        """
        Model prints with the number of parameters.
        """
        all_parameters = sum([p.numel() for p in self.parameters()])
        trainable_parameters = sum(
            [p.numel() for p in self.parameters() if p.requires_grad]
        )

        result_info = super().__repr__()
        result_info = result_info + f"\nAll parameters: {all_parameters}"
        result_info = result_info + f"\nTrainable parameters: {trainable_parameters}"

        return result_info
