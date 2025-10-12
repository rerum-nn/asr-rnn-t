from torch import nn
from .mhsa_relative_pos_encodings import MultiHeadSelfAttentionModule
from .feed_forward_module import FeedForwardModule
from .convolution_module import ConvolutionModule

class ConformerBlock(nn.Module):
    def __init__(self, encoder_dim, attention_heads, kernel_size, max_length, dropout_rate=0.1):
        super().__init__()

        self.encoder_dim = encoder_dim

        self.ff1 = FeedForwardModule(encoder_dim, dropout_rate)
        self.mhsa = MultiHeadSelfAttentionModule(encoder_dim, max_length, attention_heads, dropout_rate)
        self.conv = ConvolutionModule(encoder_dim, kernel_size, dropout_rate)
        self.ff2 = FeedForwardModule(encoder_dim, dropout_rate)
        self.layer_norm = nn.LayerNorm(encoder_dim)

    def forward(self, x):
        x = x + self.ff1(x) * 0.5
        x = x + self.mhsa(x)
        x = x + self.conv(x)
        x = x + self.ff2(x) * 0.5

        return self.layer_norm(x)

