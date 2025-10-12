from torch import nn

class FeedForwardModule(nn.Module):
    def __init__(self, encoder_dim, dropout_rate=0.1, expansion_factor=4):
        super().__init__()

        self.encoder_dim = encoder_dim

        self.ff_module = nn.Sequential(
            nn.LayerNorm(encoder_dim),
            nn.Linear(encoder_dim, encoder_dim * expansion_factor),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(encoder_dim * expansion_factor, encoder_dim),
            nn.Dropout(dropout_rate)
        )

    def forward(self, x):
        x = self.ff_module(x)
        return x
