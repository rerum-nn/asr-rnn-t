import torch
import pytest
from src.model.conformer.feed_forward_module import FeedForwardModule


class TestFeedForwardModule:
    def test_forward_input(self):
        model = FeedForwardModule(
            encoder_dim=144,
            dropout_rate=0.1,
            expansion_factor=4
        )
        
        batch_size = 2
        seq_len = 50
        
        x = torch.randn(batch_size, seq_len, 144)
        output = model(x)
        
        assert output.shape == (batch_size, seq_len, 144)

    def test_forward_different_sequence_lengths(self):
        model = FeedForwardModule(
            encoder_dim=256,
            dropout_rate=0.2,
            expansion_factor=8
        )
        
        batch_size = 3
        seq_len = 100
        
        x = torch.randn(batch_size, seq_len, 256)
        output = model(x)
        
        assert output.shape == (batch_size, seq_len, 256)

    def test_forward_preserves_shape(self):
        model = FeedForwardModule(
            encoder_dim=512,
            dropout_rate=0.3,
            expansion_factor=16
        )
        
        batch_size = 4
        seq_len = 200
        
        x = torch.randn(batch_size, seq_len, 512)
        output = model(x)
        
        assert output.shape == x.shape
