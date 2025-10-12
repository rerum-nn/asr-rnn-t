import torch
import pytest
from src.model.conformer.conformer_block import ConformerBlock


class TestConformerBlock:
    def test_forward_input(self):
        model = ConformerBlock(
            encoder_dim=144,
            attention_heads=4,
            kernel_size=9,
            max_length=100,
            dropout_rate=0.1
        )
        
        batch_size = 2
        seq_len = 50
        
        x = torch.randn(batch_size, seq_len, 144)
        output = model(x)
        
        assert output.shape == (batch_size, seq_len, 144)

    def test_forward_different_sequence_lengths(self):
        model = ConformerBlock(
            encoder_dim=144,
            attention_heads=4,
            kernel_size=9,
            max_length=100,
            dropout_rate=0.1
        )
        
        batch_size = 3
        seq_len = 30
        
        x = torch.randn(batch_size, seq_len, 144)
        output = model(x)
        
        assert output.shape == (batch_size, seq_len, 144)

    def test_forward_preserves_shape(self):
        model = ConformerBlock(
            encoder_dim=256,
            attention_heads=8,
            kernel_size=15,
            max_length=200,
            dropout_rate=0.2
        )
        
        batch_size = 4
        seq_len = 100
        
        x = torch.randn(batch_size, seq_len, 256)
        output = model(x)
        
        assert output.shape == x.shape
        