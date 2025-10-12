import torch
import pytest
from src.model.conformer.conformer import Conformer


class TestConformer:
    def test_forward_input(self):
        model = Conformer(
            max_length=100,
            input_dim=80,
            encoder_dim=144,
            encoder_layers=2,
            attention_heads=4,
            conv_kernel_size=9,
            dropout_rate=0.1
        )
        
        batch_size = 2
        seq_len = 80
        
        x = torch.randn(batch_size, seq_len, 80)
        input_lengths = torch.tensor([seq_len, seq_len])
        
        output, output_lengths = model(x, input_lengths)
        
        expected_length = (seq_len + 7) // 8
        assert output.shape == (batch_size, expected_length, 144)
        assert output_lengths.shape == (batch_size,)
        assert torch.all(output_lengths == expected_length)

    def test_forward_different_lengths(self):
        model = Conformer(
            max_length=100,
            input_dim=80,
            encoder_dim=144,
            encoder_layers=2,
            attention_heads=4,
            conv_kernel_size=9,
            dropout_rate=0.1
        )
        
        batch_size = 2
        
        x = torch.randn(batch_size, 80, 80)
        input_lengths = torch.tensor([60, 80])
        
        output, output_lengths = model(x, input_lengths)
        
        expected_length_1 = (60 + 7) // 8
        expected_length_2 = (80 + 7) // 8
        assert output.shape[0] == batch_size
        assert output_lengths[0] == expected_length_1
        assert output_lengths[1] == expected_length_2

