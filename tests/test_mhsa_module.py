import torch
import pytest
from src.model.conformer.mhsa_relative_pos_encodings import (
    MHSAWithRelativePosEncoding,
    MultiHeadSelfAttentionModule
)


class TestMHSAWithRelativePosEncoding:
    def test_forward_input(self):
        model = MHSAWithRelativePosEncoding(
            encoder_dim=144,
            max_length=100,
            attention_heads=4,
            dropout_rate=0.1
        )
        
        batch_size = 2
        seq_len = 50
        
        x = torch.randn(batch_size, seq_len, 144)
        output = model(x)
        
        assert output.shape == (batch_size, seq_len, 144)

    def test_forward_different_sequence_lengths(self):
        model = MHSAWithRelativePosEncoding(
            encoder_dim=256,
            max_length=200,
            attention_heads=8,
            dropout_rate=0.2
        )
        
        batch_size = 3
        seq_len = 100
        
        x = torch.randn(batch_size, seq_len, 256)
        output = model(x)
        
        assert output.shape == (batch_size, seq_len, 256)

    def test_forward_preserves_shape(self):
        model = MHSAWithRelativePosEncoding(
            encoder_dim=512,
            max_length=300,
            attention_heads=16,
            dropout_rate=0.3
        )
        
        batch_size = 4
        seq_len = 200
        
        x = torch.randn(batch_size, seq_len, 512)
        output = model(x)
        
        assert output.shape == x.shape

    def test_different_attention_heads(self):
        attention_heads = [1, 2, 4, 8, 16]
        
        for heads in attention_heads:
            model = MHSAWithRelativePosEncoding(
                encoder_dim=128,
                max_length=100,
                attention_heads=heads,
                dropout_rate=0.1
            )
            
            x = torch.randn(2, 50, 128)
            output = model(x)
            
            assert output.shape == (2, 50, 128)

    def test_pe_matrix_generation(self):
        model = MHSAWithRelativePosEncoding(
            encoder_dim=64,
            max_length=50,
            attention_heads=4,
            dropout_rate=0.1
        )
        
        pe_matrix = model._generate_pe_matrix(50, 64)
        
        assert pe_matrix.shape == (101, 64)
        
        zero_position_idx = 50
        expected_zero_position = torch.zeros(64)
        expected_zero_position[0::2] = 0.0
        expected_zero_position[1::2] = 1.0
        
        assert torch.allclose(pe_matrix[zero_position_idx], expected_zero_position, atol=1e-6)

    def test_shift_relative_pos(self):
        model = MHSAWithRelativePosEncoding(
            encoder_dim=144,
            max_length=100,
            attention_heads=4,
            dropout_rate=0.1
        )
        
        batch_size = 1
        attention_heads = 1
        i, j = 3, 4
        
        x = torch.tensor([
            [[[1, 2, 3, 4],
              [5, 6, 7, 8],
              [9, 10, 11, 12]]],
        ], dtype=torch.float32)
        
        expected = torch.tensor([
            [[[3, 4, 0, 5],
              [6, 7, 8, 0],
              [9, 10, 11, 12]]],
        ], dtype=torch.float32)
        
        shifted = model._shift_relative_pos(x)
        print(shifted)
        
        assert shifted.shape == (batch_size, attention_heads, i, j)
        assert torch.allclose(shifted, expected, atol=1e-6)


class TestMultiHeadSelfAttentionModule:
    def test_forward_input(self):
        model = MultiHeadSelfAttentionModule(
            encoder_dim=144,
            max_length=100,
            attention_heads=4,
            dropout_rate=0.1
        )
        
        batch_size = 2
        seq_len = 50
        
        x = torch.randn(batch_size, seq_len, 144)
        output = model(x)
        
        assert output.shape == (batch_size, seq_len, 144)

    def test_forward_different_sequence_lengths(self):
        model = MultiHeadSelfAttentionModule(
            encoder_dim=256,
            max_length=200,
            attention_heads=8,
            dropout_rate=0.2
        )
        
        batch_size = 3
        seq_len = 100
        
        x = torch.randn(batch_size, seq_len, 256)
        output = model(x)
        
        assert output.shape == (batch_size, seq_len, 256)

    def test_forward_preserves_shape(self):
        model = MultiHeadSelfAttentionModule(
            encoder_dim=512,
            max_length=300,
            attention_heads=16,
            dropout_rate=0.3
        )
        
        batch_size = 4
        seq_len = 200
        
        x = torch.randn(batch_size, seq_len, 512)
        output = model(x)
        
        assert output.shape == x.shape
