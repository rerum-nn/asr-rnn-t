import torch
import pytest
from src.model.conformer.convolution_module import ConvolutionModule


class TestConvolutionModule:
    def test_forward_different_sequence_lengths(self):
        model = ConvolutionModule(
            encoder_dim=256,
            kernel_size=15,
            dropout_rate=0.2
        )
        
        batch_size = 3
        seq_len = 100
        
        x = torch.randn(batch_size, seq_len, 256)
        output = model(x)
        
        assert output.shape == (batch_size, seq_len, 256)

    def test_forward_preserves_shape(self):
        model = ConvolutionModule(
            encoder_dim=512,
            kernel_size=63,
            dropout_rate=0.3
        )
        
        batch_size = 4
        seq_len = 200
        
        x = torch.randn(batch_size, seq_len, 512)
        output = model(x)
        
        assert output.shape == x.shape

    def test_different_kernel_sizes(self):
        kernel_sizes = [9, 15, 31, 63]
        
        for kernel_size in kernel_sizes:
            model = ConvolutionModule(
                encoder_dim=144,
                kernel_size=kernel_size,
                dropout_rate=0.1
            )
            
            x = torch.randn(2, 50, 144)
            output = model(x)
            
            assert output.shape == (2, 50, 144)
