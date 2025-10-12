import torch
import pytest
from src.model.conformer_rnn_t import ConformerRNNT


class TestConformerRNNT:
    def test_forward_input(self):
        model = ConformerRNNT(
            max_length=100,
            input_dim=80,
            n_tokens=1000,
            pad_idx=0,
            bos_idx=1
        )
        
        batch_size = 2
        seq_len = 50
        vocab_size = 1000
        
        x = torch.randn(batch_size, seq_len, 80)
        y = torch.randint(1, vocab_size, (batch_size, 10))
        x_lengths = torch.tensor([seq_len, seq_len])
        y_lengths = torch.tensor([10, 10])
        
        logits, logits_lengths = model(x, y, x_lengths)
        
        assert logits.size() == (batch_size, (seq_len + 7) // 8, 10, vocab_size)

