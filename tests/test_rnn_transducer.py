import torch
import pytest
from src.model.rnn_transducer.rnn_transducer import (
    PredictionNetwork,  
    JointNetwork
)


class TestPredictionNetwork:
    def test_forward_input(self):
        model = PredictionNetwork(
            input_dim=128,
            hidden_dim=256,
            output_dim=144,
            vocab_size=1000,
            pad_idx=0
        )
        
        batch_size = 2
        seq_len = 10
        
        y = torch.randint(1, 1000, (batch_size, seq_len))
        g, h, c = model(y)
        
        assert g.shape == (batch_size, seq_len, 144)
        assert h.shape == (1, batch_size, 256)
        assert c.shape == (1, batch_size, 256)

    def test_forward_with_hidden_states(self):
        model = PredictionNetwork(
            input_dim=128,
            hidden_dim=256,
            output_dim=144,
            vocab_size=1000,
            pad_idx=0
        )
        
        batch_size = 2
        seq_len = 10
        
        y = torch.randint(1, 1000, (batch_size, seq_len))
        h = torch.randn(1, batch_size, 256)
        c = torch.randn(1, batch_size, 256)
        
        g, next_h, next_c = model(y, h, c)
        
        assert g.shape == (batch_size, seq_len, 144)
        assert next_h.shape == (1, batch_size, 256)
        assert next_c.shape == (1, batch_size, 256)

class TestJointNetwork:
    def test_forward_input(self):
        model = JointNetwork(
            input_dim=128,
            hidden_dim=144,
            vocab_size=1000,
            pad_idx=0
        )
        
        batch_size = 2
        seq_len_f = 50
        seq_len_p = 10
        
        f = torch.randn(batch_size, seq_len_f, 144)
        p = torch.randn(batch_size, seq_len_p, 144)
        
        logits = model(f, p)
        
        assert logits.shape == (batch_size, seq_len_f, seq_len_p, 1000)
