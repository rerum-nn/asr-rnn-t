from torch import nn
from torchaudio.transforms import RNNTLoss
from src.text_encoder import RNNTTextEncoder

class RNNTLossWrapper(RNNTLoss):
    def __init__(self, *args, **kwargs):
        self.text_encoder = RNNTTextEncoder()
        super().__init__(fused_log_softmax=False, *args, **kwargs)

    def forward(self, log_probs, text_encoded, log_probs_length, text_encoded_length, **kwargs):
        loss = super().forward(log_probs, text_encoded[:, 1:].contiguous(), log_probs_length, text_encoded_length - 1)
        return {"loss": loss}
    