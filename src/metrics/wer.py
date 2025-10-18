from typing import List

import torch
from torch import Tensor

from src.metrics.base_metric import BaseMetric
from src.metrics.utils import calc_wer

# TODO beam search / LM versions
# Note: they can be written in a pretty way
# Note 2: overall metric design can be significantly improved


class ArgmaxWERMetric(BaseMetric):
    def __init__(self, text_encoder, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.text_encoder = text_encoder

    def __call__(
        self, log_probs: Tensor, log_probs_length: Tensor, text: List[str], **kwargs
    ):
        wers = []
        predictions = torch.argmax(log_probs.cpu(), dim=-1).numpy()
        lengths = log_probs_length.detach().cpu().numpy()
        for log_prob_vec, length, target_text in zip(predictions, lengths, text):
            target_text = self.text_encoder.normalize_text(target_text)
            pred_text = self.text_encoder.output_decode(log_prob_vec[:length])
            wers.append(calc_wer(target_text, pred_text))
        return sum(wers) / len(wers)

    def infer(self, result, text, **kwargs):
        wers = []
        for pred_text, target_text in zip(result, text):
            target_text = self.text_encoder.normalize_text(target_text)
            pred_text = self.text_encoder.decode(pred_text)
            wers.append(calc_wer(target_text, pred_text))
        return sum(wers) / len(wers)


class BeamSearchWERMetric(BaseMetric):
    def __init__(self, text_encoder, beam_size=10, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.text_encoder = text_encoder
        self.beam_size = beam_size

    def __call__(
        self, log_probs: Tensor, log_probs_length: Tensor, text: List[str], **kwargs
    ):
        wers = []
        lengths = log_probs_length.detach().cpu().numpy()

        for log_prob_vec, length, target_text in zip(log_probs, lengths, text):
            target_text = self.text_encoder.normalize_text(target_text)

            pred_text = self.text_encoder.beam_search_decode(
                log_prob_vec[:length], beam_size=self.beam_size
            )

            wers.append(calc_wer(target_text, pred_text))
        return sum(wers) / len(wers)

    def infer(self, result_beam, text, **kwargs):
        wers = []
        for pred_text, target_text in zip(result_beam, text):
            target_text = self.text_encoder.normalize_text(target_text)
            pred_text = self.text_encoder.decode(pred_text)
            wers.append(calc_wer(target_text, pred_text))
        return sum(wers) / len(wers)
