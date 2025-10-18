from typing import List

import torch
from torch import Tensor

from src.metrics.base_metric import BaseMetric
from src.metrics.utils import calc_cer

# TODO add beam search/lm versions
# Note: they can be written in a pretty way
# Note 2: overall metric design can be significantly improved


class ArgmaxCERMetric(BaseMetric):
    def __init__(self, text_encoder, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.text_encoder = text_encoder

    def __call__(
        self, log_probs: Tensor, log_probs_length: Tensor, text: List[str], **kwargs
    ):
        cers = []
        predictions = torch.argmax(log_probs.cpu(), dim=-1).numpy()
        lengths = log_probs_length.detach().cpu().numpy()
        for log_prob_vec, length, target_text in zip(predictions, lengths, text):
            target_text = self.text_encoder.normalize_text(target_text)
            pred_text = self.text_encoder.output_decode(log_prob_vec[:length])
            cers.append(calc_cer(target_text, pred_text))
        return sum(cers) / len(cers)

    def infer(self, result, text):
        cers = []
        for pred_text, target_text in zip(result, text):
            target_text = self.text_encoder.normalize_text(target_text)
            pred_text = self.text_encoder.output_decode(pred_text)
            cers.append(calc_cer(target_text, pred_text))
        return sum(cers) / len(cers)


class BeamSearchCERMetric(BaseMetric):
    def __init__(self, text_encoder, beam_size=10, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.text_encoder = text_encoder
        self.beam_size = beam_size

    def __call__(
        self, log_probs: Tensor, log_probs_length: Tensor, text: List[str], **kwargs
    ):
        cers = []
        lengths = log_probs_length.detach().cpu().numpy()

        for log_prob_vec, length, target_text in zip(log_probs, lengths, text):
            target_text = self.text_encoder.normalize_text(target_text)

            pred_text = self.text_encoder.beam_search_decode(
                log_prob_vec[:length], beam_size=self.beam_size
            )

            cers.append(calc_cer(target_text, pred_text))
        return sum(cers) / len(cers)

    def infer(self, result_beam, text):
        cers = []
        for pred_text, target_text in zip(result_beam, text):
            target_text = self.text_encoder.normalize_text(target_text)
            pred_text = self.text_encoder.output_decode(pred_text)
            cers.append(calc_cer(target_text, pred_text))
        return sum(cers) / len(cers)
