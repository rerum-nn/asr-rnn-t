import re
from itertools import groupby
from string import ascii_lowercase

import torch

# TODO add CTC decode
# TODO add BPE, LM, Beam Search support
# Note: think about metrics and encoder
# The design can be remarkably improved
# to calculate stuff more efficiently and prettier


class CTCTextEncoder:
    EMPTY_TOK = ""

    def __init__(self, alphabet=None, **kwargs):
        """
        Args:
            alphabet (list): alphabet for language. If None, it will be
                set to ascii
        """

        if alphabet is None:
            alphabet = list(ascii_lowercase + " ")

        self.alphabet = alphabet
        self.vocab = [self.EMPTY_TOK] + list(self.alphabet)

        self.ind2char = dict(enumerate(self.vocab))
        self.char2ind = {v: k for k, v in self.ind2char.items()}

        self.empty_tok = self.char2ind[self.EMPTY_TOK]

    def __len__(self):
        return len(self.vocab)

    def __getitem__(self, item: int):
        assert type(item) is int
        return self.ind2char[item]

    def encode(self, text) -> torch.Tensor:
        text = self.normalize_text(text)
        try:
            return torch.Tensor([self.char2ind[char] for char in text]).unsqueeze(0)
        except KeyError:
            unknown_chars = set([char for char in text if char not in self.char2ind])
            raise Exception(
                f"Can't encode text '{text}'. Unknown chars: '{' '.join(unknown_chars)}'"
            )

    def decode(self, inds) -> str:
        """
        Raw decoding without CTC.
        Used to validate the CTC decoding implementation.

        Args:
            inds (list): list of tokens.
        Returns:
            raw_text (str): raw text with empty tokens and repetitions.
        """
        return "".join([self.ind2char[int(ind)] for ind in inds]).strip()

    def ctc_decode(self, inds) -> str:
        return "".join([self.ind2char[k] for k, _ in groupby(inds) if k != 0])

    def beam_search_decode(self, log_probs, beam_size=10) -> str:
        hypos = {"": (self.empty_tok, 1.0)}

        for i in range(len(log_probs)):
            frame = log_probs[i]

            for hypo_key, hypo in hypos.items():
                for token in frame:
                    if token == hypo[0]:
                        new_hypo_key = hypo_key
                    else:
                        new_hypo_key = hypo_key + self.ind2char[token]

                    new_hypo_prob = hypo[1] * frame[token].exp()
                    if new_hypo_key not in hypos:
                        hypos[new_hypo_key] = new_hypo_prob
                    else:
                        hypos[new_hypo_key] += new_hypo_prob

            hypos = dict(
                sorted(hypos.items(), key=lambda x: x[1][1], reverse=True)[:beam_size]
            )

        return max(hypos.items(), key=lambda x: x[1][1])[0]

    def output_decode(self, inds) -> str:
        return self.ctc_decode(inds)

    @staticmethod
    def normalize_text(text: str):
        text = text.lower()
        text = re.sub(r"[^a-z ]", "", text)
        return text
