import re
from string import ascii_lowercase

import torch
# import youtokentome as yttm

# TODO add BPE, LM, Beam Search support
# Note: think about metrics and encoder
# The design can be remarkably improved
# to calculate stuff more efficiently and prettier


class RNNTTextEncoder:
    EMPTY_TOK = ""
    BOS_TOK = "<bos>"

    def __init__(self, alphabet=None, **kwargs):
        """
        Args:
            alphabet (list): alphabet for language. If None, it will be
                set to ascii
        """

        if alphabet is None:
            alphabet = list(ascii_lowercase + " ")

        self.alphabet = alphabet
        self.vocab = [self.EMPTY_TOK, self.BOS_TOK] + list(self.alphabet)

        self.ind2char = dict(enumerate(self.vocab))
        self.char2ind = {v: k for k, v in self.ind2char.items()}

        self.empty_tok = self.char2ind[self.EMPTY_TOK]
        self.bos_tok = self.char2ind[self.BOS_TOK]

    def __len__(self):
        return len(self.vocab)

    def __getitem__(self, item: int):
        assert type(item) is int
        return self.ind2char[item]

    def encode(self, text) -> torch.Tensor:
        text = self.normalize_text(text)
        try:
            return torch.Tensor(
                [self.char2ind[self.BOS_TOK]] + [self.char2ind[char] for char in text]
            ).unsqueeze(0)
        except KeyError:
            unknown_chars = set([char for char in text if char not in self.char2ind])
            raise Exception(
                f"Can't encode text '{text}'. Unknown chars: '{' '.join(unknown_chars)}'"
            )

    def decode(self, inds) -> str:
        """
        Args:
            inds (list): list of tokens.
        Returns:
            raw_text (str): raw text with empty tokens and repetitions.
        """
        return "".join([self.ind2char[int(ind)] for ind in inds]).strip()

    def rnnt_decode(self, inds) -> str:
        parsed_inds = []
        j = 0
        print(inds)
        for i in range(inds.shape[1]):
            row = inds[:, i]
            while j < len(row) and row[j] == self.empty_tok:
                j += 1
            if j < len(row):
                parsed_inds.append(row[j])
        return self.decode(parsed_inds)

    def output_decode(self, inds) -> str:
        return self.rnnt_decode(inds)

    @staticmethod
    def normalize_text(text: str):
        text = text.lower()
        text = re.sub(r"[^a-z ]", "", text)
        return text


# class RNNTTextEncoderBPE(RNNTTextEncoder):
#     def __init__(self, bpe_model_path=None, **kwargs):
#         self.bpe_model = yttm.BPE(bpe_model_path)
#         self.bos_tok = 1
#         self.empty_tok = 0

#     def __len__(self):
#         return self.bpe_model.vocab_size()

#     def __getitem__(self, item: int):
#         assert type(item) is int
#         return self.bpe_model.id_to_subword(item)

#     def encode(self, text):
#         return torch.Tensor(
#             self.bpe_model.encode(text, output_type=yttm.OutputType.ID, bos=True)
#         ).unsqueeze(0)

#     def decode(self, inds):
#         if len(inds) == 0:
#             return ""
#         return RNNTTextEncoder.normalize_text(
#             self.bpe_model.decode([int(i) for i in inds])[0]
#         )
