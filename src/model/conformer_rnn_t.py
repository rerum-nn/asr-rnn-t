import torch
import torch.nn.functional as F
from torch import nn

from .conformer.conformer import Conformer
from .rnn_transducer.rnn_transducer import JointNetwork, PredictionNetwork


class ConformerRNNT(nn.Module):
    def __init__(
        self,
        max_length,
        input_dim,
        n_tokens,
        pad_idx,
        bos_idx,
        encoding_dim=512,
        conformer_encoder_dim=144,
        conformer_subsampling_dim=256,
        conformer_encoder_layers=16,
        conformer_attention_heads=4,
        conformer_conv_kernel_size=31,
        conformer_dropout_rate=0.1,
        num_lstm_layers=1,
        lstm_hidden_dim=256,
        lstm_dropout_rate=0.3,
        max_tokens_per_frame=5,
    ):
        super().__init__()

        self.max_length = max_length
        self.vocab_size = n_tokens
        self.bos_idx = bos_idx
        self.pad_idx = pad_idx

        self.max_tokens_per_frame = max_tokens_per_frame

        self.conformer = Conformer(
            max_length=max_length,
            input_dim=input_dim,
            output_dim=encoding_dim,
            encoder_dim=conformer_encoder_dim,
            subsampling_dim=conformer_subsampling_dim,
            encoder_layers=conformer_encoder_layers,
            attention_heads=conformer_attention_heads,
            conv_kernel_size=conformer_conv_kernel_size,
            dropout_rate=conformer_dropout_rate,
        )
        self.prediction_network = PredictionNetwork(
            hidden_dim=lstm_hidden_dim,
            output_dim=encoding_dim,
            num_lstm_layers=num_lstm_layers,
            vocab_size=n_tokens,
            pad_idx=pad_idx,
            dropout_rate=lstm_dropout_rate,
        )
        self.joint_network = JointNetwork(encoding_dim, n_tokens)

    def forward(self, x, text_encoded, spectrogram_length, **kwargs):
        f, x_lengths = self.conformer(x.transpose(1, 2), spectrogram_length)
        g, _, _ = self.prediction_network(text_encoded)
        logits = self.joint_network(f, g)
        log_probs = F.log_softmax(logits, dim=-1)

        return {"log_probs": log_probs, "log_probs_length": x_lengths}

    def infer(self, x, spectrogram_length, **kwargs):
        with torch.no_grad():
            batch, x_lengths = self.conformer(x.transpose(1, 2), spectrogram_length)

            result = []
            for i in range(len(batch)):
                sample = batch[i]
                length = x_lengths[i]

                h = None
                c = None
                prediction = []
                last_token = self.bos_idx
                for frame in sample[:length]:
                    for _ in range(self.max_tokens_per_frame):
                        g, next_h, next_c = self.prediction_network(
                            torch.tensor(last_token, device=frame.device)
                            .unsqueeze(0)
                            .unsqueeze(0),
                            h,
                            c,
                        )
                        logits = self.joint_network.infer(frame, g)
                        log_probs = F.log_softmax(logits, dim=-1)
                        next_token = log_probs.argmax(dim=-1)
                        if next_token.item() == self.pad_idx:
                            break
                        last_token = next_token.item()
                        h = next_h
                        c = next_c
                        prediction.append(next_token)
                result.append(
                    torch.tensor(
                        [predict.item() for predict in prediction], device=frame.device
                    )
                )

        return {"result": result}

    def infer_beam_search(self, x, spectrogram_length, beam_size=10, **kwargs):
        result = []
        with torch.no_grad():
            encodings, x_lengths = self.conformer(x.transpose(1, 2), spectrogram_length)

            for i in range(len(encodings)):
                encoding = encodings[i]
                length = x_lengths[i]

                
                bos_token = torch.tensor(self.bos_idx, device=encoding.device)
                g, h, c = self.prediction_network(bos_token.unsqueeze(0).unsqueeze(0), None, None)
                
                # (probability, g, hidden state, cell state, token_count)
                b_hypos = {
                    "1": (
                        torch.tensor(0.0, device=encoding.device),
                        g,
                        h,
                        c,
                        0
                    )
                }

                for frame in encoding[:length]:
                    a_hypos = b_hypos
                    b_hypos = {}

                    for _ in range(self.max_tokens_per_frame):
                        if len(a_hypos) == 0:
                            break
                        
                        for hypo_key, hypo in a_hypos.items():
                            logits = self.joint_network.infer(frame, hypo[2])
                            log_probs = F.log_softmax(logits, dim=-1).squeeze(0, 1)

                            # generate blank hypothesis
                            blank_hyp = (
                                hypo[0].logaddexp(log_probs[self.pad_idx]),
                                hypo[1],
                                hypo[2],
                                hypo[3],
                                hypo[4]
                            )

                            b_hypos[hypo_key] = blank_hyp

                            # generate new hypotheses
                            for token_id in torch.topk(
                                log_probs, beam_size
                            ).indices:  # optimization, take only top-k probs
                                if token_id.item() == self.pad_idx:
                                    continue

                                new_g, new_h, new_c = self.prediction_network(
                                    torch.tensor(token_id.item(), device=frame.device)
                                    .unsqueeze(0)
                                    .unsqueeze(0),
                                    hypo[2],
                                    hypo[3],
                                )
                                logits = self.joint_network.infer(frame, new_g)
                                log_probs = F.log_softmax(logits, dim=-1).squeeze(0, 1)

                                new_hypo_key = hypo_key + f" {token_id}"
                                new_hypo_prob = hypo[0].logaddexp(log_probs[token_id])
                                new_hypo_last_token = token_id

                                a_hypos[new_hypo_key] = (
                                    new_hypo_prob,
                                    new_hypo_last_token,
                                    new_g,
                                    new_h,
                                    new_c,
                                    hypo[4] + 1
                                )

                    b_hypos = dict(
                        sorted(b_hypos.items(), key=lambda x: x[1][0] / x[1][4], reverse=True)[
                            :beam_size
                        ]
                    )

                result.append(
                    (
                        int(r)
                        for r in max(b_hypos.items(), key=lambda x: x[1][0] / x[1][4])[0][1:].split()
                    )
                )

        return {"result_beam": result}

    def __str__(self):
        """
        Model prints with the number of parameters.
        """
        all_parameters = sum([p.numel() for p in self.parameters()])
        trainable_parameters = sum(
            [p.numel() for p in self.parameters() if p.requires_grad]
        )

        result_info = super().__str__()
        result_info = result_info + f"\nAll parameters: {all_parameters}"
        result_info = result_info + f"\nTrainable parameters: {trainable_parameters}"

        return result_info
