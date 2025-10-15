from pathlib import Path

import pandas as pd
import torch
from torch.profiler import record_function

from src.logger.utils import plot_spectrogram
from src.metrics.tracker import MetricTracker
from src.metrics.utils import calc_cer, calc_wer
from src.trainer.base_trainer import BaseTrainer


class Trainer(BaseTrainer):
    """
    Trainer class. Defines the logic of batch logging and processing.
    """

    def process_batch(self, batch_idx, batch, metrics: MetricTracker):
        """
        Run batch through the model, compute metrics, compute loss,
        and do training step (during training stage).

        The function expects that criterion aggregates all losses
        (if there are many) into a single one defined in the 'loss' key.

        Args:
            batch (dict): dict-based batch containing the data from
                the dataloader.
            metrics (MetricTracker): MetricTracker object that computes
                and aggregates the metrics. The metrics depend on the type of
                the partition (train or inference).
        Returns:
            batch (dict): dict-based batch containing the data from
                the dataloader (possibly transformed via batch transform),
                model outputs, and losses.
        """
        with record_function("move_batch_to_device"):
            batch = self.move_batch_to_device(batch)

        with record_function("transform_batch"):
            batch = self.transform_batch(batch)  # transform batch on device -- faster

        metric_funcs = self.metrics["inference"]
        if self.is_train:
            metric_funcs = self.metrics["train"]

        with record_function("model_forward"):
            outputs = self.model(**batch)
        batch.update(outputs)

        with record_function("loss_computation"):
            all_losses = self.criterion(**batch)
        batch.update(all_losses)

        if self.is_train:
            original_loss = batch["loss"]
            batch["loss"] = batch["loss"] / self.gradient_accumulation_steps
            with record_function("backward_pass"):
                batch["loss"].backward()
            batch["loss"] = original_loss

        # update metrics for each loss (in case of multiple losses)
        if batch_idx % self.log_step == 0:
            with record_function("metrics_update"):
                for loss_name in self.config.writer.loss_names:
                    metrics.update(loss_name, batch[loss_name].item())

                for met in metric_funcs:
                    metrics.update(met.name, met(**batch))
        return batch

    def _log_batch(self, batch_idx, batch, mode="train"):
        """
        Log data from batch. Calls self.writer.add_* to log data
        to the experiment tracker.

        Args:
            batch_idx (int): index of the current batch.
            batch (dict): dict-based batch after going through
                the 'process_batch' function.
            mode (str): train or inference. Defines which logging
                rules to apply.
        """
        # method to log data from you batch
        # such as audio, text or images, for example

        # logging scheme might be different for different partitions
        if mode == "train":  # the method is called only every self.log_step steps
            self.log_spectrogram(**batch)
            self.log_predictions(**batch)
            self.log_audio(**batch)
        else:
            # Log Stuff
            self.log_spectrogram(**batch)
            self.log_predictions(**batch)
            self.log_audio(**batch)

    def log_audio(self, instance_audio, audio, **batch):
        audio_to_log = audio[0]
        self.writer.add_audio("audio", audio_to_log, sample_rate=16000)
        audio_to_log = instance_audio[0]
        self.writer.add_audio("instance_audio", audio_to_log, sample_rate=16000)

    def log_spectrogram(self, instance_spectrogram, spectrogram, **batch):
        spectrogram_for_plot = spectrogram[0].detach().cpu()
        image = plot_spectrogram(spectrogram_for_plot)
        self.writer.add_image("spectrogram", image)
        spectrogram_for_plot = instance_spectrogram[0].detach().cpu()
        image = plot_spectrogram(spectrogram_for_plot)
        self.writer.add_image("instance_spectrogram", image)

    def log_predictions(
        self, text, log_probs, log_probs_length, audio_path, examples_to_log=10, **batch
    ):
        # TODO add beam search
        # Note: by improving text encoder and metrics design
        # this logging can also be improved significantly

        argmax_inds = log_probs.cpu().argmax(-1).numpy()
        argmax_inds = [
            inds[: int(ind_len)]
            for inds, ind_len in zip(argmax_inds, log_probs_length.cpu().numpy())
        ]
        argmax_texts = [self.text_encoder.output_decode(inds) for inds in argmax_inds]
        tuples = list(zip(argmax_texts, text, audio_path))

        rows = {}
        for pred, target, audio_path in tuples[:examples_to_log]:
            target = self.text_encoder.normalize_text(target)
            wer = calc_wer(target, pred) * 100
            cer = calc_cer(target, pred) * 100

            rows[Path(audio_path).name] = {
                "target": target,
                "predictions": pred,
                "wer": wer,
                "cer": cer,
                "step": self.writer.step,
            }
        self.writer.add_table(
            "predictions", pd.DataFrame.from_dict(rows, orient="index")
        )
