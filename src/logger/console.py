from datetime import datetime

import numpy as np
import pandas as pd


class ConsoleWriter:
    def __init__(
        self,
        logger,
        project_config,
        project_name,
        run_id=None,
        run_name=None,
        **kwargs,
    ):
        self.logger = logger
        self.project_name = project_name
        self.run_id = run_id
        self.run_name = run_name
        self.project_config = project_config

        if run_name:
            self.logger.info(f"Run name: {run_name}")
        if run_id:
            self.logger.info(f"Run ID: {run_id}")

        self.step = 0
        self.mode = ""
        self.timer = datetime.now()

    def set_step(self, step, mode="train"):
        self.mode = mode
        previous_step = self.step
        self.step = step
        if step == 0:
            self.timer = datetime.now()
        else:
            duration = datetime.now() - self.timer
            self.add_scalar(
                "steps_per_sec", (self.step - previous_step) / duration.total_seconds()
            )
            self.timer = datetime.now()

    def _object_name(self, object_name):
        return f"{object_name}_{self.mode}"

    def add_checkpoint(self, checkpoint_path, save_dir):
        raise NotImplementedError()

    def add_scalar(self, scalar_name, scalar):
        formatted_name = self._object_name(scalar_name)
        self.logger.info(f"Step {self.step} | {formatted_name}: {scalar}")

    def add_scalars(self, scalars):
        formatted_scalars = {
            self._object_name(scalar_name): scalar
            for scalar_name, scalar in scalars.items()
        }
        scalars_str = " | ".join(
            [f"{name}: {value}" for name, value in formatted_scalars.items()]
        )
        self.logger.info(f"Step {self.step} | {scalars_str}")

    def add_image(self, image_name, image):
        formatted_name = self._object_name(image_name)
        self.logger.info(f"Step {self.step} | Image logged: {formatted_name}")

    def add_audio(self, audio_name, audio, sample_rate=None):
        formatted_name = self._object_name(audio_name)
        sample_rate_str = f" (sample_rate: {sample_rate})" if sample_rate else ""
        self.logger.info(
            f"Step {self.step} | Audio logged: {formatted_name}{sample_rate_str}"
        )

    def add_text(self, text_name, text):
        formatted_name = self._object_name(text_name)
        self.logger.info(f"Step {self.step} | Text logged: {formatted_name}")
        self.logger.info(f"Text content: {text}")

    def add_histogram(self, hist_name, values_for_hist, bins=None):
        formatted_name = self._object_name(hist_name)
        values_for_hist = values_for_hist.detach().cpu().numpy()
        hist_info = f"min: {values_for_hist.min():.4f}, max: {values_for_hist.max():.4f}, mean: {values_for_hist.mean():.4f}, std: {values_for_hist.std():.4f}"
        self.logger.info(
            f"Step {self.step} | Histogram logged: {formatted_name} | {hist_info}"
        )

    def add_table(self, table_name, table: pd.DataFrame):
        # formatted_name = self._object_name(table_name)
        # self.logger.info(f"Step {self.step} | Table logged: {formatted_name}")
        # self.logger.info(f"Table shape: {table.shape}")
        # self.logger.info(f"Table columns: {list(table.columns)}")
        pass

    def add_images(self, image_names, images):
        raise NotImplementedError()

    def add_pr_curve(self, curve_name, curve):
        raise NotImplementedError()

    def add_embedding(self, embedding_name, embedding):
        raise NotImplementedError()
