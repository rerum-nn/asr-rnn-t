import logging
import random
from pathlib import Path

import torchaudio

from src.datasets.librispeech_dataset import LibrispeechDataset
from src.logger.cometml import CometMLWriter
from src.logger.logger import setup_logging
from src.text_encoder import CTCTextEncoder
from src.transforms.spec_augs import (
    FrequencyMasking,
    Logarithm,
    NormalizeSpec,
    TimeMasking,
)
from src.transforms.wav_augs import (
    Gain,
    GaussianNoise,
    NormalizeRMS,
    PitchShift,
    TimeStretch,
)


def main():
    setup_logging(Path("/home/rerum-nn/asr-rnn-t/outputs"))
    logger = logging.getLogger(__name__)

    text_encoder = CTCTextEncoder()

    instance_transforms = {
        "get_spectrogram": torchaudio.transforms.MelSpectrogram(
            sample_rate=16000, n_fft=400, hop_length=160, n_mels=80
        ),
        "spectrogram": Logarithm(),
    }

    dataset = LibrispeechDataset(
        part="test-clean",
        limit=5,
        shuffle_index=True,
        text_encoder=text_encoder,
        instance_transforms=instance_transforms,
    )

    wave_augmentations = {
        "gain": Gain(min_gain_in_db=-6.0, max_gain_in_db=6.0, p=1.0),
        "noise": GaussianNoise(max_std=0.01, p=1.0),
        "normalize_rms": NormalizeRMS(target_rms=0.1),
        "pitch_shift": PitchShift(min_semitones=-2, max_semitones=2, sr=16000, p=1.0),
        "time_stretch": TimeStretch(
            min_time_stretch=0.85, max_time_stretch=1.15, p=1.0
        ),
    }

    spec_augmentations = {
        "frequency_masking": FrequencyMasking(frequency_mask_param=10, p=1.0),
        "time_masking": TimeMasking(time_mask_param=100, p=1.0, masks_number=1),
        "normalize_spec": NormalizeSpec(params_path="normalization_params_clean.json"),
    }

    config = {"trainer": {"resume_from": None}}

    comet_writer = CometMLWriter(
        logger=logger,
        project_config=config,
        project_name="pytorch-asr-conformer-rnnt",
        run_name=f"demo_{random.randint(1000, 9999)}",
    )

    for i in range(len(dataset)):
        logger.info(f"{i+1}/{len(dataset)}")

        sample = dataset[i]
        original_audio = sample["audio"]
        original_spec = sample["spectrogram"]
        text = sample["text"]

        comet_writer.set_step(i, mode="demo")
        comet_writer.add_audio(f"original_{i}", original_audio, sample_rate=16000)
        comet_writer.add_image(f"original_spec_{i}", original_spec)
        comet_writer.add_text(f"text_{i}", f"<b>Текст:</b> {text}")

        for aug_name, aug in wave_augmentations.items():
            augmented_audio = aug(original_audio.clone())
            augmented_spec = instance_transforms["get_spectrogram"](augmented_audio)
            augmented_log_spec = instance_transforms["spectrogram"](augmented_spec)

            comet_writer.add_audio(
                f"{aug_name}_audio_{i}", augmented_audio, sample_rate=16000
            )
            comet_writer.add_image(f"{aug_name}_spec_{i}", augmented_log_spec)

        for aug_name, aug in spec_augmentations.items():
            augmented_log_spec = aug(original_spec.clone())

            comet_writer.add_image(f"{aug_name}_spec_{i}", augmented_log_spec)


if __name__ == "__main__":
    main()
