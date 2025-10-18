from pathlib import Path

import torchaudio

from src.datasets.base_dataset import BaseDataset


class CustomDirAudioDataset(BaseDataset):
    def __init__(self, audio_dir, transcription_dir=None, *args, **kwargs):
        data = []
        for path in Path(audio_dir).iterdir():
            entry = {}
            if path.suffix in [".mp3", ".wav", ".flac", ".m4a"]:
                entry["path"] = str(path)

                try:
                    audio_info = torchaudio.info(str(path))
                    entry["audio_len"] = audio_info.num_frames / audio_info.sample_rate
                except Exception as e:
                    print(f"Warning: Could not load audio info for {path}: {e}")
                    continue

                if transcription_dir and Path(transcription_dir).exists():
                    transc_path = Path(transcription_dir) / (path.stem + ".txt")
                    if transc_path.exists():
                        with transc_path.open() as f:
                            entry["text"] = f.read().strip()
                    else:
                        continue
                else:
                    continue
            if len(entry) > 0:
                data.append(entry)
        super().__init__(data, *args, **kwargs)
