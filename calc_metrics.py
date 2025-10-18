"""
NameOfTheDirectoryWithUtterances/
├── audio/
│   ├── UtteranceID1.wav  # can be flac or mp3
│   ├── UtteranceID2.wav
│   └── ...
└── transcriptions/  # ground truth
    ├── UtteranceID1.txt
    ├── UtteranceID2.txt
    └── ...

Usage:
    python calc_metrics.py --dataset_dir <path> --predictions <path> [options]
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

from src.metrics.utils import calc_cer, calc_wer
from src.text_encoder import RNNTTextEncoder


def get_utterance_id_from_filename(filename: str) -> str:
    return Path(filename).stem


def parse_dataset_directory(dataset_dir: Path) -> Dict[str, str]:
    audio_dir = dataset_dir / "audio"
    transcriptions_dir = dataset_dir / "transcriptions"

    if not audio_dir.exists():
        raise FileNotFoundError(f"Audio directory not found: {audio_dir}")

    if not transcriptions_dir.exists():
        raise FileNotFoundError(
            f"Transcriptions directory not found: {transcriptions_dir}"
        )

    audio_extensions = {".wav", ".flac", ".mp3", ".m4a", ".ogg"}
    audio_files = []
    for ext in audio_extensions:
        audio_files.extend(audio_dir.glob(f"*{ext}"))

    if not audio_files:
        raise ValueError(f"No audio files found in {audio_dir}")

    utterance_ids = {get_utterance_id_from_filename(f.name) for f in audio_files}

    ground_truth = {}
    missing_transcriptions = []

    for utterance_id in utterance_ids:
        transcription_file = transcriptions_dir / f"{utterance_id}.txt"
        if transcription_file.exists():
            try:
                with open(transcription_file, "r", encoding="utf-8") as f:
                    ground_truth[utterance_id] = f.read().strip()
            except IOError as e:
                print(f"Warning: Could not read {transcription_file}: {e}")
                missing_transcriptions.append(utterance_id)
        else:
            missing_transcriptions.append(utterance_id)

    if missing_transcriptions:
        print(
            f"Warning: Missing transcriptions for {len(missing_transcriptions)} utterances"
        )

    if not ground_truth:
        raise ValueError("No ground truth transcriptions found")

    print(
        f"Found {len(ground_truth)} ground truth transcriptions out of {len(utterance_ids)} audio files"
    )
    return ground_truth


def parse_predictions_file(predictions_path: Path) -> Dict[str, str]:
    if not predictions_path.exists():
        raise FileNotFoundError(f"Predictions directory not found: {predictions_path}")

    if not predictions_path.is_dir():
        raise ValueError(f"Predictions path is not a directory: {predictions_path}")

    txt_files = list(predictions_path.glob("*.txt"))
    if not txt_files:
        raise ValueError(
            f"No .txt files found in predictions directory: {predictions_path}"
        )

    predictions = {}
    for txt_file in txt_files:
        utterance_id = get_utterance_id_from_filename(txt_file.name)
        try:
            with open(txt_file, "r", encoding="utf-8") as f:
                predictions[utterance_id] = f.read().strip()
        except IOError as e:
            print(f"Warning: Could not read {txt_file}: {e}")

    return predictions


def calculate_metrics_from_dicts(
    ground_truth: Dict[str, str],
    predictions: Dict[str, str],
    text_encoder: RNNTTextEncoder,
) -> Tuple[List[float], List[float], float, float, List[str]]:
    common_ids = set(ground_truth.keys()) & set(predictions.keys())

    if not common_ids:
        raise ValueError(
            "No common utterance IDs found between ground truth and predictions"
        )

    missing_predictions = set(ground_truth.keys()) - set(predictions.keys())
    if missing_predictions:
        print(f"Warning: Missing predictions for {len(missing_predictions)} utterances")

    extra_predictions = set(predictions.keys()) - set(ground_truth.keys())
    if extra_predictions:
        print(f"Warning: Extra predictions for {len(extra_predictions)} utterances")

    individual_wers = []
    individual_cers = []
    utterance_ids = []
    valid_samples = 0

    for utterance_id in sorted(common_ids):
        gt_text = ground_truth[utterance_id]
        pred_text = predictions[utterance_id]

        gt_text = text_encoder.normalize_text(gt_text)
        pred_text = text_encoder.normalize_text(pred_text)

        wer = calc_wer(gt_text, pred_text)
        cer = calc_cer(gt_text, pred_text)

        if wer == -1 or cer == -1:
            print(f"Warning: Skipping utterance {utterance_id} with empty ground truth")
            continue

        individual_wers.append(wer)
        individual_cers.append(cer)
        utterance_ids.append(utterance_id)
        valid_samples += 1

    if valid_samples == 0:
        raise ValueError("No valid samples found (all ground truth texts are empty)")

    average_wer = sum(individual_wers) / len(individual_wers)
    average_cer = sum(individual_cers) / len(individual_cers)

    return individual_wers, individual_cers, average_wer, average_cer, utterance_ids


def save_results(
    individual_wers: List[float],
    individual_cers: List[float],
    average_wer: float,
    average_cer: float,
    utterance_ids: List[str],
    output_path: Path = None,
) -> None:
    """Save results to file or print to console."""
    results = {
        "average_wer": average_wer,
        "average_cer": average_cer,
        "num_samples": len(individual_wers),
        "individual_results": [
            {"utterance_id": uid, "wer": wer, "cer": cer}
            for uid, wer, cer in zip(utterance_ids, individual_wers, individual_cers)
        ],
    }

    if output_path:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"Results saved to: {output_path}")
    else:
        print("Results:")
        print(f"Number of samples: {len(individual_wers)}")
        print(f"\tAverage WER: {average_wer:.4f} ({average_wer*100:.2f}%)")
        print(f"\tAverage CER: {average_cer:.4f} ({average_cer*100:.2f}%)")


def main():
    parser = argparse.ArgumentParser(
        description="Calculate WER and CER metrics",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python calc_metrics.py --dataset_dir /path/to/dataset --predictions /path/to/predictions_dir
  python calc_metrics.py --dataset_dir /path/to/dataset --predictions /path/to/predictions_dir --output results.json
  python calc_metrics.py --dataset_dir /path/to/dataset --predictions /path/to/predictions_dir --verbose
        """,
    )

    mode_group = parser.add_mutually_exclusive_group(required=True)

    mode_group.add_argument(
        "--dataset_dir",
        type=Path,
        help="Path to dataset directory with audio/ and transcriptions/ folders",
    )

    parser.add_argument(
        "--predictions",
        type=Path,
        required=True,
        help="Path to predictions directory with individual .txt files",
    )

    parser.add_argument(
        "--output", type=Path, help="Path to save results as JSON file (optional)"
    )

    parser.add_argument(
        "--verbose", action="store_true", help="Print individual sample results"
    )

    args = parser.parse_args()

    try:
        text_encoder = RNNTTextEncoder()

        print(f"Dataset dir: {args.dataset_dir}")
        ground_truth = parse_dataset_directory(args.dataset_dir)
        predictions = parse_predictions_file(args.predictions)

        (
            individual_wers,
            individual_cers,
            average_wer,
            average_cer,
            utterance_ids,
        ) = calculate_metrics_from_dicts(
            ground_truth=ground_truth,
            predictions=predictions,
            text_encoder=text_encoder,
        )

        save_results(
            individual_wers=individual_wers,
            individual_cers=individual_cers,
            average_wer=average_wer,
            average_cer=average_cer,
            utterance_ids=utterance_ids,
            output_path=args.output,
        )

        if args.verbose:
            print("\nIndividual sample results:")
            print("Utterance ID\tWER\tCER")
            print("-" * 40)
            for uid, wer, cer in zip(utterance_ids, individual_wers, individual_cers):
                print(f"{uid}\t{wer:.4f}\t{cer:.4f}")

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
