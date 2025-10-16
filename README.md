# ASR RNN-T

PyTorch implementation of Automatic Speech Recognition using RNN-Transducer with Conformer encoder.


Implemented models: Conformer-RNN-T, Conformer-CTC, DeepSpeech2, and Baseline models


## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/rerum-nn/asr-rnn-t.git
   cd asr-rnn-t
   ```

2. Create virtual environment:
   ```bash
   conda create -n dla python=3.10
   conda activate dla
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

Train a model:
```bash
python train.py -cn=conformer_ctc
```

Run inference:
```bash
python inference.py -cn=inference ++datasets.test.dir=path/to/custom_dataset
```

See `demo.ipynb` for examples.

## Models

**Conformer-RNN-T**: Main architecture with Conformer encoder, LSTM prediction network, and joint network.

**Conformer-CTC**: CTC variant with Conformer encoder.

**DeepSpeech2**: DeepSpeech2 implementation.

## Configuration

Uses Hydra for configuration. Main configs in `src/configs/`:
- `conformer_ctc.yaml`: Main configuration
- `conformer_rnn_t.yaml`: RNN-T configuration
- `datasets/`: Dataset configs

## Training

```bash
# Basic training
python train.py -cn=conformer_rnn_t

# Custom parameters
python train.py -cn=conformer_rnn_t ++trainer.n_epochs=50 ++optimizer.lr=1e-3

# Resume from checkpoint
python train.py -cn=conformer_rnn_t ++trainer.resume_from=saved/conformer_m
```

## Inference

```bash
# Evaluate model
python inference.py -cn=conformer_rnn_t ++datasets.test.dir=path/to/custom_dataset

```

## Calc metrics

```bash
python calc_metrics.py --predictions ./predictions/dir --dataset_dir ./dataset/dir
```

Metrics: WER, CER

## Project Structure

```
src/
├── configs/           # Configuration files
├── datasets/          # Dataset implementations
├── logger/            # Experiment tracking
├── loss/              # Loss functions (CTC, RNN-T)
├── metrics/           # Evaluation metrics
├── model/             # Model architectures
├── text_encoder/      # Text encoding (char, BPE)
├── trainer/           # Training logic
├── transforms/        # Audio transformations
└── utils/             # Utility functions
```

## License

MIT License - see [LICENSE](LICENSE) file.

## Acknowledgments

- Based on [PyTorch Project Template](https://github.com/Blinorot/pytorch_project_template)
- Part of [HSE DLA course](https://github.com/markovka17/dla) ASR homework
- LibriSpeech dataset from [OpenSLR](https://www.openslr.org/12/)
- Conformer architecture from [Conformer paper](https://arxiv.org/abs/2005.08100)
