# ASR RNN-T

PyTorch implementation of Automatic Speech Recognition using RNN-Transducer with Conformer encoder.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/rerum-nn/asr-rnn-t.git
   cd asr-rnn-t
   ```

2. Create conda environment (strongly recommended):
   ```bash
   conda create -n dla python=3.10
   conda activate dla
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. If you have troubles with library `youtokentome` installinng, please delete this dependency from `requirements.txt` and try next command:
   ```bash
   conda install youtokentome -c conda-forge
   pip install -r requierements.txt
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

## Configuration

Uses Hydra for configuration. Main configs in `src/configs/`:
- `conformer_rnn_t.yaml`: RNN-T configuration
- `train_[clean|clean2|other].yaml`: training configs.
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

Test on LibriSpeech datasets:
```bash
python inference.py -cn=conformer_rnn_t datasets=librispeech ++datasets.test.part=test-other
```

## Calc metrics

```bash
python calc_metrics.py --predictions ./predictions/dir --dataset_dir ./dataset/dir
```

Metrics: WER, CER

### File structure

Dataset directory should have the following structure:
```
dataset_dir/
├── audio/
│   ├── utterance1.wav  # can be .flac, .mp3, .m4a, .ogg
│   ├── utterance2.wav
│   └── ...
└── transcriptions/  # ground truth
    ├── utterance1.txt
    ├── utterance2.txt
    └── ...
```

Predictions directory should contain individual .txt files:
```
predictions_dir/
├── utterance1.txt
├── utterance2.txt
└── ...
```

## Project Structure

```
src/
├── configs/           # Configuration files
├── datasets/          # Dataset implementations
├── logger/            # Experiment tracking
├── loss/              # Loss functions (RNN-T)
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
