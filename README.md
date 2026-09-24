# Limited-Dataset-Training

## LOCI-1: MicroData Benchmark — High-Accuracy Learning Under Data Scarcity

### Overview

This project investigates how modern transfer-learning models perform under extreme data scarcity. Rather than maximizing dataset size, the focus is on sample efficiency, robustness, and reliable evaluation.

Using a deliberately constrained subset of a public chest X-ray image dataset, this benchmark studies whether strong classification performance can be achieved with as few as **50 samples per class**.

> This project is a computer vision benchmark only. It does not perform medical diagnosis and makes no clinical claims.

### Objectives

- Study high-accuracy learning with very small datasets
- Evaluate transfer learning under strict data constraints
- Analyze the effect of fine-tuning and decision thresholds
- Emphasize reliable evaluation using balanced validation splits

## Methodology

### Model

- **Architecture:** EfficientNet-B0 (`torchvision`)
- **Initialization:** ImageNet pretrained weights
- **Training strategy:**
  - Phase 1: train the classification head with a frozen backbone (lr 5e-3)
  - Phase 2: fine-tune the top 4 backbone blocks + head (lr 1e-3, cosine decay)
- **Loss:** Binary Cross-Entropy with logits (single output logit, PNEUMONIA = positive class)
- **Optimizer:** AdamW (weight decay 1e-4)
- BatchNorm layers in frozen blocks stay in eval mode so tiny batches don't corrupt their running statistics.
- Augmentation (train only): random resized crop (scale 0.8–1.0), ±10° rotation, light brightness/contrast jitter.

### Dataset Setup

- **Dataset:** [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) — public Kaggle dataset, downloaded via `kagglehub`
- **Classes:** NORMAL, PNEUMONIA
- **Training subset:** 50 images per class (100 total), randomly sampled from the original `train` split
- **Validation subset:** balanced 50 per class (100 total)

The original dataset's `val` split has only 8 images per class, so `data/chest_xray/val/` is built from the original `val` + `test` splits (242 NORMAL / 398 PNEUMONIA). Both are disjoint from the original `train` split. Balanced subsampling is deterministic given `--seed`.

### Evaluation Metrics

- Accuracy, Precision, Recall, F1-score, ROC-AUC
- Confusion matrix
- Validation-based threshold optimization (sweep 0.05–0.95, step 0.01, maximize F1)

## Key Results (N = 50 per class)

Primary experiment (command below, seed 42, Apple M-series GPU via MPS):

| Metric | Threshold 0.50 | Best-F1 threshold (0.34) |
|---|---|---|
| Accuracy | 0.830 | 0.840 |
| Precision | 0.824 | 0.815 |
| Recall | 0.840 | 0.880 |
| F1-score | **0.832** | **0.846** |
| ROC-AUC | 0.886 | 0.886 |

Confusion matrix at the best threshold: **TN = 40, FP = 10, FN = 6, TP = 44**.

**Seed variance.** With only 100 training images, *which* 100 you draw matters a lot. Across seeds {0, 1, 2, 3, 42} (each seed also changes the sampled subset), the same configuration gives:

| Seed | F1 @ 0.50 | Best F1 | ROC-AUC |
|---|---|---|---|
| 0 | 0.923 | 0.936 | 0.982 |
| 1 | 0.822 | 0.889 | 0.955 |
| 2 | 0.828 | 0.854 | 0.907 |
| 3 | 0.901 | 0.928 | 0.971 |
| 42 | 0.832 | 0.846 | 0.886 |
| **Mean** | **0.861** | **0.891** | **0.940** |

On the full 640-image validation pool (seed-42 model, `--val-max-samples-per-class 0`): F1 0.861 @ 0.50, ROC-AUC 0.904.

Note: the "best threshold" is chosen on the same validation set it is reported on, so best-threshold numbers are optimistic. Use the threshold-0.50 column and ROC-AUC as the unbiased figures.

## Project Structure

```
LOCI-1/
├── data/
│   └── chest_xray/
│       ├── train/            # NORMAL/, PNEUMONIA/
│       └── val/              # NORMAL/, PNEUMONIA/
├── src/
│   ├── datasets/chest_xray.py      # dataset, balanced subsampling, transforms
│   ├── models/efficientnet.py      # EfficientNet-B0 + freeze/unfreeze helpers
│   ├── train/train_classifier.py   # two-phase training entry point
│   ├── eval/metrics.py             # metrics, threshold sweep, plots
│   ├── eval/evaluate.py            # re-evaluate a saved run
│   └── utils/common.py             # seeding, device, JSON I/O
├── scripts/
│   ├── download_data.py
│   └── inspect_data.py
├── outputs/
│   └── runs/
├── requirements.txt
└── README.md
```

## Setup

Install dependencies:

```bash
pip install -r requirements.txt
```

Download the dataset via API:

```bash
python scripts/download_data.py
```

This uses the `kagglehub` Python API and does not require manual dataset downloads (public datasets download without Kaggle credentials; ~2.3 GB, cached in `~/.cache/kagglehub`). Add `--symlink` to link into the cache instead of copying.

Inspect class balance and image sizes:

```bash
python scripts/inspect_data.py
```

## Training (Primary Experiment)

```bash
python -m src.train.train_classifier --max-samples-per-class 50 --val-max-samples-per-class 50 --batch-size 32 --phase1-epochs 3 --phase2-epochs 12 --seed 42 --num-workers 0
```

Runs in about a minute on an Apple Silicon GPU (auto-selects CUDA > MPS > CPU; override with `--device`). Each run writes to `outputs/runs/<timestamp>_n50_seed42/`:

| File | Contents |
|---|---|
| `model.pt` | final model weights + config |
| `config.json` | all CLI arguments and sampled class counts |
| `history.json` | per-epoch train/val loss, accuracy, F1, AUC |
| `metrics.json` | metrics at threshold 0.50 and at the best-F1 threshold |
| `threshold_sweep.csv` / `.png` | metrics at every threshold |
| `val_predictions.csv` | per-image probability |
| `roc_curve.png`, `confusion_matrix.png` | plots |

Other useful flags: `--phase1-lr`, `--phase2-lr`, `--unfreeze-blocks`, `--weight-decay`, `--dropout`, `--image-size`, `--run-name`.

## Threshold Optimization

After training, validation probabilities are evaluated across multiple decision thresholds to identify the value that maximizes F1-score. This highlights how decision strategy impacts performance, especially in small-data settings. It runs automatically at the end of training, and can be redone for any saved run:

```bash
python -m src.eval.evaluate --run-dir outputs/runs/<run_name>
python -m src.eval.evaluate --run-dir outputs/runs/<run_name> --val-max-samples-per-class 0   # full val pool
```

## Why Small-Data Learning Matters

In many real-world ML systems, data is expensive or limited, labels are scarce, and reliability matters more than scale. This project explores how to build ML pipelines that remain effective even when data is minimal.

## Disclaimer

This repository is intended solely for machine learning research and benchmarking. It is not a medical system and should not be used for diagnostic purposes.

## Author

Independent machine learning project focused on sample-efficient learning and reliable AI systems.
