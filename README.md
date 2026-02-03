# Limited-Dataset-Training

# LOCI-1: MicroData Benchmark — High-Accuracy Learning Under Data Scarcity

## Overview
This project investigates how modern transfer-learning models perform under extreme data scarcity. Rather than maximizing dataset size, the focus is on sample efficiency, robustness, and reliable evaluation.

Using a deliberately constrained subset of a public chest X-ray image dataset, this benchmark studies whether strong classification performance can be achieved with as few as 50 samples per class.

This project is a computer vision benchmark only. It does not perform medical diagnosis and makes no clinical claims.

## Objectives
- Study high-accuracy learning with very small datasets
- Evaluate transfer learning under strict data constraints
- Analyze the effect of fine-tuning and decision thresholds
- Emphasize reliable evaluation using balanced validation splits

## Methodology

### Model
- Architecture: EfficientNet-B0
- Initialization: ImageNet pretrained weights
- Training strategy:
  - Phase 1: Train classification head with frozen backbone
  - Phase 2: Fine-tune upper backbone layers
- Loss: Binary Cross-Entropy with logits
- Optimizer: AdamW

### Dataset Setup
- Dataset: Public chest X-ray image dataset (Kaggle)
- Classes: NORMAL, PNEUMONIA
- Training subset: 50 images per class (100 total)
- Validation subset: Balanced 50 per class (100 total)

### Evaluation Metrics
- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Confusion matrix
- Validation-based threshold optimization

## Key Results (N = 50 per class)

F1-score at threshold 0.50: 0.8889  
Best F1-score with threshold optimization: approximately 0.90  
ROC-AUC: approximately 0.93  

Confusion matrix at best threshold:
- True Negatives ≈ 44
- False Positives ≈ 6
- False Negatives ≈ 4
- True Positives ≈ 46

These results demonstrate that strong performance is achievable even under severe data constraints.

## Project Structure
LOCI-1/
├── data/
│   └── chest_xray/
│       ├── train/
│       └── val/
├── src/
│   ├── datasets/
│   ├── models/
│   ├── train/
│   ├── eval/
│   └── utils/
├── scripts/
│   ├── download_data.py
│   └── inspect_data.py
├── outputs/
│   └── runs/
├── requirements.txt
└── README.md

## Setup

Install dependencies:
pip install -r requirements.txt

Download dataset via API:
python scripts/download_data.py

This uses the kagglehub Python API and does not require manual dataset downloads.

## Training (Primary Experiment)

python -m src.train.train_classifier --max-samples-per-class 50 --val-max-samples-per-class 50 --batch-size 32 --phase1-epochs 3 --phase2-epochs 12 --seed 42 --num-workers 0

## Threshold Optimization
After training, validation probabilities are evaluated across multiple decision thresholds to identify the value that maximizes F1-score. This highlights how decision strategy impacts performance, especially in small-data settings.

## Why Small-Data Learning Matters
In many real-world ML systems, data is expensive or limited, labels are scarce, and reliability matters more than scale. This project explores how to build ML pipelines that remain effective even when data is minimal.

## Disclaimer
This repository is intended solely for machine learning research and benchmarking. It is not a medical system and should not be used for diagnostic purposes.

## Author
Independent machine learning project focused on sample-efficient learning and reliable AI systems.
