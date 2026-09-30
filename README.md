# Hindi Spoken Digit Recognition

A machine learning pipeline for classifying Hindi spoken digits (0-9) from speech audio using MFCC-based features, an RBF-SVM baseline, and a lightweight 2D convolutional neural network.

The project uses a speaker-independent evaluation protocol, meaning speakers in the test set are completely unseen during training.

## Dataset & Experimental Protocol

- 20,174 labeled audio clips
- 15,817 training clips
- 4,357 test clips
- 86 training speakers
- 22 test speakers
- 0 speaker overlap between train and test
- 10 digit classes (0-9)
- Audio standardized to 16 kHz mono
- Fixed audio duration of 1 second

The train/test split uses `GroupShuffleSplit` with speaker ID as the grouping variable, preventing the same speaker from appearing in both training and test sets.

## Feature Extraction

The SVM uses a 78-dimensional feature vector consisting of:

- 13 MFCC coefficients
- MFCC mean and standard deviation
- Delta mean and standard deviation
- Delta-delta mean and standard deviation

The CNN uses MFCC, delta, and delta-delta representations while preserving their time-frequency structure as a 2D tensor.

## Models

### V1 - RBF SVM Baseline

- StandardScaler
- RBF kernel
- C = 10
- gamma = scale

### V2 - Lightweight 2D CNN

A lightweight convolutional neural network designed to learn patterns from the MFCC-based time-frequency representation.

- 620,810 trainable parameters
- Approximately 0.58M parameters

## Results

Both models are evaluated on the same speaker-independent test set of 4,357 clips from 22 unseen speakers.

| Model | Accuracy | Macro Precision | Macro Recall | Macro F1 |
|---|---:|---:|---:|---:|
| RBF SVM | 53.16% | 53.45% | 53.25% | 52.96% |
| Lightweight 2D CNN | **78.75%** | **79.79%** | **78.83%** | **78.76%** |

The CNN improves test accuracy by **25.59 percentage points** over the SVM baseline.

### CNN Per-Class Results

| Digit | Precision | Recall | F1 |
|---|---:|---:|---:|
| 0 | 71.75% | 87.81% | 78.97% |
| 1 | 82.52% | 81.19% | 81.85% |
| 2 | 69.01% | 78.59% | 73.49% |
| 3 | 90.24% | 78.17% | 83.77% |
| 4 | 87.21% | 59.91% | 71.03% |
| 5 | 89.97% | 78.48% | 83.83% |
| 6 | 87.35% | 80.86% | 83.98% |
| 7 | 72.09% | 87.84% | 79.19% |
| 8 | 73.27% | 80.67% | 76.79% |
| 9 | 74.50% | 74.83% | 74.66% |

Confusion matrices are available in the `results/` directory.

## Repository Structure

```text
hindi-spoken-digit-recognition/

├── configs/
│   └── config.yaml
├── data/
│   ├── raw/
│   └── split_manifest.json
├── models/
│   ├── cnn2d_best.pth
│   ├── cnn2d_meta.json
│   └── svm_baseline.joblib
├── results/
│   ├── cnn2d_results.json
│   ├── svm_baseline_results.json
│   ├── confusion_matrix_cnn2d.png
│   └── confusion_matrix_svm.png
├── src/
│   ├── inspect_dataset.py
│   ├── dataset.py
│   ├── features.py
│   ├── features_2d.py
│   └── models/
│       └── cnn2d.py
├── train_svm.py
├── train_cnn2d.py
├── evaluate.py
├── evaluate_cnn2d.py
├── requirements.txt
└── README.md
````

## How to Run

### Install dependencies

```bash
pip install -r requirements.txt
```

### Train SVM

```bash
python train_svm.py
```

### Evaluate SVM

```bash
python evaluate.py
```

### Train CNN

```bash
python train_cnn2d.py
```

### Evaluate CNN

```bash
python evaluate_cnn2d.py
```

## Notes

The raw audio dataset is not included in the repository.

Feature caches are also excluded because they are generated artifacts.

The trained SVM and CNN models are included so that the final models do not need to be retrained just to use the repository.
