# Network Intrusion Detection System (Machine Learning)

## Overview

This project implements a machine learning–based Intrusion Detection System (IDS) using the CIC-IDS2017 dataset.
The system classifies network flows as either benign or malicious.

The main goal is to build a realistic IDS pipeline and evaluate it under temporal generalization conditions.

---

## Dataset

- Name: CIC-IDS2017
- Source: Canadian Institute for Cybersecurity
- Size: ~2.8 million network flows
- Features: 78 statistical traffic features
- Classes: Benign, Attack

Raw CSV files are stored locally and are not included in the repository.

---

## Project Structure

```
ids-ml-project/
│
├── data/
│   ├── raw/            # Original dataset (local only)
│   └── processed/      # Cleaned merged dataset (local only)
│
├── models/             # Trained models
│
├── src/                # Source code
│   ├── merge_data.py
│   └── train.py
│
├── app.py              # Streamlit demo (optional)
├── requirements.txt
└── README.md
```

## Methodology

### 1. Data Preprocessing

- Merged 8 CSV files into a unified dataset
- Removed infinite and missing values
- Converted labels to binary format
- Added source file metadata to avoid data leakage

Final dataset size: 2,827,876 rows

### 2. Train-Test Split

To prevent temporal data leakage, data was split by capture day:

- Training: Monday to Thursday
- Testing: Friday

This simulates real-world deployment.

Random splitting was avoided due to severe leakage.

### 3. Model

- Algorithm: XGBoost (Gradient Boosted Trees)
- Tree method: Histogram-based
- Class imbalance handled using scale_pos_weight

### 4. Imbalance Handling

Attack traffic is under-represented in training data.
Class weights were computed using:


and integrated into XGBoost.

### 5. Evaluation

Evaluation was performed using:

- Precision
- Recall
- F1-score
- Accuracy

Focus was placed on attack recall due to security relevance.

---

## Results

Evaluation on unseen Friday traffic:

| Class  | Precision | Recall | F1-score |
|--------|-----------|--------|----------|
| Benign | 0.67      | 0.99   | 0.80     |
| Attack | 0.99      | 0.28   | 0.44     |

Overall Accuracy: 70%

These results reflect realistic generalization performance under temporal distribution shift.

---

## Key Findings

- Random train-test splits produced near-100% accuracy due to data leakage
- Day-based splitting revealed realistic performance
- Class weighting improved attack recall
- Performance plateaued due to domain shift in attack patterns

---

## Limitations

- Limited generalization to unseen attack types
- Dataset contains correlated and duplicated flows
- No real-time packet capture
- Offline batch processing only

---

## Future Work

- Feature selection and stability analysis
- Domain adaptation methods
- Per-attack-type classifiers
- Online learning pipeline
- Integration with real packet capture

---

## How to Run

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Merge Dataset

```bash
python src/merge_data.py
```

### 3. Train Model

```bash
python src/train.py
```