# KNN From Scratch

This project implements the K-Nearest Neighbors (KNN) algorithm from scratch in Python, with parameter tuning using scikit-learn’s GridSearchCV.

---

## Project Overview

- Custom KNN classifier supporting Euclidean and Manhattan distances.
- Data loading and splitting with Iris dataset.
- Pipeline includes optional data scaling and hyperparameter tuning (`k`, distance metric, scaler).
- Evaluation with accuracy and detailed classification reports.

---

## Files

- `model.py` — KNN algorithm implementation (`fit`, `predict`, `evaluate`).
- `data_loader.py` — Loads Iris dataset and splits into train, validation, and test sets.
- `pipeline.py` — Defines pipeline and GridSearchCV for hyperparameter tuning.
- `main.py` — Runs training, prints best parameters, evaluates and reports results.

---

## Usage

1. Install dependencies: `numpy`, `scikit-learn`, `pandas`
2. Run the main script:  
   ```bash
   python main.py
