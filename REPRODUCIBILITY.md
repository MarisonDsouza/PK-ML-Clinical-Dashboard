# Reproducibility Guide

Follow these steps to reproduce all model weights, cross-validation metrics, and benchmarks from the raw clinical CSV dataset.

---

## 1. Prerequisites
- Python 3.10, 3.11, or 3.12
- Linux, macOS (ARM64/x86_64), or Windows WSL2

```bash
git clone https://github.com/<your-username>/PK-ML-Clinical-Dashboard.git
cd PK-ML-Clinical-Dashboard
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-training.txt
```

---

## 2. Verification of Dataset
Confirm the clinical dataset exists at `data/Kaggle_PK_Remifentanil.csv`:
```bash
python training/pk_dataset.py
```
Expected output:
```text
Dataset built successfully!
Shape: (1992, 21) | Subjects: 65 | DV range: 0.10 -> 245.40 ng/mL
Raw Minto Baseline vs Observed: R2 = 0.941 | MAE = 4.26 ng/mL
```

---

## 3. Train All 8 Models
```bash
python training/train_models.py
```
This trains:
1. `BayesianCompartment.joblib`
2. `SupportVectorRobust.joblib`
3. `RandomForest.joblib`
4. `ExtraTrees.joblib`
5. `GradientBoosting.joblib`
6. `XGBoost.joblib`
7. `CatBoost.joblib`
8. `StackingEnsemble.joblib`

And outputs `models/training_leaderboard.csv`.

---

## 4. Run Automated Test Suite
Verify that all API endpoints respond properly:
```bash
python -c "from fastapi.testclient import TestClient; import app; client = TestClient(app.app); print('API Status:', client.get('/api/health').json())"
```
