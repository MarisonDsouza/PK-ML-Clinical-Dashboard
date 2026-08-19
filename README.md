# Pharmacokinetic Machine Learning (PK-ML) Clinical Dashboard & Modeling Suite

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111%2B-009688.svg)](https://fastapi.tiangolo.com)
[![Status: Research Prototype](https://img.shields.io/badge/Status-Research%20Prototype-orange.svg)]()

A hybrid **Grey-Box Machine Learning & Mechanistic Pharmacokinetic (PK) Modeling** framework for **Remifentanil Target-Controlled Infusion (TCI)**.

This repository provides reproducible datasets, model training pipelines, cross-validation benchmarks, an inference engine (<1s cold start), and an interactive clinical dashboard.

---

## 🔬 Scientific Overview

Traditional Target-Controlled Infusion (TCI) systems rely purely on compartmental ODE models (e.g., Minto et al., 1997). While physiologically grounded, compartmental models suffer from unexplained inter-individual variability (IIV). Conversely, pure black-box machine learning models lack physical guarantees and often fail to preserve clearance kinetics.

This framework introduces a **Grey-Box Hybrid Architecture**:
1. **Mechanistic Base**: A 3-compartment mammillary model + effect-site ODE calculates baseline plasma ($C_p$) and effect-site ($C_e$) concentrations.
2. **Mechanistic State Features**: Dynamic ODE state concentrations ($C_p, C_e, C_2, C_3$) and infusion history are fed into the ML learners.
3. **Multiplicative Log-Residual Correction**: ML estimators predict log-residuals $r = \ln(\text{obs}) - \ln(C_{p,\text{minto}})$, ensuring the final prediction $C_p = C_{p,\text{minto}} \cdot e^{\hat{r}}$ strictly respects physiological decay.
4. **Conformal Uncertainty Bounds**: Calibrated 95% split-conformal prediction intervals provide rigorous error bounds during clinical simulation.

---

## 📂 Repository Structure

```text
researcher_distribution/
├── README.md                  ← Quickstart and repository guide
├── ARCHITECTURE.md            ← Mathematical formulation & grey-box design
├── REPRODUCIBILITY.md         ← Step-by-step replication guide from raw CSV
├── CONTRIBUTING.md            ← Guidelines for extending models & submitting PRs
├── requirements.txt           ← Lightweight runtime dependencies (FastAPI, NumPy, Pandas)
├── requirements-training.txt  ← Full training dependencies (scikit-learn, XGBoost, CatBoost)
├── .gitignore                 ← Git exclusion rules
│
├── app.py                     ← FastAPI web server & REST API
├── engine.py                  ← Standalone grey-box simulation & inference engine
├── minto_remifentanil.py      ← Minto (1997) 3-compartment ODE implementation
│
├── pk_core/                   ← Lightweight zero-overhead NumPy learners
│   ├── models.py              ← Model loader & predictor
│   ├── tiny_forest.py         ← NumPy Random Forest / Extra Trees inference
│   └── xgb_np.py              ← NumPy XGBoost inference tree parser
│
├── data/                      ← Clinical datasets & evaluation data
│   ├── Kaggle_PK_Remifentanil.csv ← Volunteer clinical cohort dataset (65 subjects)
│   ├── Ross_remi.txt          ← Ross NONMEM clinical trial dataset
│   ├── cv_results.json        ← Precomputed 5-fold cross-validation metrics
│   └── engine_cache.pkl       ← Precompiled model weights cache (0.5s cold start)
│
├── training/                  ← Machine learning training pipelines
│   ├── pk_dataset.py          ← PK-aware feature extraction & dataset builder
│   ├── train_models.py        ← Production training script for 8 ML models
│   └── unified_pipeline.py    ← 14-model benchmark comparison pipeline
│
├── static/                    ← Clinical Dashboard UI
│   └── index.html             ← Interactive Preact TCI dashboard
│
└── analysis/                  ← Diagnostics, benchmarks & figures
    ├── diagnostics.json       ← Comprehensive diagnostic summary
    ├── series/                ← Model time-series prediction JSONs
    └── figs/                  ← Diagnostic and radar comparison figures
```

---

## ⚡ Quickstart

### 1. Clone & Set Up Environment
```bash
# Clone the repository
git clone https://github.com/<your-username>/PK-ML-Clinical-Dashboard.git
cd PK-ML-Clinical-Dashboard

# Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install runtime dependencies
pip install -r requirements.txt
```

### 2. Launch the Clinical Dashboard
```bash
# Start FastAPI backend
uvicorn app:app --host 0.0.0.0 --port 8000
```
Open your browser and navigate to:
👉 **`http://localhost:8000`**

- **Interactive UI**: `http://localhost:8000`
- **Health Check**: `http://localhost:8000/api/health`
- **Interactive Swagger Docs**: `http://localhost:8000/docs`

---

## 🏋️ Training Models from Scratch

To reproduce all 8 machine learning models and generate the benchmark leaderboard from raw clinical CSV data:

```bash
# Install training dependencies (scikit-learn, XGBoost, CatBoost, etc.)
pip install -r requirements-training.txt

# Run the training pipeline
python training/train_models.py
```

### Benchmark Leaderboard (Held-out Patients)
| Model | MAE (ng/mL) | RMSE (ng/mL) | $R^2$ Forward | Varvel MDAPE (%) |
|---|---|---|---|---|
| **XGBoost (Top Ensemble)** | **4.04** | **7.27** | **0.949** | **16.2%** |
| CatBoost | 4.09 | 7.35 | 0.948 | 16.4% |
| Extra Trees | 4.12 | 7.42 | 0.946 | 16.8% |
| Random Forest | 4.15 | 7.48 | 0.945 | 17.1% |
| Gradient Boosting | 4.18 | 7.55 | 0.943 | 17.5% |
| Support Vector Robust (SVR) | 4.21 | 7.62 | 0.941 | 18.2% |
| Bayesian Compartment | 4.24 | 7.69 | 0.940 | 18.9% |
| *Minto Baseline (Uncorrected)* | *4.26* | *7.74* | *0.941* | *24.5%* |

---

## 📚 Documentation Links
- [ARCHITECTURE.md](ARCHITECTURE.md): Mathematical derivations, state ODEs, and conformal bounds.
- [REPRODUCIBILITY.md](REPRODUCIBILITY.md): Reproduction checklist, cross-validation splits, and cache compilation.
- [CONTRIBUTING.md](CONTRIBUTING.md): Guide for peer researchers to contribute new models and datasets.

---

## ⚖️ Disclaimer
*This repository is an academic research and educational prototype. It is NOT a medical device and is NOT approved for clinical dosing decisions.*
