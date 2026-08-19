# Contributing to PK-ML Research

We welcome contributions from pharmacometricians, machine learning researchers, and clinical anesthesiologists.

---

## 🔬 Research Extension Ideas

1. **New Drug PK Models**:
   - Propofol (Eleveld, Schnider, Marsh models)
   - Dexmedetomidine (Hannivoort model)
   - Fentanyl / Sufentanil

2. **Alternative ML Architectures**:
   - Physics-Informed Neural Networks (PINNs)
   - Neural Ordinary Differential Equations (Neural ODEs)
   - Conformalized Quantile Regression (CQR)

3. **Multi-Study Validation**:
   - Benchmarking across diverse demographic cohorts (pediatric, elderly, obese populations).

---

## 🛠️ Contribution Workflow

1. **Fork the Repository** and create a feature branch (`git checkout -b feature/new-estimator`).
2. **Implement your changes**:
   - Add new ML models into `training/train_models.py`.
   - Add mechanistic ODE equations into `minto_remifentanil.py` or new drug module.
3. **Verify locally**:
   - Ensure `python training/train_models.py` runs without errors.
   - Check that `python training/pk_dataset.py` builds cleanly.
4. **Submit a Pull Request** with a detailed explanation of your methodology and validation metrics.
