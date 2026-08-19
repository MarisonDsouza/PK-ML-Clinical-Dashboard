#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_models.py — Train 8 Machine Learning Residual Models for PK Simulation.

Estimators trained:
  - BayesianCompartment (BayesianRidge)
  - SupportVectorRobust (SVR RBF kernel)
  - RandomForest (300 trees)
  - ExtraTrees (300 trees)
  - GradientBoosting (400 estimators)
  - XGBoost (Histogram gradient boosting)
  - CatBoost (Gradient boosting on decision trees)
  - StackingEnsemble (Stacking meta-regressor)
"""
from __future__ import annotations
import os, sys, json, argparse
from pathlib import Path
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import BayesianRidge, Ridge
from sklearn.svm import SVR
from sklearn.ensemble import (RandomForestRegressor, ExtraTreesRegressor,
                              GradientBoostingRegressor, StackingRegressor)
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
from pk_dataset import build, GREYBOX_FEATURES  # noqa: E402

MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
LLOQ = 0.1


def get_estimators():
    est = {
        "BayesianCompartment": BayesianRidge(),
        "SupportVectorRobust": SVR(kernel="rbf", C=10.0, gamma="scale", epsilon=0.1),
        "RandomForest": RandomForestRegressor(n_estimators=300, max_depth=14,
                                              min_samples_leaf=3, n_jobs=-1, random_state=42),
        "ExtraTrees": ExtraTreesRegressor(n_estimators=300, max_depth=16,
                                          min_samples_leaf=2, n_jobs=-1, random_state=42),
        "GradientBoosting": GradientBoostingRegressor(n_estimators=400, learning_rate=0.03,
                                                      max_depth=3, subsample=0.85, random_state=42),
    }
    try:
        import xgboost as xgb
        est["XGBoost"] = xgb.XGBRegressor(n_estimators=250, learning_rate=0.05, max_depth=4,
                                          reg_lambda=1.0, gamma=1.0, subsample=0.7, colsample_bytree=0.8,
                                          min_child_weight=10, tree_method="hist", random_state=42, verbosity=0)
    except BaseException as e:
        print(f"[NOTE] XGBoost library not loaded ({e}).")
        print("       (On macOS: `brew install libomp` to enable native XGBoost C++ acceleration)")

    try:
        import catboost as cb
        est["CatBoost"] = cb.CatBoostRegressor(iterations=250, learning_rate=0.05, depth=6,
                                               l2_leaf_reg=9, subsample=0.8, random_seed=42, verbose=0)
    except BaseException:
        pass

    # Stacking ensemble
    base = [("rf", est["RandomForest"]), ("et", est["ExtraTrees"]), ("gbm", est["GradientBoosting"])]
    if "XGBoost" in est:
        base.append(("xgb", est["XGBoost"]))
    est["StackingEnsemble"] = StackingRegressor(estimators=base, final_estimator=Ridge(alpha=1.0), cv=5, n_jobs=-1)
    return est


def varvel_metrics(y_true, y_pred):
    ok = (y_true > LLOQ) & (y_pred > LLOQ)
    pe = 100.0 * (y_true[ok] - y_pred[ok]) / y_pred[ok]
    return float(np.median(pe)), float(np.median(np.abs(pe)))


def main(data_path: str | None = None):
    print("Loading and preparing PK dataset...")
    df = build(data_path)
    print(f"Loaded {len(df)} records across {df.SID.nunique()} patients.")

    X = df[GREYBOX_FEATURES].values
    r = (np.log(df.DV.clip(lower=1e-3)) - np.log(df.Cp_minto.clip(lower=1e-3))).values
    groups = df.SID.values

    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=4)
    tr_idx, te_idx = next(gss.split(X, r, groups))

    scaler = StandardScaler().fit(X[tr_idx])
    Xtr, Xte = scaler.transform(X[tr_idx]), scaler.transform(X[te_idx])
    rtr = r[tr_idx]
    cp_te = df.Cp_minto.values[te_idx]
    y_te = df.DV.values[te_idx]

    joblib.dump(scaler, MODEL_DIR / "scaler.joblib")
    joblib.dump(GREYBOX_FEATURES, MODEL_DIR / "feature_names.joblib")

    est = get_estimators()
    rows = []
    print("\\n================ Training ML Residual Models ================")
    for name, model in est.items():
        model.fit(Xtr, rtr)
        p = np.clip(cp_te * np.exp(model.predict(Xte)), 1e-3, None)
        mae = mean_absolute_error(y_te, p)
        rmse = float(np.sqrt(mean_squared_error(y_te, p)))
        r2 = r2_score(y_te, p)
        mdpe, mdape = varvel_metrics(y_te, p)

        joblib.dump(model, MODEL_DIR / f"{name}.joblib")
        rows.append({
            "model": name,
            "MAE": round(mae, 3),
            "RMSE": round(rmse, 3),
            "R2": round(r2, 4),
            "MDPE_%": round(mdpe, 2),
            "MDAPE_%": round(mdape, 2)
        })
        print(f"  {name:22s} | MAE: {mae:5.2f} ng/mL | RMSE: {rmse:5.2f} | R2: {r2:.3f} | MDAPE: {mdape:4.1f}%")

    res_df = pd.DataFrame(rows).sort_values("MAE")
    print("\\n================ Final Leaderboard Summary ================")
    print(res_df.to_string(index=False))
    res_df.to_csv(MODEL_DIR / "training_leaderboard.csv", index=False)
    print(f"\\nAll models and metrics saved to: {MODEL_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=None, help="Optional custom CSV path")
    args = parser.parse_args()
    main(args.data)
