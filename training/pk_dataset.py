#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pk_dataset.py — PK-aware dataset builder for Remifentanil.

Reconstructs dosing history and mechanistic state features (Cp, Ce, C2, C3)
from the Minto 3-compartment ODE so that ML learners predict log-residuals:
    r = log(DV) - log(Cp_minto)  -->  pred = Cp_minto * exp(r_hat)
"""
from __future__ import annotations
import sys, os
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
from minto_remifentanil import minto_parameters, lbm_james  # noqa: E402

DATA_PATH = ROOT / "data" / "Kaggle_PK_Remifentanil.csv"
LLOQ = 0.1

GREYBOX_FEATURES = [
    "Cp_minto", "Ce_minto", "C2_minto", "C3_minto", "logCp_minto",
    "AGE", "WGHT", "HGHT", "SEX", "BMI", "LBM", "BSA",
    "RATE", "Rate_per_Kg", "Cum_Dose", "is_infusion", "time_post_inf"
]


def _simulate_states(age, wt, ht, sex, segments, query_times, dt=0.05):
    """Integrate Minto ODE; return Cp, Ce, C2, C3 (ng/mL) at query_times."""
    lbm = lbm_james(wt, ht, sex)
    p = minto_parameters(age, lbm)
    k10, k12, k21, k13, k31, ke0 = (p["k10"], p["k12"], p["k21"],
                                    p["k13"], p["k31"], p["ke0"])
    V1, V2, V3 = p["V1"], p["V2"], p["V3"]
    tmax = float(max(query_times.max(), max((e for _, e, _ in segments), default=0)))
    n = int(tmax / dt) + 2
    grid = np.arange(n) * dt
    Cp = np.zeros(n); Ce = np.zeros(n); C2 = np.zeros(n); C3 = np.zeros(n)
    A = np.zeros(3); ce = 0.0

    def rate_at(t):
        r = 0.0
        for s, e, rr in segments:
            if s <= t < e:
                r += rr
        return r

    def deriv(Ast, t):
        A1, A2, A3 = Ast
        return np.array([rate_at(t) - (k10 + k12 + k13) * A1 + k21 * A2 + k31 * A3,
                         k12 * A1 - k21 * A2,
                         k13 * A1 - k31 * A3])

    for i in range(1, n):
        t = grid[i - 1]
        A = np.maximum(0.0, A + dt * deriv(A, t))
        cp = A[0] / V1
        ce = cp + (ce - cp) * np.exp(-ke0 * dt)
        Cp[i] = cp
        Ce[i] = ce
        C2[i] = A[1] / V2
        C3[i] = A[2] / V3

    q = lambda arr: np.interp(query_times, grid, arr)
    return q(Cp), q(Ce), q(C2), q(C3), p, lbm


def build(csv_path: str | Path | None = None) -> pd.DataFrame:
    """Parse raw dataset, compute mechanistic features, and construct log-residuals."""
    path = Path(csv_path) if csv_path else DATA_PATH
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at {path}")
        
    df = pd.read_csv(path)
    df = df.rename(columns={"Time": "TIME", "Rate": "RATE", "Amt": "AMT",
                            "conc": "DV", "Age": "AGE", "Ht": "HGHT",
                            "Wt": "WGHT", "Subject": "SID"})
    df["SEX"] = df["Sex"].map({"Male": 1, "Female": 0}).fillna(1).astype(int)

    rows = []
    for sid, g in df.groupby("SID"):
        g = g.sort_values("TIME").reset_index(drop=True)
        age = float(g["AGE"].iloc[0])
        wt = float(g["WGHT"].iloc[0])
        ht = float(g["HGHT"].iloc[0])
        sex = int(g["SEX"].iloc[0])
        
        segments = []
        for i in range(len(g) - 1):
            r = float(g["RATE"].iloc[i])
            if r > 0:
                segments.append((float(g["TIME"].iloc[i]), float(g["TIME"].iloc[i+1]), r))
                
        cum = g["AMT"].fillna(0).cumsum().values
        inf_end = max((e for _, e, _ in segments), default=0.0)
        qt = g["TIME"].values.astype(float)
        Cp, Ce, C2, C3, p, lbm = _simulate_states(age, wt, ht, sex, segments, qt)

        bmi = wt / (ht/100.)**2
        bsa = np.sqrt(ht*wt/3600.)
        for i in range(len(g)):
            dv = g["DV"].iloc[i]
            if pd.isna(dv):
                continue
            t = qt[i]
            rows.append({
                "SID": sid, "TIME": t, "DV": float(dv),
                "Cp_minto": Cp[i], "Ce_minto": Ce[i],
                "C2_minto": C2[i], "C3_minto": C3[i], "logCp_minto": np.log(max(Cp[i], 1e-3)),
                "AGE": age, "WGHT": wt, "HGHT": ht, "SEX": sex,
                "BMI": bmi, "LBM": lbm, "BSA": bsa,
                "RATE": float(g["RATE"].iloc[i]), "Rate_per_Kg": float(g["RATE"].iloc[i])/wt,
                "Cum_Dose": float(cum[i]), "is_infusion": 1.0 if g["RATE"].iloc[i] > 0 else 0.0,
                "time_post_inf": max(0.0, t - inf_end), "logTIME": np.log1p(t),
            })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    d = build()
    print("Dataset built successfully!")
    print(f"Shape: {d.shape} | Subjects: {d.SID.nunique()} | DV range: {d.DV.min():.2f} -> {d.DV.max():.2f} ng/mL")
    err = d.DV - d.Cp_minto
    r2 = 1 - np.sum(err**2) / np.sum((d.DV - d.DV.mean())**2)
    mae = np.mean(np.abs(err))
    print(f"Raw Minto Baseline vs Observed: R2 = {r2:.3f} | MAE = {mae:.2f} ng/mL")
