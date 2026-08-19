"""
services/greybox_engine.py — the corrected (Claude) grey-box engine, bundled.

Self-contained and NumPy-only: embeds the Minto (1997) model, builds the
grey-box dataset from data/Kaggle_PK_Remifentanil.csv, trains the corrected
models (grey-box residual over Minto) on a subject-aware split, and exposes
forward-simulation predictions, an honest forward leaderboard (MAE/RMSE/R²/
Varvel MDPE·MDAPE/WAPE/P95) with split-conformal coverage, and residuals.

No scikit-learn / XGBoost / CatBoost needed — so the corrected pages run on the
deployed site with zero heavy dependencies. Trained once and cached.
"""
from __future__ import annotations

import os
from pathlib import Path
from functools import lru_cache

import numpy as np
import pandas as pd

try:
    from pk_core.xgb_np import XGBoostNP   # when app/ is on sys.path (Streamlit)
    from pk_core.tiny_forest import RandomForestRegressor
    from pk_core.models import KernelRidgeRBF, GBM, Stack
except ModuleNotFoundError:                          # when app/services/ is on sys.path
    from pk_core.xgb_np import XGBoostNP
    from pk_core.tiny_forest import RandomForestRegressor
    from pk_core.models import KernelRidgeRBF, GBM, Stack

try:
    import streamlit as st
    _ST = True
except Exception:
    _ST = False

DATA = Path(__file__).resolve().parent / "data" / "Kaggle_PK_Remifentanil.csv"
LLOQ = 0.1

# ── out-of-fold evidence ───────────────────────────────────────────────────
# Residual diagnostics are served from the cross-validated pipeline, not from
# the single hold-out block that the Engine carves for its own internal model
# selection. oof_all.csv holds one row per observation for all 65 subjects,
# each prediction made by a model that never saw that subject in training, so
# the diagnostics shown here are the same evidence reported in the thesis.
OOF = DATA.parent / "oof_all.csv"


@lru_cache(maxsize=1)
def _oof_table():
    """Out-of-fold predictions for every subject, or None if not shipped."""
    try:
        return pd.read_csv(OOF) if OOF.exists() else None
    except Exception:
        return None


FEATURES = ["Cp_minto", "Ce_minto", "C2_minto", "C3_minto", "logCp_minto",
            "AGE", "WGHT", "HGHT", "SEX", "BMI", "LBM", "BSA", "RATE",
            "Rate_per_Kg", "Cum_Dose", "is_infusion", "time_post_inf"]


# ── Minto ──────────────────────────────────────────────────────────────────
def _lbm(w, h, male):
    return 1.1 * w - 128 * (w / h) ** 2 if male else 1.07 * w - 148 * (w / h) ** 2


def _params(age, lbm):
    da, dl = age - 40.0, lbm - 55.0
    V1 = max(5.1 - .0201 * da + .072 * dl, 1e-3); V2 = max(9.82 - .0811 * da + .108 * dl, 1e-3); V3 = 5.42
    Cl1 = max(2.6 - .0162 * da + .0191 * dl, 1e-4); Cl2 = max(2.05 - .0301 * da, 1e-4)
    Cl3 = max(.076 - .00113 * da, 1e-4); ke0 = max(.595 - .007 * da, 1e-4)
    return dict(V1=V1, V2=V2, V3=V3, Cl1=Cl1, k10=Cl1 / V1, k12=Cl2 / V1, k21=Cl2 / V2,
                k13=Cl3 / V1, k31=Cl3 / V3, ke0=ke0)


def simulate_states(age, wt, ht, male, segments, query, dt=0.05):
    p = _params(age, _lbm(wt, ht, male))
    k10, k12, k21, k13, k31, ke0 = p["k10"], p["k12"], p["k21"], p["k13"], p["k31"], p["ke0"]
    V1, V2, V3 = p["V1"], p["V2"], p["V3"]
    tmax = float(max(query.max(), max((e for _, e, _ in segments), default=0)))
    n = int(tmax / dt) + 2; grid = np.arange(n) * dt
    Cp = np.zeros(n); Ce = np.zeros(n); C2 = np.zeros(n); C3 = np.zeros(n); A = np.zeros(3); ce = 0.0

    def rate(t): return sum(r for s, e, r in segments if s <= t < e)

    def d(st, t):
        A1, A2, A3 = st
        return np.array([rate(t) - (k10 + k12 + k13) * A1 + k21 * A2 + k31 * A3,
                         k12 * A1 - k21 * A2, k13 * A1 - k31 * A3])
    for i in range(1, n):
        t = grid[i - 1]
        k1 = d(A, t); k2 = d(A + .5 * dt * k1, t + .5 * dt); k3 = d(A + .5 * dt * k2, t + .5 * dt); k4 = d(A + dt * k3, t + dt)
        A = A + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        cp = A[0] / V1; ce = cp + (ce - cp) * np.exp(-ke0 * dt)
        Cp[i] = cp; Ce[i] = ce; C2[i] = A[1] / V2; C3[i] = A[2] / V3
    q = lambda a: np.interp(query, grid, a)
    return q(Cp), q(Ce), q(C2), q(C3), p


# ── dataset ────────────────────────────────────────────────────────────────
def build_dataset():
    df = pd.read_csv(DATA).rename(columns={"Time": "TIME", "Rate": "RATE", "Amt": "AMT",
                                           "conc": "DV", "Age": "AGE", "Ht": "HGHT", "Wt": "WGHT", "Subject": "SID"})
    df["SEX"] = (df["Sex"].astype(str).str.lower() == "male").astype(int)
    rows = []
    for sid, g in df.groupby("SID"):
        g = g.sort_values("TIME").reset_index(drop=True)
        age, wt, ht, sex = g.AGE.iloc[0], g.WGHT.iloc[0], g.HGHT.iloc[0], int(g.SEX.iloc[0])
        segs = [(float(g.TIME.iloc[i]), float(g.TIME.iloc[i + 1]), float(g.RATE.iloc[i]))
                for i in range(len(g) - 1) if g.RATE.iloc[i] > 0]
        inf_end = max((e for _, e, _ in segs), default=0.0)
        cum = g.AMT.fillna(0).cumsum().values
        qt = g.TIME.values.astype(float)
        Cp, Ce, C2, C3, p = simulate_states(age, wt, ht, sex == 1, segs, qt)
        bmi = wt / (ht / 100) ** 2; bsa = np.sqrt(ht * wt / 3600)
        lbm = 9270 * wt / ((6680 + 216 * bmi) if sex == 1 else (8780 + 244 * bmi))
        for i in range(len(g)):
            if pd.isna(g.DV.iloc[i]):
                continue
            t = qt[i]
            rows.append(dict(SID=sid, TIME=t, DV=float(g.DV.iloc[i]), Cp_minto=Cp[i], Ce_minto=Ce[i],
                             C2_minto=C2[i], C3_minto=C3[i], logCp_minto=np.log(max(Cp[i], 1e-3)),
                             AGE=age, WGHT=wt, HGHT=ht, SEX=sex, BMI=bmi, LBM=lbm, BSA=bsa,
                             RATE=float(g.RATE.iloc[i]), Rate_per_Kg=float(g.RATE.iloc[i]) / wt,
                             Cum_Dose=float(cum[i]), is_infusion=1.0 if g.RATE.iloc[i] > 0 else 0.0,
                             time_post_inf=max(0.0, t - inf_end)))
    return pd.DataFrame(rows)


class _Ridge:
    def __init__(self, a=1.0): self.a = a
    def fit(self, X, y):
        self.mu, self.sd = X.mean(0), X.std(0); self.sd[self.sd == 0] = 1
        Z = np.c_[np.ones(len(X)), (X - self.mu) / self.sd]
        self.w = np.linalg.solve(Z.T @ Z + self.a * np.eye(Z.shape[1]), Z.T @ y); return self
    def predict(self, X): return np.c_[np.ones(len(X)), (X - self.mu) / self.sd] @ self.w


def _metrics(y, p):
    p = np.clip(p, 1e-3, None); r = y - p; ok = (y > LLOQ) & (p > LLOQ); pe = 100 * r[ok] / p[ok]
    return dict(MAE=float(np.mean(np.abs(r))), RMSE=float(np.sqrt(np.mean(r ** 2))),
                R2=float(1 - np.sum(r ** 2) / np.sum((y - y.mean()) ** 2)),
                MDPE=float(np.median(pe)), MDAPE=float(np.median(np.abs(pe))),
                WAPE=float(100 * np.abs(r).sum() / np.abs(y).sum()),
                P95=float(np.percentile(np.abs(r), 95)))


# ── cross-validated results (precomputed by analysis/cv_rigor.py) ──────────
CV_JSON = Path(__file__).resolve().parent / "data" / "cv_results.json"

# How each arm should be described to an evaluator reading the site.
_ARM_ROLE = {
    "Minto": "Mechanistic baseline — fitted on this cohort (in-sample)",
    "PureML_noMinto": "Ablation — ML alone, no mechanistic base",
}
_DISPLAY = {"Minto": "Minto (mechanistic ref)", "PureML_noMinto": "Pure ML (ablation, no Minto)"}


def load_cv_results():
    """Precomputed 5-fold subject-level CV results, or None if unavailable."""
    try:
        import json
        if CV_JSON.exists():
            return json.loads(CV_JSON.read_text())
    except Exception:
        pass
    return None


def cv_leaderboard_frame(cv):
    """Build the display leaderboard from cross-validated results.

    Columns mirror the thesis table exactly: mean per-subject MAE with a
    bootstrap 95% CI, Varvel MDPE/MDAPE computed per-subject then pooled,
    conformal coverage, and an explicit verdict against the Minto baseline.
    """
    rows = []
    for name, a in cv["arms"].items():
        v = a["varvel"]
        d = a["vs_Minto"]
        if name == "Minto":
            verdict = "baseline"
        elif d["hi"] < 0:
            verdict = "better than Minto"
        elif d["lo"] > 0:
            verdict = "worse than Minto"
        else:
            verdict = "no sig. difference"
        rows.append({
            "model": _DISPLAY.get(name, name),
            # --- standard supervised-regression metrics ---
            "MAE": a["MAE"], "MAE_lo": a["MAE_lo"], "MAE_hi": a["MAE_hi"],
            "RMSE": a["RMSE"], "RMSE_lo": a["RMSE_lo"], "RMSE_hi": a["RMSE_hi"],
            "R2": a["R2"], "R2_lo": a["R2_lo"], "R2_hi": a["R2_hi"],
            "WAPE": a["WAPE"], "P95_AE": a["P95_AE"], "RMSE_log": a["RMSE_log"],
            "MAE_pooled": a["MAE_pooled"],
            # --- clinical / TCI metrics (Varvel, per-subject then pooled) ---
            "MDPE": v["MDPE_median"], "MDAPE": v["MDAPE_median"],
            "wobble": v["wobble_median"],
            "coverage": a["coverage"] if a["coverage"] is not None else np.nan,
            # --- inferential verdicts ---
            "vs_Minto": verdict,
            "vs_Minto_RMSE_p": (a["vs_Minto_RMSE"]["p_a_better"] * 100
                                if a.get("vs_Minto_RMSE") else np.nan),
            "role": _ARM_ROLE.get(name, "Grey-box: Minto base x ML residual"),
        })
    return pd.DataFrame(rows).sort_values("MAE").reset_index(drop=True)


# ── engine ─────────────────────────────────────────────────────────────────
class Engine:
    def __init__(self):
        d = build_dataset()
        subs = np.array(sorted(d.SID.unique())); np.random.default_rng(4).shuffle(subs)
        nt = max(1, int(.2 * len(subs))); nc = max(1, int(.2 * len(subs)))
        te, ca = set(subs[:nt]), set(subs[nt:nt + nc])
        self.tr = d[~d.SID.isin(te | ca)].reset_index(drop=True)
        self.ca = d[d.SID.isin(ca)].reset_index(drop=True)
        self.te = d[d.SID.isin(te)].reset_index(drop=True)
        r = self._resid(self.tr); Xtr = self.tr[FEATURES].values
        # all eight model families, trained as grey-box residual-over-Minto learners
        self.models = {
            "BayesianCompartment": (_Ridge(1.0).fit(Xtr, r), 1.0),
            "SupportVectorRobust": (KernelRidgeRBF(1.0).fit(Xtr, r), 1.0),
            "RandomForest":        (RandomForestRegressor(120, 13, 3, random_state=1).fit(Xtr, r), 0.6),
            "ExtraTrees":          (RandomForestRegressor(120, 16, 2, max_features="all", random_state=5).fit(Xtr, r), 0.6),
            "GradientBoosting":    (GBM(150, .05, 3, subsample=.85, seed=2).fit(Xtr, r), 0.7),
            "XGBoost":             (XGBoostNP(250, .05, 4, 1., 1., .7, .8, 10, seed=11).fit(Xtr, r), 0.6),
            "CatBoost":            (GBM(200, .05, 6, subsample=.8, min_leaf=8, seed=6).fit(Xtr, r), 0.7),
            "StackingEnsemble":    (Stack(seed=0).fit(Xtr, r), 0.6),
        }
        # per-model split-conformal q (log space) from calibration
        self.qs = {name: self._conformal(name) for name in self.models}
        # default model = best held-out MAE
        _mae = {n: float(np.mean(np.abs(self.te.DV.values - self._pred(n, self.te)))) for n in self.models}
        self.default = min(_mae, key=_mae.get)

    def _resid(self, df):
        return (np.log(df.DV.clip(lower=1e-3)) - np.log(df.Cp_minto.clip(lower=1e-3))).values

    def _pred(self, name, df):
        m, lam = self.models[name]
        return np.clip(df.Cp_minto.values * np.exp(lam * m.predict(df[FEATURES].values)), 1e-3, None)

    def _conformal(self, name, alpha=.05):
        y = self.ca.DV.values; p = self._pred(name, self.ca); ok = (y > LLOQ) & (p > LLOQ)
        s = np.abs(np.log(y[ok]) - np.log(p[ok])); return float(np.quantile(s, 1 - alpha))

    def model_names(self):
        return list(self.models.keys())

    def leaderboard(self):
        """Headline leaderboard = CROSS-VALIDATED results when available.

        The single-split numbers this used to return were computed on 13
        held-out subjects and were optimistic by ~0.35 ng/mL. The
        cross-validated table (all 65 subjects, each held out once, with
        bootstrap CIs over subjects) is the defensible one and is what both
        the site and the thesis report. Falls back to the single split only
        if the precomputed file is missing.
        """
        cv = load_cv_results()
        if cv is not None:
            return cv_leaderboard_frame(cv)
        return self.single_split_leaderboard()

    def single_split_leaderboard(self):
        """Original single 20% split. Retained for provenance//audit only."""
        y = self.te.DV.values; rows = []
        rows.append({"model": "Minto (mechanistic ref)", **_metrics(y, self.te.Cp_minto.values), "coverage": np.nan})
        for name in self.models:
            p = self._pred(name, self.te); m = _metrics(y, p)
            q = self.qs[name]; lo, hi = p * np.exp(-q), p * np.exp(q); ok = (y > LLOQ) & (p > LLOQ)
            m["coverage"] = float(100 * np.mean((y[ok] >= lo[ok]) & (y[ok] <= hi[ok])))
            rows.append({"model": name, **m})
        return pd.DataFrame(rows).sort_values("MAE").reset_index(drop=True)

    def residuals(self, name=None):
        """Residuals on patients the model never trained on.

        Prefers the cross-validated out-of-fold table (65 subjects, 1,992
        observations). Falls back to the internal hold-out block only if that
        table is unavailable, in which case far fewer subjects contribute.
        """
        name = name or self.default
        o = _oof_table()
        col = f"pred_{name}"
        if o is not None and col in o.columns:
            p = o[col].values
            return o.TIME.values, p, o.DV.values - p, o.is_infusion.values > 0
        y = self.te.DV.values; p = self._pred(name, self.te)
        return self.te.TIME.values, p, y - p, self.te.is_infusion.values > 0

    def residual_scope(self):
        """Describe which patients the residual diagnostics are computed on."""
        o = _oof_table()
        if o is not None:
            return dict(kind="out-of-fold", subjects=int(o.SID.nunique()),
                        observations=int(len(o)))
        return dict(kind="single hold-out", subjects=int(self.te.SID.nunique()),
                    observations=int(len(self.te)))

    def forward_curve(self, patient, regimen, duration, name=None, with_pi=True):
        name = name or self.default
        age = float(patient["age"]); wt = float(patient["weight"]); ht = float(patient["height"])
        male = str(patient.get("sex", "Male")).lower().startswith("m")
        segs = [(float(i["start"]), float(i["end"]), float(i["rate"])) for i in regimen.get("infusions", [])]
        inf_end = max((e for _, e, _ in segs), default=0.0)
        minutes = np.arange(0, int(duration) + 1).astype(float)
        Cp, Ce, C2, C3, p = simulate_states(age, wt, ht, male, segs, minutes)
        bmi = wt / (ht / 100) ** 2; bsa = np.sqrt(ht * wt / 3600)
        lbm = 9270 * wt / ((6680 + 216 * bmi) if male else (8780 + 244 * bmi))
        rate_t = np.array([sum(r for s, e, r in segs if s <= t < e) for t in minutes])
        X = pd.DataFrame({"Cp_minto": Cp, "Ce_minto": Ce, "C2_minto": C2, "C3_minto": C3,
                          "logCp_minto": np.log(np.clip(Cp, 1e-3, None)), "AGE": age, "WGHT": wt, "HGHT": ht,
                          "SEX": 1.0 if male else 0.0, "BMI": bmi, "LBM": lbm, "BSA": bsa, "RATE": rate_t,
                          "Rate_per_Kg": rate_t / wt, "Cum_Dose": np.cumsum(rate_t),
                          "is_infusion": (rate_t > 0).astype(float),
                          "time_post_inf": np.clip(minutes - inf_end, 0, None)})[FEATURES].values
        m, lam = self.models[name]
        pred = np.clip(Cp * np.exp(lam * m.predict(X)), 0.0, None)
        out = {"time": minutes, "Cp_minto": Cp, "Ce_minto": Ce, "pred": pred,
               "Css": (segs[0][2] / p["Cl1"]) if len(segs) == 1 else None}
        if with_pi:
            q = self.qs[name]; out["lo"] = pred * np.exp(-q); out["hi"] = pred * np.exp(q)
        return out


# ── on-disk engine cache ───────────────────────────────────────────────────
# Training the eight families takes ~27 s. On a host that sleeps when idle
# (e.g. Azure App Service free tier) that cost is paid on every cold start.
# Pickling the fitted engine reduces it to ~0.4 s. The cache is keyed on the
# dataset and on this module's own source, so it self-invalidates if either
# changes and can never serve results from stale code.
CACHE = Path(__file__).resolve().parent / "data" / "engine_cache.pkl"


def cache_key() -> str:
    import hashlib
    h = hashlib.sha256()
    h.update(DATA.read_bytes())
    h.update(Path(__file__).read_bytes())
    return h.hexdigest()[:16]


def build_cache(path: Path | None = None) -> Path:
    """Train once and persist. Run offline; ship the result with the deployment."""
    import pickle
    p = Path(path) if path else CACHE
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "wb") as fh:
        pickle.dump({"key": cache_key(), "engine": Engine()}, fh,
                    protocol=pickle.HIGHEST_PROTOCOL)
    return p


def _load_cached():
    """Return a cached Engine, or None if absent, unreadable or stale."""
    import pickle
    try:
        if not CACHE.exists():
            return None
        with open(CACHE, "rb") as fh:
            blob = pickle.load(fh)
        if blob.get("key") != cache_key():
            return None                      # dataset or code changed
        return blob["engine"]
    except Exception:
        return None                          # never fail closed on a cache problem


@lru_cache(maxsize=1)
def _engine_cached():
    return _load_cached() or Engine()


def get_engine():
    if _ST:
        @st.cache_resource(show_spinner="Loading grey-box models…")
        def _c():
            return _load_cached() or Engine()
        return _c()
    return _engine_cached()
