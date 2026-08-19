#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Minto (1997) remifentanil pharmacokinetic / pharmacodynamic model.

Reference
---------
Minto CF, Schnider TW, Egan TD, et al. "Influence of age and gender on the
pharmacokinetics and pharmacodynamics of remifentanil. I. Model development."
Anesthesiology 1997; 86:10-23.

This is a 3-compartment mammillary model with a first-order effect-site
(ke0) link. Population parameters are covariate functions of AGE (years) and
lean body mass (LBM, kg). LBM here uses the James equation, as in the original
Minto/Schnider work (Janmahasatian is offered as an option for consistency
with the ML pipeline's feature engineering).

Units (kept explicit — a unit slip is one of the failure modes this baseline
is meant to expose):
    dose / infusion rate : micrograms  (ug) and ug/min
    volumes              : litres (L)
    clearances           : L/min
    amounts A1..A3       : ug
    concentration        : ug/L  ==  ng/mL   (typical remifentanil scale)

The public entry point `simulate()` takes the SAME patient + regimen shape the
FastAPI app already uses, so it can be dropped in as a side-by-side baseline.
"""

from __future__ import annotations
import numpy as np


# ----------------------------------------------------------------------------- 
# Body composition
# ----------------------------------------------------------------------------- 
def lbm_james(weight_kg: float, height_cm: float, sex: str) -> float:
    """James (1976) lean body mass — the equation used by Minto/Schnider."""
    male = str(sex).lower() in ("male", "m", "1")
    r = weight_kg / height_cm
    if male:
        return 1.1 * weight_kg - 128.0 * (weight_kg / height_cm) ** 2
    return 1.07 * weight_kg - 148.0 * (weight_kg / height_cm) ** 2


def lbm_janmahasatian(weight_kg: float, height_cm: float, sex: str) -> float:
    """Janmahasatian (2005) fat-free mass — used by the project's ML features."""
    male = str(sex).lower() in ("male", "m", "1")
    bmi = weight_kg / (height_cm / 100.0) ** 2
    if male:
        return 9270.0 * weight_kg / (6680.0 + 216.0 * bmi)
    return 9270.0 * weight_kg / (8780.0 + 244.0 * bmi)


# ----------------------------------------------------------------------------- 
# Minto population parameters
# ----------------------------------------------------------------------------- 
def minto_parameters(age: float, lbm: float) -> dict:
    """Return {V1,V2,V3,Cl1,Cl2,Cl3,ke0} and micro-rate constants."""
    da = age - 40.0
    dl = lbm - 55.0

    V1 = 5.1 - 0.0201 * da + 0.072 * dl
    V2 = 9.82 - 0.0811 * da + 0.108 * dl
    V3 = 5.42
    Cl1 = 2.6 - 0.0162 * da + 0.0191 * dl     # elimination clearance
    Cl2 = 2.05 - 0.0301 * da                  # rapid distribution
    Cl3 = 0.076 - 0.00113 * da                # slow distribution
    ke0 = 0.595 - 0.007 * da                  # effect-site rate constant

    # guard against non-physiological negatives at covariate extremes
    V1, V2, V3 = max(V1, 1e-3), max(V2, 1e-3), max(V3, 1e-3)
    Cl1, Cl2, Cl3 = max(Cl1, 1e-4), max(Cl2, 1e-4), max(Cl3, 1e-4)
    ke0 = max(ke0, 1e-4)

    return {
        "V1": V1, "V2": V2, "V3": V3,
        "Cl1": Cl1, "Cl2": Cl2, "Cl3": Cl3, "ke0": ke0,
        "k10": Cl1 / V1, "k12": Cl2 / V1, "k21": Cl2 / V2,
        "k13": Cl3 / V1, "k31": Cl3 / V3,
        "Css_per_ugmin": 1.0 / Cl1,   # steady-state plasma conc per 1 ug/min infusion
    }


# ----------------------------------------------------------------------------- 
# Simulation
# ----------------------------------------------------------------------------- 
def simulate(patient: dict, regimen: dict, duration_min: float = 90.0,
             dt: float = 0.05, lbm_formula: str = "james") -> dict:
    """
    Integrate the 3-compartment + effect-site model (RK4).

    patient : {"age","weight","height","sex"}
    regimen : {"infusions":[{"start","end","rate"}], "boluses":[{"time","amt"}]}
              rate in ug/min, amt in ug, times in minutes.
    Returns time (min), Cp (ng/mL), Ce (ng/mL), parameters, and summary.
    """
    age = float(patient["age"])
    wt = float(patient["weight"])
    ht = float(patient["height"])
    sex = patient.get("sex", "male")

    lbm = (lbm_james(wt, ht, sex) if lbm_formula == "james"
           else lbm_janmahasatian(wt, ht, sex))
    p = minto_parameters(age, lbm)
    k10, k12, k21, k13, k31, ke0 = (p["k10"], p["k12"], p["k21"],
                                    p["k13"], p["k31"], p["ke0"])
    V1 = p["V1"]

    infusions = regimen.get("infusions", [])
    boluses = regimen.get("boluses", [])

    def rate_at(t):  # ug/min entering the central compartment
        r = 0.0
        for inf in infusions:
            if inf["start"] <= t < inf["end"]:
                r += float(inf["rate"])
        return r

    n = int(round(duration_min / dt)) + 1
    times = np.linspace(0.0, duration_min, n)
    A = np.zeros(3)          # [A1, A2, A3] in ug
    Ce = 0.0                 # effect-site conc (ng/mL)
    Cp_arr = np.zeros(n)
    Ce_arr = np.zeros(n)

    # apply a bolus scheduled at t=0 before the first sample
    def apply_boluses(t0, t1):
        for b in boluses:
            if t0 < float(b["time"]) <= t1 or (t0 == 0.0 and float(b["time"]) == 0.0):
                A[0] += float(b["amt"])

    apply_boluses(-dt, 0.0)
    Cp_arr[0] = A[0] / V1
    Ce_arr[0] = Ce

    def deriv(A_state, t):
        A1, A2, A3 = A_state
        dA1 = rate_at(t) - (k10 + k12 + k13) * A1 + k21 * A2 + k31 * A3
        dA2 = k12 * A1 - k21 * A2
        dA3 = k13 * A1 - k31 * A3
        return np.array([dA1, dA2, dA3])

    for i in range(1, n):
        t = times[i - 1]
        k1 = deriv(A, t)
        k2 = deriv(A + 0.5 * dt * k1, t + 0.5 * dt)
        k3 = deriv(A + 0.5 * dt * k2, t + 0.5 * dt)
        k4 = deriv(A + dt * k3, t + dt)
        A = A + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

        # discrete boluses landing in (t, t+dt]
        apply_boluses(t, t + dt)

        Cp = A[0] / V1
        # effect-site (RK4 on a scalar linear ODE is overkill; use exact step)
        Ce = Cp + (Ce - Cp) * np.exp(-ke0 * dt)
        Cp_arr[i] = Cp
        Ce_arr[i] = Ce

    # summary
    cmax = float(np.max(Cp_arr))
    tmax = float(times[int(np.argmax(Cp_arr))])
    auc = float(np.trapz(Cp_arr, times))
    # predicted plateau for a single constant infusion
    css = None
    if len(infusions) == 1:
        css = float(infusions[0]["rate"]) * p["Css_per_ugmin"]

    return {
        "time": times, "Cp": Cp_arr, "Ce": Ce_arr,
        "lbm": lbm, "parameters": p,
        "summary": {"c_max": cmax, "t_max": tmax, "auc": auc, "css_plasma": css},
    }


if __name__ == "__main__":
    # Standard reference patient: 40 y, 70 kg, 170 cm, male
    patient = {"age": 40, "weight": 70, "height": 170, "sex": "male"}
    # 0.25 ug/kg/min = 17.5 ug/min for 20 min, then off
    rate = 0.25 * patient["weight"]
    regimen = {"infusions": [{"start": 0, "end": 20, "rate": rate}], "boluses": []}
    out = simulate(patient, regimen, duration_min=90)
    p = out["parameters"]
    s = out["summary"]
    print(f"LBM (James)      : {out['lbm']:.1f} kg")
    print(f"V1={p['V1']:.2f} L  Cl1={p['Cl1']:.2f} L/min  ke0={p['ke0']:.3f} /min")
    print(f"Infusion         : {rate:.1f} ug/min (0.25 ug/kg/min) x 20 min")
    print(f"Predicted Css    : {s['css_plasma']:.2f} ng/mL  (= rate / Cl1)")
    print(f"C_max            : {s['c_max']:.2f} ng/mL at t={s['t_max']:.1f} min")
