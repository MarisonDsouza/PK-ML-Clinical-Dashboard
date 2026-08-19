"""
app.py — Azure App Service entry point.

Serves the dashboard at / and the JSON API under /api. All estimation is done
offline: the fitted engine is loaded from data/engine_cache.pkl, so a cold
start is under a second rather than the ~27 s a retrain would cost.

Data-protection posture: inputs are de-identified physiological covariates
only (age, weight, height, sex) plus a dosing regimen. No protected health
information is accepted, transmitted or stored, and nothing is persisted.
This is a research and educational prototype, not a medical device.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import List, Optional

import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from starlette.requests import Request
from pydantic import BaseModel

import engine as gbe

HERE = Path(__file__).resolve().parent
STATIC = HERE / "static"
ANALYSIS = HERE / "analysis"

app = FastAPI(title="Remifentanil Grey-box TCI Dashboard", version="1.0")

_origins = os.environ.get("ALLOWED_ORIGINS", "*").split(",")
app.add_middleware(CORSMiddleware, allow_origins=_origins,
                   allow_methods=["GET", "POST"], allow_headers=["Content-Type"])


@app.middleware("http")
async def security_headers(request: Request, call_next):
    r = await call_next(request)
    r.headers["X-Content-Type-Options"] = "nosniff"
    r.headers["X-Frame-Options"] = "DENY"
    r.headers["Referrer-Policy"] = "no-referrer"
    r.headers["Cache-Control"] = "no-store"
    return r


def _j(v):
    if isinstance(v, np.ndarray):
        return [None if (isinstance(x, float) and (np.isnan(x) or np.isinf(x)))
                else float(x) for x in v]
    if isinstance(v, (np.floating, np.integer)):
        return float(v)
    return v


class Patient(BaseModel):
    age: float; weight: float; height: float; sex: str


class Infusion(BaseModel):
    start: float; end: float; rate: float


class Bolus(BaseModel):
    time: float; amt: float


class Regimen(BaseModel):
    infusions: List[Infusion] = []
    boluses: List[Bolus] = []


class TrajectoryRequest(BaseModel):
    patient: Patient
    regimen: Regimen
    duration: int = 90
    model: Optional[str] = None


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.get("/api/models")
def models():
    e = gbe.get_engine()
    return {"models": e.model_names(), "default": e.default}


@app.get("/api/disclaimer")
def disclaimer():
    return {
        "intended_use": "Research and educational prototype for remifentanil "
                        "pharmacokinetics. NOT a medical device; not for clinical dosing.",
        "phi": "No protected health information is collected, transmitted or stored. "
               "Inputs are de-identified physiological covariates only.",
        "model": "Grey-box: Minto (1997) three-compartment + effect-site mechanistic "
                 "base x ML residual, with 95% split-conformal prediction intervals.",
        "review": "Unpublished - requires peer review (Dr. Prathvi Shenoy).",
    }


@app.get("/api/leaderboard")
def leaderboard():
    lb = gbe.get_engine().leaderboard()
    recs = lb.to_dict(orient="records")
    for r in recs:
        for k, v in r.items():
            r[k] = None if (isinstance(v, float) and v != v) else (
                float(v) if isinstance(v, (int, float)) else v)
    return recs


@app.get("/api/validation")
def validation(model: Optional[str] = None):
    t, p, res, inf = gbe.get_engine().residuals(model)
    return {"time": _j(t), "predicted": _j(p), "residual": _j(res),
            "is_infusion": [bool(x) for x in inf],
            "bias": float(np.mean(res)),
            "bias_infusion": float(np.mean(res[inf])) if inf.any() else None,
            "bias_washout": float(np.mean(res[~inf])) if (~inf).any() else None,
            "scope": gbe.get_engine().residual_scope()}


@app.post("/api/trajectory")
def trajectory(req: TrajectoryRequest):
    g = gbe.get_engine().forward_curve(
        req.patient.model_dump(),
        {"infusions": [i.model_dump() for i in req.regimen.infusions],
         "boluses": [b.model_dump() for b in req.regimen.boluses]},
        int(req.duration), name=req.model)
    return {k: _j(v) for k, v in g.items()}


@app.get("/api/diagnostics")
def diagnostics(model: Optional[str] = None):
    f = ANALYSIS / "diagnostics.json"
    if not f.exists():
        return JSONResponse({"error": "diagnostics unavailable"}, status_code=404)
    d = json.loads(f.read_text())
    arms = {k: v for k, v in d.items() if not k.startswith("_")}
    if model:
        if model not in arms:
            return JSONResponse({"error": f"unknown model {model}"}, status_code=404)
        a = arms[model]
        return {"model": model, "toast": a.get("toast"), "log": a["log"],
                "natural": a["natural"], "scale_comparison": a.get("scale_comparison"),
                "figures": a.get("figures")}
    return {"models": list(arms),
            "toasts": {k: v.get("toast") for k, v in arms.items()},
            "interpretations": {k: v.get("interpretation") for k, v in arms.items()},
            "radar": d.get("_radar"),
            "scale": {k: v.get("scale_comparison") for k, v in arms.items()},
            "figures": {k: v.get("figures") for k, v in arms.items()}}


@app.get("/api/diagnostics/series")
def diagnostic_series(model: str):
    p = (ANALYSIS / "series" / f"{model}.json").resolve()
    root = (ANALYSIS / "series").resolve()
    if not (p.is_file() and str(p).startswith(str(root))):
        return JSONResponse({"error": f"series not found for {model}"}, status_code=404)
    return json.loads(p.read_text())


@app.get("/api/figure")
def figure(model: str, kind: str = "diagnostics"):
    name = "fig_diagnostics_" if kind == "diagnostics" else "fig_scale_"
    p = (ANALYSIS / "figs" / f"{name}{model}.png").resolve()
    root = (ANALYSIS / "figs").resolve()
    if not (p.is_file() and str(p).startswith(str(root))):
        return JSONResponse({"error": "figure not found"}, status_code=404)
    return FileResponse(str(p), media_type="image/png")


@app.get("/")
def index():
    return FileResponse(str(STATIC / "index.html"), media_type="text/html")
