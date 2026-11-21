"""
SafeMedTwin — FastAPI Backend
================================
REST API tying together disease prediction (ML), explainability (SHAP),
the MedGuard safety layer, the Digital Twin simulator, and the LLM-vs-SLM
comparison framework.

Run:
    uvicorn backend.main:app --reload --port 8000
Docs:
    http://localhost:8000/docs
"""
import os
import sys
from typing import Dict, List, Optional

import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from models.explainability import explain_prediction, explanation_to_sentence
from safety.medguard import validate_prediction, risk_band
from digital_twin.simulator import PatientTwin, simulate_scenario, multi_scenario_sweep
from llm_slm.compare import compare as compare_llm_slm

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
DISEASES = ["diabetes", "cvd", "ckd", "liver", "hypertension"]

app = FastAPI(
    title="SafeMedTwin API",
    description="Hallucination-aware medical AI: multi-disease risk prediction, "
                 "explainability, digital twin simulation, and safety validation.",
    version="1.0.0",
)

_MODEL_CACHE: Dict[str, dict] = {}
_TWINS: Dict[str, PatientTwin] = {}  # in-memory patient twin store, keyed by patient_id


def get_bundle(disease: str) -> dict:
    if disease not in DISEASES:
        raise HTTPException(404, f"Unknown disease '{disease}'. Choose from {DISEASES}")
    if disease not in _MODEL_CACHE:
        path = os.path.join(MODEL_DIR, f"{disease}_model.joblib")
        if not os.path.exists(path):
            raise HTTPException(500, f"Model for '{disease}' not trained yet. Run models/train_models.py")
        _MODEL_CACHE[disease] = joblib.load(path)
        # Defensive: see matching comment in streamlit_app.py's load_bundle.
        _MODEL_CACHE[disease]["model"].get_booster().feature_names = _MODEL_CACHE[disease]["feature_cols"]
    return _MODEL_CACHE[disease]


# ---------------------------------------------------------------- schemas --
class PredictRequest(BaseModel):
    disease: str = Field(..., description="One of: diabetes, cvd, ckd, liver, hypertension")
    patient_id: str = Field(..., description="Identifier used to track this patient's Digital Twin")
    features: Dict[str, float]


class ScenarioRequest(BaseModel):
    disease: str
    baseline: Dict[str, float]
    changes: Dict[str, float]


class SweepRequest(BaseModel):
    disease: str
    baseline: Dict[str, float]
    feature: str
    values: List[float]


class RecommendationRequest(BaseModel):
    disease: str
    risk_band: str
    top_factor: str
    contributions_summary: Optional[str] = ""


# --------------------------------------------------------------- endpoints --
@app.get("/")
def root():
    return {"service": "SafeMedTwin API", "diseases": DISEASES, "docs": "/docs"}


@app.get("/diseases/{disease}/features")
def feature_schema(disease: str):
    bundle = get_bundle(disease)
    return {"disease": disease, "features": bundle["feature_cols"], "metrics": bundle["metrics"]}


@app.post("/predict")
def predict(req: PredictRequest):
    bundle = get_bundle(req.disease)
    feature_cols = bundle["feature_cols"]
    missing = [c for c in feature_cols if c not in req.features]
    if missing:
        raise HTTPException(400, f"Missing features: {missing}")

    row = pd.DataFrame([{c: req.features[c] for c in feature_cols}])
    probability = float(bundle["model"].predict_proba(row)[0, 1])
    band = risk_band(probability)

    safety = validate_prediction(req.disease, req.features, probability)
    if safety.blocked:
        raise HTTPException(422, {"message": "Prediction blocked by MedGuard", "warnings": safety.warnings})

    contributions = explain_prediction(bundle, row, top_k=5)
    explanation_sentence = explanation_to_sentence(req.disease, contributions)

    # record in the patient's digital twin
    twin = _TWINS.setdefault(req.patient_id, PatientTwin(patient_id=req.patient_id, baseline=req.features))
    twin.record(req.disease, probability, req.features)

    return {
        "disease": req.disease,
        "probability": round(probability, 4),
        "risk_band": band,
        "safety_warnings": safety.warnings,
        "top_contributions": contributions,
        "explanation": explanation_sentence,
    }


@app.post("/simulate/scenario")
def scenario(req: ScenarioRequest):
    bundle = get_bundle(req.disease)
    return simulate_scenario(bundle, req.baseline, req.changes)


@app.post("/simulate/sweep")
def sweep(req: SweepRequest):
    bundle = get_bundle(req.disease)
    df = multi_scenario_sweep(bundle, req.baseline, req.feature, req.values)
    return df.to_dict(orient="records")


@app.get("/twin/{patient_id}/history")
def twin_history(patient_id: str, disease: Optional[str] = None):
    twin = _TWINS.get(patient_id)
    if not twin:
        raise HTTPException(404, "No twin found for this patient_id yet — call /predict first")
    if disease:
        return twin.trend(disease).to_dict(orient="records")
    return twin.history


@app.post("/compare/llm-slm")
def compare_endpoint(req: RecommendationRequest):
    return compare_llm_slm(req.disease, req.risk_band, req.top_factor, req.contributions_summary or "")


@app.get("/health")
def health():
    return {"status": "ok", "models_loaded": list(_MODEL_CACHE.keys())}
