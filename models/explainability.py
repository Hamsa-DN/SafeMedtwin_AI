"""
SafeMedTwin — Explainability Module (XAI)
===========================================
Wraps a fitted SHAP TreeExplainer to produce patient-friendly, ranked
feature-contribution explanations for a single prediction.
"""
from typing import Dict, List
import numpy as np
import pandas as pd


def explain_prediction(bundle: dict, patient_row: pd.DataFrame, top_k: int = 5) -> List[Dict]:
    """Return the top_k features driving this patient's risk score, ranked
    by absolute SHAP contribution, each with a plain-language direction."""
    explainer = bundle["explainer"]
    feature_cols = bundle["feature_cols"]

    shap_values = explainer.shap_values(patient_row[feature_cols])
    if isinstance(shap_values, list):  # binary classifier sometimes returns [class0, class1]
        shap_values = shap_values[-1]
    values = np.array(shap_values).flatten()

    contributions = []
    for feat, val, shap_val in zip(feature_cols, patient_row[feature_cols].values.flatten(), values):
        contributions.append({
            "feature": feat,
            "patient_value": float(val),
            "shap_value": float(shap_val),
            "direction": "increases risk" if shap_val > 0 else "decreases risk",
            "magnitude": abs(float(shap_val)),
        })

    contributions.sort(key=lambda c: c["magnitude"], reverse=True)
    return contributions[:top_k]


def explanation_to_sentence(disease: str, contributions: List[Dict]) -> str:
    """Turn ranked SHAP contributions into one readable sentence for the
    dashboard / patient-facing summary."""
    risk_factors = [c for c in contributions if c["shap_value"] > 0][:3]
    protective = [c for c in contributions if c["shap_value"] < 0][:2]

    parts = []
    if risk_factors:
        names = ", ".join(f"{c['feature'].replace('_', ' ')} ({c['patient_value']:g})" for c in risk_factors)
        parts.append(f"the biggest drivers of elevated {disease} risk are {names}")
    if protective:
        names = ", ".join(c["feature"].replace("_", " ") for c in protective)
        parts.append(f"while {names} are working in the patient's favor")

    if not parts:
        return f"No strong individual risk factors were identified for {disease}."
    return "For this patient, " + "; ".join(parts) + "."
