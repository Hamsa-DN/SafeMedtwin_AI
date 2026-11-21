"""
SafeMedTwin — Digital Twin Simulation Engine
===============================================
Maintains a lightweight virtual representation of a patient and lets the
model re-score them under hypothetical lifestyle/clinical changes
("what-if I lower my BMI by 3 points and quit smoking?"). This is not a
biophysical simulation — it's a decision-support tool: it re-runs the
trained risk model on a modified feature vector and reports the delta,
which is the same principle real clinical digital twins use for
lifestyle-intervention counselling.

A simple in-memory time series lets the frontend show longitudinal trend.
"""
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List
import pandas as pd


@dataclass
class PatientTwin:
    patient_id: str
    baseline: Dict[str, float]
    history: List[Dict] = field(default_factory=list)

    def record(self, disease: str, probability: float, feature_snapshot: Dict[str, float]):
        self.history.append({
            "timestamp": datetime.utcnow().isoformat(),
            "disease": disease,
            "probability": probability,
            "features": dict(feature_snapshot),
        })

    def trend(self, disease: str) -> pd.DataFrame:
        rows = [h for h in self.history if h["disease"] == disease]
        if not rows:
            return pd.DataFrame(columns=["timestamp", "probability"])
        return pd.DataFrame([{"timestamp": r["timestamp"], "probability": r["probability"]} for r in rows])


def simulate_scenario(model_bundle: dict, baseline: Dict[str, float], changes: Dict[str, float]) -> Dict:
    """Apply `changes` on top of `baseline`, re-score with the trained model,
    and report both the new probability and the delta vs baseline."""
    model = model_bundle["model"]
    feature_cols = model_bundle["feature_cols"]

    # Force float64 explicitly: if a feature happens to be integer-valued in
    # the baseline (e.g. age=55), pandas infers an int64 column, and newer
    # pandas raises a TypeError rather than upcasting when a sweep later
    # assigns a float into it (e.g. age=54.7 mid-sweep). Casting up front
    # avoids that — every feature here is numeric, so this is always safe.
    baseline_row = pd.DataFrame([{c: float(baseline.get(c, 0)) for c in feature_cols}], dtype="float64")
    scenario_row = baseline_row.copy()
    for k, v in changes.items():
        if k in feature_cols:
            scenario_row.at[0, k] = float(v)

    baseline_prob = float(model.predict_proba(baseline_row)[0, 1])
    scenario_prob = float(model.predict_proba(scenario_row)[0, 1])

    return {
        "baseline_probability": round(baseline_prob, 4),
        "scenario_probability": round(scenario_prob, 4),
        "delta": round(scenario_prob - baseline_prob, 4),
        "changes_applied": changes,
        "scenario_features": scenario_row.iloc[0].to_dict(),
    }


def multi_scenario_sweep(model_bundle: dict, baseline: Dict[str, float], feature: str,
                          value_range: List[float]) -> pd.DataFrame:
    """Sweep a single feature across a range of values, holding others fixed —
    powers the 'what happens to my risk as X changes' chart in the dashboard."""
    rows = []
    for v in value_range:
        result = simulate_scenario(model_bundle, baseline, {feature: v})
        rows.append({feature: v, "probability": result["scenario_probability"]})
    return pd.DataFrame(rows)
