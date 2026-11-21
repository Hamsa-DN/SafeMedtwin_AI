"""
SafeMedTwin — Alert Engine
============================
Threshold-based anomaly detection over a patient's prediction history:
flags newly-crossed risk bands and clinically concerning raw values (e.g.
glucose spikes) as they're recorded. Simple, auditable, rule-based — no
ML in the alerting path itself, matching the MedGuard design philosophy.
"""
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional

# raw-value thresholds that warrant a standalone alert regardless of the
# model's computed probability (a clinician would flag these on sight)
CRITICAL_VALUE_RULES = {
    "glucose": (250, "Glucose reading is critically high (>250 mg/dL)"),
    "blood_pressure": (110, "Diastolic-equivalent reading is critically high"),
    "resting_bp": (160, "Resting blood pressure is critically high"),
    "serum_creatinine": (4.0, "Serum creatinine is critically high — possible acute kidney injury"),
    "total_bilirubin": (5.0, "Bilirubin is critically high"),
}

RISK_BAND_ORDER = {"low": 0, "moderate": 1, "high": 2}


@dataclass
class Alert:
    severity: str  # "info" | "warning" | "critical"
    message: str
    disease: str
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())


def check_critical_values(disease: str, features: Dict[str, float]) -> List[Alert]:
    alerts = []
    for feature, (threshold, message) in CRITICAL_VALUE_RULES.items():
        if feature in features and features[feature] >= threshold:
            alerts.append(Alert(severity="critical", message=f"{message} (value: {features[feature]})", disease=disease))
    return alerts


def check_risk_band_escalation(disease: str, previous_band: Optional[str], current_band: str) -> List[Alert]:
    """Fires when a patient's risk band gets worse between two consecutive
    assessments for the same disease — the core 'trend' alert."""
    if previous_band is None or previous_band == current_band:
        return []
    if RISK_BAND_ORDER.get(current_band, 0) > RISK_BAND_ORDER.get(previous_band, 0):
        return [Alert(
            severity="warning" if current_band != "high" else "critical",
            message=f"{disease} risk escalated from {previous_band} to {current_band} since your last assessment",
            disease=disease,
        )]
    return []


def evaluate(disease: str, features: Dict[str, float], current_band: str,
             previous_band: Optional[str] = None) -> List[Alert]:
    """Run all alert rules for a single new prediction. Call this right
    after a prediction is recorded, passing the patient's previous risk
    band for the same disease (None if this is their first assessment)."""
    alerts = []
    alerts.extend(check_critical_values(disease, features))
    alerts.extend(check_risk_band_escalation(disease, previous_band, current_band))
    if current_band == "high" and not alerts:
        alerts.append(Alert(
            severity="warning",
            message=f"{disease} risk is currently high — consider discussing this with a physician soon",
            disease=disease,
        ))
    return alerts
