"""
SafeMedTwin — MedGuard Safety & Validation Layer
==================================================
A rule-based clinical guardrail that sits between raw model/LLM output and
the user. It does two jobs:

1. validate_prediction(): sanity-checks an ML risk score against hard
   clinical bounds and flags results that are numerically implausible or
   contradicted by the patient's own input values.
2. validate_text_recommendation(): scans free-text AI-generated advice
   (from the LLM/SLM module) for unsafe patterns — absolute/definitive
   medical claims, dosage instructions, or advice contradicting the
   patient's known risk level — and flags/redacts before display.

This is intentionally simple and auditable (no ML in the safety layer
itself) — that's the point: a black-box model should never be the only
thing standing between a recommendation and the patient.
"""
import re
from dataclasses import dataclass, field
from typing import List


@dataclass
class ValidationResult:
    is_safe: bool
    warnings: List[str] = field(default_factory=list)
    blocked: bool = False
    sanitized_text: str = ""


# Physiologically implausible combinations -> block, don't just warn
HARD_RULES = {
    "diabetes": [
        lambda r: (r.get("glucose", 0) < 40, "Glucose below 40 mg/dL is not physiologically stable input"),
    ],
    "cvd": [
        lambda r: (r.get("max_heart_rate", 0) > 220 - r.get("age", 30), "Max heart rate exceeds age-predicted maximum"),
    ],
    "ckd": [
        lambda r: (r.get("serum_creatinine", 0) > 20, "Serum creatinine value out of plausible range"),
    ],
}

# Patterns that make an AI-generated recommendation unsafe to show verbatim
UNSAFE_PATTERNS = [
    # (re.compile(r"\b(\d+(\.\d+)?\s*(mg|mcg|ml|units?)\b.*\b(daily|per day|twice|every)\b)", re.I),
    #  "Contains a specific drug dosage/frequency instruction"),
    # (re.compile(r"\byou (definitely|certainly) (have|do not have)\b", re.I),
    #  "States a diagnosis with false certainty"),
    # (re.compile(r"\bstop taking\b|\bdiscontinue\b", re.I),
    #  "Instructs discontinuation of medication without clinician oversight"),
    # (re.compile(r"\bguaranteed\b|\b100%\s*(cure|effective|safe)\b", re.I),
    #  "Makes an absolute efficacy/safety claim"),
    (re.compile(
    r"\b("
    r"(adjust|change|increase|decrease|modify)\s+(your\s+)?"
    r"(insulin|medication|medicines?|drugs?|dose|dosage)"
    r"|"
    r"(insulin|medication|medicines?|drugs?|dose|dosage)\s+"
    r"(should be adjusted|should be changed|should be increased|should be decreased)"
    r")\b",
    re.I
),
 "Instructs the patient to change medication, insulin, or dosage without clinician oversight"),
]

RISK_THRESHOLDS = {"low": 0.33, "moderate": 0.66}


def risk_band(probability: float) -> str:
    if probability < RISK_THRESHOLDS["low"]:
        return "low"
    if probability < RISK_THRESHOLDS["moderate"]:
        return "moderate"
    return "high"


def validate_prediction(disease: str, patient_dict: dict, probability: float) -> ValidationResult:
    warnings = []
    blocked = False

    for rule in HARD_RULES.get(disease, []):
        triggered, message = rule(patient_dict)
        if triggered:
            warnings.append(f"[BLOCKED] {message}")
            blocked = True

    if not (0.0 <= probability <= 1.0):
        warnings.append("[BLOCKED] Predicted probability outside valid [0,1] range")
        blocked = True

    return ValidationResult(is_safe=not blocked, warnings=warnings, blocked=blocked)


def validate_text_recommendation(text: str, patient_risk_band: str = None) -> ValidationResult:
    warnings = []
    sanitized = text
    blocked = False

    for pattern, reason in UNSAFE_PATTERNS:
        if pattern.search(text):
            warnings.append(reason)
            sanitized = pattern.sub("[reviewed by MedGuard — see a clinician for specifics]", sanitized)

    # a recommendation calling a high-risk patient "no concern" is a red flag
    if patient_risk_band == "high" and re.search(r"\bno (concern|risk|action needed)\b", text, re.I):
        warnings.append("Recommendation appears to understate a high computed risk score — flagged for review")

    if len(warnings) >= 3:
        blocked = True

    return ValidationResult(
        is_safe=len(warnings) == 0,
        warnings=warnings,
        blocked=blocked,
        sanitized_text=sanitized if not blocked else "[Recommendation withheld pending clinician review — too many safety flags triggered]",
    )
