"""
SafeMedTwin — Diet & Lifestyle Recommendation Engine
=======================================================
Two layers, same pattern as the LLM/SLM split elsewhere in this project:

1. A rule-based knowledge base (always available, offline, MedGuard-safe
   by construction — no drug names/dosages, general lifestyle guidance
   only) keyed by disease and risk band.
2. An optional LLM personalization pass that takes the rule-based bullets
   plus the patient's specific top SHAP risk factors and turns them into
   a short, personalized paragraph — still validated through MedGuard
   before display.
"""
from typing import Dict, List, Optional
from safety.medguard import validate_text_recommendation
from llm_slm.llm_provider import call_text

DIET_KNOWLEDGE_BASE = {
    "diabetes": {
        "diet": [
            "Favor high-fiber, low-glycemic-index foods (whole grains, legumes, non-starchy vegetables)",
            "Limit refined sugars and sugary beverages",
            "Spread carbohydrate intake evenly across meals rather than large single servings",
            "Include lean protein at each meal to slow glucose absorption",
        ],
        "lifestyle": [
            "Aim for at least 150 minutes/week of moderate aerobic activity",
            "Monitor blood glucose as advised by your physician",
            "Prioritize consistent sleep timing — poor sleep worsens insulin resistance",
            "Maintain a healthy waist circumference through gradual, sustainable weight management",
        ],
    },
    "cvd": {
        "diet": [
            "Adopt a Mediterranean-style or DASH-style eating pattern",
            "Reduce saturated fat and trans fat (fried foods, processed meats)",
            "Increase omega-3 rich foods (fatty fish, walnuts, flaxseed)",
            "Limit sodium to support healthy blood pressure",
        ],
        "lifestyle": [
            "Engage in regular cardiovascular exercise as cleared by your physician",
            "Avoid tobacco use in all forms",
            "Manage stress through relaxation techniques, adequate sleep, or counseling",
            "Track blood pressure and cholesterol at recommended intervals",
        ],
    },
    "ckd": {
        "diet": [
            "Discuss appropriate protein intake with a renal dietitian — needs vary by CKD stage",
            "Limit sodium, potassium, and phosphorus if advised by your care team",
            "Stay adequately hydrated within limits set by your physician",
            "Limit processed foods, which are often high in hidden sodium and phosphorus additives",
        ],
        "lifestyle": [
            "Keep blood pressure and blood glucose well controlled — both protect kidney function",
            "Avoid overuse of NSAIDs and other nephrotoxic substances without medical guidance",
            "Get regular kidney function monitoring (creatinine, eGFR) as scheduled",
            "Maintain moderate, physician-approved physical activity",
        ],
    },
    "liver": {
        "diet": [
            "Avoid alcohol entirely, or strictly as advised by your physician",
            "Favor a balanced diet rich in vegetables, whole grains, and lean protein",
            "Limit foods high in added sugar and saturated fat, which contribute to fatty liver",
            "Maintain a healthy body weight — excess weight is a major driver of liver strain",
        ],
        "lifestyle": [
            "Avoid unnecessary or unsupervised use of medications/supplements metabolized by the liver",
            "Get vaccinated for hepatitis A/B if not already immune (discuss with your physician)",
            "Engage in regular physical activity to support metabolic health",
            "Schedule regular liver function testing as recommended",
        ],
    },
    "hypertension": {
        "diet": [
            "Follow a DASH-style diet — high in fruits, vegetables, whole grains, and low-fat dairy",
            "Limit sodium intake to the level recommended by your physician",
            "Limit alcohol consumption",
            "Increase potassium-rich foods (if not contraindicated by kidney function)",
        ],
        "lifestyle": [
            "Engage in regular aerobic exercise",
            "Practice stress-reduction techniques — chronic stress elevates blood pressure",
            "Monitor blood pressure regularly at home if advised",
            "Maintain a healthy weight and limit tobacco use",
        ],
    },
}

RISK_BAND_FRAMING = {
    "low": "Your current risk is low — these are general habits to help keep it that way.",
    "moderate": "Your current risk is moderate — these changes can meaningfully lower it over time.",
    "high": "Your current risk is elevated — please prioritize these alongside prompt medical follow-up.",
    "unknown": "Run an assessment first so these recommendations can be tailored to your actual risk level.",
}


def get_rule_based_plan(disease: str, risk_band: str = "unknown") -> Dict:
    kb = DIET_KNOWLEDGE_BASE.get(disease)
    if not kb:
        return {"framing": "Unknown disease.", "diet": [], "lifestyle": []}
    return {
        "framing": RISK_BAND_FRAMING.get(risk_band, RISK_BAND_FRAMING["unknown"]),
        "diet": kb["diet"],
        "lifestyle": kb["lifestyle"],
    }


def get_personalized_plan(disease: str, risk_band: str, top_factors: Optional[List[str]] = None) -> Dict:
    """Rule-based plan plus an LLM-personalized paragraph woven around the
    patient's specific top risk factors. Falls back cleanly to the pure
    rule-based plan if the LLM call fails (no API key / offline)."""
    base = get_rule_based_plan(disease, risk_band)
    factors_str = ", ".join(top_factors) if top_factors else "their overall risk profile"

    try:
        prompt = (
            f"A patient has a {risk_band} computed risk of {disease}. Their top contributing "
            f"factors are: {factors_str}. Using ONLY general diet and lifestyle guidance (no drug "
            f"names, no dosages, no definitive diagnosis), write a warm, encouraging 3-sentence "
            f"personalized note connecting their specific risk factors to 1-2 concrete, actionable "
            f"habits. End by recommending they discuss specifics with their physician or a dietitian."
        )
        text = call_text("You are a cautious health and wellness assistant.", prompt, max_tokens=220)
        validation = validate_text_recommendation(text, patient_risk_band=risk_band)
        base["personalized_note"] = validation.sanitized_text if validation.warnings else text
        base["personalization_source"] = "llm"
    except Exception:
        base["personalized_note"] = (
            f"Based on your {risk_band} {disease} risk (influenced by {factors_str}), "
            f"focus on the diet and lifestyle habits below, and discuss a specific plan with "
            f"your physician or a registered dietitian."
        )
        base["personalization_source"] = "fallback"

    return base
