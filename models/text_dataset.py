"""
SafeMedTwin — Text Dataset Builder for the DistilBERT Risk Classifier
========================================================================
DistilBERT is an encoder — it classifies, it doesn't generate free text.
That's a different (complementary) job from TinyLlama's role in the SLM
arm of the LLM-vs-SLM comparison. Here, DistilBERT's job is: given a
short natural-language description of a patient, classify their risk
band (low/moderate/high) directly from text — the kind of task a fast,
small encoder model is genuinely well-suited for.

This module builds that text dataset from the same structured data the
XGBoost models train on, so the classifier's "ground truth" risk bands
come from the same trained model's predictions (via risk_band()) that
the rest of the app already treats as authoritative — this keeps the
DistilBERT classifier a second, independent way of arriving at the same
label space, not a different standard entirely.
"""
import os
import sys
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from data.generate_datasets import load_or_generate, GENERATORS
from safety.medguard import risk_band

# Human-readable phrasing for each numeric feature, used to build the text
# description. Centralized here so adding a 6th disease only needs an
# entry added, not changes to the conversion logic itself.
FEATURE_LABELS = {
    "age": "age {v:.0f}", "bmi": "BMI {v:.1f}", "glucose": "glucose {v:.0f} mg/dL",
    "blood_pressure": "blood pressure {v:.0f}", "insulin": "insulin {v:.0f}",
    "skin_thickness": "skin thickness {v:.0f}", "pregnancies": "{v:.0f} pregnancies",
    "physical_activity_hrs_wk": "{v:.1f} hours/week physical activity",
    "cholesterol": "cholesterol {v:.0f} mg/dL", "resting_bp": "resting BP {v:.0f}",
    "max_heart_rate": "max heart rate {v:.0f}",
    "blood_urea": "blood urea {v:.0f}", "serum_creatinine": "serum creatinine {v:.2f}",
    "hemoglobin": "hemoglobin {v:.1f}", "albumin_grade": "urine albumin grade {v:.0f}",
    "blood_glucose": "blood glucose {v:.0f}",
    "total_bilirubin": "total bilirubin {v:.2f}", "alk_phosphotase": "alkaline phosphatase {v:.0f}",
    "sgpt_alt": "ALT {v:.0f}", "sgot_ast": "AST {v:.0f}", "albumin": "albumin {v:.2f}",
    "sodium_intake_mg": "sodium intake {v:.0f}mg/day",
    "stress_level": "stress level {v:.1f}/10",
}

# Binary (0/1) features are phrased as present/absent rather than
# "feature_name 1" / "feature_name 0" — this is what makes the text real
# natural language rather than a thinly-disguised feature vector, which
# matters for genuinely testing a language-understanding model.
BINARY_LABELS = {
    "family_history": "family history of the condition", "sex_male": "male",
    "smoker": "smoker", "diabetes_flag": "history of diabetes",
    "exercise_angina": "exercise-induced angina", "hypertension": "history of hypertension",
    "alcohol_use": "alcohol use", "hepatitis_history": "history of hepatitis",
    "salt_sensitivity": "salt sensitivity",
}
BINARY_FEATURES = set(BINARY_LABELS.keys())

DISEASE_NAMES = {
    "diabetes": "diabetes", "cvd": "cardiovascular disease", "ckd": "chronic kidney disease",
    "liver": "liver disease", "hypertension": "hypertension",
}


def row_to_text(disease: str, row: pd.Series) -> str:
    """Convert one patient's structured feature row into a short natural-
    language description, e.g. 'Patient, age 55, BMI 34.0, glucose 165
    mg/dL, family history of the condition. Being assessed for diabetes.'"""
    parts = []
    for feat, value in row.items():
        if feat == "target":
            continue
        if feat in BINARY_FEATURES:
            if value >= 1:
                parts.append(BINARY_LABELS[feat])
        elif feat in FEATURE_LABELS:
            parts.append(FEATURE_LABELS[feat].format(v=value))
        else:
            parts.append(f"{feat.replace('_', ' ')} {value}")

    description = "Patient, " + ", ".join(parts) + "."
    return f"{description} Being assessed for {DISEASE_NAMES.get(disease, disease)}."


def build_text_dataset(disease: str) -> pd.DataFrame:
    """Returns a DataFrame with columns [text, label, disease] where label
    is the risk band (low/moderate/high), computed via the same trained
    model + risk_band() logic the rest of the app treats as ground truth."""
    import joblib
    bundle_path = os.path.join(os.path.dirname(__file__), f"{disease}_model.joblib")
    bundle = joblib.load(bundle_path)
    feature_cols = bundle["feature_cols"]

    df = load_or_generate(disease)
    X = df[feature_cols]
    probabilities = bundle["model"].predict_proba(X)[:, 1]

    texts = [row_to_text(disease, df.iloc[i][feature_cols]) for i in range(len(df))]
    labels = [risk_band(p) for p in probabilities]

    return pd.DataFrame({"text": texts, "label": labels, "disease": disease})


def build_all_text_datasets() -> pd.DataFrame:
    """Combined dataset across all 5 diseases — one multi-disease text
    classifier rather than 5 separate ones, since the risk-band vocabulary
    and reasoning pattern is shared across diseases."""
    frames = [build_text_dataset(d) for d in GENERATORS.keys()]
    combined = pd.concat(frames, ignore_index=True)
    return combined.sample(frac=1, random_state=42).reset_index(drop=True)  # shuffle
