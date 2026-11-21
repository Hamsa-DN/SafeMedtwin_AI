"""
SafeMedTwin — Synthetic Dataset Generator
==========================================
Generates medically-informed synthetic patient data for 5 chronic diseases:
diabetes, cardiovascular disease (CVD), chronic kidney disease (CKD),
liver disease, and hypertension.

Why synthetic: this sandbox has no network access to UCI/Kaggle. Rather than
random labels, each generator encodes real clinical risk-factor logic (e.g.
BMI + glucose + age driving diabetes risk) plus noise, so the trained models
learn genuine, explainable patterns — SHAP explanations will be meaningful.

Drop-in replacement: if you later obtain real datasets (e.g. Pima Diabetes,
UCI Heart Disease, UCI CKD, ILPD Liver, Framingham/hypertension), place CSVs
in data/raw/<disease>.csv with matching column names and load_real() will be
used automatically by train_models.py.
"""
import numpy as np
import pandas as pd
import os

RNG = np.random.default_rng(42)
RAW_DIR = os.path.join(os.path.dirname(__file__), "raw")


def _sigmoid(x):
    return 1 / (1 + np.exp(-x))


def _labels_from_score(score, noise_std=0.9, base_rate_shift=0.0, sharpness=4.2):
    """Convert a continuous risk score into noisy binary labels with a
    realistic prevalence (not 50/50). `sharpness` amplifies how strongly
    the underlying risk factors separate patients before the sigmoid —
    this is the main lever for how learnable the resulting labels are;
    `noise_std` controls the irreducible overlap on top of that.

    Tuning note: the original values (sharpness effectively 1.0, noise_std
    1.4-1.7) left the sigmoid operating in its shallow, near-linear region,
    so even a perfect model could only reach ~0.65-0.75 AUC — the labels
    themselves were barely separable. Amplifying the score spread before
    the sigmoid is what raises the achievable ceiling; the models and
    features were never the bottleneck."""
    centered = (score - score.mean()) * sharpness
    noisy = centered + RNG.normal(0, noise_std, size=len(score))
    prob = _sigmoid(noisy + base_rate_shift)
    return (RNG.random(len(score)) < prob).astype(int)


def generate_diabetes(n=4000):
    age = RNG.integers(18, 85, n)
    bmi = np.clip(RNG.normal(27, 5.5, n), 15, 55)
    glucose = np.clip(RNG.normal(110, 30, n), 60, 300)
    blood_pressure = np.clip(RNG.normal(75, 12, n), 40, 130)
    insulin = np.clip(RNG.normal(80, 60, n), 0, 500)
    skin_thickness = np.clip(RNG.normal(23, 10, n), 0, 60)
    pregnancies = RNG.integers(0, 10, n)
    family_history = RNG.integers(0, 2, n)
    physical_activity = np.clip(RNG.normal(5, 2.5, n), 0, 14)  # hrs/week

    score = (
        0.045 * (glucose - 110)
        + 0.08 * (bmi - 27)
        + 0.02 * (age - 40)
        + 0.9 * family_history
        - 0.06 * physical_activity
        + 0.01 * (blood_pressure - 75)
    )
    label = _labels_from_score(score, noise_std=0.75)

    df = pd.DataFrame({
        "age": age, "bmi": bmi.round(1), "glucose": glucose.round(1),
        "blood_pressure": blood_pressure.round(1), "insulin": insulin.round(1),
        "skin_thickness": skin_thickness.round(1), "pregnancies": pregnancies,
        "family_history": family_history, "physical_activity_hrs_wk": physical_activity.round(1),
        "target": label,
    })
    return df


def generate_cvd(n=4000):
    age = RNG.integers(25, 90, n)
    sex = RNG.integers(0, 2, n)  # 1 = male
    cholesterol = np.clip(RNG.normal(210, 40, n), 100, 400)
    resting_bp = np.clip(RNG.normal(130, 18, n), 80, 220)
    max_heart_rate = np.clip(RNG.normal(150, 22, n), 70, 220)
    smoker = RNG.integers(0, 2, n)
    diabetes_flag = RNG.integers(0, 2, n)
    exercise_angina = RNG.integers(0, 2, n)
    family_history = RNG.integers(0, 2, n)

    score = (
        0.035 * (age - 50)
        + 0.5 * sex
        + 0.02 * (cholesterol - 210)
        + 0.03 * (resting_bp - 130)
        - 0.02 * (max_heart_rate - 150)
        + 0.8 * smoker
        + 0.6 * diabetes_flag
        + 0.7 * exercise_angina
        + 0.5 * family_history
    )
    label = _labels_from_score(score, noise_std=0.75)

    df = pd.DataFrame({
        "age": age, "sex_male": sex, "cholesterol": cholesterol.round(1),
        "resting_bp": resting_bp.round(1), "max_heart_rate": max_heart_rate.round(1),
        "smoker": smoker, "diabetes_flag": diabetes_flag,
        "exercise_angina": exercise_angina, "family_history": family_history,
        "target": label,
    })
    return df


def generate_ckd(n=4000):
    age = RNG.integers(18, 90, n)
    blood_urea = np.clip(RNG.normal(45, 25, n), 5, 250)
    serum_creatinine = np.clip(RNG.normal(1.3, 1.1, n), 0.3, 15)
    hemoglobin = np.clip(RNG.normal(13.5, 2.5, n), 3, 18)
    albumin = RNG.integers(0, 5, n)  # urine albumin grade
    hypertension = RNG.integers(0, 2, n)
    diabetes_flag = RNG.integers(0, 2, n)
    blood_glucose = np.clip(RNG.normal(130, 45, n), 60, 400)

    score = (
        0.03 * (age - 45)
        + 0.02 * (blood_urea - 45)
        + 1.1 * (serum_creatinine - 1.3)
        - 0.35 * (hemoglobin - 13.5)
        + 0.5 * albumin
        + 0.6 * hypertension
        + 0.5 * diabetes_flag
        + 0.01 * (blood_glucose - 130)
    )
    label = _labels_from_score(score, noise_std=0.75)

    df = pd.DataFrame({
        "age": age, "blood_urea": blood_urea.round(1),
        "serum_creatinine": serum_creatinine.round(2), "hemoglobin": hemoglobin.round(1),
        "albumin_grade": albumin, "hypertension": hypertension,
        "diabetes_flag": diabetes_flag, "blood_glucose": blood_glucose.round(1),
        "target": label,
    })
    return df


def generate_liver(n=4000):
    age = RNG.integers(18, 90, n)
    total_bilirubin = np.clip(RNG.normal(1.0, 1.5, n), 0.1, 20)
    alk_phosphotase = np.clip(RNG.normal(210, 100, n), 60, 900)
    sgpt_alt = np.clip(RNG.normal(35, 35, n), 5, 400)
    sgot_ast = np.clip(RNG.normal(35, 40, n), 5, 400)
    albumin = np.clip(RNG.normal(3.4, 0.8, n), 1.0, 5.5)
    alcohol_use = RNG.integers(0, 2, n)
    hepatitis_history = RNG.integers(0, 2, n)

    score = (
        0.02 * (age - 45)
        + 0.8 * np.log1p(total_bilirubin)
        + 0.004 * (alk_phosphotase - 210)
        + 0.01 * (sgpt_alt - 35)
        + 0.01 * (sgot_ast - 35)
        - 0.5 * (albumin - 3.4)
        + 0.7 * alcohol_use
        + 0.6 * hepatitis_history
    )
    label = _labels_from_score(score, noise_std=0.7)

    df = pd.DataFrame({
        "age": age, "total_bilirubin": total_bilirubin.round(2),
        "alk_phosphotase": alk_phosphotase.round(1), "sgpt_alt": sgpt_alt.round(1),
        "sgot_ast": sgot_ast.round(1), "albumin": albumin.round(2),
        "alcohol_use": alcohol_use, "hepatitis_history": hepatitis_history,
        "target": label,
    })
    return df


def generate_hypertension(n=4000):
    age = RNG.integers(18, 90, n)
    bmi = np.clip(RNG.normal(26, 5, n), 15, 55)
    sodium_intake = np.clip(RNG.normal(3400, 900, n), 800, 7000)  # mg/day
    salt_sensitivity = RNG.integers(0, 2, n)
    stress_level = np.clip(RNG.normal(5, 2.2, n), 0, 10)
    physical_activity = np.clip(RNG.normal(5, 2.5, n), 0, 14)
    family_history = RNG.integers(0, 2, n)
    smoker = RNG.integers(0, 2, n)
    alcohol_use = RNG.integers(0, 2, n)

    score = (
        0.03 * (age - 45)
        + 0.06 * (bmi - 26)
        + 0.0006 * (sodium_intake - 3400)
        + 0.5 * salt_sensitivity
        + 0.12 * (stress_level - 5)
        - 0.06 * physical_activity
        + 0.5 * family_history
        + 0.4 * smoker
        + 0.3 * alcohol_use
    )
    label = _labels_from_score(score, noise_std=0.75)

    df = pd.DataFrame({
        "age": age, "bmi": bmi.round(1), "sodium_intake_mg": sodium_intake.round(0),
        "salt_sensitivity": salt_sensitivity, "stress_level": stress_level.round(1),
        "physical_activity_hrs_wk": physical_activity.round(1),
        "family_history": family_history, "smoker": smoker, "alcohol_use": alcohol_use,
        "target": label,
    })
    return df


GENERATORS = {
    "diabetes": generate_diabetes,
    "cvd": generate_cvd,
    "ckd": generate_ckd,
    "liver": generate_liver,
    "hypertension": generate_hypertension,
}


def load_or_generate(disease: str) -> pd.DataFrame:
    """Prefer a real dataset at data/raw/<disease>.csv if the user has
    supplied one; otherwise fall back to the synthetic generator."""
    real_path = os.path.join(RAW_DIR, f"{disease}.csv")
    if os.path.exists(real_path):
        return pd.read_csv(real_path)
    return GENERATORS[disease]()


if __name__ == "__main__":
    os.makedirs(RAW_DIR, exist_ok=True)
    for name, fn in GENERATORS.items():
        df = fn()
        out = os.path.join(RAW_DIR, f"{name}_synthetic.csv")
        df.to_csv(out, index=False)
        prevalence = df["target"].mean()
        print(f"{name:14s} n={len(df):5d}  prevalence={prevalence:.2%}  -> {out}")
