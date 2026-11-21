"""
SafeMedTwin — Model Training Pipeline
======================================
Trains an XGBoost classifier per disease, evaluates it, and persists both
the model and a fitted SHAP TreeExplainer via joblib so the backend can
load them instantly without retraining or recomputing background data.
"""
import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score
from xgboost import XGBClassifier
import shap

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from data.generate_datasets import load_or_generate, GENERATORS

MODEL_DIR = os.path.dirname(__file__)
METRICS_PATH = os.path.join(MODEL_DIR, "metrics.json")


def train_one(disease: str):
    df = load_or_generate(disease)
    feature_cols = [c for c in df.columns if c != "target"]
    X, y = df[feature_cols], df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.08,
        subsample=0.9,
        colsample_bytree=0.9,
        eval_metric="logloss",
        random_state=42,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "roc_auc": round(roc_auc_score(y_test, y_proba), 4),
        "precision": round(precision_score(y_test, y_pred), 4),
        "recall": round(recall_score(y_test, y_pred), 4),
        "f1": round(f1_score(y_test, y_pred), 4),
        "n_train": len(X_train),
        "n_test": len(X_test),
        "prevalence": round(float(y.mean()), 4),
    }

    # SHAP explainer. tree_path_dependent perturbation avoids requiring a
    # background dataset and sidesteps XGBoost's categorical-split limitation.
    explainer = shap.TreeExplainer(model, feature_perturbation="tree_path_dependent")

    bundle = {
        "model": model,
        "explainer": explainer,
        "feature_cols": feature_cols,
        "metrics": metrics,
    }
    out_path = os.path.join(MODEL_DIR, f"{disease}_model.joblib")
    joblib.dump(bundle, out_path)
    return disease, metrics, out_path


def main():
    all_metrics = {}
    for disease in GENERATORS.keys():
        name, metrics, path = train_one(disease)
        all_metrics[name] = metrics
        print(f"[{name:14s}] AUC={metrics['roc_auc']:.3f}  Acc={metrics['accuracy']:.3f}  "
              f"F1={metrics['f1']:.3f}  -> {os.path.basename(path)}")

    with open(METRICS_PATH, "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"\nMetrics written to {METRICS_PATH}")


if __name__ == "__main__":
    main()
