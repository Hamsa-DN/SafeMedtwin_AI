"""
SafeMedTwin — DistilBERT Risk Classifier Inference
======================================================
Loads the fine-tuned classifier from models/train_distilbert_classifier.py
and classifies risk band directly from a text description — this is the
DistilBERT counterpart to the SLM's generative role, used as an additional
comparison arm: "how does a small, fast text classifier's risk band
compare to the SHAP-explained XGBoost prediction, and how fast is it?"
"""
import os
import time
from typing import Dict, Optional

MODEL_PATH = os.path.join(os.path.dirname(__file__), "distilbert_risk_classifier", "final")

_model = None
_tokenizer = None


def is_available() -> bool:
    """Whether a fine-tuned classifier has actually been trained and
    saved yet — run models/train_distilbert_classifier.py first."""
    return os.path.exists(os.path.join(MODEL_PATH, "config.json"))


def _load():
    global _model, _tokenizer
    if _model is None:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        _tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        _model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
        _model.eval()
    return _model, _tokenizer


def classify_risk_text(text: str) -> Dict:
    """Returns {"label": "low"|"moderate"|"high", "confidence": float,
    "latency_ms": float, "probabilities": {label: prob}}. Raises if the
    classifier hasn't been trained yet — check is_available() first, or
    catch the exception (matches the fail-loud pattern used by the
    LLM/SLM provider modules elsewhere in this project)."""
    import torch

    if not is_available():
        raise FileNotFoundError(
            f"No fine-tuned classifier found at {MODEL_PATH}. "
            "Run: python models/train_distilbert_classifier.py"
        )

    model, tokenizer = _load()
    start = time.time()

    inputs = tokenizer(text, truncation=True, max_length=128, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1)[0]

    latency_ms = (time.time() - start) * 1000
    id2label = model.config.id2label
    prob_dict = {id2label[i]: float(probs[i]) for i in range(len(probs))}
    best_id = int(probs.argmax())

    return {
        "label": id2label[best_id],
        "confidence": float(probs[best_id]),
        "latency_ms": round(latency_ms, 1),
        "probabilities": prob_dict,
    }


def classify_patient(disease: str, patient_row) -> Optional[Dict]:
    """Convenience wrapper: takes a disease name + a patient's feature row
    (same shape used elsewhere in the app) and returns the classification,
    or None if the classifier isn't trained yet (graceful, not an
    exception, since this is meant to be called from UI code that should
    just hide the feature rather than error)."""
    if not is_available():
        return None
    from models.text_dataset import row_to_text
    text = row_to_text(disease, patient_row)
    result = classify_risk_text(text)
    result["text"] = text
    return result
