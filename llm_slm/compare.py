"""
SafeMedTwin — LLM vs SLM Evaluation Framework
================================================
Compares a cloud LLM against a lightweight local "SLM" responder on the
same medical query, scoring both on accuracy (keyword overlap with a
curated medical knowledge base), completeness, latency, and hallucination
rate (via MedGuard cross-referencing).

IMPORTANT — environment note: this sandbox cannot reach huggingface.co, so
a true local SLM (DistilBERT/TinyLlama) can't be downloaded here. By
default slm_generate() below uses a fast, offline, template + knowledge-
base grounded responder that plays the same *role* an on-device SLM would
(fast, offline, constrained) so the comparison framework and metrics are
real and swappable.

On your own machine (which has normal internet access), set
SAFEMEDTWIN_SLM_BACKEND=transformers to route through a real local model
via llm_slm/slm_transformers.py (TinyLlama by default) instead — no code
changes needed, just the env var and `pip install transformers torch`.
The LLM arm calls the real Anthropic API.
"""
import os
import time
import re
from dataclasses import dataclass
from typing import Dict, List

from safety.medguard import validate_text_recommendation
from llm_slm.llm_provider import call_text

# --- A small curated medical knowledge base for keyword-accuracy scoring ---
KNOWLEDGE_BASE = {
    "diabetes": ["blood sugar", "glucose", "insulin", "hba1c", "diet", "exercise", "monitoring", "physician"],
    "cvd": ["cholesterol", "blood pressure", "heart", "smoking", "exercise", "diet", "cardiologist"],
    "ckd": ["kidney", "creatinine", "fluid", "protein intake", "blood pressure", "nephrologist"],
    "liver": ["liver", "alcohol", "bilirubin", "diet", "hepatologist", "avoid alcohol"],
    "hypertension": ["blood pressure", "sodium", "salt", "stress", "exercise", "monitoring", "physician"],
}

SLM_TEMPLATES = {
    "low": "Based on the available data, your {disease} risk appears low. Maintain a healthy diet, "
           "regular exercise, and routine check-ups. Continue monitoring relevant markers periodically "
           "and consult a physician if symptoms change.",
    "moderate": "Your {disease} risk is moderate. Consider discussing your {top_factor} with a physician, "
                "adopt lifestyle changes such as improved diet and increased physical activity, and "
                "schedule follow-up monitoring in the near term.",
    "high": "Your {disease} risk is elevated. This warrants a consultation with a specialist soon. "
            "Key contributing factors include {top_factor}. Please seek professional medical evaluation; "
            "this summary does not replace clinical diagnosis.",
}


@dataclass
class ModelResponse:
    source: str  # "LLM" or "SLM"
    text: str
    latency_ms: float
    accuracy_score: float
    completeness_score: float
    hallucination_flags: List[str]


# def _keyword_accuracy(text: str, disease: str) -> float:
#     kb = KNOWLEDGE_BASE.get(disease, [])
#     if not kb:
#         return 0.0
#     text_l = text.lower()
#     hits = sum(1 for kw in kb if kw in text_l)
#     return round(hits / len(kb), 3)
# def _keyword_accuracy(text: str, disease: str) -> float:
#     """
#     Estimate grounded accuracy using the curated knowledge base.

#     Handles disease names such as:
#     - diabetes
#     - type 2 diabetes
#     - diabetic
#     - hypertension
#     - high blood pressure
#     """

#     disease_l = disease.lower().strip()

#     # Map common disease-name variations to the knowledge-base key.
#     disease_aliases = {
#         "type 2 diabetes": "diabetes",
#         "type ii diabetes": "diabetes",
#         "type 1 diabetes": "diabetes",
#         "diabetic": "diabetes",
#         "high blood pressure": "hypertension",
#         "high bp": "hypertension",
#         "cardiovascular disease": "cvd",
#         "cardiovascular": "cvd",
#         "chronic kidney disease": "ckd",
#         "kidney disease": "ckd",
#         "fatty liver disease": "liver",
#     }

#     kb_key = disease_aliases.get(disease_l, disease_l)

#     # Also try partial matching if the exact name was not found.
#     if kb_key not in KNOWLEDGE_BASE:
#         for key in KNOWLEDGE_BASE:
#             if key in disease_l or disease_l in key:
#                 kb_key = key
#                 break

#     kb = KNOWLEDGE_BASE.get(kb_key, [])

#     if not kb:
#         return 0.0

#     text_l = text.lower()

#     hits = sum(1 for kw in kb if kw.lower() in text_l)

#     return round(hits / len(kb), 3)


# def _keyword_accuracy(text: str, disease: str) -> float:
#     """
#     Concept-based accuracy score.

#     The previous implementation required exact keyword matches, which
#     unfairly scored valid medical responses very low. For example,
#     "blood glucose" is semantically equivalent to "blood sugar" for
#     this evaluation, but exact keyword matching treated them differently.

#     Each disease therefore has groups of equivalent/related concepts.
#     A concept receives credit when at least one phrase from that group
#     appears in the response.
#     """
#     concept_groups = {
#         "diabetes": [
#             ["glucose", "blood glucose", "blood sugar"],
#             ["diabetes"],
#             ["risk", "elevated", "higher than normal"],
#             ["diet", "healthy lifestyle", "lifestyle"],
#             ["exercise", "physical activity"],
#             ["monitor", "monitoring", "follow-up"],
#             ["physician", "doctor", "healthcare provider", "healthcare professional"],
#         ],

#         "cvd": [
#             ["cholesterol"],
#             ["heart", "cardiovascular"],
#             ["blood pressure"],
#             ["smoking", "tobacco"],
#             ["exercise", "physical activity"],
#             ["diet", "healthy diet", "lifestyle"],
#             ["physician", "doctor", "cardiologist", "healthcare provider"],
#         ],

#         "ckd": [
#             ["kidney", "renal"],
#             ["creatinine"],
#             ["fluid", "hydration"],
#             ["protein"],
#             ["blood pressure"],
#             ["monitor", "monitoring", "follow-up"],
#             ["physician", "doctor", "nephrologist", "healthcare provider"],
#         ],

#         "liver": [
#             ["liver", "hepatic"],
#             ["bilirubin"],
#             ["alcohol"],
#             ["diet", "healthy diet", "lifestyle"],
#             ["monitor", "monitoring", "follow-up"],
#             ["physician", "doctor", "hepatologist", "healthcare provider"],
#         ],

#         "hypertension": [
#             ["blood pressure", "hypertension"],
#             ["sodium", "salt"],
#             ["stress"],
#             ["exercise", "physical activity"],
#             ["diet", "healthy diet", "lifestyle"],
#             ["monitor", "monitoring"],
#             ["physician", "doctor", "healthcare provider"],
#         ],
#     }


def _keyword_accuracy(text: str, disease: str) -> float:
    """
    Task-specific concept coverage score.

    Each disease has a set of medically relevant concepts.
    Synonyms/related terms are grouped together so that correct
    wording is not penalized merely because the model used a
    different phrase.
    """

    CONCEPTS = {
        "diabetes": [
            ["blood sugar", "blood glucose", "glucose", "blood sugar levels"],
            ["hba1c", "hemoglobin a1c", "glycated hemoglobin"],
            ["diet", "nutrition", "healthy eating", "dietary"],
            ["exercise", "physical activity", "physical activity levels"],
            ["monitoring", "monitor", "monitoring blood sugar"],
            ["physician", "doctor", "healthcare provider", "clinician"],
        ],

        "cvd": [
            ["cholesterol", "ldl", "lipid"],
            ["blood pressure", "hypertension"],
            ["heart", "cardiovascular"],
            ["smoking", "tobacco"],
            ["exercise", "physical activity"],
            ["diet", "nutrition"],
            ["cardiologist", "physician", "doctor"],
        ],

        "ckd": [
            ["kidney", "renal"],
            ["creatinine"],
            ["blood pressure", "hypertension"],
            ["fluid", "hydration"],
            ["protein intake", "protein"],
            ["nephrologist", "physician", "doctor"],
        ],

        "liver": [
            ["liver", "hepatic"],
            ["bilirubin"],
            ["alcohol", "avoid alcohol"],
            ["diet", "nutrition"],
            ["hepatologist", "physician", "doctor"],
        ],

        "hypertension": [
            ["blood pressure", "hypertension"],
            ["sodium", "salt"],
            ["stress", "stress management"],
            ["exercise", "physical activity"],
            ["diet", "nutrition"],
            ["monitoring", "monitor"],
            ["physician", "doctor", "clinician"],
        ],
    }

    concepts = CONCEPTS.get(disease, [])

    if not concepts:
        return 0.0

    text_l = text.lower()

    matched = 0

    for synonyms in concepts:
        if any(term.lower() in text_l for term in synonyms):
            matched += 1

    return round(matched / len(concepts), 3)

    groups = concept_groups.get(disease)

    if not groups:
        return 0.0

    text_l = text.lower()

    matched = 0

    for group in groups:
        if any(phrase in text_l for phrase in group):
            matched += 1

    return round(matched / len(groups), 3)


def _completeness(text: str) -> float:
    """
    Evaluate whether the response contains the main components expected
    from a safe patient-facing clinical summary:

    1. Identifies the relevant condition/risk.
    2. Mentions an important factor or finding.
    3. Provides an appropriate next step.
    4. Uses safe/non-diagnostic language.
    """

    text_l = text.lower()

    checks = [
        # Risk / condition
        bool(re.search(
            r"risk|diabetes|elevated|higher than normal|increased",
            text_l
        )),

        # Finding / contributing factor
        bool(re.search(
            r"glucose|blood sugar|cholesterol|blood pressure|"
            r"factor|driver|contribut|level|marker|result",
            text_l
        )),

        # Appropriate follow-up
        bool(re.search(
            r"physician|doctor|clinician|specialist|"
            r"healthcare provider|healthcare professional|"
            r"follow.?up|appointment|consult",
            text_l
        )),

        # Safe language / avoids definitive diagnosis
        bool(re.search(
            r"not a diagnosis|not a final diagnosis|"
            r"does not diagnose|may|suggest|indicate|"
            r"could|appears|assessment",
            text_l
        )),
    ]

    return round(sum(checks) / len(checks), 3)


# def _completeness(text: str) -> float:
#     """Rough completeness heuristic: does the response cover risk level,
#     a contributing factor, and an actionable next step?"""
#     checks = [
#         bool(re.search(r"risk", text, re.I)),
#         bool(re.search(r"factor|driver|contribut", text, re.I)),
#         bool(re.search(r"physician|doctor|clinician|specialist|consult", text, re.I)),
#     ]
#     return round(sum(checks) / len(checks), 3)
def _completeness(text: str) -> float:
    """
    Check whether the recommendation contains the three expected elements:
    1. Risk interpretation
    2. Contributing factor
    3. Appropriate follow-up/action
    """

    checks = [
        bool(re.search(
            r"risk|elevated|moderate|low|high",
            text,
            re.I
        )),
        bool(re.search(
            r"factor|driver|contribut|influenc|associated|due to|related",
            text,
            re.I
        )),
        bool(re.search(
            r"physician|doctor|clinician|specialist|consult|follow.?up|monitor",
            text,
            re.I
        )),
    ]

    return round(sum(checks) / len(checks), 3)


def slm_generate(disease: str, risk_band: str, top_factor: str) -> str:
    """Dispatches to a real local transformers model (TinyLlama etc.) if
    SAFEMEDTWIN_SLM_BACKEND=transformers is set and the package + model are
    available; otherwise uses the fast offline template responder below.
    See llm_slm/slm_transformers.py for setup instructions."""
    if os.environ.get("SAFEMEDTWIN_SLM_BACKEND", "template") == "transformers":
        try:
            from llm_slm.slm_transformers import slm_generate_transformers, is_available
            if is_available():
                return slm_generate_transformers(disease, risk_band, top_factor)
        except Exception:
            pass  # fall through to template backend if the model isn't ready
    template = SLM_TEMPLATES.get(risk_band, SLM_TEMPLATES["moderate"])
    return template.format(disease=disease, top_factor=top_factor)


# def llm_generate(disease: str, risk_band: str, top_factor: str, contributions_summary: str) -> str:
#     """Calls the configured LLM provider (Claude or Gemini) for a richer, more contextual response."""
#     try:
#         prompt = (
#             f"A clinical decision-support system computed a {risk_band} risk of {disease} for a patient. "
#             f"{contributions_summary} "
#             "Write a short (3-4 sentence), safe, non-alarmist patient-facing summary. "
#             "Do not give specific drug names, dosages, or a definitive diagnosis. "
#             "Recommend appropriate follow-up with a healthcare professional."
#         )
#         return call_text("You are a cautious clinical decision-support assistant.", prompt, max_tokens=250)
#     except Exception as e:
#         # Graceful fallback if no API key / network issue — keeps the demo running
#         from llm_slm.llm_provider import friendly_error
#         return (f"[LLM unavailable — {friendly_error(e)}] Fallback summary: your {disease} risk "
#                 f"is {risk_band}, largely influenced by {top_factor}. Please consult a physician "
#                 f"for guidance.")
def llm_generate(disease: str, risk_band: str, top_factor: str, contributions_summary: str) -> str:
    """Generate a short patient-facing recommendation using the configured cloud LLM."""

    try:
        prompt = (
            f"Write exactly 3 short sentences for a patient whose calculated "
            f"{disease} risk is {risk_band}. "
            f"The main contributing factor is {top_factor}. "
            f"{contributions_summary} "
            "Mention the risk level, the contributing factor, and appropriate "
            "follow-up with a physician. "
            "Do not diagnose. Do not mention medications or dosages. "
            "Do not invent patient history. "
            "Return ONLY the 3-sentence recommendation."
        )

        result = call_text(
            "You are a cautious clinical decision-support assistant. "
            "Give concise, safe, patient-facing information.",
            prompt,
            max_tokens=150,
        )

        if not result or not result.strip():
            return "[LLM unavailable — empty response from provider]"

        return result.strip()

    except Exception as e:
        from llm_slm.llm_provider import friendly_error
        return (
            f"[LLM unavailable — {friendly_error(e)}]"
        )


def consistency_test(disease: str, risk_band: str, top_factor: str, n_runs: int = 3) -> Dict:
    """Runs the SLM (deterministic template — will show ~0 variance, as
    expected for a rule-based responder) and the LLM (stochastic — will
    show genuine run-to-run variance) n_runs times each, and reports a
    simple textual-similarity-based consistency score for each. This is
    the 'Consistency Testing' arm named in the project report."""
    from difflib import SequenceMatcher

    def _avg_pairwise_similarity(texts: List[str]) -> float:
        if len(texts) < 2:
            return 1.0
        scores = []
        for i in range(len(texts)):
            for j in range(i + 1, len(texts)):
                scores.append(SequenceMatcher(None, texts[i], texts[j]).ratio())
        return round(sum(scores) / len(scores), 3)

    slm_outputs = [slm_generate(disease, risk_band, top_factor) for _ in range(n_runs)]
    llm_outputs = [llm_generate(disease, risk_band, top_factor, "") for _ in range(n_runs)]

    return {
        "disease": disease, "risk_band": risk_band, "n_runs": n_runs,
        "slm": {"outputs": slm_outputs, "consistency_score": _avg_pairwise_similarity(slm_outputs)},
        "llm": {"outputs": llm_outputs, "consistency_score": _avg_pairwise_similarity(llm_outputs)},
    }


# def compare(disease: str, risk_band: str, top_factor: str, contributions_summary: str = "") -> Dict:
#     results = {}

#     for source, fn in [
#         ("SLM", lambda: slm_generate(disease, risk_band, top_factor)),
#         ("LLM", lambda: llm_generate(disease, risk_band, top_factor, contributions_summary)),
#     ]:
#         start = time.time()
#         text = fn()
#         latency_ms = round((time.time() - start) * 1000, 1)

#         validation = validate_text_recommendation(text, patient_risk_band=risk_band)
#         response = ModelResponse(
#             source=source,
#             text=validation.sanitized_text if validation.warnings else text,
#             latency_ms=latency_ms,
#             accuracy_score=_keyword_accuracy(text, disease),
#             completeness_score=_completeness(text),
#             hallucination_flags=validation.warnings,
#         )
#         results[source] = response

#     return {
#         "disease": disease, "risk_band": risk_band,
#         "slm": vars(results["SLM"]), "llm": vars(results["LLM"]),
#     }

def compare(disease: str, risk_band: str, top_factor: str, contributions_summary: str = "") -> Dict:
    results = {}

    for source, fn in [
        ("SLM", lambda: slm_generate(disease, risk_band, top_factor)),
        ("LLM", lambda: llm_generate(disease, risk_band, top_factor, contributions_summary)),
    ]:
        start = time.time()
        text = fn()
        latency_ms = round((time.time() - start) * 1000, 1)

        unavailable = (
            source == "LLM"
            and text.startswith("[LLM unavailable")
        )

        if unavailable:
            response = ModelResponse(
                source=source,
                text=text,
                latency_ms=latency_ms,
                accuracy_score=None,
                completeness_score=None,
                hallucination_flags=[],
            )
        else:
            validation = validate_text_recommendation(
                text,
                patient_risk_band=risk_band
            )

            response = ModelResponse(
                source=source,
                text=validation.sanitized_text if validation.warnings else text,
                latency_ms=latency_ms,
                accuracy_score=_keyword_accuracy(text, disease),
                completeness_score=_completeness(text),
                hallucination_flags=validation.warnings,
            )

        results[source] = response

    return {
        "disease": disease,
        "risk_band": risk_band,
        "slm": vars(results["SLM"]),
        "llm": vars(results["LLM"]),
    }
