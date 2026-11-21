"""
SafeMedTwin — Health Chatbot
==============================
A conversational assistant that has access to the logged-in patient's
latest risk assessments (probabilities, risk bands, top SHAP factors) as
context, so it can answer questions like "why is my CVD risk high?" or
"what should I focus on first?" grounded in their actual data rather than
generic advice. Every response is passed through MedGuard before display.

This is a support tool, not a clinician — the system prompt and MedGuard
both enforce that framing (no diagnoses, no dosages, always defer specifics
to a physician).
"""
from typing import Dict, List, Optional
from safety.medguard import validate_text_recommendation
from llm_slm.llm_provider import call_chat

SYSTEM_PROMPT_TEMPLATE = """You are the SafeMedTwin health assistant, embedded in a clinical \
decision-support dashboard. You help the patient understand their own computed risk assessments \
and general preventive health concepts.

Hard rules:
- Never state or imply a diagnosis ("you have X") — only discuss computed RISK levels from the \
tool's models, and always note these are statistical estimates, not medical diagnoses.
- Never give specific drug names, dosages, or medication instructions.
- Never tell a patient to stop or start a medication.
- For anything beyond general lifestyle/education, recommend they consult their physician.
- If asked something outside your scope (unrelated to this patient's health data or general \
health education), say so plainly and redirect.

Here is this patient's current data from the SafeMedTwin system (may be partial if they haven't \
assessed every disease yet):

{patient_context}

Answer the patient's questions conversationally, grounded in the data above where relevant. Keep \
responses concise (typically under 150 words) unless they ask for more detail."""


def _format_patient_context(disease_results: Dict) -> str:
    if not disease_results:
        return "No assessments have been run yet for this patient."
    lines = []
    for disease, r in disease_results.items():
        pct = round(r.get("probability", 0) * 100)
        band = r.get("risk_band", "unknown")
        lines.append(f"- {disease}: {pct}% predicted risk ({band} risk band)")
    return "\n".join(lines)


def chat(user_message: str, disease_results: Dict, conversation_history: Optional[List[Dict]] = None) -> str:
    """conversation_history: list of {"role": "user"|"assistant", "content": str}, most recent last.
    Returns the MedGuard-validated assistant reply text (never raises — returns an error string)."""
    try:
        system_prompt = SYSTEM_PROMPT_TEMPLATE.format(
            patient_context=_format_patient_context(disease_results)
        )
        messages = list(conversation_history or [])
        messages.append({"role": "user", "content": user_message})

        text = call_chat(system_prompt, messages, max_tokens=400)

        # use the worst risk band present as the validation context, so MedGuard
        # is appropriately strict if the reply understates a high-risk finding
        bands_present = [r.get("risk_band") for r in disease_results.values()]
        worst_band = "high" if "high" in bands_present else ("moderate" if "moderate" in bands_present else "low")
        validation = validate_text_recommendation(text, patient_risk_band=worst_band)
        return validation.sanitized_text if validation.warnings else text

    except Exception as e:
        from llm_slm.llm_provider import friendly_error
        return f"Sorry — {friendly_error(e)}"
