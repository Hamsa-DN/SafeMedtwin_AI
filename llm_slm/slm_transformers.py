"""
SafeMedTwin — Local Transformers SLM Backend
===============================================
A real Hugging Face `transformers` backend for the SLM side of the LLM-vs-
SLM comparison, as an in-place swap for the template stand-in in
compare.py. Not usable in the build sandbox (no huggingface.co access
there) — this is meant to run on your own machine, which has normal
internet access.

Setup (on your machine, not the sandbox):
    pip install transformers torch --upgrade

Recommended models (small enough for CPU, still genuinely generative):
    - "TinyLlama/TinyLlama-1.1B-Chat-v1.0"   (~1.1B params, chat-tuned, best fit)
    - "microsoft/Phi-3-mini-4k-instruct"      (~3.8B, stronger but heavier)
    - "distilgpt2"                            (~82M, fastest, weakest)

Note on DistilBERT specifically: it's an *encoder* (BERT-style), built for
classification/embeddings, not text generation — it can't directly write
the free-text recommendations this module produces. Your report lists it
as an SLM option alongside TinyLlama; in practice, use TinyLlama (or
another small causal/generative model) for the free-text arm, and
DistilBERT would fit a *different* role — e.g. fine-tuned as a fast risk-
category classifier. If you want that second path built out (a DistilBERT
classifier trained on the SHAP top-factors -> risk-band mapping), let me
know and I'll add it as an additional comparison arm.

Usage: set the environment variable before running the app:
    export SAFEMEDTWIN_SLM_BACKEND=transformers
    export SAFEMEDTWIN_SLM_MODEL=TinyLlama/TinyLlama-1.1B-Chat-v1.0
compare.py will then route slm_generate() through here automatically.
"""
import os
import threading
import re

_MODEL_NAME = os.environ.get("SAFEMEDTWIN_SLM_MODEL", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")

_pipeline = None
_pipeline_lock = threading.Lock()


# def _get_pipeline():
#     global _pipeline
#     if _pipeline is None:
#         with _pipeline_lock:
#             if _pipeline is None:  # re-check inside the lock
#                 from transformers import pipeline
#                 _pipeline = pipeline(
#                     "text-generation",
#                     model=_MODEL_NAME,
#                     device_map="auto",  # uses GPU if available, else CPU
#                 )
#     return _pipeline

def _get_pipeline():
    global _pipeline
    if _pipeline is None:
        with _pipeline_lock:
            if _pipeline is None:
                from transformers import pipeline

                _pipeline = pipeline(
                    "text-generation",
                    model=_MODEL_NAME,
                    device_map="auto",
                )

                # TinyLlama does not define a separate pad token.
                # Use EOS as the padding token for controlled generation.
                if _pipeline.tokenizer.pad_token_id is None:
                    _pipeline.tokenizer.pad_token_id = _pipeline.tokenizer.eos_token_id
                if getattr(_pipeline.model.generation_config, "max_length", None) is not None:
                    _pipeline.model.generation_config.max_length = None

    return _pipeline


def is_available() -> bool:
    """Cheap check for whether the transformers backend can even be attempted
    (package installed) — does not guarantee the model weights will download
    successfully (needs internet on first run)."""
    try:
        import transformers  # noqa: F401
        return True
    except ImportError:
        return False


# def slm_generate_transformers(disease: str, risk_band: str, top_factor: str) -> str:
#     """Real local-model generation, matching the signature of
#     llm_slm.compare.slm_generate() so it's a drop-in replacement."""
#     pipe = _get_pipeline()
#     prompt = (
#         f"<|system|>\nYou are a cautious health assistant. Never give drug names or dosages, "
#         f"never diagnose. Keep it to 3 sentences.</s>\n"
#         f"<|user|>\nA patient has {risk_band} risk of {disease}, mainly driven by {top_factor}. "
#         f"Write a brief, safe, encouraging summary and suggest they follow up with a physician.</s>\n"
#         f"<|assistant|>\n"
#     )
#     result = pipe(prompt, max_new_tokens=120, do_sample=True, temperature=0.7, top_p=0.9)
#     generated = result[0]["generated_text"]
#     # strip the prompt prefix, keep only the newly generated continuation
#     return generated[len(prompt):].strip() if generated.startswith(prompt) else generated.strip()
# def slm_generate_transformers(disease: str, risk_band: str, top_factor: str) -> str:
#     """Generate a short patient-facing recommendation using TinyLlama."""

#     pipe = _get_pipeline()

#     prompt = f"""<|system|>
# You are a cautious clinical decision-support assistant.

# Your task is to write ONLY a short medical risk recommendation.

# Rules:
# - Do NOT write a dialogue.
# - Do NOT use "Patient:" or "Physician:".
# - Do NOT introduce yourself.
# - Do NOT invent patient history.
# - Do NOT diagnose.
# - Do NOT recommend medicines or dosages.
# - Mention the risk level.
# - Mention the main contributing factor.
# - Recommend appropriate follow-up with a physician.
# - Write exactly 3 short sentences.
# - Output ONLY the recommendation text.

# <|user|>
# The patient's calculated risk of {disease} is {risk_band}.
# The main contributing factor is {top_factor}.

# Write the 3-sentence recommendation now.

# <|assistant|>"""

#     result = pipe(
#         prompt,
#         max_new_tokens=100,
#         do_sample=False,
#         return_full_text=False,
#     )

#     generated = result[0]["generated_text"].strip()

#     # Remove accidental dialogue markers if the model still produces them.
#     generated = re.sub(
#         r"^(Patient|Physician|Assistant)\s*:\s*",
#         "",
#         generated,
#         flags=re.IGNORECASE,
#     )

#     return generated


def slm_generate_transformers(disease: str, risk_band: str, top_factor: str) -> str:
    """Generate a short, focused, patient-facing recommendation using the
    local TinyLlama model."""

    pipe = _get_pipeline()

    prompt = (
        "<|system|>\n"
        "You are a concise medical decision-support assistant. "
        "Write exactly 3 short sentences. "
        "Do not diagnose the patient. "
        "Do not mention medicines, drugs, or dosages. "
        "Do not invent patient information. "
        "State the risk level, identify the main contributing factor, "
        "and recommend appropriate follow-up with a physician. "
        # "Do not begin with phrases such as 'As a health assistant'. "
        "Start directly with the patient's risk assessment. "
        "Do not include disclaimers, introductions, or statements about what you cannot do. " 
        "</s>\n"
        "<|user|>\n"
        f"Disease: {disease}\n"
        f"Risk level: {risk_band}\n"
        f"Main contributing factor: {top_factor}\n"
        "Generate the 3-sentence patient-facing recommendation now."
        "</s>\n"
        "<|assistant|>\n"
    )

    result = pipe(
        prompt,
        max_new_tokens=80,
        do_sample=False,
        return_full_text=False,
        pad_token_id=pipe.tokenizer.eos_token_id,
    )

    generated = result[0]["generated_text"].strip()

    # Remove any accidental chat markers produced by the model.
    generated = generated.replace("<|assistant|>", "").strip()
    generated = generated.replace("<|user|>", "").strip()
    generated = generated.replace("<|system|>", "").strip()

    return generated