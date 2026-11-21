"""
SafeMedTwin — Medical Report Image Scanner
=============================================
Accepts an uploaded image of a lab report / medical document and uses
Claude's vision capability to (1) extract structured lab values where
legible, and (2) produce a cautious, plain-language summary of what's
in the report. This is explicitly NOT a diagnosis — the prompt and the
MedGuard pass both enforce that framing, and extracted values are
offered back to the user to optionally feed into the prediction form
rather than auto-submitted.

Works with either Claude or Gemini, whichever is configured (see
llm_slm/llm_provider.py). If neither is available, returns a clear error
rather than fabricating results.
"""
# import json
# import re
# from typing import Dict
import io
import json
import re
from typing import Dict

from PIL import Image
import pytesseract

from safety.medguard import validate_text_recommendation
# from llm_slm.llm_provider import call_vision
from llm_slm.llm_provider import call_text, call_vision

SYSTEM_PROMPT = """You are a careful medical-report reading assistant embedded in a clinical \
decision-support tool. You are NOT a diagnostic system.

Given an image of a medical/lab report, do two things:
1. Extract any clearly legible lab values (e.g. glucose, cholesterol, creatinine, hemoglobin, \
blood pressure, bilirubin, etc.) as a JSON object under the key "extracted_values" — only include \
values you can actually read; do not guess or invent numbers.
2. Write a short, cautious, plain-language "summary" (3-5 sentences) describing what the report \
appears to show in general terms (e.g. "several values are within normal range" or "glucose \
appears elevated relative to the reference range shown"). Do NOT state or imply a diagnosis. Do \
NOT recommend medications or dosages. Always end the summary by recommending the patient discuss \
the full report with their physician.

Respond ONLY with valid JSON in this exact shape, nothing else:
{"extracted_values": {...}, "summary": "...", "legible": true|false}

If the image is not a medical report or is too illegible to read, set "legible": false and explain \
briefly in "summary" why, with an empty extracted_values object."""


def _media_type_from_bytes(data: bytes) -> str:
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png"
    if data[:3] == b"\xff\xd8\xff":
        return "image/jpeg"
    if data[:6] in (b"GIF87a", b"GIF89a"):
        return "image/gif"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    return "image/png"  # sane default; Streamlit's uploader mostly gives png/jpeg


def _extract_json(raw_text: str) -> Dict:
    """Robustly pull a JSON object out of a model response. Handles the
    common failure modes: markdown code fences (anywhere, not just at the
    exact start/end of the string), and explanatory text before/after the
    JSON (Gemini in particular tends to add a sentence of preamble even
    when told not to)."""
    text = re.sub(r"```(?:json)?", "", raw_text).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # fall back: grab everything between the first { and the last } and
    # try that — handles "Here's the JSON: {...} Let me know if..." style
    # wrapping that a simple fence-strip won't catch.
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return json.loads(text[start:end + 1])  # let this raise; caller catches
    raise json.JSONDecodeError("No JSON object found in response", text, 0)


# def scan_report_image(image_bytes: bytes) -> Dict:
#     try:
#         media_type = _media_type_from_bytes(image_bytes)
#         raw_text = call_vision(
#             SYSTEM_PROMPT,
#             "Read this medical report image and respond with the JSON format specified.",
#             images=[(image_bytes, media_type)],
#             max_tokens=900,
#         )

#         if not raw_text or not raw_text.strip():
#             return {"error": "The model returned an empty response — this can happen if the image "
#                               "was flagged by safety filters (medical images sometimes are) or was "
#                               "too large/unclear. Try a clearer, cropped photo of just the report."}

#         try:
#             parsed = _extract_json(raw_text)
#         except json.JSONDecodeError:
#             return {"error": "Could not parse a structured response from the model.", "raw": raw_text}

#         summary = parsed.get("summary", "")
#         validation = validate_text_recommendation(summary, patient_risk_band=None)
#         parsed["summary"] = validation.sanitized_text if validation.warnings else summary
#         parsed["safety_flags"] = validation.warnings
#         return parsed

#     # except Exception as e:
#     #     return {"error": f"Report scan failed: {e}"}
#     except Exception as e:
#         return {
#             "error": (
#                 "Automated image interpretation is temporarily unavailable. "
#                 "The AI vision service is currently busy or unavailable. "
#                 "No medical interpretation was performed. "
#                 "Please try again later or upload a text-based PDF."
#             )
#         }

def scan_report_image(image_bytes: bytes) -> Dict:
    """
    Read a medical report image using local OCR first, then use the
    configured text LLM to interpret the extracted text.

    This avoids depending on the LLM's vision endpoint for image reading.
    """

    try:
        from PIL import Image
        import io
        import pytesseract

        # 1. Convert uploaded bytes into an image
        image = Image.open(io.BytesIO(image_bytes))

        # 2. Extract text locally using Tesseract OCR
        extracted_text = pytesseract.image_to_string(image).strip()

        if not extracted_text:
            return {
                "error": (
                    "Could not extract readable text from this report image. "
                    "Try uploading a clearer image with better lighting/resolution."
                )
            }

        # 3. Send ONLY the extracted text to the normal text LLM.
        #    This avoids the Gemini vision endpoint that was returning 503.
        # raw_text = call_text(
        #     # TEXT_SYSTEM_PROMPT,
        #     SYSTEM_PROMPT,
        #     f"Report text extracted using OCR:\n\n{extracted_text[:8000]}",
        #     max_tokens=900,
        # )
                # 3. Send the OCR text directly to the local TinyLlama model.
        from llm_slm.slm_transformers import _get_pipeline

        pipe = _get_pipeline()

        prompt = f"""<|system|>
You are a careful medical report reading assistant.
You are NOT a diagnostic system.

Read the OCR text from a medical/lab report.

Extract only values that are explicitly present.
Do not invent, estimate, or guess any values.

Return ONLY valid JSON in exactly this format:
{{"extracted_values": {{}}, "summary": "...", "legible": true}}

The summary must be cautious.
Do not diagnose the patient.
Do not recommend medicines or dosages.

<|user|>
OCR text from the report:

{extracted_text[:6000]}

<|assistant|>
"""

        result = pipe(
            prompt,
            max_new_tokens=250,
            do_sample=False,
        )

        raw_text = result[0]["generated_text"]

        if raw_text.startswith(prompt):
            raw_text = raw_text[len(prompt):].strip()

        # 4. Convert the model response into our expected JSON structure
        # try:
        #     parsed = _extract_json(raw_text)
        # except json.JSONDecodeError:
        #     return {
        #         "error": "Could not parse a structured response from the model.",
        #         "raw": raw_text,
        #     }
                # 4. Convert the model response into our expected JSON structure.
        # TinyLlama may sometimes return useful plain text instead of
        # perfectly formatted JSON, so try JSON first and then fall back
        # to a safe text-based result.
        try:
            parsed = _extract_json(raw_text)

        # except json.JSONDecodeError:
        #     parsed = {
        #         "extracted_values": {},
        #         "summary": raw_text.strip(),
        #         "legible": True,
        #     }
        except json.JSONDecodeError:
            parsed = {
                "extracted_values": {},
                "summary": raw_text.strip(),
                "legible": True,
            }

            # Extract common laboratory values from TinyLlama's response.
            value_patterns = {
                "glucose_fasting": r"GLUCOSE,\s*FASTING[,\s:]+(\d+(?:\.\d+)?)",
                "hemoglobin_a1c": r"HEMOGLOBIN\s*A1C[,\s:]+(\d+(?:\.\d+)?)",
                "creatinine": r"CREATININE[,\s:]+(\d+(?:\.\d+)?)",
                "total_cholesterol": r"TOTAL\s+CHOLESTEROL[,\s:]+(\d+(?:\.\d+)?)",
                "ldl_cholesterol": r"LDL\s+CHOLESTEROL[,\s:]+(\d+(?:\.\d+)?)",
                "triglycerides": r"TRIGLYCERIDES[,\s:]+(\d+(?:\.\d+)?)",
            }

            for key, pattern in value_patterns.items():
                match = re.search(pattern, raw_text, re.IGNORECASE)

                if match:
                    parsed["extracted_values"][key] = float(match.group(1))

        # 5. Mark how the report was processed
        parsed["read_method"] = "ocr_text_extraction"

        # 6. Apply the existing MedGuard safety validation
        summary = parsed.get("summary", "")
        validation = validate_text_recommendation(
            summary,
            patient_risk_band=None
        )

        parsed["summary"] = (
            validation.sanitized_text
            if validation.warnings
            else summary
        )
        parsed["safety_flags"] = validation.warnings

        return parsed

    except Exception as e:
        return {
            "error": f"Report OCR scan failed: {e}"
        }