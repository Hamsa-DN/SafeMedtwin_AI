"""
SafeMedTwin — PDF Medical Report Reader
==========================================
Reads an uploaded PDF lab report two ways, picking whichever fits:

1. Text-based PDFs (the common case — most lab reports are digitally
   generated, not scanned): extract text directly with pdfplumber, send
   the text to the configured LLM for structured extraction + a cautious
   summary.
2. Scanned/image-only PDFs (little or no extractable text): rasterize the
   first few pages to images and reuse the same vision path as
   vision/report_reader.py, sending all pages in one message.

Works with either Claude or Gemini, whichever is configured (see
llm_slm/llm_provider.py). Same safety posture as the image scanner: never
a diagnosis, MedGuard-validated summary, extracted values offered back
for the user to review rather than auto-submitted.
"""
import io
import json
import re
from typing import Dict

from safety.medguard import validate_text_recommendation
from llm_slm.llm_provider import call_text, call_vision

MIN_TEXT_CHARS = 40  # below this, treat the PDF as scanned/image-only

TEXT_SYSTEM_PROMPT = """You are a careful medical-report reading assistant embedded in a clinical \
decision-support tool. You are NOT a diagnostic system.

Given the raw extracted text of a lab report, do two things:
1. Extract any clearly stated lab values (e.g. glucose, cholesterol, creatinine, hemoglobin, \
blood pressure, bilirubin, etc.) as a JSON object under the key "extracted_values" — only include \
values actually present in the text; do not guess or invent numbers.
2. Write a short, cautious, plain-language "summary" (3-5 sentences) describing what the report \
appears to show in general terms. Do NOT state or imply a diagnosis. Do NOT recommend medications \
or dosages. Always end the summary by recommending the patient discuss the full report with their \
physician.

Respond ONLY with valid JSON in this exact shape, nothing else:
{"extracted_values": {...}, "summary": "...", "legible": true}"""

VISION_SYSTEM_PROMPT = """You are a careful medical-report reading assistant embedded in a clinical \
decision-support tool. You are NOT a diagnostic system.

Given page images of a scanned lab report, do two things:
1. Extract any clearly legible lab values as a JSON object under "extracted_values" — only include \
values you can actually read; do not guess or invent numbers.
2. Write a short, cautious, plain-language "summary" (3-5 sentences). Do NOT state or imply a \
diagnosis. Do NOT recommend medications or dosages. Always end by recommending the patient discuss \
the full report with their physician.

Respond ONLY with valid JSON in this exact shape, nothing else:
{"extracted_values": {...}, "summary": "...", "legible": true|false}

If the pages are too illegible to read, set "legible": false and explain briefly in "summary", \
with an empty extracted_values object."""


def _clean_json(raw_text: str) -> Dict:
    """Robustly pull a JSON object out of a model response — handles code
    fences anywhere in the text and explanatory preamble/postamble text
    around the JSON, which Gemini in particular tends to add even when
    told not to."""
    text = re.sub(r"```(?:json)?", "", raw_text).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return json.loads(text[start:end + 1])  # let this raise; caller catches
    raise json.JSONDecodeError("No JSON object found in response", text, 0)


def _validate_and_finish(parsed: Dict) -> Dict:
    summary = parsed.get("summary", "")
    validation = validate_text_recommendation(summary, patient_risk_band=None)
    parsed["summary"] = validation.sanitized_text if validation.warnings else summary
    parsed["safety_flags"] = validation.warnings
    return parsed


def scan_report_pdf(pdf_bytes: bytes, max_scanned_pages: int = 3) -> Dict:
    try:
        import pdfplumber
    except ImportError:
        return {"error": "The 'pdfplumber' package is not installed. Run: pip install pdfplumber"}

    # ---- 1. try text extraction first (fast, cheap, no vision tokens) ----
    try:
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            text_parts = [page.extract_text() or "" for page in pdf.pages]
        full_text = "\n".join(text_parts).strip()
    except Exception as e:
        return {"error": f"Could not open this PDF: {e}"}

    raw_text = None
    try:
        # if len(full_text) >= MIN_TEXT_CHARS:
        #     raw_text = call_text(TEXT_SYSTEM_PROMPT, f"Report text:\n\n{full_text[:8000]}", max_tokens=900)
        #     parsed = _clean_json(raw_text)
        #     parsed["read_method"] = "text_extraction"
        #     return _validate_and_finish(parsed)
        if len(full_text) >= MIN_TEXT_CHARS:
            try:
                raw_text = call_text(
                    TEXT_SYSTEM_PROMPT,
                    f"Report text:\n\n{full_text[:8000]}",
                    max_tokens=900,
                )
                parsed = _clean_json(raw_text)
                parsed["read_method"] = "text_extraction"
                return _validate_and_finish(parsed)

            except Exception:
                return {
                    "extracted_values": {},
                    "summary": (
                        "The PDF text was extracted locally, but automated "
                        "interpretation is currently unavailable. No medical "
                        "interpretation or risk assessment was performed. "
                        "Review the report with your physician.\n\n"
                        "Extracted report text (not interpreted):\n"
                        + full_text[:2500]
                    ),
                    "legible": True,
                    "read_method": "local_text_extraction",
                    "safety_flags": [],
                }

        # ---- 2. fall back to rasterizing pages for scanned/image PDFs ----
        try:
            import pypdfium2 as pdfium
        except ImportError:
            return {"error": "This PDF has little extractable text (likely scanned) and "
                              "'pypdfium2' isn't installed for image fallback. Run: pip install pypdfium2"}

        doc = pdfium.PdfDocument(io.BytesIO(pdf_bytes))
        images = []
        for i in range(min(len(doc), max_scanned_pages)):
            bitmap = doc[i].render(scale=2.0)
            pil_image = bitmap.to_pil()
            buf = io.BytesIO()
            pil_image.save(buf, format="PNG")
            images.append((buf.getvalue(), "image/png"))

        raw_text = call_vision(
            VISION_SYSTEM_PROMPT,
            "Read these scanned report pages and respond with the JSON format specified.",
            images=images, max_tokens=900,
        )
        parsed = _clean_json(raw_text)
        parsed["read_method"] = "vision_fallback"
        return _validate_and_finish(parsed)

    except json.JSONDecodeError:
        return {"error": "Could not parse a structured response from the model.", "raw": raw_text}
    except Exception as e:
        return {"error": f"PDF report scan failed: {e}"}
