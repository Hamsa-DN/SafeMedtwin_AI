
import io
import json
import unittest
from unittest.mock import patch

from PIL import Image
from vision.report_reader import scan_report_image


class TestReportReaderIntegration(unittest.TestCase):

    def create_test_image(self):
        image = Image.new("RGB", (100, 100), "white")
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        return buffer.getvalue()

    @patch("llm_slm.slm_transformers._get_pipeline")
    @patch("vision.report_reader.pytesseract.image_to_string")
    def test_ocr_to_medguard_safe_recommendation(
        self, mock_ocr, mock_pipeline
    ):
        mock_ocr.return_value = (
            "Glucose fasting: 126 mg/dL"
        )

        response = {
            "extracted_values": {"glucose_fasting": 126},
            "summary": "Discuss the report with your physician.",
            "legible": True
        }

        mock_pipeline.return_value = lambda prompt, **kwargs: [
            {"generated_text": prompt + json.dumps(response)}
        ]

        result = scan_report_image(self.create_test_image())

        self.assertEqual(
            result["extracted_values"]["glucose_fasting"], 126
        )
        self.assertEqual(result["read_method"], "ocr_text_extraction")
        self.assertEqual(result["safety_flags"], [])

    @patch("llm_slm.slm_transformers._get_pipeline")
    @patch("vision.report_reader.pytesseract.image_to_string")
    def test_unsafe_recommendation_is_sanitized(
        self, mock_ocr, mock_pipeline
    ):
        mock_ocr.return_value = "Patient medical report"

        response = {
            "extracted_values": {},
            "summary": "Adjust your medication without consulting a doctor.",
            "legible": True
        }

        mock_pipeline.return_value = lambda prompt, **kwargs: [
            {"generated_text": prompt + json.dumps(response)}
        ]

        result = scan_report_image(self.create_test_image())

        self.assertGreater(len(result["safety_flags"]), 0)
        self.assertNotIn(
            "Adjust your medication",
            result["summary"]
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
