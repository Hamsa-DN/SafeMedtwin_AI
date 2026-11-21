
import unittest
from unittest.mock import patch

from vision.report_reader import (
    _media_type_from_bytes,
    _extract_json,
    scan_report_image,
)


class TestReportReader(unittest.TestCase):

    # Image format detection
    def test_png_detection(self):
        self.assertEqual(
            _media_type_from_bytes(b"\x89PNG\r\n\x1a\n"),
            "image/png"
        )

    def test_jpeg_detection(self):
        self.assertEqual(
            _media_type_from_bytes(b"\xff\xd8\xff"),
            "image/jpeg"
        )

    def test_gif_detection(self):
        self.assertEqual(
            _media_type_from_bytes(b"GIF89a"),
            "image/gif"
        )

    def test_webp_detection(self):
        data = b"RIFF" + b"\x00\x00\x00\x00" + b"WEBP"
        self.assertEqual(
            _media_type_from_bytes(data),
            "image/webp"
        )

    # JSON extraction
    def test_valid_json(self):
        raw = '{"extracted_values": {}, "summary": "Report reviewed", "legible": true}'
        result = _extract_json(raw)
        self.assertTrue(result["legible"])

    def test_json_inside_markdown_fence(self):
        raw = '```json\n{"summary": "Report reviewed"}\n```'
        result = _extract_json(raw)
        self.assertEqual(result["summary"], "Report reviewed")

    def test_json_with_extra_text(self):
        raw = 'Here is the result: {"summary": "Report reviewed"} Hope this helps.'
        result = _extract_json(raw)
        self.assertEqual(result["summary"], "Report reviewed")

    # OCR processing
    @patch("vision.report_reader.pytesseract.image_to_string")
    @patch("vision.report_reader.Image.open")
    def test_empty_ocr_text(self, mock_image, mock_ocr):
        mock_image.return_value = object()
        mock_ocr.return_value = ""

        result = scan_report_image(b"test image")

        self.assertIn("error", result)
        self.assertIn("readable text", result["error"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
