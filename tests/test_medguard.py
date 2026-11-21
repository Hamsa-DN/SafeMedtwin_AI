
import unittest

from safety.medguard import (
    risk_band,
    validate_prediction,
    validate_text_recommendation,
)


class TestMedGuard(unittest.TestCase):

    # Risk band tests
    def test_low_risk(self):
        self.assertEqual(risk_band(0.20), "low")

    def test_moderate_risk(self):
        self.assertEqual(risk_band(0.50), "moderate")

    def test_high_risk(self):
        self.assertEqual(risk_band(0.80), "high")

    # Prediction validation tests
    def test_valid_prediction(self):
        result = validate_prediction(
            "diabetes", {"glucose": 100}, 0.75
        )
        self.assertTrue(result.is_safe)
        self.assertFalse(result.blocked)

    def test_invalid_probability(self):
        result = validate_prediction(
            "diabetes", {"glucose": 100}, 1.5
        )
        self.assertTrue(result.blocked)
        self.assertFalse(result.is_safe)

    def test_implausible_glucose(self):
        result = validate_prediction(
            "diabetes", {"glucose": 30}, 0.75
        )
        self.assertTrue(result.blocked)

    def test_implausible_creatinine(self):
        result = validate_prediction(
            "ckd", {"serum_creatinine": 25}, 0.50
        )
        self.assertTrue(result.blocked)

    # Recommendation validation tests
    def test_safe_recommendation(self):
        result = validate_text_recommendation(
            "Maintain a balanced diet and exercise regularly."
        )
        self.assertTrue(result.is_safe)
        self.assertEqual(result.warnings, [])

    def test_unsafe_medication_advice(self):
        result = validate_text_recommendation(
            "Adjust your medication without consulting a doctor."
        )
        self.assertFalse(result.is_safe)
        self.assertGreater(len(result.warnings), 0)
        self.assertNotIn(
            "Adjust your medication",
            result.sanitized_text
        )

    def test_high_risk_contradiction(self):
        result = validate_text_recommendation(
            "There is no concern.",
            patient_risk_band="high"
        )
        self.assertFalse(result.is_safe)
        self.assertGreater(len(result.warnings), 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
