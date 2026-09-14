import json
import sys
import unittest
from pathlib import Path

import joblib
import numpy as np


API_SERVER_ROOT = Path(__file__).resolve().parents[1]
if str(API_SERVER_ROOT) not in sys.path:
    sys.path.insert(0, str(API_SERVER_ROOT))

from backend.main import DetectorService, EXPECTED_LABEL_MAPPING  # noqa: E402
from ml.feature_extraction import (  # noqa: E402
    FEATURE_NAMES,
    extract_features_for_prediction,
    extract_handcrafted_features,
)


ARTIFACT_DIR = API_SERVER_ROOT / "ml" / "artifacts"
CHALLENGE_FIXTURE = (
    API_SERVER_ROOT / "tests" / "fixtures" / "challenge_domains.json"
)
EXPECTED_FEATURE_COUNT = 19 + 10_000


class ReorderedProbabilityModel:
    """Minimal classifier double that exposes a non-positional class order."""

    classes_ = np.array([1, 0])

    def predict(self, features):
        return np.array([1] * len(features))

    def predict_proba(self, features):
        return np.tile(np.array([[0.83, 0.17]]), (len(features), 1))


class DetectorContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.service = DetectorService()
        cls.service.load()
        cls.fixture = json.loads(CHALLENGE_FIXTURE.read_text(encoding="utf-8"))

    def test_saved_model_and_persisted_mapping_have_stable_class_contract(self):
        model = joblib.load(ARTIFACT_DIR / "model_combined.pkl")
        metrics = json.loads(
            (ARTIFACT_DIR / "model_comparison_combined.json").read_text(
                encoding="utf-8"
            )
        )
        metadata = json.loads(
            (ARTIFACT_DIR / "feature_metadata.json").read_text(encoding="utf-8")
        )

        np.testing.assert_array_equal(model.classes_, np.array([0, 1]))
        self.assertEqual(metrics["label_mapping"], EXPECTED_LABEL_MAPPING)
        self.assertEqual(metadata["label_mapping"], EXPECTED_LABEL_MAPPING)

    def test_prediction_matches_probability_argmax_using_model_classes(self):
        model = self.service.model
        vectorizer = self.service.vectorizer
        self.assertIsNotNone(model)
        self.assertIsNotNone(vectorizer)

        for category, cases in self.fixture.items():
            for case in cases:
                with self.subTest(category=category, domain=case["domain"]):
                    features = extract_features_for_prediction(
                        case["domain"], vectorizer
                    )
                    probabilities = np.asarray(model.predict_proba(features))[0]
                    argmax_label = int(
                        np.asarray(model.classes_)[int(np.argmax(probabilities))]
                    )
                    predicted_label = int(model.predict(features)[0])

                    self.assertEqual(predicted_label, argmax_label)
                    self.assertEqual(
                        self.service._predicted_label(features), argmax_label
                    )
                    result = self.service.predict(case["domain"])
                    self.assertEqual(
                        result["prediction"],
                        EXPECTED_LABEL_MAPPING[str(argmax_label)],
                    )

    def test_probability_helpers_select_values_by_class_label(self):
        service = DetectorService()
        service.model = ReorderedProbabilityModel()
        features = np.zeros((1, EXPECTED_FEATURE_COUNT))

        self.assertEqual(service._predicted_label(features), 1)
        self.assertAlmostEqual(service._malicious_probability(features), 0.83)
        self.assertAlmostEqual(service._decision_score(features), 0.66)

    def test_challenge_fixture_covers_domain_length_and_class_boundaries(self):
        expected_categories = {
            "short_dga",
            "medium_dga",
            "legitimate",
            "legitimate_short",
        }
        self.assertEqual(set(self.fixture), expected_categories)

        for category, cases in self.fixture.items():
            for case in cases:
                label = case["domain"].split(".", 1)[0]
                with self.subTest(category=category, domain=case["domain"]):
                    if category == "short_dga":
                        self.assertIn(len(label), range(4, 7))
                        self.assertEqual(case["expected_label"], 1)
                    elif category == "medium_dga":
                        self.assertIn(len(label), range(7, 16))
                        self.assertEqual(case["expected_label"], 1)
                    elif category == "legitimate_short":
                        self.assertLessEqual(len(label), 6)
                        self.assertEqual(case["expected_label"], 0)
                    else:
                        self.assertEqual(case["expected_label"], 0)

                    result = self.service.predict(case["domain"])
                    self.assertEqual(
                        result["prediction"],
                        EXPECTED_LABEL_MAPPING[str(case["expected_label"])],
                    )

    def test_inference_vector_is_handcrafted_then_tfidf(self):
        domain = self.fixture["short_dga"][0]["domain"]
        vectorizer = self.service.vectorizer
        self.assertIsNotNone(vectorizer)

        features = extract_features_for_prediction(domain, vectorizer)
        handcrafted = extract_handcrafted_features(domain)
        tfidf = vectorizer.transform([domain]).toarray()[0]

        self.assertEqual(features.shape, (1, EXPECTED_FEATURE_COUNT))
        self.assertEqual(len(FEATURE_NAMES), 19)
        self.assertEqual(tfidf.shape, (10_000,))
        np.testing.assert_array_equal(features[0, :19], handcrafted)
        np.testing.assert_allclose(features[0, 19:], tfidf)


if __name__ == "__main__":
    unittest.main()