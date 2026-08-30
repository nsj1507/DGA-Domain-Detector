from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from fastapi import APIRouter, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict, Field

SERVICE_ROOT = Path(__file__).resolve().parent.parent
if str(SERVICE_ROOT) not in sys.path:
    sys.path.insert(0, str(SERVICE_ROOT))

from ml.feature_extraction import (  # noqa: E402
    FEATURE_NAMES,
    extract_features_for_prediction,
)

MODEL_DIR = SERVICE_ROOT / "ml" / "artifacts"
MODEL_FILE = MODEL_DIR / "model_combined.pkl"
VECTOR_FILE = MODEL_DIR / "tfidf_vectorizer.pkl"
METRICS_FILE = MODEL_DIR / "model_comparison_combined.json"
METADATA_FILE = MODEL_DIR / "feature_metadata.json"

DOMAIN_PATTERN = re.compile(
    r"^[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?"
    r"(?:\.[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?)+$"
)


class DomainPredictionInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    domain: str = Field(min_length=1, max_length=253)


class DetectorService:
    def __init__(self) -> None:
        self.model: Any | None = None
        self.vectorizer: Any | None = None
        self.metrics: dict[str, Any] | None = None
        self.metadata: dict[str, Any] | None = None

    def load(self) -> None:
        if self.model is not None:
            return
        if not MODEL_FILE.exists() or not VECTOR_FILE.exists():
            raise HTTPException(
                status_code=503,
                detail="The combined detector model is still being prepared.",
            )
        self.model = joblib.load(MODEL_FILE)
        self.vectorizer = joblib.load(VECTOR_FILE)
        self.metrics = json.loads(METRICS_FILE.read_text(encoding="utf-8"))
        self.metadata = json.loads(METADATA_FILE.read_text(encoding="utf-8"))

    def model_info(self) -> dict[str, Any]:
        self.load()
        assert self.metrics is not None
        assert self.metadata is not None
        model_name = self.metrics["model_name"]
        return {
            "model_name": model_name,
            "feature_count": self.metrics["feature_count"],
            "handcrafted_feature_count": self.metrics[
                "handcrafted_feature_count"
            ],
            "ngram_feature_count": self.metrics["ngram_feature_count"],
            "vectorizer": self.metadata["vectorizer"],
            "metrics": {
                key: self.metrics["results"][model_name][key]
                for key in (
                    "accuracy",
                    "precision",
                    "recall",
                    "f1_score",
                    "false_positive_rate",
                    "false_negative_rate",
                )
            },
            "explainability_method": self._explainability_method(),
            "handcrafted_feature_names": FEATURE_NAMES,
        }

    def predict(self, original_domain: str) -> dict[str, Any]:
        self.load()
        assert self.model is not None
        assert self.vectorizer is not None
        assert self.metrics is not None

        normalized = original_domain.lower().strip()
        if not DOMAIN_PATTERN.fullmatch(normalized):
            raise HTTPException(
                status_code=400,
                detail=(
                    "Enter a valid domain such as example.com. "
                    "Protocols, paths, spaces, and underscores are not accepted."
                ),
            )

        features = extract_features_for_prediction(normalized, self.vectorizer)
        expected_count = int(self.metrics["feature_count"])
        if features.shape != (1, expected_count):
            raise HTTPException(
                status_code=503,
                detail=(
                    f"Feature pipeline produced {features.shape[1]} values; "
                    f"the model requires {expected_count}."
                ),
            )

        predicted_label = int(self.model.predict(features)[0])
        decision_score = self._decision_score(features)
        confidence = self._confidence(features)
        prediction = "Malicious" if predicted_label == 1 else "Legitimate"
        feature_values = {
            name: float(value)
            for name, value in zip(FEATURE_NAMES, features[0][: len(FEATURE_NAMES)])
        }

        return {
            "domain": original_domain,
            "normalized_domain": normalized,
            "prediction": prediction,
            "confidence": confidence,
            # The selected MLP exposes probabilities, but no calibration
            # artifact was trained, so these remain raw model probabilities.
            "confidence_is_calibrated": False,
            "decision_score": decision_score,
            "risk_level": "High" if predicted_label == 1 else "Low",
            "reasons": self._reasons(features, predicted_label),
            "explainability_method": self._explainability_method(),
            "features": feature_values,
            "feature_count": expected_count,
            "model_name": self.metrics["model_name"],
        }

    def _decision_score(self, features: np.ndarray) -> float | None:
        assert self.model is not None
        if hasattr(self.model, "decision_function"):
            value = np.asarray(self.model.decision_function(features)).reshape(-1)[0]
            return float(value)
        if hasattr(self.model, "predict_proba"):
            probabilities = np.asarray(self.model.predict_proba(features))[0]
            classes = np.asarray(getattr(self.model, "classes_", [0, 1]))
            malicious_positions = np.flatnonzero(classes == 1)
            if malicious_positions.size:
                malicious_probability = probabilities[malicious_positions[0]]
                return float(2 * malicious_probability - 1)
        return None

    def _confidence(self, features: np.ndarray) -> float | None:
        assert self.model is not None
        if not hasattr(self.model, "predict_proba"):
            return None
        probabilities = np.asarray(self.model.predict_proba(features))[0]
        return float(np.max(probabilities))

    def _explainability_method(self) -> str:
        assert self.model is not None
        if hasattr(self.model, "coef_"):
            return "Local additive contributions from the LinearSVM coefficient vector."
        if hasattr(self.model, "feature_importances_"):
            return (
                "Global feature importance is available, but no per-domain "
                "causal attribution is claimed."
            )
        return (
            "No per-domain attribution is available for the selected model; "
            "handcrafted values are shown as analysis metadata."
        )

    def _reasons(self, features: np.ndarray, predicted_label: int) -> list[str]:
        assert self.model is not None
        if not hasattr(self.model, "coef_"):
            return []

        coefficients = np.asarray(self.model.coef_)[0][: len(FEATURE_NAMES)]
        contributions = features[0][: len(FEATURE_NAMES)] * coefficients
        ranked = np.argsort(contributions)
        ordered = ranked[::-1] if predicted_label == 1 else ranked
        direction = "malicious" if predicted_label == 1 else "legitimate"
        reasons = []
        for index in ordered[:3]:
            contribution = float(contributions[index])
            if abs(contribution) < 1e-6:
                continue
            reasons.append(
                f"{FEATURE_NAMES[index].replace('_', ' ')} had a "
                f"{'positive' if contribution > 0 else 'negative'} local "
                f"contribution toward the {direction} classification "
                f"({contribution:.3f})."
            )
        return reasons


service = DetectorService()
app = FastAPI(title="DGA Domain Detector API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

router = APIRouter(prefix="/api")


@router.get("/healthz")
def healthz() -> dict[str, str]:
    return {"status": "ok"}


@router.post("/predict")
def predict_domain(payload: DomainPredictionInput) -> dict[str, Any]:
    return service.predict(payload.domain)


@router.get("/model-info")
def model_info() -> dict[str, Any]:
    return service.model_info()


app.include_router(router)