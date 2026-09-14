"""Train the DGA detector on the intended 10,019-feature representation.

The representation is deliberately built here from the raw train/test domain
files instead of relying on the previous handcrafted CSV export. That keeps
each feature row aligned with its domain and guarantees that prediction uses
the same feature order.
"""

from __future__ import annotations

import json
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC

from feature_extraction import (
    FEATURE_NAMES,
    extract_handcrafted_features,
)

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
ARTIFACT_DIR = ROOT / "artifacts"
VECTOR_FILE = ARTIFACT_DIR / "tfidf_vectorizer.pkl"
MODEL_FILE = ARTIFACT_DIR / "model_combined.pkl"
METRICS_FILE = ARTIFACT_DIR / "model_comparison_combined.json"
METADATA_FILE = ARTIFACT_DIR / "feature_metadata.json"

RANDOM_STATE = 42
NGRAM_FEATURE_COUNT = 10_000
LABEL_MAPPING = {"0": "Legitimate", "1": "Malicious"}
SUPPLEMENTAL_FILE = DATA_DIR / "verified_short_dga.csv"


def load_training_data(
    train: pd.DataFrame,
    test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load verified supplemental rows while enforcing leakage invariants."""
    if not SUPPLEMENTAL_FILE.exists():
        raise FileNotFoundError(
            f"Verified supplemental data is required at {SUPPLEMENTAL_FILE}"
        )
    supplemental = pd.read_csv(SUPPLEMENTAL_FILE)
    required_columns = {
        "domain",
        "label",
        "source_dataset",
        "source_threat",
        "source_label",
        "source_length",
        "length_band",
    }
    if not required_columns.issubset(supplemental.columns):
        raise ValueError(
            f"Supplemental data is missing columns: "
            f"{sorted(required_columns - set(supplemental.columns))}"
        )

    train_rows = train[["domain", "label"]].copy()
    supplemental_rows = supplemental[["domain", "label"]].copy()
    combined = pd.concat([train_rows, supplemental_rows], ignore_index=True)
    combined["domain"] = combined["domain"].astype(str).str.lower().str.strip()
    combined["label"] = combined["label"].astype(int)
    normalized_test = test["domain"].astype(str).str.lower().str.strip()

    if set(combined["label"].unique()) != {0, 1}:
        raise ValueError("Combined training data must contain labels 0 and 1.")
    if combined["domain"].duplicated().any():
        raise ValueError("Duplicate domains exist in the combined training data.")
    if set(combined["domain"]) & set(normalized_test):
        raise ValueError("Training and test domains overlap after normalization.")
    if not set(supplemental["label"].astype(int).unique()) == {1}:
        raise ValueError("Verified supplemental rows must all be labeled DGA (1).")

    return combined, supplemental


def handcrafted_matrix(domains: pd.Series) -> np.ndarray:
    """Apply the teammate-provided extractor to every domain."""
    return np.vstack(
        [extract_handcrafted_features(domain) for domain in domains]
    ).astype(np.float64, copy=False)


def build_features(domains: pd.Series, vectorizer) -> sp.csr_matrix:
    """Build [19 handcrafted | 10,000 TF-IDF] features in that exact order."""
    handcrafted = sp.csr_matrix(handcrafted_matrix(domains))
    ngrams = vectorizer.transform(domains.astype(str).str.lower().str.strip())
    if ngrams.shape[1] != NGRAM_FEATURE_COUNT:
        raise ValueError(
            f"Expected {NGRAM_FEATURE_COUNT} n-gram features, got {ngrams.shape[1]}"
        )
    return sp.hstack([handcrafted, ngrams], format="csr")


def registered_label_length(domain: str) -> int:
    """Use the same host-label convention for all evaluation buckets."""
    return len(str(domain).lower().strip().rsplit(".", 1)[0])


def evaluate_length_bands(
    domains: pd.Series,
    labels: np.ndarray,
    predictions: np.ndarray,
) -> dict[str, dict[str, float | int | None]]:
    """Report DGA recall/FNR and benign FPR by registered-label length."""
    lengths = domains.map(registered_label_length).to_numpy()
    bands = {
        "le_10": lengths <= 10,
        "11_12": (lengths >= 11) & (lengths <= 12),
        "13_15": (lengths >= 13) & (lengths <= 15),
        "gt_15": lengths > 15,
    }
    report: dict[str, dict[str, float | int | None]] = {}
    for name, band_mask in bands.items():
        dga_mask = band_mask & (labels == 1)
        benign_mask = band_mask & (labels == 0)
        dga_count = int(dga_mask.sum())
        benign_count = int(benign_mask.sum())
        dga_recall = (
            float((predictions[dga_mask] == 1).mean())
            if dga_count
            else None
        )
        benign_fpr = (
            float((predictions[benign_mask] == 1).mean())
            if benign_count
            else None
        )
        report[name] = {
            "dga_count": dga_count,
            "dga_recall": dga_recall,
            "dga_false_negative_rate": (
                float(1.0 - dga_recall) if dga_recall is not None else None
            ),
            "legitimate_count": benign_count,
            "legitimate_false_positive_rate": benign_fpr,
        }
    return report


def evaluate(name: str, model, X_test, y_test) -> dict:
    started = time.perf_counter()
    predictions = model.predict(X_test)
    inference_time = time.perf_counter() - started
    tn, fp, fn, tp = confusion_matrix(
        y_test, predictions, labels=[0, 1]
    ).ravel()
    negatives = tn + fp
    positives = fn + tp
    metrics = {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision": float(precision_score(y_test, predictions, zero_division=0)),
        "recall": float(recall_score(y_test, predictions, zero_division=0)),
        "f1_score": float(f1_score(y_test, predictions, zero_division=0)),
        "false_positive_rate": float(fp / negatives if negatives else 0.0),
        "false_negative_rate": float(fn / positives if positives else 0.0),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
        "inference_time_sec": round(inference_time, 4),
    }
    # Recall and false negatives are intentionally weighted in model selection.
    metrics["selection_score"] = float(
        0.35 * metrics["f1_score"]
        + 0.30 * metrics["recall"]
        + 0.20 * metrics["accuracy"]
        + 0.15 * metrics["precision"]
    )
    print(f"{name}: {json.dumps(metrics, sort_keys=True)}")
    return metrics


def main() -> None:
    raw_train = pd.read_csv(DATA_DIR / "train.csv")
    test = pd.read_csv(DATA_DIR / "test.csv")
    if list(raw_train.columns) != ["domain", "label"] or list(test.columns) != [
        "domain",
        "label",
    ]:
        raise ValueError("Training and test CSVs must contain domain,label columns")

    train, supplemental = load_training_data(raw_train, test)
    train_labels = set(train["label"].astype(int).unique())
    test_labels = set(test["label"].astype(int).unique())
    if train_labels != {0, 1} or test_labels != {0, 1}:
        raise ValueError(
            f"Expected binary labels {{0, 1}}, got train={train_labels}, test={test_labels}"
        )

    training_domains = train["domain"]
    y_train = train["label"].astype(int).to_numpy()
    vectorizer = TfidfVectorizer(
        analyzer="char",
        ngram_range=(2, 4),
        min_df=2,
        max_features=NGRAM_FEATURE_COUNT,
    )
    vectorizer.fit(training_domains.astype(str).str.lower().str.strip())
    joblib.dump(vectorizer, VECTOR_FILE)

    X_train = build_features(training_domains, vectorizer)
    X_test = build_features(test["domain"], vectorizer)
    y_test = test["label"].astype(int).to_numpy()

    expected_feature_count = len(FEATURE_NAMES) + NGRAM_FEATURE_COUNT
    if X_train.shape[1] != expected_feature_count:
        raise ValueError(
            f"Expected {expected_feature_count} features, got {X_train.shape[1]}"
        )
    print(
        f"Train shape: {X_train.shape}; test shape: {X_test.shape}; "
        f"verified supplemental DGA rows: {len(supplemental)}"
    )

    models = {
        "RandomForest": RandomForestClassifier(
            n_estimators=200,
            max_depth=None,
            n_jobs=-1,
            random_state=RANDOM_STATE,
        ),
        "LinearSVM": LinearSVC(
            C=1.0,
            max_iter=5000,
            random_state=RANDOM_STATE,
        ),
        "NeuralNetwork": MLPClassifier(
            hidden_layer_sizes=(128, 64),
            activation="relu",
            max_iter=100,
            early_stopping=True,
            random_state=RANDOM_STATE,
        ),
    }

    results = {}
    trained_models = {}
    for name, model in models.items():
        print(f"Training {name} ...")
        started = time.perf_counter()
        model.fit(X_train, y_train)
        trained_models[name] = model
        metrics = evaluate(name, model, X_test, y_test)
        metrics["train_time_sec"] = round(time.perf_counter() - started, 2)
        results[name] = metrics

    probability_candidates = [
        candidate
        for candidate, model in trained_models.items()
        if hasattr(model, "predict_proba")
    ]
    if not probability_candidates:
        raise RuntimeError("At least one trained model must expose predict_proba().")
    best_name = max(
        probability_candidates,
        key=lambda candidate: (
            results[candidate]["selection_score"],
            results[candidate]["recall"],
            results[candidate]["f1_score"],
        ),
    )
    best_model = trained_models[best_name]

    joblib.dump(best_model, MODEL_FILE)
    with METRICS_FILE.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "model_name": best_name,
                "feature_count": expected_feature_count,
                "handcrafted_feature_count": len(FEATURE_NAMES),
                "ngram_feature_count": NGRAM_FEATURE_COUNT,
                "used_handcrafted_features": True,
                "label_mapping": LABEL_MAPPING,
                "training_rows": int(len(training_domains)),
                "verified_short_dga_rows": int(len(supplemental)),
                "selection_policy": (
                    "Highest selection score among models exposing predict_proba"
                ),
                "length_metrics": evaluate_length_bands(
                    test["domain"],
                    y_test,
                    best_model.predict(X_test),
                ),
                "results": results,
            },
            handle,
            indent=2,
        )
    with METADATA_FILE.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "feature_order": "handcrafted_then_tfidf",
                "handcrafted_feature_names": FEATURE_NAMES,
                "normalization": "lowercase and strip surrounding whitespace",
                "vectorizer": {
                    "analyzer": "char",
                    "ngram_range": [2, 4],
                    "min_df": 2,
                    "max_features": NGRAM_FEATURE_COUNT,
                    "lowercase": True,
                },
                "model_name": best_name,
                "feature_count": expected_feature_count,
                "label_mapping": LABEL_MAPPING,
                "selection_policy": (
                    "Highest selection score among models exposing predict_proba"
                ),
                "verified_short_dga": {
                    "enabled": True,
                    "rows": int(len(supplemental)),
                    "dataset": "ExtraHop/DGA-Detection-Training-Dataset",
                    "manifest": "verified_dga_source_manifest.json",
                },
            },
            handle,
            indent=2,
        )
    print(f"Selected {best_name}; saved {MODEL_FILE}")


if __name__ == "__main__":
    main()