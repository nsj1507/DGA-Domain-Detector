"""Train the DGA detector on the intended 10,019-feature representation.

The representation is deliberately built here from the raw train/test domain
files instead of relying on the previous handcrafted CSV export. That keeps
each feature row aligned with its domain and guarantees that prediction uses
the same feature order.
"""

from __future__ import annotations

import json
import random
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
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer

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
SYNTHETIC_DGA_SEED = 20260910
SYNTHETIC_DGA_PER_LENGTH_BAND = 1_000
DGA_LENGTH_BANDS = ((4, 6), (7, 10), (11, 15), (16, 25))
LABEL_MAPPING = {"0": "Legitimate", "1": "Malicious"}


def generate_synthetic_dga_domains(
    per_length_band: int = SYNTHETIC_DGA_PER_LENGTH_BAND,
    seed: int = SYNTHETIC_DGA_SEED,
) -> list[str]:
    """Generate deterministic DGA-like training rows without domain allowlists.

    These rows expand the positive class into lengths absent from the source
    data. The generator intentionally mixes letters and digits so the model
    cannot reduce the task to a single "contains a digit" rule.
    """
    rng = random.Random(seed)
    tlds = ("com", "org", "net", "info")
    letters = "abcdefghijklmnopqrstuvwxyz"
    consonants = "bcdfghjklmnpqrstvwxyz"
    digits = "0123456789"
    generated: list[str] = []
    seen: set[str] = set()

    for minimum, maximum in DGA_LENGTH_BANDS:
        while sum(
            minimum <= len(domain.rsplit(".", 1)[0]) <= maximum
            for domain in generated
        ) < per_length_band:
            length = rng.randint(minimum, maximum)
            characters: list[str] = []
            include_digit = rng.random() < 0.65
            for _ in range(length):
                roll = rng.random()
                if roll < 0.20:
                    character = rng.choice(digits)
                elif roll < 0.78:
                    character = rng.choice(consonants)
                else:
                    character = rng.choice(letters)
                characters.append(character)
            if include_digit and not any(character.isdigit() for character in characters):
                characters[rng.randrange(length)] = rng.choice(digits)
            domain = f"{''.join(characters)}.{rng.choice(tlds)}"
            if domain not in seen:
                seen.add(domain)
                generated.append(domain)
    return generated


def augment_training_data(
    train: pd.DataFrame,
) -> tuple[pd.Series, np.ndarray, int]:
    """Add generated positive examples only to cover missing short DGA lengths."""
    synthetic_domains = generate_synthetic_dga_domains()
    domains = pd.concat(
        [train["domain"], pd.Series(synthetic_domains, dtype="string")],
        ignore_index=True,
    )
    labels = np.concatenate(
        [
            train["label"].astype(int).to_numpy(),
            np.ones(len(synthetic_domains), dtype=int),
        ]
    )
    return domains, labels, len(synthetic_domains)


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
    train = pd.read_csv(DATA_DIR / "train.csv")
    test = pd.read_csv(DATA_DIR / "test.csv")
    if list(train.columns) != ["domain", "label"] or list(test.columns) != [
        "domain",
        "label",
    ]:
        raise ValueError("Training and test CSVs must contain domain,label columns")

    train_labels = set(train["label"].astype(int).unique())
    test_labels = set(test["label"].astype(int).unique())
    if train_labels != {0, 1} or test_labels != {0, 1}:
        raise ValueError(
            f"Expected binary labels {{0, 1}}, got train={train_labels}, test={test_labels}"
        )

    training_domains, y_train, synthetic_count = augment_training_data(train)
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
        f"synthetic DGA rows: {synthetic_count}"
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

    best_name = max(
        results,
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
                "synthetic_dga_rows": synthetic_count,
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
                "synthetic_dga": {
                    "enabled": True,
                    "seed": SYNTHETIC_DGA_SEED,
                    "rows_per_length_band": SYNTHETIC_DGA_PER_LENGTH_BAND,
                    "length_bands": [list(band) for band in DGA_LENGTH_BANDS],
                },
            },
            handle,
            indent=2,
        )
    print(f"Selected {best_name}; saved {MODEL_FILE}")


if __name__ == "__main__":
    main()