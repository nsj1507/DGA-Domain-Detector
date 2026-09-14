"""Evaluate the selected detector on the verified source-derived challenge set."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.main import service  # noqa: E402

DATA_FILE = ROOT / "ml" / "data" / "verified_dga_challenge.csv"
OUTPUT_FILE = ROOT / "ml" / "artifacts" / "verified_dga_challenge_predictions.json"


def main() -> None:
    cases = pd.read_csv(DATA_FILE)
    results: list[dict[str, object]] = []
    for row in cases.to_dict("records"):
        prediction = service.predict(str(row["domain"]))
        results.append(
            {
                "domain": row["domain"],
                "expected_label": int(row["label"]),
                "category": row["category"],
                "source_dataset": row["source_dataset"],
                "source_length": int(row["source_length"]),
                "prediction": prediction["prediction"],
                "predicted_label": 1 if prediction["prediction"] == "Malicious" else 0,
                "malicious_probability": prediction["malicious_probability"],
                "confidence": prediction["confidence"],
                "risk_level": prediction["risk_level"],
            }
        )

    summary: dict[str, dict[str, float | int | None]] = {}
    for category in cases["category"].unique():
        category_results = [
            result for result in results if result["category"] == category
        ]
        expected = [int(result["expected_label"]) for result in category_results]
        predicted = [int(result["predicted_label"]) for result in category_results]
        correct = sum(actual == guess for actual, guess in zip(expected, predicted))
        dga_indices = [index for index, label in enumerate(expected) if label == 1]
        benign_indices = [index for index, label in enumerate(expected) if label == 0]
        summary[category] = {
            "count": len(category_results),
            "accuracy": correct / len(category_results) if category_results else None,
            "dga_recall": (
                sum(predicted[index] == 1 for index in dga_indices) / len(dga_indices)
                if dga_indices
                else None
            ),
            "legitimate_false_positive_rate": (
                sum(predicted[index] == 1 for index in benign_indices)
                / len(benign_indices)
                if benign_indices
                else None
            ),
        }

    output = {
        "dataset": "ExtraHop/DGA-Detection-Training-Dataset",
        "model_name": service.metrics["model_name"] if service.metrics else None,
        "summary": summary,
        "results": results,
    }
    OUTPUT_FILE.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")

    print("domain\tcategory\tprediction\tmalicious_probability\tconfidence\trisk")
    for result in results:
        print(
            f"{result['domain']}\t{result['category']}\t{result['prediction']}\t"
            f"{result['malicious_probability']}\t{result['confidence']}\t"
            f"{result['risk_level']}"
        )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()