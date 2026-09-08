"""Re-evaluate exported pipelines on the untouched chronological test split."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.features.feature_schema import MODEL_FEATURES, TARGETS, chronological_split
from src.features.historical_features import build_historical_features
from src.models.inference import load_model
from src.models.train_models import metrics


ROOT = Path(__file__).resolve().parents[2]


def evaluate_exported_models(input_path: Path, model_dir: Path, output_path: Path) -> pd.DataFrame:
    raw = pd.read_csv(input_path, low_memory=False)
    raw = raw.drop_duplicates(["player_id", "match_id", "map"], keep="last").reset_index(drop=True)
    featured = build_historical_features(raw)
    test = chronological_split(featured).test
    rows = []
    for target in TARGETS:
        frame = test.dropna(subset=[target])
        bundle = load_model(target, model_dir)
        prediction = bundle["pipeline"].predict(frame[list(MODEL_FEATURES)])
        row = metrics(frame[target], prediction)
        row.update({"target": target, "model": bundle["metadata"].get("model_name"), "split": "test"})
        rows.append(row)
    result = pd.DataFrame(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "data/processed/player_match_dataset_cleaned.csv")
    parser.add_argument("--model-dir", type=Path, default=ROOT / "models")
    parser.add_argument("--output", type=Path, default=ROOT / "reports/modeling/exported_model_metrics.csv")
    args = parser.parse_args()
    print(evaluate_exported_models(args.input, args.model_dir, args.output).to_string(index=False))
