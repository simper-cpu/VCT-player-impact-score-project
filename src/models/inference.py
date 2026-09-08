"""Standalone inference for exported VCT target pipelines."""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.features.feature_schema import CATEGORICAL_FEATURES, MODEL_FEATURES, MODEL_NUMERIC_FEATURES
from src.features.historical_features import build_historical_features


ROOT = Path(__file__).resolve().parents[2]


def model_path(target: str, model_dir: Path) -> Path:
    normalized = target if target.endswith("_all") else f"{target}_all"
    if normalized not in {"rating2_all", "acs_all", "kda_all"}:
        raise ValueError(f"Unsupported target: {target}")
    stem = {"rating2_all": "rating", "acs_all": "acs", "kda_all": "kda"}[normalized]
    return model_dir / f"vct_{stem}_pipeline.joblib"


def load_model(target: str, model_dir: Path = ROOT / "models") -> dict:
    path = model_path(target, model_dir)
    if not path.exists():
        raise FileNotFoundError(f"Exported model not found: {path}")
    return joblib.load(path)


def prepare_inference_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Prepare either a full feature row or a sparse future-map request.

    Sparse requests are intentionally filled with missing historical values;
    the fitted train imputer handles cold-start players and unseen categories.
    """
    if set(MODEL_FEATURES).issubset(frame.columns):
        return frame[list(MODEL_FEATURES)].copy()

    prepared = pd.DataFrame(index=frame.index)
    for column in CATEGORICAL_FEATURES:
        prepared[column] = frame[column] if column in frame else "__unknown__"
    for column in MODEL_NUMERIC_FEATURES:
        if column in frame:
            prepared[column] = pd.to_numeric(frame[column], errors="coerce")
        else:
            prepared[column] = np.nan
    return prepared[list(MODEL_FEATURES)]


def predict_frame(frame: pd.DataFrame, target: str, model_dir: Path = ROOT / "models") -> pd.DataFrame:
    bundle = load_model(target, model_dir)
    prepared = prepare_inference_frame(frame)
    prediction = bundle["pipeline"].predict(prepared)
    result = frame.copy()
    result[f"prediction_{target if target.endswith('_all') else target + '_all'}"] = prediction
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", required=True, choices=["rating2_all", "acs_all", "kda_all", "rating2", "acs", "kda"])
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, default=ROOT / "models")
    args = parser.parse_args()
    result = predict_frame(pd.read_csv(args.input, low_memory=False), args.target, args.model_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    print(f"Predictions saved to {args.output}")


if __name__ == "__main__":
    main()
