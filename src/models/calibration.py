"""Leakage-safe post-hoc calibration for regression predictions."""

from __future__ import annotations

import numpy as np


def _mae(actual: np.ndarray, prediction: np.ndarray) -> float:
    return float(np.mean(np.abs(actual - prediction)))


def _rmse(actual: np.ndarray, prediction: np.ndarray) -> float:
    return float(np.sqrt(np.mean((actual - prediction) ** 2)))


def fit_affine_calibration(actual, prediction, *, min_slope: float = 0.25, max_slope: float = 3.0) -> dict:
    """Fit a conservative affine recalibration on a validation split only.

    The slope is clipped to avoid turning validation noise into extreme
    forecasts.  Calibration is disabled when it does not improve validation
    MAE, preserving the original model output in that case.
    """
    y = np.asarray(actual, dtype=float)
    raw = np.asarray(prediction, dtype=float)
    if len(y) != len(raw) or len(y) < 3:
        return {
            "method": "affine_validation",
            "enabled": False,
            "slope": 1.0,
            "intercept": 0.0,
            "reason": "insufficient_validation_rows",
        }

    variance = float(np.var(raw))
    if not np.isfinite(variance) or variance <= 1e-12:
        return {
            "method": "affine_validation",
            "enabled": False,
            "slope": 1.0,
            "intercept": 0.0,
            "reason": "constant_raw_prediction",
        }

    slope = float(np.cov(raw, y, ddof=0)[0, 1] / variance)
    if not np.isfinite(slope) or slope <= 0:
        return {
            "method": "affine_validation",
            "enabled": False,
            "slope": 1.0,
            "intercept": 0.0,
            "reason": "non_positive_validation_slope",
        }
    slope = float(np.clip(slope, min_slope, max_slope))
    intercept = float(y.mean() - slope * raw.mean())
    calibrated = intercept + slope * raw
    raw_mae = _mae(y, raw)
    calibrated_mae = _mae(y, calibrated)

    enabled = bool(calibrated_mae < raw_mae)
    if not enabled:
        slope = 1.0
        intercept = 0.0
        calibrated = raw

    return {
        "method": "affine_validation",
        "enabled": enabled,
        "slope": slope,
        "intercept": intercept,
        "reason": "validation_mae_improved" if enabled else "validation_mae_not_improved",
        "validation_raw_mae": raw_mae,
        "validation_calibrated_mae": _mae(y, calibrated),
        "validation_raw_rmse": _rmse(y, raw),
        "validation_calibrated_rmse": _rmse(y, calibrated),
        "validation_actual_std": float(np.std(y)),
        "validation_raw_std": float(np.std(raw)),
        "validation_calibrated_std": float(np.std(calibrated)),
    }


def apply_prediction_calibration(prediction, calibration: dict | None):
    """Apply exported calibration metadata; old bundles remain backward-compatible."""
    raw = np.asarray(prediction, dtype=float)
    if not calibration or not calibration.get("enabled", False):
        return raw
    return float(calibration.get("intercept", 0.0)) + float(calibration.get("slope", 1.0)) * raw

