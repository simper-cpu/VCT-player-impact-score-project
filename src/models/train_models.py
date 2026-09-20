"""Train, evaluate, and export the three leakage-safe VCT player pipelines."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import warnings
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import TransformedTargetRegressor

from src.features.feature_schema import (
    CATEGORICAL_FEATURES,
    MODEL_FEATURES,
    MODEL_NUMERIC_FEATURES,
    TARGETS,
    FEATURE_SCHEMA_VERSION,
    chronological_split,
    ensure_no_leakage,
    make_match_timestamp,
    rolling_time_splits,
    validate_dataset,
)
from src.features.historical_features import build_historical_features
from src.models.calibration import apply_prediction_calibration, fit_affine_calibration


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = ROOT / "data/processed/player_match_dataset_cleaned.csv"
DEFAULT_FEATURE_OUTPUT = ROOT / "data/processed/player_match_features.csv"
DEFAULT_MODEL_DIR = ROOT / "models"
DEFAULT_REPORT_DIR = ROOT / "reports/modeling"
DEFAULT_MANIFEST_DIR = ROOT / "reports/manifests"
DATASET_VERSION = "player_match_dataset_cleaned-v4-context-patch"

FEATURE_GROUPS = {
    "player_only": tuple(column for column in MODEL_FEATURES if column.startswith("player_") or column in {"match_year", "match_month", "match_dayofweek"}),
    "player_agent_role": tuple(column for column in MODEL_FEATURES if column.startswith("player_") or column in {"agent", "role", "role_confidence", "form_fallback_level", "patch", "match_year", "match_month", "match_dayofweek"}),
    "team": tuple(column for column in MODEL_FEATURES if column in {"team", "team_pick", "team_elo", "team_rating_last_5", "team_rating_last_10", "team_acs_last_5", "team_kda_last_5", "elo_gap", "match_importance"}),
    "opponent": tuple(column for column in MODEL_FEATURES if column in {"opponent_team", "opponent_elo", "opponent_rating_last_5", "opponent_rating_last_10", "opponent_acs_last_5", "opponent_kda_last_5", "opponent_map_strength"}),
    "map": tuple(column for column in MODEL_FEATURES if column in {"map", "patch", "player_map_rating_last", "player_map_rating_last_3", "player_map_acs_last_5", "player_map_kda_last_5", "player_map_matches_played", "team_map_win_rate", "opponent_map_win_rate"}),
    "roster_interactions": tuple(column for column in MODEL_FEATURES if column in {"lineup_continuity", "veterans_remaining", "team_roster_synergy", "team_role_duelist_share", "team_role_initiator_share", "team_role_controller_share", "team_role_sentinel_share", "player_agent_map_rating_last_5", "player_role_map_rating_last_5", "team_map_rating_last_5"}),
}

DEFAULT_CANDIDATES = {
    "rating2_all": ("rf", "xgb", "xgb_tuned", "stacking", "catboost"),
    "acs_all": ("rf", "xgb", "xgb_tuned", "stacking", "catboost"),
    "kda_all": ("rf", "xgb", "xgb_tuned", "stacking", "xgb_huber", "xgb_quantile", "catboost"),
}


def make_preprocessor() -> ColumnTransformer:
    numeric = Pipeline(
        [("imputer", SimpleImputer(strategy="median", add_indicator=True))]
    )
    categorical = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
        ]
    )
    return ColumnTransformer(
        [("numeric", numeric, list(MODEL_NUMERIC_FEATURES)),
         ("categorical", categorical, list(CATEGORICAL_FEATURES))],
        remainder="drop",
        sparse_threshold=1.0,
        verbose_feature_names_out=True,
    )


def make_estimator(kind: str, random_state: int = 42, params: dict | None = None):
    params = dict(params or {})
    if kind == "rf":
        defaults = {
            "n_estimators": 30,
            "max_features": 0.5,
            "min_samples_leaf": 2,
            "max_depth": 10,
            "n_jobs": 2,
            "random_state": random_state,
        }
        defaults.update(params)
        return RandomForestRegressor(**defaults)
    if kind in {"xgb", "xgb_tuned", "xgb_optuna"}:
        try:
            from xgboost import XGBRegressor
        except ImportError as exc:
            raise RuntimeError("XGBoost is not installed") from exc
        defaults = {
            "n_estimators": 140 if kind == "xgb" else 220,
            "max_depth": 4 if kind == "xgb" else 3,
            "min_child_weight": 1 if kind == "xgb" else 5,
            "learning_rate": 0.05 if kind == "xgb" else 0.03,
            "subsample": 0.85 if kind == "xgb" else 0.9,
            "colsample_bytree": 0.85 if kind == "xgb" else 0.9,
            "reg_lambda": 1.0 if kind == "xgb" else 2.0,
            "objective": "reg:squarederror",
            "tree_method": "hist",
            "n_jobs": 2,
            "random_state": random_state,
        }
        defaults.update(params)
        return XGBRegressor(**defaults)
    if kind in {"xgb_huber", "xgb_quantile"}:
        try:
            from xgboost import XGBRegressor
        except ImportError as exc:
            raise RuntimeError("XGBoost is not installed") from exc
        defaults = {
            "n_estimators": 180,
            "max_depth": 3,
            "learning_rate": 0.04,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "tree_method": "hist",
            "n_jobs": 2,
            "random_state": random_state,
        }
        if kind == "xgb_huber":
            defaults["objective"] = "reg:pseudohubererror"
        else:
            defaults["objective"] = "reg:quantileerror"
            defaults["quantile_alpha"] = 0.5
        defaults.update({key: value for key, value in params.items() if key not in {"objective", "quantile_alpha"}})
        return XGBRegressor(**defaults)
    if kind == "catboost":
        try:
            from catboost import CatBoostRegressor
        except ImportError as exc:
            raise RuntimeError("CatBoost is not installed") from exc
        defaults = {
            "iterations": 250,
            "depth": 6,
            "learning_rate": 0.05,
            "loss_function": "RMSE",
            "verbose": False,
            "random_seed": random_state,
        }
        defaults.update(params)
        return CatBoostRegressor(**defaults)
    if kind == "stacking":
        return StackingRegressor(
            estimators=[
                ("rf", RandomForestRegressor(
                    n_estimators=40, max_depth=10, min_samples_leaf=2,
                    n_jobs=2, random_state=random_state,
                )),
                ("ridge", Ridge(alpha=1.0)),
            ],
            final_estimator=Ridge(alpha=1.0),
            cv=3,
            n_jobs=2,
            **params,
        )
    raise ValueError(f"Unknown model kind: {kind}")


def make_pipeline(kind: str, log_target: bool = False, estimator_params: dict | None = None) -> Pipeline:
    estimator = make_estimator(kind, params=estimator_params)
    model = estimator
    if log_target:
        model = TransformedTargetRegressor(
            regressor=estimator, func=np.log1p, inverse_func=np.expm1
        )
    return Pipeline([("preprocessor", make_preprocessor()), ("model", model)])


def log_options(target: str, kind: str) -> tuple[bool, ...]:
    if target == "kda_all" and kind not in {"xgb_quantile"}:
        return (False, True)
    return (False,)


def tune_xgb_with_optuna(
    train_frame: pd.DataFrame,
    validation_frame: pd.DataFrame,
    target: str,
    n_trials: int = 0,
    random_state: int = 42,
) -> tuple[dict, dict] | None:
    """Tune a compact XGBoost search space on the validation block.

    Optuna is optional at import time.  The function returns ``None`` when it
    is unavailable or disabled, so the regular candidate comparison remains
    usable in lightweight environments.
    """
    if n_trials <= 0:
        return None
    try:
        import optuna
    except ImportError:
        return None

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 80, 260, step=20),
            "max_depth": trial.suggest_int("max_depth", 2, 6),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.15, log=True),
            "subsample": trial.suggest_float("subsample", 0.70, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.70, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 10.0, log=True),
        }
        pipeline = make_pipeline("xgb_optuna", target == "kda_all", estimator_params=params)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pipeline.fit(train_frame[list(MODEL_FEATURES)], train_frame[target])
        prediction = pipeline.predict(validation_frame[list(MODEL_FEATURES)])
        return metrics(validation_frame[target], prediction)["mae"]

    sampler = optuna.samplers.TPESampler(seed=random_state)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return dict(study.best_params), {
        "enabled": True,
        "n_trials": n_trials,
        "best_value": float(study.best_value),
        "best_params": dict(study.best_params),
    }


def metrics(y_true, prediction) -> dict:
    actual = np.asarray(y_true, dtype=float)
    pred = np.asarray(prediction, dtype=float)
    error = actual - pred
    correlation = spearmanr(actual, pred, nan_policy="omit").statistic
    return {
        "n": int(len(actual)),
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "r2": float(1 - (np.sum(error**2) / np.sum((actual - actual.mean()) ** 2)))
        if np.sum((actual - actual.mean()) ** 2) > 0
        else np.nan,
        "spearman": float(correlation) if pd.notna(correlation) else np.nan,
    }


def _reference_mean(train: pd.DataFrame, target: str) -> float:
    return float(train[target].mean())


def _historical_column(target: str, rolling: bool = False) -> str:
    if target == "rating2_all":
        return "player_rating_last_3" if rolling else "player_rating_last_10"
    if target == "acs_all":
        return "player_acs_last_5"
    return "player_kda_last_5"


def baseline_predictions(train: pd.DataFrame, frame: pd.DataFrame, target: str) -> dict:
    """Compute non-leaky baselines using train-only constants/coefficients."""
    global_mean = _reference_mean(train, target)
    global_median = float(train[target].median())
    result = {
        "global_mean": np.full(len(frame), global_mean),
        "global_median": np.full(len(frame), global_median),
    }

    history_column = _historical_column(target)
    history = pd.to_numeric(frame[history_column], errors="coerce")
    result["player_historical_mean"] = history.fillna(global_mean).to_numpy()
    rolling_column = _historical_column(target, rolling=True)
    rolling = pd.to_numeric(frame[rolling_column], errors="coerce")
    result["player_rolling_mean"] = rolling.fillna(global_mean).to_numpy()

    def linear_prediction(columns: list[str], name: str) -> None:
        x_train = train[columns].apply(pd.to_numeric, errors="coerce")
        x_frame = frame[columns].apply(pd.to_numeric, errors="coerce")
        impute = x_train.median().fillna(0)
        estimator = LinearRegression().fit(x_train.fillna(impute), train[target])
        result[name] = estimator.predict(x_frame.fillna(impute))

    linear_prediction(["elo_gap"], "elo_only")
    linear_prediction(
        ["team_rating_last_5", "opponent_rating_last_5", "elo_gap"],
        "team_opponent_historical",
    )
    return result


def _group_feature(name: str) -> str:
    raw = name.split("__", 1)[-1]
    for prefix in ("map_", "agent_", "team_", "opponent_team_", "team_pick_"):
        if raw.startswith(prefix):
            return prefix.rstrip("_")
    return raw


def save_importance(pipeline: Pipeline, x_test: pd.DataFrame, y_test, target: str, output_dir: Path) -> None:
    preprocessor = pipeline.named_steps["preprocessor"]
    model = pipeline.named_steps["model"]
    estimator = model.regressor_ if isinstance(model, TransformedTargetRegressor) else model
    names = np.asarray(preprocessor.get_feature_names_out())
    if hasattr(estimator, "feature_importances_"):
        raw = pd.DataFrame({"feature": names, "importance": estimator.feature_importances_})
    else:
        # Stacking/Ridge models have no native tree importance.  Keep the same
        # report contract and rely on permutation importance for them below.
        raw = pd.DataFrame({"feature": names, "importance": np.nan})
    raw["group"] = raw["feature"].map(_group_feature)
    raw.sort_values("importance", ascending=False).to_csv(
        output_dir / f"{target}_feature_importance_raw.csv", index=False
    )
    grouped = (
        raw.groupby("group", as_index=False)["importance"].sum()
        .sort_values("importance", ascending=False)
    )
    grouped.to_csv(output_dir / f"{target}_feature_importance_grouped.csv", index=False)

    sample = x_test.sample(min(len(x_test), 600), random_state=42)
    y_sample = y_test.loc[sample.index]
    permutation = permutation_importance(
        pipeline, sample, y_sample, n_repeats=2, random_state=42,
        scoring="neg_mean_absolute_error", n_jobs=1,
    )
    # permutation_importance permutes columns of the original X, not the
    # expanded one-hot matrix returned by the fitted preprocessor.
    perm = pd.DataFrame({"feature": list(sample.columns), "importance_mean": permutation.importances_mean,
                         "importance_std": permutation.importances_std})
    perm["group"] = perm["feature"].map(_group_feature)
    perm.sort_values("importance_mean", ascending=False).to_csv(
        output_dir / f"{target}_permutation_importance.csv", index=False
    )

    if grouped["importance"].notna().any():
        top = grouped.head(20).sort_values("importance")
    else:
        permutation_groups = (
            perm.groupby("group", as_index=False)["importance_mean"].sum()
            .rename(columns={"importance_mean": "importance"})
            .sort_values("importance", ascending=False)
        )
        top = permutation_groups.head(20).sort_values("importance")
    plt.figure(figsize=(9, 7))
    plt.barh(top["group"], top["importance"])
    plt.title(f"{target}: grouped Random Forest importance")
    plt.tight_layout()
    plt.savefig(output_dir / f"{target}_feature_importance.png", dpi=150)
    plt.close()


def save_error_reports(predictions: pd.DataFrame, target: str, output_dir: Path) -> None:
    predictions = predictions.copy()
    predictions["abs_error"] = predictions["residual"].abs()
    predictions.to_csv(output_dir / f"{target}_test_predictions.csv", index=False)
    for column, filename in (
        ("map", "map"), ("agent", "agent"), ("role", "role"),
        ("experience_bucket", "experience"), ("time_period", "time_period"),
        ("known_opponent", "opponent_knowledge"), ("known_map", "map_knowledge"),
        ("roster_stability", "roster_stability"), ("role_experience", "role_experience"),
    ):
        if column not in predictions:
            continue
        grouped = predictions.groupby(column, dropna=False).apply(
            lambda group: pd.Series(metrics(group["actual"], group["prediction"])),
            include_groups=False,
        ).reset_index()
        grouped.to_csv(output_dir / f"{target}_errors_by_{filename}.csv", index=False)

    plt.figure(figsize=(8, 5))
    plt.scatter(predictions["actual"], predictions["prediction"], alpha=0.18, s=8)
    limits = [predictions[["actual", "prediction"]].min().min(), predictions[["actual", "prediction"]].max().max()]
    plt.plot(limits, limits, "k--")
    plt.xlabel("Actual")
    plt.ylabel("Prediction")
    plt.title(f"{target}: test predictions")
    plt.tight_layout()
    plt.savefig(output_dir / f"{target}_residual_plot.png", dpi=150)
    plt.close()


def _json_default(value):
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, (np.ndarray,)):
        return value.tolist()
    return str(value)


def _package_versions() -> dict:
    names = ("numpy", "pandas", "scikit-learn", "joblib", "scipy", "xgboost")
    return {name: importlib.metadata.version(name) for name in names if _installed(name)}


def _installed(name: str) -> bool:
    try:
        importlib.metadata.version(name)
        return True
    except importlib.metadata.PackageNotFoundError:
        return False


def evaluate_rolling_candidates(
    featured: pd.DataFrame,
    target: str,
    candidates: list[tuple],
) -> tuple[list[dict], list[dict]]:
    """Score the validation shortlist on several forward-only time windows."""
    windows = rolling_time_splits(featured, n_splits=3)
    detail_rows: list[dict] = []
    selection_rows: list[dict] = []
    for candidate in candidates:
        label, kind, log_target = candidate[:3]
        estimator_params = candidate[3] if len(candidate) > 3 else None
        model_scores = []
        baseline_scores = []
        for window_number, window in enumerate(windows, start=1):
            train_window = window.train.dropna(subset=[target]).copy()
            validation_window = window.validation.dropna(subset=[target]).copy()
            if train_window.empty or validation_window.empty:
                continue
            pipeline = make_pipeline(kind, log_target, estimator_params=estimator_params)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                pipeline.fit(train_window[list(MODEL_FEATURES)], train_window[target])
            prediction = pipeline.predict(validation_window[list(MODEL_FEATURES)])
            score = metrics(validation_window[target], prediction)
            score.update({"target": target, "model": label, "split": f"rolling_{window_number}"})
            detail_rows.append(score)
            model_scores.append(score)

            baseline_prediction = np.full(len(validation_window), float(train_window[target].mean()))
            baseline_score = metrics(validation_window[target], baseline_prediction)
            baseline_score.update({
                "target": target,
                "model": "global_mean",
                "candidate": label,
                "split": f"rolling_{window_number}",
            })
            detail_rows.append(baseline_score)
            baseline_scores.append(baseline_score)

        if not model_scores:
            continue
        summary = {
            "target": target,
            "model": label,
            "split": "rolling_mean",
            "n_windows": len(model_scores),
            "mae": float(np.mean([row["mae"] for row in model_scores])),
            "rmse": float(np.mean([row["rmse"] for row in model_scores])),
            "mae_std": float(np.std([row["mae"] for row in model_scores])),
            "baseline_mae": float(np.mean([row["mae"] for row in baseline_scores])),
        }
        summary["mae_improvement_vs_global_mean"] = summary["baseline_mae"] - summary["mae"]
        selection_rows.append(summary)
    return detail_rows, selection_rows


def train_all(
    input_path: Path = DEFAULT_INPUT,
    model_dir: Path = DEFAULT_MODEL_DIR,
    report_dir: Path = DEFAULT_REPORT_DIR,
    manifest_dir: Path = DEFAULT_MANIFEST_DIR,
    candidate_kinds: tuple[str, ...] | None = None,
    optuna_trials: int = 0,
) -> dict:
    raw = pd.read_csv(input_path, low_memory=False)
    raw = raw.drop_duplicates(["player_id", "match_id", "map"], keep="last").reset_index(drop=True)
    validate_dataset(raw, require_targets=True)
    featured = build_historical_features(raw)
    DEFAULT_FEATURE_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    featured.to_csv(DEFAULT_FEATURE_OUTPUT, index=False)
    ensure_no_leakage(MODEL_FEATURES, "__no_target__")
    split = chronological_split(featured)
    rolling_windows = rolling_time_splits(featured, n_splits=3)
    train = split.train.copy()
    validation = split.validation.copy()
    test = split.test.copy()
    model_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    manifest_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "dataset_version": DATASET_VERSION,
        "input": str(input_path),
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "features": list(MODEL_FEATURES),
        "feature_groups": {name: list(columns) for name, columns in FEATURE_GROUPS.items()},
        "targets": list(TARGETS),
        "split": {
            "train_match_ids": list(split.train_match_ids),
            "validation_match_ids": list(split.validation_match_ids),
            "test_match_ids": list(split.test_match_ids),
            "train_cutoff": split.train_cutoff,
            "validation_cutoff": split.validation_cutoff,
            "test_cutoff": split.test_cutoff,
        },
        "rows": {"train": len(train), "validation": len(validation), "test": len(test)},
        "training": {"optuna_trials_per_target": int(optuna_trials)},
        "rolling_windows": [
            {
                "train_match_ids": list(window.train_match_ids),
                "validation_match_ids": list(window.validation_match_ids),
                "train_cutoff": window.train_cutoff,
                "validation_cutoff": window.validation_cutoff,
            }
            for window in rolling_windows
        ],
    }
    (manifest_dir / "dataset_manifest.json").write_text(
        json.dumps(manifest, indent=2, default=_json_default), encoding="utf-8"
    )

    all_metrics = []
    rolling_detail_rows: list[dict] = []
    rolling_selection_rows: list[dict] = []
    calibration_rows: list[dict] = []
    optuna_rows: list[dict] = []
    for target in TARGETS:
        train_target = train.dropna(subset=[target]).copy()
        validation_target = validation.dropna(subset=[target]).copy()
        test_target = test.dropna(subset=[target]).copy()
        candidate_results = []
        candidate_pipelines = {}
        candidate_validation_predictions = {}
        kinds = candidate_kinds if candidate_kinds is not None else DEFAULT_CANDIDATES[target]
        tuned_params = None
        tuning_metadata = {"enabled": False, "n_trials": 0}
        if optuna_trials > 0 and not train_target.empty and not validation_target.empty:
            try:
                tuned = tune_xgb_with_optuna(
                    train_target, validation_target, target, n_trials=optuna_trials,
                )
            except (RuntimeError, ValueError, ImportError):
                tuned = None
            if tuned is not None:
                tuned_params, tuning_metadata = tuned
                if "xgb_optuna" not in kinds:
                    kinds = tuple(kinds) + ("xgb_optuna",)
        optuna_rows.append({"target": target, **tuning_metadata})
        for kind in kinds:
            for log_target in log_options(target, kind):
                label = f"{kind}_{'log' if log_target else 'raw'}"
                try:
                    estimator_params = tuned_params if kind == "xgb_optuna" else None
                    pipeline = make_pipeline(kind, log_target, estimator_params=estimator_params)
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        pipeline.fit(train_target[list(MODEL_FEATURES)], train_target[target])
                    prediction = pipeline.predict(validation_target[list(MODEL_FEATURES)])
                    score = metrics(validation_target[target], prediction)
                    score.update({"target": target, "model": label, "split": "validation"})
                    candidate_results.append(score)
                    candidate_pipelines[label] = (kind, log_target, estimator_params)
                    candidate_validation_predictions[label] = prediction
                except (RuntimeError, ValueError):
                    continue
        if not candidate_results:
            raise RuntimeError(f"No candidate model could be trained for {target}")

        # Only the two strongest candidates on the first validation block are
        # refit on three expanding rolling windows. This keeps tuning tractable
        # while making the final choice less dependent on one time period.
        shortlist = sorted(candidate_results, key=lambda item: item["mae"])[:2]
        rolling_candidates = [
            (item["model"], *candidate_pipelines[item["model"]]) for item in shortlist
        ]
        rolling_detail, rolling_summary = evaluate_rolling_candidates(
            featured, target, rolling_candidates
        )
        rolling_detail_rows.extend(rolling_detail)
        rolling_selection_rows.extend(rolling_summary)
        all_metrics.extend(rolling_detail)
        all_metrics.extend(rolling_summary)
        if rolling_summary:
            selected_summary = min(
                rolling_summary,
                key=lambda item: (item["mae"], item["mae_std"]),
            )
            best = next(
                item for item in candidate_results if item["model"] == selected_summary["model"]
            )
        else:
            selected_summary = None
            best = min(candidate_results, key=lambda item: item["mae"])
        best_kind, best_log, best_params = candidate_pipelines[best["model"]]
        validation_prediction = candidate_validation_predictions[best["model"]]
        calibration = fit_affine_calibration(validation_target[target], validation_prediction)

        # Refit only the selected configuration on train + validation; test stays untouched.
        refit = pd.concat([train_target, validation_target], ignore_index=True)
        final_pipeline = make_pipeline(best_kind, best_log, estimator_params=best_params)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            final_pipeline.fit(refit[list(MODEL_FEATURES)], refit[target])
        raw_prediction = final_pipeline.predict(test_target[list(MODEL_FEATURES)])
        prediction = apply_prediction_calibration(raw_prediction, calibration)
        raw_test_score = metrics(test_target[target], raw_prediction)
        raw_test_score.update({"target": target, "model": f"{best['model']}_raw_uncalibrated", "split": "test"})
        test_score = metrics(test_target[target], prediction)
        test_score.update({"target": target, "model": best["model"], "split": "test"})
        all_metrics.extend(candidate_results)
        all_metrics.append(raw_test_score)
        all_metrics.append(test_score)

        calibration_rows.append({
            "target": target,
            "model": best["model"],
            "enabled": calibration.get("enabled", False),
            "slope": calibration.get("slope", 1.0),
            "intercept": calibration.get("intercept", 0.0),
            "validation_raw_mae": calibration.get("validation_raw_mae"),
            "validation_calibrated_mae": calibration.get("validation_calibrated_mae"),
            "test_raw_mae": raw_test_score["mae"],
            "test_calibrated_mae": test_score["mae"],
            "test_actual_std": float(np.std(test_target[target].to_numpy(dtype=float))),
            "test_raw_std": float(np.std(raw_prediction)),
            "test_calibrated_std": float(np.std(prediction)),
        })

        baseline_rows = []
        for split_name, frame, baseline_train in (
            ("validation", validation_target, train_target),
            ("test", test_target, refit),
        ):
            for baseline_name, baseline_prediction in baseline_predictions(baseline_train, frame, target).items():
                score = metrics(frame[target], baseline_prediction)
                score.update({"target": target, "model": baseline_name, "split": split_name})
                baseline_rows.append(score)
        all_metrics.extend(baseline_rows)

        timestamp = make_match_timestamp(test_target)
        errors = pd.DataFrame({
            "player_id": test_target["player_id"].to_numpy(),
            "match_id": test_target["match_id"].to_numpy(),
            "map": test_target["map"].to_numpy(),
            "agent": test_target["agent"].to_numpy(),
            "role": test_target["role"].to_numpy(),
            "actual": test_target[target].to_numpy(),
            "prediction_raw": raw_prediction,
            "prediction": prediction,
            "residual": test_target[target].to_numpy() - prediction,
            "player_matches_played": test_target["player_matches_played"].to_numpy(),
            "match_timestamp": timestamp.to_numpy(),
        })
        errors["experience_bucket"] = pd.cut(
            errors["player_matches_played"], bins=[-1, 0, 5, 20, np.inf],
            labels=["cold_start", "1-5", "6-20", "21+"],
        ).astype("string")
        errors["known_opponent"] = np.where(test_target["opponent_rating_last_5"].notna(), "known", "unknown")
        errors["known_map"] = np.where(test_target["player_map_matches_played"] > 0, "known", "unknown")
        errors["roster_stability"] = np.where(test_target["lineup_continuity"].fillna(0) >= 0.8, "stable", "changed_or_unknown")
        errors["role_experience"] = pd.cut(
            test_target["player_role_matches_played"], bins=[-1, 0, 5, 20, np.inf],
            labels=["cold_start", "1-5", "6-20", "21+"],
        ).astype("string")
        errors["time_period"] = pd.qcut(
            errors["match_timestamp"].rank(method="first"), q=min(4, len(errors)), labels=False
        ) if len(errors) else pd.Series(dtype="float64")
        save_error_reports(errors, target, report_dir)
        save_importance(final_pipeline, test_target[list(MODEL_FEATURES)], test_target[target], target, report_dir)

        metadata = {
            "target": target,
            "features": list(MODEL_FEATURES),
            "feature_schema_version": FEATURE_SCHEMA_VERSION,
            "feature_groups": {name: list(columns) for name, columns in FEATURE_GROUPS.items()},
            "cutoff_date": split.test_cutoff,
            "model_name": best["model"],
            "model_parameters": final_pipeline.named_steps["model"].get_params(deep=True),
            "metrics": {
                "validation": best,
                "validation_calibration": calibration,
                "test_raw_uncalibrated": raw_test_score,
                "rolling_selection": selected_summary,
                "test": test_score,
            },
            "calibration": calibration,
            "dataset_version": DATASET_VERSION,
            "package_versions": _package_versions(),
            "optuna": tuning_metadata,
        }
        stem = {"rating2_all": "rating", "acs_all": "acs", "kda_all": "kda"}[target]
        joblib.dump({"pipeline": final_pipeline, "metadata": metadata}, model_dir / f"vct_{stem}_pipeline.joblib")

        (manifest_dir / f"{target}_model_manifest.json").write_text(
            json.dumps(metadata, indent=2, default=_json_default), encoding="utf-8"
        )

    pd.DataFrame(rolling_detail_rows).to_csv(
        report_dir / "rolling_validation_metrics.csv", index=False
    )
    pd.DataFrame(rolling_selection_rows).to_csv(
        report_dir / "rolling_selection.csv", index=False
    )
    pd.DataFrame(calibration_rows).to_csv(
        report_dir / "calibration_metrics.csv", index=False
    )
    pd.DataFrame(optuna_rows).to_csv(report_dir / "optuna_trials.csv", index=False)
    metrics_frame = pd.DataFrame(all_metrics)
    metrics_frame.to_csv(report_dir / "metrics.csv", index=False)
    return {"metrics": metrics_frame, "manifest": manifest}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--manifest-dir", type=Path, default=DEFAULT_MANIFEST_DIR)
    parser.add_argument("--rf-only", action="store_true", help="Skip XGBoost comparison")
    parser.add_argument(
        "--optuna-trials", type=int, default=0,
        help="Run this many Optuna trials per target (0 disables optional tuning)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    result = train_all(
        args.input, args.model_dir, args.report_dir, args.manifest_dir,
        candidate_kinds=("rf",) if args.rf_only else None,
        optuna_trials=args.optuna_trials,
    )
    print(result["metrics"].to_string(index=False))
