"""Schema, leakage rules, and chronological split helpers for the VCT pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import pandas as pd
import numpy as np


TARGETS = ("rating2_all", "acs_all", "kda_all")
KEY_COLUMNS = ("player_id", "match_id", "map")

# These columns are useful for auditing, but are not allowed in model X.
DIRECT_ID_COLUMNS = ("player_id", "team_id", "opponent_team_id", "match_id", "event_id")
POST_MAP_COLUMNS = (
    "rating2_all", "rating2_t", "rating2_ct",
    "acs_all", "acs_t", "acs_ct",
    "kills_all", "kills_t", "kills_ct",
    "deaths_all", "deaths_t", "deaths_ct",
    "assists_all", "assists_t", "assists_ct",
    "adr_all", "adr_t", "adr_ct",
    "kast_all", "kast_t", "kast_ct",
    "hsp_all", "hsp_t", "hsp_ct",
    "fb_all", "fb_t", "fb_ct",
    "fd_all", "fd_t", "fd_ct", "kda_all",
    "map_winner", "first_half_ct_win_rounds", "first_half_t_win_rounds",
    "second_half_ct_win_rounds", "second_half_t_win_rounds",
    "ot_team_ct_win", "ot_team_t_win",
)

CATEGORICAL_FEATURES = (
    "map", "agent", "team", "opponent_team", "team_pick",
    "event_stage", "event_round", "region",
)
HISTORICAL_FEATURES = (
    "player_rating_last", "player_rating_last_3", "player_rating_last_5",
    "player_rating_last_10", "player_acs_last_5", "player_kda_last_5",
    "player_acs_last_3", "player_acs_last_10", "player_kda_last_3", "player_kda_last_10",
    "player_rating_ewm_3", "player_rating_ewm_5", "player_rating_ewm_10",
    "player_acs_ewm_3", "player_acs_ewm_5", "player_acs_ewm_10",
    "player_kda_ewm_3", "player_kda_ewm_5", "player_kda_ewm_10",
    "player_rating_std", "player_matches_played", "player_form_trend",
    "player_rating_long_term", "player_acs_long_term", "player_kda_long_term",
    "player_rating_form_gap", "player_acs_form_gap", "player_kda_form_gap",
    "team_rating_last_5", "team_rating_last_10",
    "opponent_rating_last_5", "opponent_rating_last_10",
    "team_elo", "opponent_elo", "elo_gap",
    "player_map_rating_last", "player_map_rating_last_3",
    "player_map_acs_last_5", "player_map_kda_last_5", "player_map_matches_played",
    "player_agent_rating_last_5", "player_agent_acs_last_5", "player_agent_kda_last_5",
    "player_agent_matches_played", "player_rest_days",
    "player_matches_last_7d", "player_matches_last_14d", "player_matches_last_30d",
    "player_cold_start",
)
TEMPORAL_FEATURES = ("match_year", "match_month", "match_dayofweek")
MODEL_NUMERIC_FEATURES = HISTORICAL_FEATURES + TEMPORAL_FEATURES
MODEL_FEATURES = MODEL_NUMERIC_FEATURES + CATEGORICAL_FEATURES


def make_match_timestamp(df: pd.DataFrame) -> pd.Series:
    """Parse the source date/time without using any post-map information."""
    date = df.get("match_date", pd.Series(pd.NaT, index=df.index)).astype("string")
    time = df.get("match_time", pd.Series("", index=df.index)).fillna("").astype("string")
    timestamp = pd.to_datetime(date + " " + time, format="mixed", errors="coerce")
    return timestamp.fillna(pd.to_datetime(date, format="mixed", errors="coerce"))


def validate_dataset(df: pd.DataFrame, require_targets: bool = True) -> dict:
    """Return a compact schema report and fail early for missing core columns."""
    required = set(KEY_COLUMNS) | {"match_date", "team_id", "agent"}
    if require_targets:
        required.update(TARGETS)
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")

    duplicates = int(df.duplicated(list(KEY_COLUMNS)).sum())
    timestamp = make_match_timestamp(df)
    return {
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
        "duplicate_player_match_map_rows": duplicates,
        "missing_key_values": int(df[list(KEY_COLUMNS)].isna().any(axis=1).sum()),
        "missing_timestamp_rows": int(timestamp.isna().sum()),
        "target_missing": {target: int(df[target].isna().sum()) for target in TARGETS if target in df},
        "feature_schema_version": "v2-ewma-context",
    }


def ensure_no_leakage(features: Iterable[str], target: str) -> None:
    """Raise if a proposed model feature is a target, ID, or post-map field."""
    feature_set = set(features)
    forbidden = set(DIRECT_ID_COLUMNS) | set(POST_MAP_COLUMNS) | set(TARGETS)
    leaked = sorted(feature_set.intersection(forbidden))
    if leaked:
        raise ValueError(f"Forbidden/leaky model features: {leaked}")


@dataclass(frozen=True)
class TimeSplit:
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame
    train_match_ids: tuple
    validation_match_ids: tuple
    test_match_ids: tuple
    train_cutoff: str | None
    validation_cutoff: str | None
    test_cutoff: str | None


@dataclass(frozen=True)
class RollingTimeWindow:
    train: pd.DataFrame
    validation: pd.DataFrame
    train_match_ids: tuple
    validation_match_ids: tuple
    train_cutoff: str | None
    validation_cutoff: str | None


def chronological_split(
    df: pd.DataFrame, train_fraction: float = 0.70, validation_fraction: float = 0.15
) -> TimeSplit:
    """Split whole matches in time order; no match can cross a split boundary."""
    if not 0 < train_fraction < 1 or not 0 <= validation_fraction < 1:
        raise ValueError("Split fractions must be between 0 and 1")
    if train_fraction + validation_fraction >= 1:
        raise ValueError("Train plus validation fractions must be less than 1")

    work = df.copy()
    work["_match_timestamp"] = make_match_timestamp(work)
    match_table = (
        work[["match_id", "_match_timestamp"]]
        .drop_duplicates("match_id")
        .sort_values(["_match_timestamp", "match_id"], na_position="last")
    )
    if match_table["_match_timestamp"].isna().any():
        raise ValueError("Cannot create a chronological split with missing match timestamps")

    match_ids = match_table["match_id"].tolist()
    n_matches = len(match_ids)
    train_end = max(1, int(n_matches * train_fraction))
    validation_end = max(train_end + 1, int(n_matches * (train_fraction + validation_fraction)))
    validation_end = min(validation_end, n_matches - 1) if n_matches > 2 else n_matches
    if validation_end <= train_end and n_matches > 2:
        validation_end = train_end + 1

    train_ids = tuple(match_ids[:train_end])
    validation_ids = tuple(match_ids[train_end:validation_end])
    test_ids = tuple(match_ids[validation_end:])
    train = work[work["match_id"].isin(train_ids)].drop(columns="_match_timestamp")
    validation = work[work["match_id"].isin(validation_ids)].drop(columns="_match_timestamp")
    test = work[work["match_id"].isin(test_ids)].drop(columns="_match_timestamp")

    def cutoff(ids: tuple) -> str | None:
        if not ids:
            return None
        value = match_table.loc[match_table["match_id"] == ids[-1], "_match_timestamp"].iloc[0]
        return value.isoformat()

    return TimeSplit(
        train=train,
        validation=validation,
        test=test,
        train_match_ids=train_ids,
        validation_match_ids=validation_ids,
        test_match_ids=test_ids,
        train_cutoff=cutoff(train_ids),
        validation_cutoff=cutoff(validation_ids),
        test_cutoff=cutoff(test_ids),
    )


def rolling_time_splits(
    df: pd.DataFrame,
    n_splits: int = 3,
    min_train_fraction: float = 0.45,
    validation_fraction: float = 0.10,
    max_end_fraction: float = 0.85,
) -> list[RollingTimeWindow]:
    """Create expanding-train, forward-only validation windows.

    The final ``1 - max_end_fraction`` portion is never touched here, so it
    can remain a genuine final test set.
    """
    if n_splits < 1 or not 0 < min_train_fraction < max_end_fraction <= 1:
        raise ValueError("Invalid rolling split configuration")
    if validation_fraction <= 0 or min_train_fraction + validation_fraction > max_end_fraction:
        raise ValueError("Validation windows do not fit before the final test set")

    work = df.copy()
    work["_match_timestamp"] = make_match_timestamp(work)
    match_table = (
        work[["match_id", "_match_timestamp"]]
        .drop_duplicates("match_id")
        .sort_values(["_match_timestamp", "match_id"])
    )
    if match_table["_match_timestamp"].isna().any():
        raise ValueError("Cannot create rolling splits with missing timestamps")
    match_ids = match_table["match_id"].tolist()
    n_matches = len(match_ids)
    train_floor = max(1, int(n_matches * min_train_fraction))
    horizon = max(1, int(n_matches * validation_fraction))
    last_start = max(train_floor, int(n_matches * max_end_fraction) - horizon)
    starts = sorted(set(int(value) for value in np.linspace(train_floor, last_start, n_splits)))
    if not starts:
        starts = [train_floor]
    windows = []
    for start in starts:
        end = min(start + horizon, n_matches)
        train_ids = tuple(match_ids[:start])
        validation_ids = tuple(match_ids[start:end])
        if not validation_ids:
            continue
        train = work[work["match_id"].isin(train_ids)].drop(columns="_match_timestamp")
        validation = work[work["match_id"].isin(validation_ids)].drop(columns="_match_timestamp")

        def cutoff(ids: tuple) -> str | None:
            if not ids:
                return None
            value = match_table.loc[match_table["match_id"] == ids[-1], "_match_timestamp"].iloc[0]
            return value.isoformat()

        windows.append(RollingTimeWindow(
            train=train,
            validation=validation,
            train_match_ids=train_ids,
            validation_match_ids=validation_ids,
            train_cutoff=cutoff(train_ids),
            validation_cutoff=cutoff(validation_ids),
        ))
    return windows
