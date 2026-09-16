"""Online state construction backed by the offline historical feature logic."""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd

from .feature_schema import MODEL_FEATURES, TARGETS, make_match_timestamp
from .historical_features import build_historical_features
from .role_mapping import add_role_columns


def _id_series(frame: pd.DataFrame, value_column: str, fallback: str) -> pd.Series:
    if value_column in frame:
        values = frame[value_column]
    else:
        values = frame.get(fallback, pd.Series(index=frame.index, dtype="object"))
    return pd.Series(pd.factorize(values.fillna("__unknown__").astype(str))[0], index=frame.index, dtype="float64")


def _prepare_request(request: pd.DataFrame, history: pd.DataFrame | None) -> pd.DataFrame:
    result = request.copy().reset_index(drop=True)
    result = add_role_columns(result)
    if "player_id" not in result:
        result["player_id"] = _id_series(result, "player_id", "player")
    if "team_id" not in result:
        result["team_id"] = _id_series(result, "team_id", "team")
    if "opponent_team_id" not in result:
        result["opponent_team_id"] = _id_series(result, "opponent_team_id", "opponent_team")
    if "team" not in result:
        result["team"] = result.get("team_name", "__unknown__")
    if "opponent_team" not in result:
        result["opponent_team"] = result.get("opponent", "__unknown__")
    if "map" not in result:
        result["map"] = result.get("map_name", "__unknown__")
    result["map"] = result["map"].fillna("__unknown__")
    if "agent" not in result:
        result["agent"] = "__unknown__"
    result["agent"] = result["agent"].fillna("__unknown__")
    result["match_id"] = [f"__inference_{index}" for index in result.index]
    if "match_date" not in result:
        history_timestamp = make_match_timestamp(history).max() if history is not None and len(history) else pd.NaT
        base = history_timestamp + pd.Timedelta(seconds=1) if pd.notna(history_timestamp) else pd.Timestamp(datetime.now(timezone.utc).replace(tzinfo=None))
        result["match_date"] = base.date().isoformat()
        result["match_time"] = base.strftime("%H:%M:%S")
    elif "match_time" not in result:
        result["match_time"] = "00:00:00"
    result["match_date"] = result["match_date"].fillna(pd.Timestamp.now().date().isoformat())
    result["match_time"] = result["match_time"].fillna("00:00:00")
    for target in TARGETS:
        if target not in result:
            result[target] = np.nan
    for column in ("team_elo", "opponent_elo", "elo_gap"):
        if column not in result:
            result[column] = np.nan
    return result


def build_inference_state(request: pd.DataFrame, completed_history: pd.DataFrame | None = None) -> pd.DataFrame:
    """Build request snapshots using exactly the historical offline semantics.

    ``completed_history`` must contain only maps that are complete.  Request
    rows are assigned synthetic match IDs and missing targets, so they can be
    snapshotted without ever updating the state used by another request row.
    """
    request_base = _prepare_request(request, completed_history)
    if completed_history is None or completed_history.empty:
        combined = request_base
    else:
        history = completed_history.copy()
        history = add_role_columns(history)
        for target in TARGETS:
            if target not in history:
                history[target] = np.nan
        combined = pd.concat([history, request_base], ignore_index=True, sort=False)
    featured = build_historical_features(combined)
    featured = featured.iloc[-len(request_base):].reset_index(drop=True)
    for column in MODEL_FEATURES:
        if column not in featured:
            featured[column] = "__unknown__" if column in {"map", "agent", "role", "team", "opponent_team", "team_pick", "event_stage", "event_round", "region", "form_fallback_level"} else np.nan
    return featured[list(MODEL_FEATURES)]
