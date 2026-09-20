"""Data preparation helpers for the local Streamlit VCT explorer.

The functions in this module deliberately do not import Streamlit.  That keeps
the data contract easy to test and lets the UI use Streamlit caching around
these pure-ish operations.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
import pandas as pd

from src.features.feature_schema import TARGETS, make_match_timestamp, validate_dataset
from src.features.role_mapping import add_role_columns
from src.models.inference import load_model, predict_bundle
from src.features.state_builder import build_inference_state


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_PATH = ROOT / "data" / "processed" / "player_match_dataset_cleaned_all_regions.csv"
FALLBACK_DATA_PATH = ROOT / "data" / "processed" / "player_match_dataset_cleaned.csv"
DEFAULT_MODEL_DIR = ROOT / "models"

_NUMERIC_COLUMNS = (
    "player_id", "team_id", "opponent_team_id", "match_id", "event_id",
    "team_elo", "opponent_elo", "elo_gap", *TARGETS,
    "kills_all", "deaths_all", "assists_all", "adr_all", "kast_all",
    "fb_all", "fd_all", "first_half_ct_win_rounds", "first_half_t_win_rounds",
    "second_half_ct_win_rounds", "second_half_t_win_rounds", "ot_team_ct_win",
    "ot_team_t_win",
    "match_importance",
)
_TEXT_COLUMNS = (
    "region", "team", "opponent_team", "agent", "map", "team_pick",
    "map_winner", "event_stage", "event_round", "team_1", "team_2",
    "patch",
)
_TBD_VALUES = {"", "nan", "none", "null", "tbd", "unknown", "__unknown__"}


def _clean_text(value: object) -> object:
    if pd.isna(value):
        return np.nan
    text = str(value).strip()
    return text if text.lower() not in _TBD_VALUES else np.nan


def normalize_dataset(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize source types and validate the minimum app data contract."""
    work = frame.copy()
    if "player_name" not in work.columns and "name" in work.columns:
        work = work.rename(columns={"name": "player_name"})

    for column in _TEXT_COLUMNS:
        if column in work.columns:
            work[column] = work[column].map(_clean_text)
    if "player_name" in work.columns:
        work["player_name"] = work["player_name"].astype("string").str.strip()
    if "match_date" in work.columns:
        work["match_date"] = pd.to_datetime(work["match_date"], errors="coerce")
    if "match_time" in work.columns:
        work["match_time"] = work["match_time"].fillna("").astype("string").str.strip()
    for column in _NUMERIC_COLUMNS:
        if column in work.columns:
            work[column] = pd.to_numeric(work[column], errors="coerce")

    if "match_id" in work and "map" in work and "player_id" in work:
        work = work.drop_duplicates(["player_id", "match_id", "map"], keep="last")
    work = add_role_columns(work)
    validate_dataset(work, require_targets=True)
    return work.reset_index(drop=True)


def load_local_dataset(
    preferred_path: Path = DEFAULT_DATA_PATH,
    fallback_path: Path = FALLBACK_DATA_PATH,
) -> tuple[pd.DataFrame, Path]:
    """Read the preferred all-regions dataset, falling back to the single-region file."""
    source = Path(preferred_path) if Path(preferred_path).exists() else Path(fallback_path)
    if not source.exists():
        raise FileNotFoundError(f"No local dataset found at {preferred_path} or {fallback_path}")
    # The explorer never needs crawler-only columns.  Selecting them here is
    # particularly helpful for the all-regions player-stat file.
    columns = {
        "player_name", "name", "region", "player_id", "team", "team_id",
        "opponent_team", "opponent_team_id", "agent", "map", "match_id",
        "event_id", "event_stage", "event_round", "match_date", "match_time",
        "team_1", "team_2", "team_pick", "map_winner", "patch",
        "match_importance", *TARGETS, "kills_all", "deaths_all", "assists_all",
        "adr_all", "kast_all", "fb_all", "fd_all",
        "first_half_ct_win_rounds", "first_half_t_win_rounds",
        "second_half_ct_win_rounds", "second_half_t_win_rounds",
        "ot_team_ct_win", "ot_team_t_win", "role", "role_confidence",
    }
    return normalize_dataset(_read_csv_selected(source, columns)), source


def _read_csv_selected(path: Path, columns: set[str]) -> pd.DataFrame:
    """Read only columns used by the app, preferring pyarrow when available."""
    header = pd.read_csv(path, nrows=0)
    usecols = [column for column in header.columns if column in columns]
    try:
        return pd.read_csv(path, usecols=usecols, engine="pyarrow")
    except (ImportError, ModuleNotFoundError, ValueError, TypeError):
        return pd.read_csv(path, usecols=usecols, low_memory=False)


def filter_completed_history(
    frame: pd.DataFrame,
    cutoff: pd.Timestamp | datetime | str | None = None,
) -> pd.DataFrame:
    """Keep only completed, non-TBD maps with known targets before ``cutoff``.

    A cutoff is exclusive: a row at the prediction timestamp is not treated as
    completed history.  When omitted, today's end is used so local future rows
    cannot silently enter the default history.
    """
    work = frame.copy()
    timestamp = make_match_timestamp(work)
    map_name = work.get("map", pd.Series(index=work.index, dtype="object")).astype("string").str.strip().str.lower()
    winner = work.get("map_winner", pd.Series(index=work.index, dtype="object")).astype("string").str.strip().str.lower()
    valid_map = map_name.notna() & ~map_name.isin(_TBD_VALUES)
    valid_winner = winner.notna() & ~winner.isin(_TBD_VALUES)
    valid_targets = work[list(TARGETS)].notna().all(axis=1)
    valid_timestamp = timestamp.notna()
    if cutoff is None:
        cutoff_value = pd.Timestamp.now().normalize() + pd.Timedelta(days=1)
    else:
        cutoff_value = pd.Timestamp(cutoff)
    mask = valid_map & valid_winner & valid_targets & valid_timestamp & (timestamp < cutoff_value)
    result = work.loc[mask].copy()
    result["_timestamp"] = timestamp.loc[mask]
    return result.sort_values(["_timestamp", "match_id", "map"], kind="stable").drop(columns="_timestamp")


def filter_team(frame: pd.DataFrame, team_id: object, region: str | None = None) -> pd.DataFrame:
    """Filter by numeric team identity, optionally constrained to a region."""
    ids = pd.to_numeric(frame.get("team_id"), errors="coerce")
    target_id = pd.to_numeric(pd.Series([team_id]), errors="coerce").iloc[0]
    mask = ids.eq(target_id)
    if region is not None and "region" in frame:
        mask &= frame["region"].astype("string").eq(str(region))
    return frame.loc[mask].copy()


def get_team_options(frame: pd.DataFrame) -> pd.DataFrame:
    """Return one display row per region/team_id, with a readable team name."""
    rows = []
    for (region, team_id), group in frame.dropna(subset=["team_id"]).groupby(["region", "team_id"], dropna=False):
        team_name = group["team"].dropna().astype(str).iloc[0] if group["team"].notna().any() else str(team_id)
        full_names = []
        for name_column, id_column in (("team_1", "team_1_id"), ("team_2", "team_2_id")):
            if name_column in group and id_column in group:
                full_names.extend(group.loc[pd.to_numeric(group[id_column], errors="coerce").eq(team_id), name_column].dropna().astype(str))
        display_name = full_names[0] if full_names else team_name
        rows.append({
            "region": str(region), "team_id": team_id, "team": team_name,
            "display_name": display_name,
            "label": f"{display_name} ({team_name}) · id={int(team_id)}",
        })
    return pd.DataFrame(rows).sort_values(["region", "display_name", "team_id"]).reset_index(drop=True)


def get_opponent_options(
    team_frame: pd.DataFrame,
    all_teams: pd.DataFrame | None = None,
    exclude_team_id: object | None = None,
) -> pd.DataFrame:
    """Return selectable opponents, optionally from the complete team catalog.

    Passing ``all_teams`` makes teams with no previous head-to-head match
    selectable as well.  This is useful for future scenarios against a new
    opponent; the model then has no opponent-specific history and falls back
    accordingly.
    """
    if all_teams is not None and not all_teams.empty:
        result = all_teams[["team_id", "display_name", "team"]].copy()
        result["opponent_team_id"] = pd.to_numeric(result["team_id"], errors="coerce")
        result["opponent_team"] = result["display_name"].fillna(result["team"])
        result = result.drop(columns=["team_id", "display_name", "team"])
    else:
        rows = team_frame.dropna(subset=["opponent_team_id"]).copy()
        rows["opponent_team_id"] = pd.to_numeric(rows["opponent_team_id"], errors="coerce")
        rows = rows.dropna(subset=["opponent_team_id"])
        result = rows.groupby("opponent_team_id", as_index=False).agg(
            opponent_team=("opponent_team", lambda values: next((str(v) for v in values if pd.notna(v)), "Unknown")),
        )
    if exclude_team_id is not None:
        result = result[~result["opponent_team_id"].eq(pd.to_numeric(pd.Series([exclude_team_id]), errors="coerce").iloc[0])]
    result = result.drop_duplicates("opponent_team_id")
    result["label"] = result.apply(lambda row: f"{row['opponent_team']} (id={int(row['opponent_team_id'])})", axis=1)
    return result.sort_values("opponent_team").reset_index(drop=True)


def _map_keys(frame: pd.DataFrame) -> pd.DataFrame:
    timestamp = make_match_timestamp(frame)
    return (
        frame.assign(_timestamp=timestamp)
        .sort_values(["_timestamp", "match_id", "map"], kind="stable")
        [["match_id", "map", "_timestamp"]]
        .drop_duplicates(["match_id", "map"])
    )


def get_recent_roster(frame: pd.DataFrame, team_id: object, n_maps: int = 5) -> pd.DataFrame:
    """Get unique players appearing in the selected team's latest maps.

    ``agent`` remains the agent from the player's latest recorded map for
    backwards compatibility.  ``recent_agents`` is the safer UI field: it
    summarizes all recorded agents in the window instead of presenting one
    historical pick as the agent for an upcoming map.
    """
    team_frame = filter_team(frame, team_id)
    if team_frame.empty:
        return pd.DataFrame(columns=["player_id", "player_name", "role", "last_map_date", "maps_in_window"])
    keys = _map_keys(team_frame).sort_values("_timestamp", ascending=False).head(n_maps)
    recent = team_frame.merge(
        keys[["match_id", "map"]], on=["match_id", "map"], how="inner", validate="many_to_one"
    )
    recent = recent.assign(_timestamp=make_match_timestamp(recent)).sort_values("_timestamp", ascending=False)
    roster = recent.drop_duplicates("player_id", keep="first").copy()
    counts = recent.groupby("player_id")["match_id"].nunique().rename("maps_in_window")
    roster = roster.join(counts, on="player_id")
    if "agent" in recent:
        recent_agents = (
            recent.dropna(subset=["agent"])
            .groupby("player_id")["agent"]
            .apply(lambda values: " / ".join(dict.fromkeys(values.astype(str))))
            .rename("recent_agents")
        )
        roster = roster.join(recent_agents, on="player_id")
    roster["last_map_date"] = roster["_timestamp"].dt.date
    columns = [
        column for column in [
            "player_id", "player_name", "role", "last_map_date", "maps_in_window",
            "agent", "recent_agents",
        ] if column in roster
    ]
    return roster[columns].reset_index(drop=True)


def _first_non_null(values: Iterable[object]) -> object:
    for value in values:
        if pd.notna(value):
            return value
    return np.nan


def _format_round_score(row: pd.Series) -> str:
    def total(columns: tuple[str, ...]) -> float:
        values = pd.to_numeric(pd.Series([row.get(column, np.nan) for column in columns]), errors="coerce")
        return float(values.fillna(0).sum())

    ct = total(("first_half_ct_win_rounds", "second_half_ct_win_rounds", "ot_team_ct_win"))
    t = total(("first_half_t_win_rounds", "second_half_t_win_rounds", "ot_team_t_win"))
    if ct.is_integer() and t.is_integer():
        return f"CT/T {int(ct)}-{int(t)}"
    return f"CT/T {ct:.1f}-{t:.1f}"


def aggregate_match_history(team_frame: pd.DataFrame) -> pd.DataFrame:
    """Collapse player rows into one row per completed match/map."""
    if team_frame.empty:
        return pd.DataFrame()
    work = team_frame.copy()
    work["match_date"] = pd.to_datetime(work["match_date"], errors="coerce")
    group_columns = [column for column in ["match_id", "map"] if column in work]
    rows = []
    for key, group in work.groupby(group_columns, sort=False, dropna=False):
        row = group.iloc[0].copy()
        row["match_date"] = group["match_date"].min()
        row["opponent_team"] = _first_non_null(group.get("opponent_team", pd.Series(dtype="object")))
        row["map_winner"] = _first_non_null(group.get("map_winner", pd.Series(dtype="object")))
        row["team_pick"] = _first_non_null(group.get("team_pick", pd.Series(dtype="object")))
        row["event_id"] = _first_non_null(group.get("event_id", pd.Series(dtype="object")))
        row["event_stage"] = _first_non_null(group.get("event_stage", pd.Series(dtype="object")))
        row["event_round"] = _first_non_null(group.get("event_round", pd.Series(dtype="object")))
        row["round_score"] = _format_round_score(row)
        rows.append(row)
    result = pd.DataFrame(rows)
    return result.sort_values(["match_date", "match_id"], ascending=False).reset_index(drop=True)


def get_player_history(frame: pd.DataFrame, player_id: object, team_id: object | None = None) -> pd.DataFrame:
    """Return map-level history for one player, newest first."""
    ids = pd.to_numeric(frame["player_id"], errors="coerce")
    target = pd.to_numeric(pd.Series([player_id]), errors="coerce").iloc[0]
    result = frame.loc[ids.eq(target)].copy()
    if team_id is not None:
        result = filter_team(result, team_id)
    result["match_date"] = pd.to_datetime(result["match_date"], errors="coerce")
    return result.assign(_timestamp=make_match_timestamp(result)).sort_values("_timestamp", ascending=False).drop(columns="_timestamp")


def get_recent_performance_history(
    frame: pd.DataFrame,
    *,
    player_id: object | None = None,
    team_id: object | None = None,
    limit: int = 10,
) -> pd.DataFrame:
    """Return one aggregate performance row for each of the latest matches.

    Raw data is map-grain.  The UI wants a match-grain trend, so multi-map
    matches are averaged before taking the latest ``limit`` matches.
    """
    if limit < 1:
        raise ValueError("limit must be positive")
    scoped = frame.copy()
    if player_id is not None:
        scoped = get_player_history(scoped, player_id, team_id)
    elif team_id is not None:
        scoped = filter_team(scoped, team_id)
    if scoped.empty:
        return scoped
    scoped = filter_completed_history(scoped)
    if scoped.empty:
        return scoped
    scoped = scoped.assign(_timestamp=make_match_timestamp(scoped))
    rows = []
    for match_id, group in scoped.groupby("match_id", sort=False, dropna=False):
        row = group.sort_values("_timestamp", ascending=False).iloc[0].copy()
        row["match_id"] = match_id
        row["match_date"] = group["_timestamp"].min()
        for metric in ("rating2_all", "acs_all", "kda_all"):
            if metric in group:
                row[metric] = pd.to_numeric(group[metric], errors="coerce").mean()
        row["map"] = ", ".join(group["map"].dropna().astype(str).unique())
        rows.append(row)
    result = pd.DataFrame(rows).sort_values("_timestamp", ascending=False).head(limit)
    return result.sort_values("_timestamp", kind="stable").drop(columns="_timestamp").reset_index(drop=True)


CHART_METRICS = {
    "Rating": "rating2_all", "ACS": "acs_all", "KDA": "kda_all",
    "Kills": "kills_all", "Deaths": "deaths_all", "Assists": "assists_all",
    "ADR": "adr_all", "KAST": "kast_all",
}


def validate_chart_metrics(metrics: Iterable[str]) -> list[str]:
    selected = list(metrics)
    if not 1 <= len(selected) <= 2:
        raise ValueError("Select exactly one or two chart metrics")
    unknown = sorted(set(selected).difference(CHART_METRICS))
    if unknown:
        raise ValueError(f"Unknown chart metrics: {unknown}")
    return selected


def build_chart_data(player_history: pd.DataFrame, metrics: Iterable[str]) -> pd.DataFrame:
    """Build testable long-form chart data from player map history."""
    selected = validate_chart_metrics(metrics)
    id_columns = [column for column in ["match_date", "match_id", "map", "opponent_team", "agent", "role"] if column in player_history]
    pieces = []
    for metric in selected:
        source = CHART_METRICS[metric]
        if source not in player_history:
            continue
        piece = player_history[id_columns + [source]].copy()
        piece = piece.rename(columns={source: "value"})
        piece["metric"] = metric
        pieces.append(piece)
    if not pieces:
        return pd.DataFrame(columns=id_columns + ["value", "metric"])
    return pd.concat(pieces, ignore_index=True)


def build_forecast_request(
    history: pd.DataFrame,
    team_id: object,
    opponent_team_id: object,
    player_ids: Iterable[object],
    prediction_time: pd.Timestamp | datetime | str,
    map_name: str | None = None,
    agent_name: str | None = None,
    opponent_team_name: str | None = None,
    agent_by_player: Mapping[object, str | None] | None = None,
) -> pd.DataFrame:
    """Create one sparse future-map request per selected player."""
    if opponent_team_id is None or pd.isna(opponent_team_id):
        raise ValueError("opponent_team_id is required for forecast")
    prediction = pd.Timestamp(prediction_time)
    team_rows = filter_team(history, team_id)
    opponent_rows = team_rows[pd.to_numeric(team_rows["opponent_team_id"], errors="coerce").eq(float(opponent_team_id))]
    latest_team = team_rows.assign(_timestamp=make_match_timestamp(team_rows)).sort_values("_timestamp", ascending=False).iloc[0]
    latest_opponent = opponent_rows.iloc[0] if not opponent_rows.empty else pd.Series(dtype="object")
    requests = []
    for player_id in player_ids:
        player_rows = get_player_history(team_rows, player_id, team_id)
        if player_rows.empty:
            raise ValueError(f"Player {player_id} has no completed history for the selected team")
        latest = player_rows.iloc[0]
        request = {
            "player_id": latest["player_id"], "player_name": latest.get("player_name", str(player_id)),
            "team_id": latest["team_id"], "team": latest.get("team", latest_team.get("team", "__unknown__")),
            # The forecast selection is authoritative.  The player's latest
            # completed map may have had a different opponent.
            "opponent_team_id": opponent_team_id,
            "opponent_team": opponent_team_name or latest_opponent.get("opponent_team", "__unknown__"),
            "region": latest.get("region", latest_team.get("region", "__unknown__")),
            "match_date": prediction.date().isoformat(), "match_time": prediction.strftime("%H:%M:%S"),
        }
        if map_name:
            request["map"] = map_name
        selected_agent = agent_name
        if agent_by_player is not None:
            selected_agent = agent_by_player.get(player_id)
            if selected_agent is None:
                selected_agent = agent_by_player.get(str(player_id))
        if selected_agent:
            request["agent"] = selected_agent
        else:
            request["role"] = latest.get("role", "Flex/Unknown")
        requests.append(request)
    return pd.DataFrame(requests)


def select_forecast_history(history: pd.DataFrame, request: pd.DataFrame) -> pd.DataFrame:
    """Keep history that can affect the requested player/team/opponent state.

    The state builder is intentionally general and walks every row it receives.
    For an interactive forecast, rows for unrelated teams and players cannot
    change the selected player's direct, team, opponent, map, or agent history.
    Keeping the relevant identities preserves those features while avoiding a
    full-corpus rebuild on every button click.
    """
    if history.empty or request.empty:
        return history.copy()
    player_ids = set(pd.to_numeric(request.get("player_id"), errors="coerce").dropna())
    team_ids = set(pd.to_numeric(request.get("team_id"), errors="coerce").dropna())
    team_ids.update(pd.to_numeric(request.get("opponent_team_id"), errors="coerce").dropna())
    player_mask = pd.to_numeric(history.get("player_id"), errors="coerce").isin(player_ids)
    team_mask = pd.to_numeric(history.get("team_id"), errors="coerce").isin(team_ids)
    result = history.loc[player_mask | team_mask].copy()
    result = result.assign(_timestamp=make_match_timestamp(result))
    return result.sort_values(["_timestamp", "match_id"], kind="stable").drop(columns="_timestamp")


def run_forecast(
    request: pd.DataFrame,
    history: pd.DataFrame,
    model_bundles: dict[str, dict] | None = None,
    model_dir: Path = DEFAULT_MODEL_DIR,
) -> pd.DataFrame:
    """Build online state and run all three existing exported target models."""
    relevant_history = select_forecast_history(history, request)
    state = build_inference_state(request, relevant_history)
    bundles = model_bundles or {
        target: load_model(target, model_dir) for target in TARGETS
    }
    result = request[[column for column in ["player_id", "player_name", "team", "opponent_team", "map", "agent"] if column in request]].copy()
    for target in TARGETS:
        result[f"forecast_{target}"] = predict_bundle(bundles[target], state)
    for column in ["player_history_sample_size", "player_cold_start", "player_map_matches_played", "player_agent_matches_played", "form_fallback_level"]:
        if column in state:
            result[column] = state[column].to_numpy()
    result["low_confidence"] = (result["player_cold_start"].eq(1) | result["player_history_sample_size"].fillna(0).lt(5))
    result["confidence_warning"] = np.where(
        result["low_confidence"],
        "Low confidence: limited player history or cold start",
        "Use as estimated performance, not a win probability",
    )
    return result
