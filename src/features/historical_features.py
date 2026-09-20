"""Leakage-safe, pre-map historical feature generation.

The algorithm walks maps in chronological order.  A row receives a snapshot
of player/team state first; only after the map is complete are its outcomes
added to the state.  Thus every rolling value is equivalent to a shift(1).
"""

from __future__ import annotations

from collections import defaultdict, deque
from pathlib import Path

import numpy as np
import pandas as pd

from .feature_schema import HISTORICAL_FEATURES, make_match_timestamp, validate_dataset
from .role_mapping import add_role_columns


HISTORY_COLUMNS = (
    "rating2_all", "acs_all", "kda_all", "match_id", "map", "agent", "timestamp"
)


def _numeric(value) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if np.isfinite(result) else None


def _key(value, fallback="__unknown__"):
    return fallback if value is None or (isinstance(value, float) and np.isnan(value)) else value


def _mean(values, n: int | None = None) -> float:
    values = list(values)[-n:] if n else list(values)
    if values and isinstance(values[0], dict):
        values = [value.get("rating") for value in values]
    values = [value for value in values if value is not None and np.isfinite(value)]
    return float(np.mean(values)) if values else np.nan


def _metric_mean(values, key: str, n: int | None = None) -> float:
    values = list(values)[-n:] if n else list(values)
    return _mean([value.get(key) if isinstance(value, dict) else value for value in values])


def _last(values, n: int = 1) -> float:
    values = list(values)
    return _mean(values, n) if n > 1 else (values[-1] if values else np.nan)


def _ewma(values, span: int) -> float:
    """Exponentially weighted mean; the newest valid map gets the most weight."""
    values = [value for value in values if value is not None and np.isfinite(value)]
    if not values:
        return np.nan
    alpha = 2.0 / (span + 1.0)
    result = float(values[0])
    for value in values[1:]:
        result = alpha * float(value) + (1.0 - alpha) * result
    return result


def _trend(values, n: int = 5) -> float:
    values = [v for v in list(values)[-n:] if v is not None and np.isfinite(v)]
    if len(values) < 2:
        return 0.0
    return float(np.polyfit(np.arange(len(values), dtype=float), values, 1)[0])


def _state(max_history: int = 500) -> dict:
    return {
        "history": deque(maxlen=max_history),
        "all_match_ids": set(),
        "last_timestamp": None,
    }


def _feature_snapshot(
    row: pd.Series,
    player_state: dict,
    map_state: dict,
    agent_state: dict,
    role_state: dict | None,
    team_history: deque,
    opponent_history: deque,
    timestamp: pd.Timestamp,
) -> dict:
    player_records = list(player_state["history"])
    map_records = list(map_state["history"])
    agent_records = list(agent_state["history"])
    role_records = list(role_state["history"]) if role_state is not None else []

    ratings = [entry["rating"] for entry in player_records]
    acs = [entry["acs"] for entry in player_records]
    kdas = [entry["kda"] for entry in player_records]
    map_ratings = [entry["rating"] for entry in map_records]
    map_acs = [entry["acs"] for entry in map_records]
    map_kdas = [entry["kda"] for entry in map_records]
    agent_ratings = [entry["rating"] for entry in agent_records]
    agent_acs = [entry["acs"] for entry in agent_records]
    agent_kdas = [entry["kda"] for entry in agent_records]
    role_ratings = [entry["rating"] for entry in role_records]
    role_acs = [entry["acs"] for entry in role_records]
    role_kdas = [entry["kda"] for entry in role_records]

    last_timestamp = player_state["last_timestamp"]
    rest_days = np.nan
    if last_timestamp is not None and pd.notna(timestamp):
        rest_days = max(0.0, (timestamp - last_timestamp).total_seconds() / 86400.0)

    def recent_match_count(days: int) -> int:
        if pd.isna(timestamp):
            return 0
        cutoff = timestamp - pd.Timedelta(days=days)
        return len({entry["match_id"] for entry in player_records if entry["timestamp"] >= cutoff})

    std = np.nan
    valid_ratings = [value for value in ratings if value is not None]
    if len(valid_ratings) >= 2:
        std = float(np.std(valid_ratings, ddof=1))

    rating_long = _mean(ratings)
    acs_long = _mean(acs)
    kda_long = _mean(kdas)
    rating_ewm_5 = _ewma(ratings, 5)
    acs_ewm_5 = _ewma(acs, 5)
    kda_ewm_5 = _ewma(kdas, 5)

    fallback_candidates = [
        (_last(map_ratings, 5), "player-map"),
        (_mean(agent_ratings, 5), "player-agent"),
        (_mean(role_ratings, 5), "player-role"),
        (_mean(ratings, 5), "player-global"),
        (_mean(team_history, 5), "team-context"),
        (_mean(opponent_history, 5), "opponent-context"),
    ]
    fallback_value, fallback_level = next(
        ((value, source) for value, source in fallback_candidates if pd.notna(value)),
        (np.nan, "global-mean"),
    )

    return {
        "player_rating_last": _last(ratings, 1),
        "player_rating_last_3": _mean(ratings, 3),
        "player_rating_last_5": _mean(ratings, 5),
        "player_rating_last_10": _mean(ratings, 10),
        "player_acs_last_5": _mean(acs, 5),
        "player_kda_last_5": _mean(kdas, 5),
        "player_acs_last_3": _mean(acs, 3),
        "player_acs_last_10": _mean(acs, 10),
        "player_kda_last_3": _mean(kdas, 3),
        "player_kda_last_10": _mean(kdas, 10),
        "player_rating_ewm_3": _ewma(ratings, 3),
        "player_rating_ewm_5": rating_ewm_5,
        "player_rating_ewm_10": _ewma(ratings, 10),
        "player_acs_ewm_3": _ewma(acs, 3),
        "player_acs_ewm_5": acs_ewm_5,
        "player_acs_ewm_10": _ewma(acs, 10),
        "player_kda_ewm_3": _ewma(kdas, 3),
        "player_kda_ewm_5": kda_ewm_5,
        "player_kda_ewm_10": _ewma(kdas, 10),
        "player_rating_std": std,
        "player_matches_played": len(player_state["all_match_ids"]),
        "player_form_trend": _trend(ratings, 5),
        "player_rating_long_term": rating_long,
        "player_acs_long_term": acs_long,
        "player_kda_long_term": kda_long,
        "player_rating_form_gap": rating_ewm_5 - rating_long if pd.notna(rating_ewm_5) and pd.notna(rating_long) else np.nan,
        "player_acs_form_gap": acs_ewm_5 - acs_long if pd.notna(acs_ewm_5) and pd.notna(acs_long) else np.nan,
        "player_kda_form_gap": kda_ewm_5 - kda_long if pd.notna(kda_ewm_5) and pd.notna(kda_long) else np.nan,
        "team_rating_last_5": _metric_mean(team_history, "rating", 5),
        "team_rating_last_10": _metric_mean(team_history, "rating", 10),
        "team_acs_last_5": _metric_mean(team_history, "acs", 5),
        "team_kda_last_5": _metric_mean(team_history, "kda", 5),
        "opponent_rating_last_5": _metric_mean(opponent_history, "rating", 5),
        "opponent_rating_last_10": _metric_mean(opponent_history, "rating", 10),
        "opponent_acs_last_5": _metric_mean(opponent_history, "acs", 5),
        "opponent_kda_last_5": _metric_mean(opponent_history, "kda", 5),
        "team_elo": _numeric(row.get("team_elo")),
        "opponent_elo": _numeric(row.get("opponent_elo")),
        "elo_gap": _numeric(row.get("elo_gap")),
        "player_map_rating_last": _last(map_ratings, 1),
        "player_map_rating_last_3": _mean(map_ratings, 3),
        "player_map_acs_last_5": _mean(map_acs, 5),
        "player_map_kda_last_5": _mean(map_kdas, 5),
        "player_map_matches_played": len(map_records),
        "player_agent_rating_last_5": _mean(agent_ratings, 5),
        "player_agent_acs_last_5": _mean(agent_acs, 5),
        "player_agent_kda_last_5": _mean(agent_kdas, 5),
        "player_agent_matches_played": len(agent_records),
        "player_role_rating_last_5": _mean(role_ratings, 5),
        "player_role_acs_last_5": _mean(role_acs, 5),
        "player_role_kda_last_5": _mean(role_kdas, 5),
        "player_role_matches_played": len(role_records),
        "player_role_share": len(role_records) / max(len(player_records), 1),
        "player_rest_days": rest_days,
        "player_matches_last_7d": recent_match_count(7),
        "player_matches_last_14d": recent_match_count(14),
        "player_matches_last_30d": recent_match_count(30),
        "player_cold_start": int(not player_records),
        "role_confidence": _numeric(row.get("role_confidence")),
        "player_rating_fallback": fallback_value,
        "player_acs_fallback": next((value for value, _ in [(_last(map_acs, 5), "player-map"), (_mean(agent_acs, 5), "player-agent"), (_mean(role_acs, 5), "player-role"), (_mean(acs, 5), "player-global"), (_mean(team_history, 5), "team-context"), (_mean(opponent_history, 5), "opponent-context")] if pd.notna(value)), np.nan),
        "player_kda_fallback": next((value for value, _ in [(_last(map_kdas, 5), "player-map"), (_mean(agent_kdas, 5), "player-agent"), (_mean(role_kdas, 5), "player-role"), (_mean(kdas, 5), "player-global"), (_mean(team_history, 5), "team-context"), (_mean(opponent_history, 5), "opponent-context")] if pd.notna(value)), np.nan),
        "player_history_sample_size": len(player_records),
        "form_fallback_level": fallback_level,
    }


def _legacy_build_historical_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add pre-map features while preserving the input row order."""
    validate_dataset(df, require_targets=True)
    work = add_role_columns(df)
    if work.duplicated(["player_id", "match_id", "map"]).any():
        raise ValueError("Historical features require unique (player_id, match_id, map) rows")

    work["timestamp_internal"] = make_match_timestamp(work)
    if work["timestamp_internal"].isna().any():
        raise ValueError("Cannot build historical features with missing timestamps")
    work["row_order_internal"] = np.arange(len(work))

    map_table = (
        work[["match_id", "map", "timestamp_internal"]]
        .drop_duplicates(["match_id", "map"])
        .sort_values(["timestamp_internal", "match_id", "map"])
    )
    groups = {(key[0], key[1]): group for key, group in work.groupby(["match_id", "map"], sort=False)}

    players = defaultdict(lambda: _state(500))
    player_maps = defaultdict(lambda: _state(100))
    player_agents = defaultdict(lambda: _state(100))
    player_roles = defaultdict(lambda: _state(100))
    player_agent_maps = defaultdict(lambda: _state(100))
    player_role_maps = defaultdict(lambda: _state(100))
    teams = defaultdict(lambda: deque(maxlen=100))
    team_maps = defaultdict(lambda: deque(maxlen=100))
    player_opponents = defaultdict(lambda: deque(maxlen=100))
    role_matchups = defaultdict(lambda: deque(maxlen=100))
    roster_history = defaultdict(lambda: deque(maxlen=20))
    previous_players = set()
    generated: list[dict] = []

    for map_record in map_table.itertuples(index=False):
        key = (map_record.match_id, map_record.map)
        map_rows = groups[key]
        timestamp = pd.Timestamp(map_record.timestamp_internal)

        # Snapshot every player before adding any outcome from this map.
        for row in map_rows.itertuples(index=False):
            row_series = pd.Series(row._asdict())
            player_id = row_series["player_id"]
            team_id = row_series["team_id"]
            opponent_id = row_series.get("opponent_team_id")
            agent = str(row_series.get("agent", "__missing__"))
            map_name = str(row_series.get("map", "__missing__"))
            snapshot = _feature_snapshot(
                row_series,
                players[player_id],
                player_maps[(player_id, map_name)],
                player_agents[(player_id, agent)],
                player_roles[(player_id, str(row_series.get("role", "Flex/Unknown")))],
                teams[team_id],
                teams[opponent_id],
                timestamp,
            )
            snapshot["row_order_internal"] = row_series["row_order_internal"]
            generated.append(snapshot)

        # Update state only after all current-map snapshots were generated.
        team_ratings: dict = defaultdict(list)
        for row in map_rows.itertuples(index=False):
            row_series = pd.Series(row._asdict())
            rating = _numeric(row_series.get("rating2_all"))
            acs = _numeric(row_series.get("acs_all"))
            kda = _numeric(row_series.get("kda_all"))
            player_id = row_series["player_id"]
            team_id = row_series["team_id"]
            agent = str(row_series.get("agent", "__missing__"))
            map_name = str(row_series.get("map", "__missing__"))
            entry = {
                "rating": rating,
                "acs": acs,
                "kda": kda,
                "match_id": row_series["match_id"],
                "timestamp": timestamp,
            }
            players[player_id]["history"].append(entry)
            players[player_id]["all_match_ids"].add(row_series["match_id"])
            players[player_id]["last_timestamp"] = timestamp
            player_maps[(player_id, map_name)]["history"].append(entry)
            player_agents[(player_id, agent)]["history"].append(entry)
            player_roles[(player_id, str(row_series.get("role", "Flex/Unknown")))]["history"].append(entry)
            if rating is not None:
                team_ratings[team_id].append(rating)

        for team_id, ratings in team_ratings.items():
            teams[team_id].append(float(np.mean(ratings)))

    result = pd.DataFrame(generated).set_index("row_order_internal")
    result = result.reindex(range(len(work)))
    for column in HISTORICAL_FEATURES:
        work[column] = result[column].to_numpy() if column in result else np.nan
    work["match_year"] = work["timestamp_internal"].dt.year.astype("float64")
    work["match_month"] = work["timestamp_internal"].dt.month.astype("float64")
    work["match_dayofweek"] = work["timestamp_internal"].dt.dayofweek.astype("float64")
    if "form_fallback_level" in result:
        work["form_fallback_level"] = result["form_fallback_level"].to_numpy()
    elif "form_fallback_level" not in work:
        work["form_fallback_level"] = "__unknown__"
    if "match_importance" not in work:
        work["match_importance"] = 0.5
    work["match_importance"] = pd.to_numeric(work["match_importance"], errors="coerce").fillna(0.5)
    for column in ("event_stage", "event_round", "patch"):
        if column not in work:
            work[column] = "__unknown__"
    for column in ("map", "agent", "role", "team", "opponent_team", "team_pick", "event_stage", "event_round", "region", "patch", "form_fallback_level"):
        if column not in work:
            work[column] = "__unknown__"
        work[column] = work[column].fillna("__unknown__")
    return work.drop(columns=["timestamp_internal", "row_order_internal"])


def build_historical_features_from_csv(input_path: Path, output_path: Path) -> pd.DataFrame:
    df = pd.read_csv(input_path, low_memory=False)
    result = build_historical_features(df)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)
    return result


if __name__ == "__main__" and False:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    build_historical_features_from_csv(args.input, args.output)


# This definition intentionally follows the small original implementation
# above.  Keeping the public function in one module preserves imports from
# older notebooks while providing the timestamp-batched state machine used by
# new training and inference runs.
def build_historical_features(df: pd.DataFrame) -> pd.DataFrame:
    validate_dataset(df, require_targets=True)
    work = add_role_columns(df)
    if work.duplicated(["player_id", "match_id", "map"]).any():
        raise ValueError("Historical features require unique (player_id, match_id, map) rows")
    work["timestamp_internal"] = make_match_timestamp(work)
    if work["timestamp_internal"].isna().any():
        raise ValueError("Cannot build historical features with missing timestamps")
    work["row_order_internal"] = np.arange(len(work))
    maps = (work[["match_id", "map", "timestamp_internal"]]
            .drop_duplicates(["match_id", "map"])
            .sort_values(["timestamp_internal", "match_id", "map"]))
    groups = {(key[0], key[1]): group for key, group in work.groupby(["match_id", "map"], sort=False)}
    players = defaultdict(lambda: _state(500))
    player_maps = defaultdict(lambda: _state(100))
    player_agents = defaultdict(lambda: _state(100))
    player_roles = defaultdict(lambda: _state(100))
    teams = defaultdict(lambda: deque(maxlen=100))
    player_agent_maps = defaultdict(lambda: _state(100))
    player_role_maps = defaultdict(lambda: _state(100))
    team_maps = defaultdict(lambda: deque(maxlen=100))
    player_opponents = defaultdict(lambda: deque(maxlen=100))
    role_matchups = defaultdict(lambda: deque(maxlen=100))
    roster_history = defaultdict(lambda: deque(maxlen=20))
    previous_players = set()
    generated = []

    for timestamp, timestamp_maps in maps.groupby("timestamp_internal", sort=True):
        batch = [(record.match_id, record.map, groups[(record.match_id, record.map)])
                 for record in timestamp_maps.itertuples(index=False)]
        # Snapshot all maps in this timestamp before updating any of them.
        for match_id, map_name, map_rows in batch:
            current_rosters = defaultdict(set)
            current_roles = defaultdict(list)
            for source in map_rows.itertuples(index=False):
                source_row = pd.Series(source._asdict())
                current_rosters[source_row["team_id"]].add(source_row["player_id"])
                current_roles[source_row["team_id"]].append(str(source_row.get("role", "Flex/Unknown")))
            for source in map_rows.itertuples(index=False):
                row = pd.Series(source._asdict())
                player_id = row["player_id"]; team_id = row["team_id"]
                opponent_id = row.get("opponent_team_id")
                agent = str(row.get("agent", "__missing__")); role = str(row.get("role", "Flex/Unknown"))
                snapshot = _feature_snapshot(
                    row, players[player_id], player_maps[(player_id, str(map_name))],
                    player_agents[(player_id, agent)], player_roles[(player_id, role)],
                    teams[team_id], teams[opponent_id], pd.Timestamp(timestamp),
                )
                team_map_history = team_maps[(team_id, str(map_name))]
                opponent_map_history = team_maps[(opponent_id, str(map_name))]
                player_opponent_history = player_opponents[(player_id, opponent_id)]
                opponent_role = str(row.get("opponent_role", "Flex/Unknown"))
                role_matchup_history = role_matchups[(role, opponent_role)]
                team_wins = [entry.get("won") for entry in team_map_history if pd.notna(entry.get("won"))]
                opponent_wins = [entry.get("won") for entry in opponent_map_history if pd.notna(entry.get("won"))]
                roster = current_rosters[team_id]
                previous_roster = roster_history[team_id][-1] if roster_history[team_id] else None
                role_counts = {name: current_roles[team_id].count(name) for name in ("Duelist", "Initiator", "Controller", "Sentinel")}
                sample_player_ratings = []
                for roster_player in roster:
                    sample_player_ratings.extend(
                        entry["rating"] for entry in players[roster_player]["history"]
                        if entry.get("rating") is not None
                    )
                snapshot.update({
                    "team_acs_last_5": _metric_mean(teams[team_id], "acs", 5),
                    "team_kda_last_5": _metric_mean(teams[team_id], "kda", 5),
                    "opponent_acs_last_5": _metric_mean(teams[opponent_id], "acs", 5),
                    "opponent_kda_last_5": _metric_mean(teams[opponent_id], "kda", 5),
                    "team_map_win_rate": float(np.mean(team_wins)) if team_wins else np.nan,
                    "opponent_map_win_rate": float(np.mean(opponent_wins)) if opponent_wins else np.nan,
                    "team_map_matches": len(team_wins), "opponent_map_matches": len(opponent_wins),
                    "team_strength_of_schedule": _metric_mean(teams[team_id], "opponent_rating", 10),
                    "opponent_map_strength": _metric_mean(opponent_map_history, "rating", 5),
                    "player_opponent_rating_last_5": _metric_mean(player_opponent_history, "rating", 5),
                    "player_opponent_matches_played": len(player_opponent_history),
                    "role_matchup_rating_last_5": _metric_mean(role_matchup_history, "rating", 5),
                    "role_matchup_matches_played": len(role_matchup_history),
                    "team_elo_confidence": min(1.0, np.sqrt(len(teams[team_id])) / np.sqrt(10)) if _numeric(row.get("team_elo")) is not None else np.nan,
                    "lineup_continuity": (len(roster & previous_roster) / max(len(roster), 1)) if previous_roster is not None else np.nan,
                    "veterans_remaining": sum(1 for roster_player in roster if roster_player in previous_players),
                    "team_role_duelist_share": role_counts["Duelist"] / max(len(roster), 1),
                    "team_role_initiator_share": role_counts["Initiator"] / max(len(roster), 1),
                    "team_role_controller_share": role_counts["Controller"] / max(len(roster), 1),
                    "team_role_sentinel_share": role_counts["Sentinel"] / max(len(roster), 1),
                    "team_roster_synergy": float(np.mean(sample_player_ratings)) if sample_player_ratings else np.nan,
                    "player_agent_map_rating_last_5": _metric_mean(player_agent_maps[(player_id, agent, str(map_name))]["history"], "rating", 5),
                    "player_role_map_rating_last_5": _metric_mean(player_role_maps[(player_id, role, str(map_name))]["history"], "rating", 5),
                    "team_map_rating_last_5": _metric_mean(team_map_history, "rating", 5),
                })
                snapshot.update({
                    "row_order_internal": row["row_order_internal"],
                    "role": role,
                    "form_fallback_level": snapshot.get("form_fallback_level", "global-mean"),
                })
                generated.append(snapshot)

        # Only completed maps enter state after the complete timestamp batch.
        for match_id, map_name, map_rows in batch:
            team_stats = defaultdict(lambda: {"rating": [], "acs": [], "kda": []})
            for source in map_rows.itertuples(index=False):
                row = pd.Series(source._asdict())
                player_id = row["player_id"]; team_id = row["team_id"]
                agent = str(row.get("agent", "__missing__")); role = str(row.get("role", "Flex/Unknown"))
                entry = {
                    "rating": _numeric(row.get("rating2_all")),
                    "acs": _numeric(row.get("acs_all")),
                    "kda": _numeric(row.get("kda_all")),
                    "match_id": row["match_id"], "timestamp": pd.Timestamp(timestamp),
                }
                players[player_id]["history"].append(entry)
                players[player_id]["all_match_ids"].add(row["match_id"])
                players[player_id]["last_timestamp"] = pd.Timestamp(timestamp)
                player_maps[(player_id, str(map_name))]["history"].append(entry)
                player_agents[(player_id, agent)]["history"].append(entry)
                player_roles[(player_id, role)]["history"].append(entry)
                player_agent_maps[(player_id, agent, str(map_name))]["history"].append(entry)
                player_role_maps[(player_id, role, str(map_name))]["history"].append(entry)
                for metric in ("rating", "acs", "kda"):
                    if entry[metric] is not None:
                        team_stats[team_id][metric].append(entry[metric])
            for team_id, metrics in team_stats.items():
                team_source = next(
                    pd.Series(source._asdict()) for source in map_rows.itertuples(index=False)
                    if source.team_id == team_id
                )
                winner = str(team_source.get("map_winner", ""))
                team_name = str(team_source.get("team", ""))
                opponent_name = str(team_source.get("opponent_team", ""))
                won = 1.0 if winner == team_name else (0.0 if winner == opponent_name else np.nan)
                entry = {
                    metric: float(np.mean(values)) if values else np.nan
                    for metric, values in metrics.items()
                }
                entry.update({"won": won, "opponent_rating": _metric_mean(teams[_key(team_source.get("opponent_team_id"))], "rating", 1)})
                teams[team_id].append(entry)
                team_maps[(team_id, str(map_name))].append(entry)
                roster_history[team_id].append(set(current_rosters[team_id]))
                previous_players.update(current_rosters[team_id])

            for source in map_rows.itertuples(index=False):
                row = pd.Series(source._asdict())
                player_id = row["player_id"]; opponent_id = row.get("opponent_team_id")
                role = str(row.get("role", "Flex/Unknown")); opponent_role = str(row.get("opponent_role", "Flex/Unknown"))
                interaction = {"rating": _numeric(row.get("rating2_all")), "acs": _numeric(row.get("acs_all")), "kda": _numeric(row.get("kda_all"))}
                player_opponents[(player_id, opponent_id)].append(interaction)
                role_matchups[(role, opponent_role)].append(interaction)

    result = pd.DataFrame(generated).set_index("row_order_internal").reindex(range(len(work)))
    for column in HISTORICAL_FEATURES:
        work[column] = result[column].to_numpy() if column in result else np.nan
    if "form_fallback_level" in result:
        work["form_fallback_level"] = result["form_fallback_level"].to_numpy()
    elif "form_fallback_level" not in work:
        work["form_fallback_level"] = "__unknown__"
    if "match_importance" not in work:
        work["match_importance"] = 0.5
    work["match_importance"] = pd.to_numeric(work["match_importance"], errors="coerce").fillna(0.5)
    for column in ("event_stage", "event_round", "patch"):
        if column not in work:
            work[column] = "__unknown__"
    for column in ("map", "agent", "role", "team", "opponent_team", "team_pick", "event_stage", "event_round", "region", "patch", "form_fallback_level"):
        if column not in work:
            work[column] = "__unknown__"
        work[column] = work[column].fillna("__unknown__")
    work["match_year"] = work["timestamp_internal"].dt.year.astype("float64")
    work["match_month"] = work["timestamp_internal"].dt.month.astype("float64")
    work["match_dayofweek"] = work["timestamp_internal"].dt.dayofweek.astype("float64")
    return work.drop(columns=["timestamp_internal", "row_order_internal"])


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    build_historical_features_from_csv(args.input, args.output)
