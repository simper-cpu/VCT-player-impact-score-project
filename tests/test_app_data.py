import numpy as np
import pandas as pd
import pytest

from src.app_data import (
    aggregate_match_history,
    build_chart_data,
    build_forecast_request,
    filter_completed_history,
    filter_team,
    get_recent_roster,
    get_opponent_options,
    get_team_options,
    load_local_dataset,
    run_forecast,
    validate_chart_metrics,
)


def _app_rows():
    rows = []
    for match_id, date, map_name, winner in [
        (1, "2024-01-01", "Ascent", "B"),
        (2, "2024-01-02", "Bind", "A"),
    ]:
        for player_id, player_name in [(1, "Alice"), (2, "Bob")]:
            rows.append({
                "player_name": player_name,
                "region": "EMEA",
                "player_id": player_id,
                "team": "A",
                "team_id": 10,
                "opponent_team": "B",
                "opponent_team_id": 20,
                "agent": "jett" if player_id == 1 else "omen",
                "map": map_name,
                "match_id": match_id,
                "event_id": 100 + match_id,
                "match_date": date,
                "match_time": "10:00 AM",
                "team_pick": "A",
                "map_winner": winner,
                "rating2_all": 1.0 + player_id / 10,
                "acs_all": 200 + player_id,
                "kda_all": 1.0,
                "kills_all": 10,
                "deaths_all": 8,
                "assists_all": 4,
                "adr_all": 140,
                "kast_all": 0.7,
            })
    return pd.DataFrame(rows)


def test_completed_history_filters_tbd_future_and_missing_targets():
    frame = _app_rows()
    frame = pd.concat([
        frame,
        frame.iloc[[0]].assign(match_id=3, map="TBD", map_winner="Draw", match_date="2024-01-03"),
        frame.iloc[[0]].assign(match_id=4, match_date="2099-01-01"),
        frame.iloc[[0]].assign(match_id=5, match_date="2024-01-04", rating2_all=np.nan),
    ], ignore_index=True)
    result = filter_completed_history(frame, cutoff="2024-02-01")
    assert set(result["match_id"]) == {1, 2}
    assert result["map"].eq("TBD").sum() == 0


def test_team_filter_roster_and_match_aggregation_are_map_level():
    frame = _app_rows()
    team = filter_team(frame, 10, "EMEA")
    roster = get_recent_roster(team, 10, n_maps=1)
    matches = aggregate_match_history(team)
    assert len(roster) == 2
    assert len(matches) == 2
    assert matches["match_id"].is_unique
    assert matches.loc[matches["match_id"].eq(1), "round_score"].iloc[0].startswith("CT/T")


def test_team_labels_are_unique_and_opponent_catalog_is_not_head_to_head_limited():
    frame = _app_rows()
    teams = get_team_options(frame)
    assert teams.groupby("region")["label"].apply(lambda values: values.is_unique).all()
    catalog = pd.DataFrame([
        {"team_id": 10, "display_name": "Alpha", "team": "A"},
        {"team_id": 20, "display_name": "Bravo", "team": "B"},
        {"team_id": 30, "display_name": "Charlie", "team": "C"},
    ])
    options = get_opponent_options(frame, all_teams=catalog, exclude_team_id=10)
    assert set(options["opponent_team_id"]) == {20, 30}


def test_load_local_dataset_falls_back_when_preferred_file_is_missing(tmp_path):
    fallback = tmp_path / "fallback.csv"
    _app_rows().to_csv(fallback, index=False)
    frame, source = load_local_dataset(tmp_path / "missing.csv", fallback)
    assert source == fallback
    assert len(frame) == 4


def test_chart_metrics_accept_only_one_or_two_variables():
    assert validate_chart_metrics(["Rating"]) == ["Rating"]
    chart = build_chart_data(_app_rows(), ["Rating", "ACS"])
    assert set(chart["metric"]) == {"Rating", "ACS"}
    with pytest.raises(ValueError):
        validate_chart_metrics([])
    with pytest.raises(ValueError):
        validate_chart_metrics(["Rating", "ACS", "KDA"])


def test_forecast_request_requires_opponent_and_supports_optional_context():
    source = _app_rows()
    source = pd.concat([
        source,
        source.iloc[[0]].assign(
            match_id=3, match_date="2024-01-03", opponent_team="Liquid", opponent_team_id=30,
        ),
    ], ignore_index=True)
    history = filter_completed_history(source, cutoff="2024-02-01")
    request = build_forecast_request(history, 10, 20, [1], "2024-02-03 12:00:00")
    assert len(request) == 1
    assert request.loc[0, "player_id"] == 1
    assert request.loc[0, "opponent_team_id"] == 20
    assert request.loc[0, "opponent_team"] == "B"
    assert "map" not in request.columns
    assert "agent" not in request.columns
    with pytest.raises(ValueError, match="opponent_team_id"):
        build_forecast_request(history, 10, None, [1], "2024-02-03 12:00:00")


def test_forecast_can_use_a_new_opponent_from_the_catalog():
    history = filter_completed_history(_app_rows(), cutoff="2024-02-01")
    request = build_forecast_request(
        history, 10, 99, [1], "2024-02-03 12:00:00", opponent_team_name="New Opponent",
    )
    assert request.loc[0, "opponent_team_id"] == 99
    assert request.loc[0, "opponent_team"] == "New Opponent"


class _DummyPipeline:
    def __init__(self, value):
        self.value = value

    def predict(self, frame):
        return np.full(len(frame), self.value)


def test_forecast_runs_with_unknown_agent_and_returns_low_confidence_metadata():
    history = filter_completed_history(_app_rows(), cutoff="2024-02-02 23:59:59")
    request = build_forecast_request(history, 10, 20, [1], "2024-02-03 12:00:00", agent_name="brand_new_agent")
    bundles = {
        "rating2_all": {"pipeline": _DummyPipeline(1.1)},
        "acs_all": {"pipeline": _DummyPipeline(210)},
        "kda_all": {"pipeline": _DummyPipeline(1.2)},
    }
    result = run_forecast(request, history, bundles)
    assert result.loc[0, "forecast_rating2_all"] == 1.1
    assert result.loc[0, "forecast_acs_all"] == 210
    assert "confidence_warning" in result
