import numpy as np
import pandas as pd
import pytest

from src.features.feature_schema import chronological_split, ensure_no_leakage, rolling_time_splits
from src.features.historical_features import build_historical_features
from src.features.role_mapping import add_role_columns, infer_role
from src.features.state_builder import build_inference_state
from src.features.feature_schema import MODEL_FEATURES
from src.preprocessing.build_player_match_dataset import clean_player_stats
from src.models.inference import prepare_inference_frame
from src.models.calibration import apply_prediction_calibration, fit_affine_calibration
from src.collection.crawl_matches import get_match_id


def _row(player_id, match_id, date, rating, team_id=10, opponent_id=20):
    return {
        "player_id": player_id,
        "match_id": match_id,
        "map": "Ascent",
        "match_date": date,
        "match_time": "10:00 AM",
        "team_id": team_id,
        "opponent_team_id": opponent_id,
        "team": "A",
        "opponent_team": "B",
        "agent": "jett",
        "team_pick": "A",
        "rating2_all": rating,
        "acs_all": 200.0,
        "kda_all": 1.0,
        "team_elo": 1500.0,
        "opponent_elo": 1500.0,
        "elo_gap": 0.0,
    }


def test_historical_features_use_previous_maps_only():
    frame = pd.DataFrame([
        _row(1, 1, "2024-01-01", 1.0),
        _row(1, 2, "2024-01-02", 2.0),
        _row(1, 3, "2024-01-03", 3.0),
    ])
    result = build_historical_features(frame)
    assert result.loc[0, "player_cold_start"] == 1
    assert pd.isna(result.loc[0, "player_rating_last"])
    assert result.loc[1, "player_rating_last"] == 1.0
    assert result.loc[2, "player_rating_last_3"] == 1.5
    assert result.loc[2, "player_rating_ewm_5"] < 2.0
    assert result.loc[2, "player_rating_form_gap"] == result.loc[2, "player_rating_ewm_5"] - 1.5
    assert result.loc[2, "player_matches_played"] == 2


def test_chronological_split_keeps_match_whole():
    frame = pd.DataFrame([
        _row(player, match, f"2024-01-{match:02d}", 1.0)
        for match in range(1, 11)
        for player in (1, 2)
    ])
    split = chronological_split(frame)
    groups = [set(split.train.match_id), set(split.validation.match_id), set(split.test.match_id)]
    assert not groups[0].intersection(groups[1])
    assert not groups[1].intersection(groups[2])
    assert not groups[0].intersection(groups[2])
    assert set(frame.match_id) == set().union(*groups)


def test_rolling_splits_are_forward_only():
    frame = pd.DataFrame([
        _row(player, match, f"2024-01-{match:02d}", 1.0)
        for match in range(1, 21)
        for player in (1, 2)
    ])
    windows = rolling_time_splits(frame, n_splits=3)
    assert len(windows) == 3
    for window in windows:
        assert max(window.train_match_ids) < min(window.validation_match_ids)
    assert max(windows[-1].validation_match_ids) < max(frame.match_id)


def test_leakage_schema_rejects_ids_and_post_map_values():
    with pytest.raises(ValueError):
        ensure_no_leakage(["elo_gap", "match_id"], "rating2_all")
    with pytest.raises(ValueError):
        ensure_no_leakage(["elo_gap", "rating2_all"], "rating2_all")


def test_sparse_inference_supports_unseen_player_and_agent():
    request = pd.DataFrame([{"map": "NewMap", "agent": "new_agent", "team": "NewTeam", "opponent_team": "Other"}])
    prepared = prepare_inference_frame(request)
    assert prepared.loc[0, "agent"] == "new_agent"
    assert prepared["player_rating_last_5"].isna().all()


def test_affine_calibration_expands_useful_validation_signal():
    actual = np.array([0.6, 0.9, 1.2, 1.5, 1.8])
    raw = np.array([0.8, 0.95, 1.1, 1.25, 1.4])
    calibration = fit_affine_calibration(actual, raw)
    calibrated = apply_prediction_calibration(raw, calibration)
    assert calibration["enabled"] is True
    assert np.std(calibrated) > np.std(raw)
    assert np.mean(np.abs(actual - calibrated)) < np.mean(np.abs(actual - raw))


def test_clean_player_stats_removes_duplicate_key():
    raw = pd.DataFrame([
        {"name": "p", "team": "A", "team_id": 1, "player_id": 1, "agent": "jett", "map": "Ascent", "match_id": 1,
         "rating2_all": 1, "acs_all": 200, "kills_all": 10, "deaths_all": 5, "assists_all": 2},
        {"name": "p", "team": "A", "team_id": 1, "player_id": 1, "agent": "jett", "map": "Ascent", "match_id": 1,
         "rating2_all": 1, "acs_all": 200, "kills_all": 10, "deaths_all": 5, "assists_all": 2},
    ])
    cleaned = clean_player_stats(raw)
    assert not cleaned.duplicated(["player_id", "match_id", "map"]).any()


def test_match_href_parser_accepts_vlr_event_links():
    assert get_match_id("/15263/aklabanlar-vs-bbl-esports") == "15263"
    assert get_match_id("/not-a-match") is None


def test_role_mapping_keeps_unknown_explicit_and_confidence():
    assert infer_role("Jett") == ("Duelist", 0.98)
    assert infer_role("brand_new_agent")[0] == "Flex/Unknown"
    result = add_role_columns(pd.DataFrame([{"agent": "brand_new_agent"}]))
    assert result.loc[0, "role"] == "Flex/Unknown"
    assert result.loc[0, "role_confidence"] == 0.0


def test_same_timestamp_maps_are_a_snapshot_barrier():
    first = _row(1, 1, "2024-01-01", 1.0, team_id=10, opponent_id=20)
    second = _row(2, 2, "2024-01-01", 99.0, team_id=20, opponent_id=10)
    result = build_historical_features(pd.DataFrame([first, second]))
    assert pd.isna(result.loc[0, "player_rating_last"])
    assert pd.isna(result.loc[1, "player_rating_last"])


def test_future_target_changes_do_not_change_past_snapshots():
    frame = pd.DataFrame([_row(1, 1, "2024-01-01", 1.0), _row(1, 2, "2024-01-02", 2.0)])
    changed = frame.copy()
    changed.loc[1, "rating2_all"] = 200.0
    left = build_historical_features(frame)
    right = build_historical_features(changed)
    assert pd.isna(left.loc[0, "player_rating_last"]) and pd.isna(right.loc[0, "player_rating_last"])
    assert left.loc[0, "player_cold_start"] == right.loc[0, "player_cold_start"]


def test_online_state_builder_matches_offline_snapshot():
    history = pd.DataFrame([_row(1, 1, "2024-01-01", 1.0)])
    request = pd.DataFrame([_row(1, 2, "2024-01-02", 2.0)])
    online = build_inference_state(request.drop(columns=["rating2_all", "acs_all", "kda_all"]), history)
    combined = pd.concat([history, request.assign(rating2_all=pd.NA, acs_all=pd.NA, kda_all=pd.NA)], ignore_index=True)
    offline = build_historical_features(combined).iloc[[-1]][list(MODEL_FEATURES)].reset_index(drop=True)
    pd.testing.assert_frame_equal(online.reset_index(drop=True), offline, check_dtype=False)
