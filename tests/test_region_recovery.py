import json

import pandas as pd
import pytest

from run_pipeline import audit_cross_region, config_signature, load_checkpoint, prune_future_raw_matches
from src.collection.crawl_parallel import completed_matches_only
from src.collection.region_config import REGIONS, semantic_region, validate_event_semantics


def test_live_region_mapping_is_versioned():
    assert REGIONS == {"EMEA": 27, "AMERICAS": 26, "PACIFIC": 28}
    assert semantic_region("VCT 2026: Americas Stage 2") == "AMERICAS"
    assert semantic_region("VCT 2026: Pacific Stage 2") == "PACIFIC"
    assert semantic_region("VCT 2026: EMEA Stage 2") == "EMEA"


def test_semantic_validation_rejects_wrong_region_even_when_region_column_is_self_assigned():
    events = pd.DataFrame({
        "region": ["AMERICAS", "AMERICAS"],
        "title": ["VCT 2026: EMEA Stage 2", "Valorant Masters London 2026"],
    })
    with pytest.raises(ValueError, match="another region"):
        validate_event_semantics(events, "AMERICAS", 26)


def test_checkpoint_is_invalidated_when_mapping_signature_changes(tmp_path):
    path = tmp_path / "crawl_checkpoint.json"
    path.write_text(json.dumps({"EMEA": {"events_URL": "completed"}, "config_signature": "old"}), encoding="utf-8")
    checkpoint = load_checkpoint(path)
    assert checkpoint["config_signature"] == config_signature()
    assert checkpoint["EMEA"] == {}


def test_cross_region_audit_blocks_unexplained_duplicate_match_ids(tmp_path):
    for region, match_id in [("EMEA", 100), ("AMERICAS", 100)]:
        pd.DataFrame([{
            "region": region, "match_id": match_id, "event_id": 1,
            "match_url": f"https://www.vlr.gg/{match_id}",
        }]).to_csv(tmp_path / f"{region}_matches.csv", index=False)
    report_path = tmp_path / "cross_region_audit.json"
    with pytest.raises(ValueError, match="match_id"):
        audit_cross_region(tmp_path, ["EMEA", "AMERICAS"], report_path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["checks"]["matches_cross_region_overlap_count"] == 1


def test_cross_region_audit_allows_same_match_from_shared_global_event(tmp_path):
    for region in ("EMEA", "AMERICAS"):
        pd.DataFrame([{
            "region": region,
            "event_id": 900,
            "title": "Valorant Masters Example",
        }]).to_csv(tmp_path / f"{region}_events_URL.csv", index=False)
        pd.DataFrame([{
            "region": region,
            "match_id": 100,
            "event_id": 900,
            "match_url": "https://www.vlr.gg/100/example",
        }]).to_csv(tmp_path / f"{region}_matches.csv", index=False)

    report = audit_cross_region(tmp_path, ["EMEA", "AMERICAS"], tmp_path / "cross_region_audit.json")

    assert report["critical_errors"] == []
    assert report["checks"]["matches_expected_global_overlap_count"] == 1


def test_stats_crawler_skips_future_and_tbd_matches_by_default():
    matches = pd.DataFrame({
        "match_id": [1, 2, 3],
        "match_date": ["2026-09-10", "2026-09-12", "TBD"],
    })
    result = completed_matches_only(matches, cutoff="2026-09-11")
    assert result["match_id"].tolist() == [1]


def test_recovery_prunes_future_match_rows_before_merge(tmp_path):
    pd.DataFrame([
        {"region": "PACIFIC", "match_id": 1, "match_date": "2026-09-10"},
        {"region": "PACIFIC", "match_id": 2, "match_date": "2026-09-12"},
    ]).to_csv(tmp_path / "PACIFIC_matches.csv", index=False)
    pd.DataFrame([{"match_id": 1, "team_id": 10}, {"match_id": 2, "team_id": 20}]).to_csv(
        tmp_path / "PACIFIC_players_stat.csv", index=False,
    )
    report = prune_future_raw_matches(tmp_path, ["PACIFIC"], cutoff="2026-09-11")
    assert report["PACIFIC"]["matches_after"] == 1
    assert pd.read_csv(tmp_path / "PACIFIC_players_stat.csv")["match_id"].tolist() == [1]
