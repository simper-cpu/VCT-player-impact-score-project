from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src.collection.collection_utils import DEFAULT_RAW_DIR, region_name
from src.collection.region_config import REGIONS, validate_event_semantics


DATASETS = {
    "events_URL": "events_URL",
    "events": "events",
    "matches": "matches",
    "players_stat": "players_stat",
    "match_maps_stat": "match_maps_stat",
}


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, low_memory=False) if path.exists() else pd.DataFrame()


def audit_region(region: str, raw_dir: Path, output_path: Path) -> dict:
    region = region_name(region)
    frames = {name: read_csv(raw_dir / f"{region}_{suffix}.csv") for name, suffix in DATASETS.items()}
    result = {"region": region, "files": {}, "checks": {}}
    for name, frame in frames.items():
        result["files"][name] = {
            "path": str(raw_dir / f"{region}_{DATASETS[name]}.csv"),
            "exists": not frame.empty,
            "rows": int(len(frame)),
            "columns": list(frame.columns),
            "missing": {str(k): int(v) for k, v in frame.isna().sum().items() if v},
            "duplicate_rows": int(frame.duplicated().sum()),
        }
        if "region" in frame:
            result["checks"][f"{name}_wrong_region"] = int((frame["region"].astype("string") != region).sum())
        else:
            result["checks"][f"{name}_missing_region_column"] = True

    events_url = frames["events_URL"]
    if not events_url.empty:
        try:
            semantic_report = validate_event_semantics(events_url, region, REGIONS[region])
            result["checks"]["events_URL_semantic"] = semantic_report
        except ValueError as error:
            result["checks"]["events_URL_semantic_error"] = str(error)

    matches = frames["matches"]
    players = frames["players_stat"]
    maps = frames["match_maps_stat"]
    if not matches.empty:
        match_ids = set(matches["match_id"].dropna().astype(str))
        result["checks"]["player_match_ids_missing_from_matches"] = int(
            (~players["match_id"].dropna().astype(str).isin(match_ids)).sum()
        ) if "match_id" in players else None
        result["checks"]["map_match_ids_missing_from_matches"] = int(
            (~maps["match_id"].dropna().astype(str).isin(match_ids)).sum()
        ) if "match_id" in maps else None
        result["checks"]["duplicate_match_ids"] = int(matches["match_id"].duplicated().sum()) if "match_id" in matches else None
    if not players.empty:
        key = [column for column in ("player_id", "match_id", "map") if column in players]
        result["checks"]["duplicate_player_match_map"] = int(players.duplicated(key).sum()) if len(key) == 3 else None
        for column in ("rating2_all", "acs_all", "kills_all", "deaths_all", "agent", "team_id"):
            if column in players:
                result["checks"][f"missing_{column}"] = int(players[column].isna().sum())
    if not maps.empty:
        key = [column for column in ("match_id", "map_name") if column in maps]
        result["checks"]["duplicate_match_map"] = int(maps.duplicated(key).sum()) if len(key) == 2 else None

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Raw audit saved to {output_path}")
    semantic_error = result["checks"].get("events_URL_semantic_error")
    if semantic_error:
        raise ValueError(semantic_error)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Audit one region's raw crawler files")
    parser.add_argument("--region", required=True)
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    region = region_name(args.region)
    output = args.output or args.raw_dir / f"{region}_raw_audit.json"
    audit_region(region, args.raw_dir, output)
