from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


REQUIRED = {
    "matches": {"region", "match_id", "match_url", "event_id", "match_date"},
    "players_stat": {
        "region", "player_id", "team_id", "agent", "map", "match_id",
        "rating2_all", "acs_all", "kills_all", "deaths_all",
    },
    "match_maps_stat": {
        "region", "match_id", "map_name", "team_1_id", "team_2_id",
        "team_pick", "map_winner",
    },
}


def _read(raw_dir: Path, name: str) -> pd.DataFrame:
    path = raw_dir / f"all_regions_{name}.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path, low_memory=False)


def validate(raw_dir: Path, report_path: Path, max_target_missing: float = 0.05) -> dict:
    frames = {name: _read(raw_dir, name) for name in REQUIRED}
    errors: list[str] = []
    warnings: list[str] = []
    report = {"raw_dir": str(raw_dir), "tables": {}, "errors": errors, "warnings": warnings}

    for name, frame in frames.items():
        missing_columns = sorted(REQUIRED[name].difference(frame.columns))
        if missing_columns:
            errors.append(f"{name}: missing columns {missing_columns}")
        wrong_region = int((frame["region"].astype("string").isna()).sum()) if "region" in frame else 0
        if wrong_region:
            errors.append(f"{name}: {wrong_region} rows have missing region")
        report["tables"][name] = {
            "rows": int(len(frame)),
            "columns": list(frame.columns),
            "duplicate_rows": int(frame.duplicated().sum()),
            "missing": {str(k): int(v) for k, v in frame.isna().sum().items() if v},
        }

    matches = frames["matches"]
    players = frames["players_stat"]
    maps = frames["match_maps_stat"]
    if "match_id" in matches:
        match_ids = set(matches["match_id"].dropna().astype("string"))
        for name, frame in (("players_stat", players), ("match_maps_stat", maps)):
            if "match_id" in frame:
                missing_refs = int((~frame["match_id"].dropna().astype("string").isin(match_ids)).sum())
                if missing_refs:
                    errors.append(f"{name}: {missing_refs} match_id values are absent from matches")
        duplicate_matches = int(matches["match_id"].duplicated().sum())
        if duplicate_matches:
            errors.append(f"matches: {duplicate_matches} duplicate match_id values")

    if all(column in players for column in ("player_id", "match_id", "map")):
        duplicate_players = int(players.duplicated(["player_id", "match_id", "map"]).sum())
        if duplicate_players:
            errors.append(f"players_stat: {duplicate_players} duplicate player-match-map rows")
        target_columns = ["rating2_all", "acs_all", "kills_all", "deaths_all"]
        missing_rates = {}
        for target in target_columns:
            missing_rate = float(players[target].isna().mean()) if target in players else 1.0
            missing_rates[target] = missing_rate
            if missing_rate > max_target_missing:
                warnings.append(
                    f"players_stat: {target} missing rate {missing_rate:.2%}; "
                    "rows without completed match stats will be removed during cleaning"
                )
        complete_target_rows = int(players[target_columns].notna().all(axis=1).sum())
        report["tables"]["players_stat"]["complete_target_rows"] = complete_target_rows
        report["tables"]["players_stat"]["target_missing_rates"] = missing_rates
        if complete_target_rows == 0:
            errors.append("players_stat: no rows contain all required target values")

    if all(column in maps for column in ("match_id", "map_name")):
        duplicate_maps = int(maps.duplicated(["match_id", "map_name"]).sum())
        if duplicate_maps:
            errors.append(f"match_maps_stat: {duplicate_maps} duplicate match-map rows")

    for name, frame in frames.items():
        if "region" in frame:
            regions = sorted(frame["region"].dropna().astype(str).unique().tolist())
            report["tables"][name]["regions"] = regions
            if not regions:
                errors.append(f"{name}: no region values found")
        if name == "matches" and "match_date" in frame:
            invalid_dates = int(pd.to_datetime(frame["match_date"], errors="coerce").isna().sum())
            if invalid_dates:
                warnings.append(f"matches: {invalid_dates} rows have unparseable match_date")

    report["status"] = "failed" if errors else "passed"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate merged raw data before preprocessing/modeling")
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--max-target-missing", type=float, default=0.05)
    args = parser.parse_args()
    result = validate(args.raw_dir, args.report, args.max_target_missing)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    raise SystemExit(1 if result["status"] == "failed" else 0)