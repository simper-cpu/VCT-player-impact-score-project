from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import timezone
from pathlib import Path

import pandas as pd

from src.collection.region_config import REGION_CONFIG_VERSION, REGIONS

ROOT = Path(__file__).resolve().parent
DATASET_SUFFIXES = ("events_URL", "events", "matches", "players_stat", "match_maps_stat")


def config_signature() -> str:
    payload = json.dumps({"version": REGION_CONFIG_VERSION, "regions": REGIONS}, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def load_checkpoint(checkpoint_path: Path) -> dict:
    if checkpoint_path.exists():
        checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        if checkpoint.get("config_signature") == config_signature():
            return checkpoint
        print("Checkpoint config changed; invalidating previous crawl stages")
    return {**{region: {} for region in REGIONS}, "config_signature": config_signature(), "config_version": REGION_CONFIG_VERSION}


def save_checkpoint(checkpoint: dict, checkpoint_path: Path) -> None:
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = checkpoint_path.with_suffix(".tmp")
    temporary.write_text(json.dumps(checkpoint, indent=2), encoding="utf-8")
    temporary.replace(checkpoint_path)


def run(command: list[str]) -> None:
    print("+", " ".join(command))
    subprocess.run(command, cwd=ROOT, check=True)


def output_path(raw_dir: Path, region: str, suffix: str) -> Path:
    return raw_dir / f"{region}_{suffix}.csv"


def shared_event_ids(raw_dir: Path, regions: list[str]) -> set[str]:
    """Return event ids intentionally listed by more than one region.

    VLR exposes global events (Masters/Champions) in every regional event
    listing.  Their matches are the same records, so seeing those match ids in
    multiple regional crawls is expected.  We only classify an event as shared
    when the regional crawls agree on its title; an accidental id collision
    therefore remains a merge-blocking error.
    """
    rows = []
    for region in regions:
        path = output_path(raw_dir, region, "events_URL")
        if not path.exists():
            continue
        frame = pd.read_csv(path, usecols=lambda column: column in {"event_id", "title"}, low_memory=False)
        if "event_id" not in frame:
            continue
        frame = frame.dropna(subset=["event_id"]).copy()
        frame["event_id"] = frame["event_id"].astype("string").str.strip()
        frame["_source_region"] = region
        if "title" not in frame:
            frame["title"] = pd.NA
        rows.append(frame[["event_id", "title", "_source_region"]])

    if not rows:
        return set()

    catalog = pd.concat(rows, ignore_index=True)
    shared = set()
    for event_id, group in catalog.groupby("event_id", sort=False):
        if group["_source_region"].nunique() < 2:
            continue
        titles = (
            group["title"]
            .dropna()
            .astype(str)
            .str.strip()
            .str.casefold()
            .drop_duplicates()
        )
        if len(titles) == 1:
            shared.add(str(event_id))
    return shared


def merge_dataset(raw_dir: Path, suffix: str, regions: list[str], key: list[str]) -> None:
    frames = []
    # Keep global-event deduplication deterministic regardless of CLI order.
    ordered_regions = sorted(set(regions))
    for region in ordered_regions:
        path = output_path(raw_dir, region, suffix)
        if not path.exists():
            raise FileNotFoundError(path)
        frame = pd.read_csv(path, low_memory=False)
        if "region" not in frame:
            frame.insert(0, "region", region)
        frames.append(frame)
    merged = pd.concat(frames, ignore_index=True)
    available_key = [column for column in key if column in merged.columns]
    if available_key:
        source_order = {region: index for index, region in enumerate(ordered_regions)}
        merged["_merge_region_order"] = merged["region"].map(source_order).fillna(len(source_order))
        merged = (
            merged.sort_values("_merge_region_order", kind="stable")
            .drop_duplicates(available_key, keep="first")
            .drop(columns="_merge_region_order")
        )
    merged.to_csv(raw_dir / f"all_regions_{suffix}.csv", index=False)
    print(f"Merged {suffix}: {len(merged)} rows")


def audit_cross_region(raw_dir: Path, regions: list[str], output_path: Path) -> dict:
    """Audit identity overlap before any multi-region merge can discard rows."""
    result = {"regions": regions, "datasets": {}, "checks": {}, "critical_errors": []}
    shared_ids = shared_event_ids(raw_dir, regions)
    result["checks"]["shared_global_event_ids"] = sorted(shared_ids)
    for suffix, key in {
        "events_URL": "event_id", "events": "id", "matches": "match_id",
        "players_stat": "match_id", "match_maps_stat": "match_id",
    }.items():
        frames = []
        for region in regions:
            path = raw_dir / f"{region}_{suffix}.csv"
            if not path.exists():
                continue
            frame = pd.read_csv(path, low_memory=False)
            frame["_source_region"] = region
            frames.append(frame)
        if not frames:
            continue
        combined = pd.concat(frames, ignore_index=True)
        if key not in combined.columns:
            continue
        identity = combined.dropna(subset=[key]).groupby(key)["_source_region"].agg(lambda values: sorted(set(values)))
        overlap = identity[identity.map(len).gt(1)]
        result["checks"][f"{suffix}_cross_region_overlap_count"] = int(len(overlap))
        result["checks"][f"{suffix}_cross_region_overlap_examples"] = [
            {"identity": str(index), "regions": regions_for_identity}
            for index, regions_for_identity in overlap.head(20).items()
        ]
        result["datasets"][suffix] = {"rows": int(len(combined)), "identity_column": key}
        if suffix == "matches" and len(overlap):
            expected_global = 0
            unexpected = []
            for match_id, group in combined.dropna(subset=[key]).groupby(key, sort=False):
                source_regions = sorted(group["_source_region"].dropna().unique().tolist())
                if len(source_regions) < 2:
                    continue
                event_ids = group["event_id"].dropna().astype("string").str.strip().unique().tolist() if "event_id" in group else []
                urls = group["match_url"].dropna().astype("string").str.strip().unique().tolist() if "match_url" in group else []
                is_expected_global = (
                    len(event_ids) == 1
                    and event_ids[0] in shared_ids
                    and len(urls) == 1
                )
                if is_expected_global:
                    expected_global += 1
                else:
                    unexpected.append(str(match_id))
            result["checks"]["matches_expected_global_overlap_count"] = expected_global
            result["checks"]["matches_unexpected_overlap_count"] = len(unexpected)
            result["checks"]["matches_unexpected_overlap_examples"] = unexpected[:20]
            if unexpected:
                result["critical_errors"].append(
                    f"{len(unexpected)} match_id values occur in more than one region"
                )

    # Team ids can legitimately recur in global events, so report them for
    # review but do not treat them as a merge-blocking error by themselves.
    team_frames = []
    for region in regions:
        path = raw_dir / f"{region}_players_stat.csv"
        if path.exists():
            frame = pd.read_csv(path, usecols=lambda column: column in {"team_id", "region"}, low_memory=False)
            frame["_source_region"] = region
            team_frames.append(frame)
    if team_frames:
        teams = pd.concat(team_frames, ignore_index=True).dropna(subset=["team_id"])
        team_overlap = teams.groupby("team_id")["_source_region"].agg(lambda values: sorted(set(values)))
        team_overlap = team_overlap[team_overlap.map(len).gt(1)]
        result["checks"]["players_stat_team_id_cross_region_overlap_count"] = int(len(team_overlap))
        result["checks"]["players_stat_team_id_cross_region_overlap_examples"] = [
            {"team_id": str(index), "regions": regions_for_identity}
            for index, regions_for_identity in team_overlap.head(20).items()
        ]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    if result["critical_errors"]:
        raise ValueError(
            "Cross-region audit blocked merge: " + "; ".join(result["critical_errors"])
        )
    return result


def prune_future_raw_matches(raw_dir: Path, regions: list[str], cutoff=None) -> dict:
    """Remove future/TBD match rows from recovery artifacts before merge."""
    cutoff = pd.Timestamp(cutoff or pd.Timestamp.now(tz=timezone.utc).tz_localize(None)).normalize()
    report = {}
    for region in regions:
        matches_path = raw_dir / f"{region}_matches.csv"
        if not matches_path.exists():
            continue
        matches = pd.read_csv(matches_path, low_memory=False)
        dates = pd.to_datetime(matches.get("match_date"), errors="coerce")
        keep = dates.notna() & dates.le(cutoff)
        valid_ids = set(matches.loc[keep, "match_id"].dropna().astype(str))
        report[region] = {"matches_before": int(len(matches)), "matches_after": int(keep.sum())}
        matches.loc[keep].to_csv(matches_path, index=False)
        for suffix in ("players_stat", "match_maps_stat"):
            path = raw_dir / f"{region}_{suffix}.csv"
            if not path.exists():
                continue
            frame = pd.read_csv(path, low_memory=False)
            if "match_id" in frame:
                frame = frame[frame["match_id"].astype(str).isin(valid_ids)]
                frame.to_csv(path, index=False)
                report[region][f"{suffix}_rows_after"] = int(len(frame))
    (raw_dir / "future_rows_pruned.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the multi-region VCT crawl with checkpoints")
    parser.add_argument("--regions", nargs="+", choices=sorted(REGIONS), default=list(REGIONS))
    parser.add_argument("--force", action="store_true", help="Rerun completed crawler stages")
    parser.add_argument("--max-pages", type=int, default=20)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--skip-modeling", action="store_true")
    parser.add_argument("--raw-dir", type=Path, default=ROOT / "data" / "raw")
    parser.add_argument("--quality-report", type=Path)
    parser.add_argument("--max-target-missing", type=float, default=0.05)
    args = parser.parse_args()

    raw_dir = args.raw_dir if args.raw_dir.is_absolute() else ROOT / args.raw_dir
    checkpoint_path = raw_dir / "crawl_checkpoint.json"
    checkpoint = load_checkpoint(checkpoint_path)
    for region in args.regions:
        region_checkpoint = checkpoint.setdefault(region, {})
        base = [sys.executable, "-m"]
        stages = [
            ("events_URL", base + ["src.collection.crawl_event_URL", "--region", region, "--region-id", str(REGIONS[region]), "--raw-dir", str(raw_dir), "--max-pages", str(args.max_pages)]),
            ("events", base + ["src.collection.crawl_events", "--region", region, "--region-id", str(REGIONS[region]), "--raw-dir", str(raw_dir), "--retries", str(args.retries)]),
            ("matches", base + ["src.collection.crawl_matches", "--region", region, "--region-id", str(REGIONS[region]), "--raw-dir", str(raw_dir), "--workers", str(args.workers), "--retries", str(args.retries)]),
            ("stats", base + ["src.collection.crawl_parallel", "--region", region, "--region-id", str(REGIONS[region]), "--raw-dir", str(raw_dir), "--target", "both", "--workers", str(args.workers), "--retries", str(args.retries)]),
            ("raw_audit", base + ["src.collection.analyze_raw", "--region", region, "--raw-dir", str(raw_dir)]),
        ]
        for stage, command in stages:
            if region_checkpoint.get(stage) == "completed" and not args.force:
                print(f"Skipping {region}/{stage}")
                continue
            run(command)
            region_checkpoint[stage] = "completed"
            save_checkpoint(checkpoint, checkpoint_path)

    regions = list(args.regions)
    prune_future_raw_matches(raw_dir, regions)
    audit_cross_region(raw_dir, regions, raw_dir / "cross_region_audit.json")
    merge_dataset(raw_dir, "matches", regions, ["match_id"])
    merge_dataset(raw_dir, "players_stat", regions, ["player_id", "match_id", "map"])
    merge_dataset(raw_dir, "match_maps_stat", regions, ["match_id", "map_name"])
    checkpoint["merge"] = "completed"
    save_checkpoint(checkpoint, checkpoint_path)

    quality_report = args.quality_report or raw_dir / "all_regions_quality_report.json"
    if not quality_report.is_absolute():
        quality_report = ROOT / quality_report
    run([
        sys.executable,
        "-m",
        "src.preprocessing.data_quality_gate",
        "--raw-dir",
        str(raw_dir),
        "--report",
        str(quality_report),
        "--max-target-missing",
        str(args.max_target_missing),
    ])
    checkpoint["quality_gate"] = "passed"
    save_checkpoint(checkpoint, checkpoint_path)

    if not args.skip_modeling:
        cleaned_path = ROOT / "data" / "processed" / "player_match_dataset_cleaned_all_regions.csv"
        run([
            sys.executable,
            "-m",
            "src.preprocessing.build_player_match_dataset",
            "--players", str(raw_dir / "all_regions_players_stat.csv"),
            "--matches", str(raw_dir / "all_regions_matches.csv"),
            "--maps", str(raw_dir / "all_regions_match_maps_stat.csv"),
            "--output", str(cleaned_path),
        ])
        run([sys.executable, "-m", "src.models.train_models", "--input", str(cleaned_path)])
        run([sys.executable, "-m", "src.models.evaluate_models", "--input", str(cleaned_path)])
        checkpoint["modeling"] = "completed"
        save_checkpoint(checkpoint, checkpoint_path)


if __name__ == "__main__":
    main()
