from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
REGIONS = {"EMEA": 27, "AMERICAS": 2, "PACIFIC": 3}
DATASET_SUFFIXES = ("events_URL", "events", "matches", "players_stat", "match_maps_stat")


def load_checkpoint(checkpoint_path: Path) -> dict:
    if checkpoint_path.exists():
        return json.loads(checkpoint_path.read_text(encoding="utf-8"))
    return {region: {} for region in REGIONS}


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


def merge_dataset(raw_dir: Path, suffix: str, regions: list[str], key: list[str]) -> None:
    frames = []
    for region in regions:
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
        merged = merged.drop_duplicates(available_key, keep="last")
    merged.to_csv(raw_dir / f"all_regions_{suffix}.csv", index=False)
    print(f"Merged {suffix}: {len(merged)} rows")


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
            "src/preprocessing/build_player_match_dataset.py",
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
