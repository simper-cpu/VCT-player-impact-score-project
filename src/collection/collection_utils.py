from __future__ import annotations

import argparse
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RAW_DIR = ROOT / "data" / "raw"


def add_region_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--region", required=True, help="Region label, for example EMEA")
    parser.add_argument("--region-id", required=True, type=int, help="VLR region query id")
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)


def region_name(value: str) -> str:
    normalized = value.strip().upper()
    if not normalized or any(character not in "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for character in normalized):
        raise ValueError(f"Invalid region name: {value!r}")
    return normalized


def region_path(raw_dir: Path, region: str, suffix: str) -> Path:
    return raw_dir / f"{region_name(region)}_{suffix}.csv"


def add_region_column(rows, region: str):
    rows = rows.copy()
    rows.insert(0, "region", region_name(region))
    return rows