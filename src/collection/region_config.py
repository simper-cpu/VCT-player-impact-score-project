"""Versioned VLR region mapping and semantic title checks."""

from __future__ import annotations

import re
from typing import Iterable

import pandas as pd


REGION_CONFIG_VERSION = "vlr-region-map-v2"
REGION_CONFIG = {
    "EMEA": {"region_id": 27, "title_markers": (r"\bemea\b", r"\beurope\b")},
    "AMERICAS": {"region_id": 26, "title_markers": (r"\bamericas\b", r"north america", r"latin america", r"\blatam\b", r"\bbrazil\b")},
    "PACIFIC": {"region_id": 28, "title_markers": (r"\bpacific\b", r"\bjapan\b", r"\bkorea\b", r"southeast asia", r"south asia", r"\bsea\b", r"\boceania\b")},
}

REGIONS = {name: config["region_id"] for name, config in REGION_CONFIG.items()}
_CHINA_MARKERS = (r"\bchina\b", r"\bchinese\b")


def semantic_region(title: object) -> str | None:
    """Infer an explicit region marker from an event title, if present."""
    value = "" if title is None else str(title).strip().lower()
    if not value or value in {"nan", "none"}:
        return None
    if any(re.search(pattern, value) for pattern in _CHINA_MARKERS):
        return "CHINA"
    for region, config in REGION_CONFIG.items():
        if any(re.search(pattern, value) for pattern in config["title_markers"]):
            return region
    return None


def validate_event_semantics(
    events: pd.DataFrame,
    requested_region: str,
    region_id: int | None = None,
    *,
    require_positive_match: bool = True,
) -> dict:
    """Validate event titles against the requested region, not crawler labels.

    Global titles such as Masters/Champions do not carry a regional marker and
    are reported as ``unclassified`` rather than incorrectly assigned.
    Explicitly conflicting titles are always errors.
    """
    region = requested_region.strip().upper()
    if region not in REGION_CONFIG:
        raise ValueError(f"Unsupported region: {requested_region!r}")
    if region_id is not None and int(region_id) != REGION_CONFIG[region]["region_id"]:
        raise ValueError(
            f"Region id mismatch for {region}: expected {REGION_CONFIG[region]['region_id']}, got {region_id}"
        )
    if "title" not in events.columns:
        raise ValueError("Event semantic validation requires a title column")

    inferred = events["title"].map(semantic_region)
    conflicting = inferred.notna() & inferred.ne(region)
    positive = inferred.eq(region)
    report = {
        "requested_region": region,
        "region_id": REGION_CONFIG[region]["region_id"],
        "rows": int(len(events)),
        "explicit_region_matches": int(positive.sum()),
        "unclassified_titles": int(inferred.isna().sum()),
        "conflicting_titles": int(conflicting.sum()),
        "conflicting_examples": events.loc[conflicting, "title"].dropna().astype(str).drop_duplicates().head(10).tolist(),
    }
    errors = []
    if conflicting.any():
        errors.append(f"{int(conflicting.sum())} event titles explicitly belong to another region")
    if require_positive_match and len(events) and not positive.any():
        errors.append("no event title explicitly matches the requested region")
    if errors:
        raise ValueError(f"Semantic region validation failed for {region}: {'; '.join(errors)}")
    return report


def expected_regions() -> tuple[str, ...]:
    return tuple(REGION_CONFIG)
