import argparse
import csv
import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from time import sleep

import pandas as pd

from crawl_map import get_map_stats, get_match_id as get_map_match_id
from crawl_player_stats import get_player, get_match_id as get_player_match_id


BASE_DIR = Path(__file__).resolve().parents[2]
INPUT_PATH = BASE_DIR / "data/raw/EMEA_matches.csv"
PLAYER_OUTPUT = BASE_DIR / "data/raw/players_stat.csv"
MAP_OUTPUT = BASE_DIR / "data/raw/match_maps_stat.csv"
LOGGER = logging.getLogger(__name__)


def fetch_with_retry(fetcher, url, retries):
    for attempt in range(1, retries + 1):
        try:
            return fetcher(url)
        except Exception as error:
            if attempt == retries:
                LOGGER.error("Failed after %d attempts: %s (%s)", retries, url, error)
                return []
            sleep(2 ** (attempt - 1))


def existing_ids(output_path, required_columns):
    if not output_path.exists() or output_path.stat().st_size == 0:
        return set(), False

    columns = set(pd.read_csv(output_path, nrows=0).columns)
    needs_rebuild = not set(required_columns).issubset(columns)
    if needs_rebuild:
        return set(), True

    existing = pd.read_csv(output_path, usecols=["match_id"], dtype={"match_id": "string"})
    return set(existing["match_id"].dropna()), False


def crawl_dataset(
    matches,
    output_path,
    fetcher,
    match_id_parser,
    required_columns,
    workers,
    retries,
    force,
):
    already_crawled, rebuild = existing_ids(output_path, required_columns)
    rebuild = force or rebuild
    jobs = [
        (str(row.match_id), row.match_url)
        for row in matches.itertuples(index=False)
        if rebuild or str(row.match_id) not in already_crawled
    ]

    if not jobs:
        LOGGER.info("No pending URLs for %s", output_path.name)
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(".tmp.csv") if rebuild else output_path
    mode = "w" if rebuild else "a"
    write_header = rebuild or not output_path.exists() or output_path.stat().st_size == 0

    with ThreadPoolExecutor(max_workers=workers) as executor:
        results = executor.map(
            lambda job: fetch_with_retry(fetcher, job[1], retries),
            jobs,
        )

        writer = None
        with temporary_path.open(mode, newline="", encoding="utf-8") as file:
            for index, ((match_id, match_url), rows) in enumerate(zip(jobs, results), start=1):
                if rows:
                    if writer is None:
                        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
                        if write_header:
                            writer.writeheader()
                            write_header = False
                    writer.writerows(rows)
                    file.flush()
                LOGGER.info("%s: %d/%d finished", output_path.name, index, len(jobs))

    if rebuild:
        temporary_path.replace(output_path)


def main():
    parser = argparse.ArgumentParser(description="Parallel VLR match data crawler")
    parser.add_argument(
        "--target",
        choices=["players", "maps", "both"],
        default="both",
        help="Dataset to crawl",
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-crawl all URLs and replace the selected output file",
    )
    args = parser.parse_args()

    if args.workers < 1 or args.retries < 1:
        parser.error("workers and retries must be positive")

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    matches = pd.read_csv(INPUT_PATH, dtype={"match_id": "string"})
    matches = matches.drop_duplicates("match_id")

    if args.target in {"players", "both"}:
        crawl_dataset(
            matches,
            PLAYER_OUTPUT,
            get_player,
            get_player_match_id,
            ["match_id", "team_id"],
            args.workers,
            args.retries,
            args.force,
        )

    if args.target in {"maps", "both"}:
        crawl_dataset(
            matches,
            MAP_OUTPUT,
            get_map_stats,
            get_map_match_id,
            ["match_id", "team_1_id", "team_2_id"],
            args.workers,
            args.retries,
            args.force,
        )


if __name__ == "__main__":
    main()