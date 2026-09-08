import argparse
from datetime import datetime
from pathlib import Path
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

import re
import pandas as pd
import requests
from bs4 import BeautifulSoup
import time
import logging
import csv

from src.collection.collection_utils import add_region_arguments, region_name, region_path

logger = logging.getLogger(__name__)
stats = Counter()

# VLR event pages use links such as /15263/team-a-vs-team-b.
MATCH_HREF_PATTERN = re.compile(r"^/(?P<match_id>\d+)(?:/|$)")
INVALID_HREF_LOG = Path("data/logs/invalid_match_hrefs.csv")

def get_match_id(href):
    match_id = re.search(r"^/(\d+)(?:/|$)", href)
    return match_id.group(1) if match_id else None

def get_match_date_time(item):
    time_tag = item.find("div", class_="match-item-time")
    match_time = time_tag.get_text(strip=True) if time_tag else None
    formatted_date = None
    wf_card = item.find_parent("div", class_="wf-card")
    if wf_card:
        date_tag = wf_card.find_previous_sibling("div", class_="wf-label mod-large")
        if date_tag:
            raw_date = date_tag.get_text(strip=True) 
            try:
                # Chuyển đổi định dạng sang yyyy-mm-dd
                parsed_date = datetime.strptime(raw_date, "%a, %B %d, %Y")
                formatted_date = parsed_date.strftime("%Y-%m-%d")
            except ValueError:
                formatted_date = raw_date 

    return formatted_date, match_time
    

def get_match_info(item, event_url):
    raw_href = item.get("href").strip()
    if not raw_href:
        stats["missing_href"] += 1 
        logger.warning("Cant find href | event_url = %s", event_url)
        return None
    
    match = MATCH_HREF_PATTERN.match(raw_href)

    if not match:
        stats["invalid_match_href"] += 1
        logger.warning("Wrong href format | event_url = %s | href = %s", event_url, raw_href,)
        INVALID_HREF_LOG.parent.mkdir(parents=True, exist_ok=True)
        is_new = not INVALID_HREF_LOG.exists()
        with INVALID_HREF_LOG.open("a", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(
                file, fieldnames=["event_url", "href", "reason"]
            )
            if is_new:
                writer.writeheader()
            writer.writerow({
                "event_url": event_url,
                "href": raw_href,
                "reason": "invalid_match_href",
            })
        return None

    stats["valid_match_href"] += 1
    match_id = match.group(1)
    match_date, match_time = get_match_date_time(item)
    return {
        "match_id": match_id,
        "match_url": f"https://www.vlr.gg{raw_href}",
        "match_date": match_date,
        "match_time": match_time
    }

def get_one_event(event_url, retries=3):
    last_error = None
    for attempt in range(retries):
        try:
            response = requests.get(event_url, timeout=30)
            response.raise_for_status()
            print(f"Crawling {event_url} | Status code: {response.status_code}")
            soup = BeautifulSoup(response.content, "html.parser")
            return soup.select("a.wf-module-item.match-item[href]")
        except requests.RequestException as error:
            last_error = error
            if attempt + 1 < retries:
                time.sleep(2 ** attempt)
    raise last_error

def crawl_event_matches(event_id, region, retries):
    event_url = f"https://www.vlr.gg/event/matches/{event_id}/"
    rows = []
    for item in get_one_event(event_url, retries):
        match_info = get_match_info(item, event_url)
        if match_info:
            rows.append({
                "region": region,
                "match_id": match_info["match_id"],
                "match_url": match_info["match_url"],
                "event_id": str(event_id),
                "match_date": match_info["match_date"],
                "match_time": match_info["match_time"],
            })
    return rows


def get_all_events(input_path, region, workers=4, retries=3):
    matches = {"region": [], "match_id": [], "match_url": [], "event_id": [], "match_date": [], "match_time": []}
    df = pd.read_csv(input_path)
    event_ids = df["event_id"].dropna().drop_duplicates().tolist()
    with ThreadPoolExecutor(max_workers=workers) as executor:
        event_results = executor.map(
            lambda event_id: crawl_event_matches(event_id, region, retries),
            event_ids,
        )
        for rows in event_results:
            for row in rows:
                for column in matches:
                    matches[column].append(row[column])
    logger.info(
        "Tổng kết | hợp lệ=%d | sai mẫu=%d | thiếu link match=%d",
        stats["valid_match_href"],
        stats["invalid_match_href"],
        stats["missing_match_href"],
    )
    return pd.DataFrame(matches).drop_duplicates("match_id")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Crawl VLR matches for one region")
    add_region_arguments(parser)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--retries", type=int, default=3)
    args = parser.parse_args()
    region = region_name(args.region)
    input_path = region_path(args.raw_dir, region, "events_URL")
    output_path = region_path(args.raw_dir, region, "matches")
    result = get_all_events(input_path, region, args.workers, args.retries)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)
    print(f"Saved {len(result)} matches to {output_path}")

