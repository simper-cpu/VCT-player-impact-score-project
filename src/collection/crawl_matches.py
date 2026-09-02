from asyncio import events
from pathlib import Path
from datetime import datetime
from collections import Counter

import re
import pandas as pd
import requests
from bs4 import BeautifulSoup
import time
import logging
import csv

logger = logging.getLogger(__name__)
stats = Counter()

INPUT_PATH = Path("data/raw/EMEA_events_URL.csv")
OUTPUT_PATH = Path("data/raw/EMEA_matches.csv")
MATCH_HREF_PATTERN = MATCH_HREF_PATTERN = re.compile(r"^/(?P<match_id>\d+)(?:/|$)")
INVALID_HREF_LOG = Path("data/logs/invalid_match_hrefs.csv")

def get_match_id(href):
    match_id = re.search(r"/match/(\d+)/", href)
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

def get_one_event(event_url):
    response = requests.get(event_url)
    response.raise_for_status()
    print("Crawling %s...., Status code: %d", event_url, response.status_code)
    soup = BeautifulSoup(response.content, "html.parser")
    items = soup.select("a.wf-module-item.match-item[href]")
    return items

def get_all_events(input_path):
    df = pd.read_csv(input_path)
    for ___, row in df.iterrows():
        event_id = row["event_id"]
        event_url = f"https://www.vlr.gg/event/matches/{event_id}/"
        for item in get_one_event(event_url):
            match_info = get_match_info(item,event_url)
            if match_info:
                matches["match_id"].append(match_info["match_id"])
                matches["match_url"].append(match_info["match_url"])
                matches["event_id"].append(event_url.split("/")[-2])
                matches["match_date"].append(match_info["match_date"])
                matches["match_time"].append(match_info["match_time"])
        time.sleep(0.1) 
    logger.info(
        "Tổng kết | hợp lệ=%d | sai mẫu=%d | thiếu link match=%d",
        stats["valid_match_href"],
        stats["invalid_match_href"],
        stats["missing_match_href"],
    )

matches = {"match_id": [], "match_url": [], "event_id": [], "match_date": [], "match_time": []}
get_all_events(INPUT_PATH)
data = pd.DataFrame(matches)

if __name__ == "__main__":
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(OUTPUT_PATH, index=False)

