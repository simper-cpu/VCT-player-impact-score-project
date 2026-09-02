from asyncio import events
from pathlib import Path
from datetime import datetime
import re

import pandas as pd
import requests
from bs4 import BeautifulSoup

URL =  "https://www.vlr.gg/events/?region=27&tier=60"
OUTPUT_PATH = Path("data/raw/EMEA_events_URL.csv")

events = []


def extract_event_id(event_url):
    if not event_url:
        return None
    match = re.search(r"/event/(\d+)(?:/|$)", event_url)
    return match.group(1) if match else None


def crawl_event_URL():
    events = []

    for page in range(1, 20):
        page_url = f"{URL}&page={page}"
        response = requests.get(page_url, timeout=30)
        response.raise_for_status()
        print(f"Scraping page {page}: {page_url} - Status code: {response.status_code}")
        soup = BeautifulSoup(response.content, "html.parser")

        status_labels = soup.select(
            "div.wf-label.mod-large.mod-upcoming, "
            "div.wf-label.mod-large.mod-ongoing, "
            "div.wf-label.mod-large.mod-completed"
        )

        if not status_labels:
            print("No event status sections found.")
            break

        for label in status_labels:
            classes = label.get("class", [])

            # 2. Xử lý triệt để cả 3 điều kiện
            if "mod-upcoming" in classes:
                status = "upcoming"
            elif "mod-ongoing" in classes:
                status = "ongoing"
            elif "mod-completed" in classes:
                status = "completed"
            else:
                continue
                
            container = label.parent
            event_items = container.find_all(
                "a",
                class_="wf-card mod-flex event-item",
                recursive=False,
            )

            for item in event_items:
                event_url = item.get("href")
                title = item.find("div", class_="event-item-title")
                status_element = item.select

                events.append({
                    "event_id": extract_event_id(event_url),
                    "title": title.get_text(strip=True) if title else None,
                    "url": f"https://www.vlr.gg{event_url}" if event_url else None,
                    "status": status,
                })

    return pd.DataFrame(events)

if __name__ == "__main__":
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    crawl_event_URL().to_csv(OUTPUT_PATH, index=False)

