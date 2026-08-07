from asyncio import events
from pathlib import Path
from datetime import datetime

import pandas as pd
import requests
from bs4 import BeautifulSoup

URL =  "https://www.vlr.gg/events/?region=27&tier=60"
OUTPUT_PATH = Path("data/raw/vlr_VCT_EMEA_events_URL.csv")

events = []

def crawl_event_URL():
    events = []

    for page in range(1, 20):
        page_url = f"{URL}&page={page}"
        response = requests.get(page_url, timeout=30)
        response.raise_for_status()

        soup = BeautifulSoup(response.content, "html.parser")

        status_labels = soup.select(
            "div.wf-label.mod-large.mod-upcoming, "
            "div.wf-label.mod-large.mod-completed"
        )

        if not status_labels:
            print("No event status sections found.")
            break

        for label in status_labels:
            classes = label.get("class", [])

            if "mod-upcoming" in classes:
                status = "upcoming"
            elif "mod-completed" in classes:
                status = "completed"
            else:
                continue

            # Đây là container đang chứa label và các event tương ứng
            container = label.parent

            # recursive=False: chỉ lấy event là con trực tiếp của container
            event_items = container.find_all(
                "a",
                class_="wf-card mod-flex event-item",
                recursive=False,
            )

            for item in event_items:
                event_url = item.get("href")
                title = item.find("div", class_="event-item-title")

                events.append({
                    "title": title.get_text(strip=True) if title else None,
                    "url": f"https://www.vlr.gg{event_url}" if event_url else None,
                    "status": status,
                })

    return pd.DataFrame(events)

if __name__ == "__main__":
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    crawl_event_URL().to_csv(OUTPUT_PATH, index=False)