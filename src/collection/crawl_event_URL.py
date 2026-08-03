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
    for page in range(1,20):
        page_url = f"{URL}&page={page}"
        response = requests.get(page_url, timeout = 30)
        response.raise_for_status()
        print(response.status_code)
        soup = BeautifulSoup(response.content, "html.parser")
        event_items = soup.find_all("a", class_="wf-card mod-flex event-item")
        if not event_items:
            print("No event items found.")
            break
        for item in event_items:
            event_url = item.get("href")
            if event_url:
                full_url = f"https://www.vlr.gg{event_url}"
            event_title = item.find("div", class_="event-item-title")
            if event_title:
                title_text = event_title.text.strip()
            events .append(
                {
                    "title": title_text, 
                    "url": full_url
                }
            )
    return pd.DataFrame(events)

if __name__ == "__main__":
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    crawl_event_URL().to_csv(OUTPUT_PATH, index=False)