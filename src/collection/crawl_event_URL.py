import re
import argparse

import pandas as pd
import requests
from bs4 import BeautifulSoup

from src.collection.collection_utils import add_region_arguments, add_region_column, region_name, region_path


def extract_event_id(event_url):
    if not event_url:
        return None
    match = re.search(r"/event/(\d+)(?:/|$)", event_url)
    return match.group(1) if match else None


def crawl_event_URL(region_id: int, region: str, max_pages: int = 20):
    events = []

    for page in range(1, max_pages + 1):
        page_url = f"https://www.vlr.gg/events/?region={region_id}&tier=60&page={page}"
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

    return add_region_column(pd.DataFrame(events), region)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Crawl VLR event URLs for one region")
    add_region_arguments(parser)
    parser.add_argument("--max-pages", type=int, default=20)
    args = parser.parse_args()
    region = region_name(args.region)
    output_path = region_path(args.raw_dir, region, "events_URL")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    crawl_event_URL(args.region_id, region, args.max_pages).to_csv(output_path, index=False)
    print(f"Saved {len(pd.read_csv(output_path))} event URLs to {output_path}")

