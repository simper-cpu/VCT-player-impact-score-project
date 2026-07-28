from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup


URL = "https://www.vlr.gg/events"
OUTPUT_PATH = Path("data/raw/vlr_events.csv")


def get_region(item):
    icon = item.find("i", class_="flag")
    classes = icon.get("class", []) if icon else []
    for css_class in classes:
        if css_class.startswith("mod-"):
            return css_class.split("-", 1)[1]
    return None


def get_date(item):
    date_el = item.find("div", class_="event-item-desc-item mod-dates")
    if not date_el:
        return None

    for text in date_el.find_all(string=True, recursive=False):
        cleaned_date = text.strip()
        if cleaned_date:
            return cleaned_date
    return None


def crawl_events():
    response = requests.get(URL, timeout=30)
    response.raise_for_status()

    events = []
    soup = BeautifulSoup(response.text, "html.parser")
    for item in soup.find_all("div", class_="event-item-inner"):
        title = item.find("div", class_="event-item-title")
        events.append(
            {
                "title": title.text.strip() if title else None,
                "date": get_date(item),
                "region": get_region(item),
            }
        )

    return pd.DataFrame(events)


if __name__ == "__main__":
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    crawl_events().to_csv(OUTPUT_PATH, index=False)
