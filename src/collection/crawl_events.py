from pathlib import Path
import re

import pandas as pd
import requests
from bs4 import BeautifulSoup


URL = "https://www.vlr.gg/events"
OUTPUT_PATH = Path("data/raw/vlr_events.csv")

MONTHS = {
    "jan": 1,
    "feb": 2,
    "mar": 3,
    "apr": 4,
    "may": 5,
    "jun": 6,
    "jul": 7,
    "aug": 8,
    "sep": 9,
    "oct": 10,
    "nov": 11,
    "dec": 12,
}


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
        date = get_date(item)
        start_date, end_date = split_event_dates(date)
        events.append(
            {
                "title": title.text.strip() if title else None,
                "date": date,
                "start_date": start_date,
                "end_date": end_date,
                "region": get_region(item),
            }
        )

    return pd.DataFrame(events)


if __name__ == "__main__":
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    crawl_events().to_csv(OUTPUT_PATH, index=False)
