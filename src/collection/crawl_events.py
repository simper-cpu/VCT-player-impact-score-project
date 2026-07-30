from pathlib import Path
from datetime import datetime

import pandas as pd
import requests
from bs4 import BeautifulSoup


URL = "https://www.vlr.gg/events/?tier=60"
OUTPUT_PATH = Path("data/raw/vlr_VCT_events.csv")

MONTHS = {
    "jan": "01",
    "feb": "02",
    "mar": "03",
    "apr": "04",
    "may": "05",
    "jun": "06",
    "jul": "07",
    "aug": "08",
    "sep": "09",
    "oct": "10",
    "nov": "11",
    "dec": "12",
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

def split_event_dates(date_str):
    if not date_str:
        return None, None
    parts = date_str.split("—")
    start_date = parts[0].strip().lower()
    start_month = start_date.split(" ")[0]
    if len(parts) == 1:
        return start_date, None
    else:
        end_part = parts[1].strip().lower()
    if end_part.isdigit():
        end_date = f"{start_month} {end_part}"
    else:
        end_date = end_part
    return start_date, end_date

def convert_date(date):
    if not date:
        return None
    date_month, date_day = date.split(" ")
    month = MONTHS[date_month]
    day = date_day.zfill(2)
    return f"{day}/{month}"


events = []
def crawl_events():
    for page in range(1, 20):
        page_url = f"{URL}&page={page}"
        response = requests.get(page_url, timeout=30)
        response.raise_for_status()
        print(response.status_code)

        soup = BeautifulSoup(response.text, "html.parser")
        event_items = soup.find_all("div", class_="event-item-inner")
        if not event_items:
            break
        for item in soup.find_all("div", class_="event-item-inner"):
            title = item.find("div", class_="event-item-title")
            date = get_date(item)
            start_date, end_date = split_event_dates(date)
            events.append(
                {
                    "title": title.text.strip() if title else None,
                    "date": date,
                    "start_date": convert_date(start_date),
                    "end_date": convert_date(end_date),
                    "region": get_region(item),
                }
            )
    return pd.DataFrame(events)


if __name__ == "__main__":
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    crawl_events().to_csv(OUTPUT_PATH, index=False)
