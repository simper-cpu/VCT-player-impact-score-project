import argparse
import re
import time
import pandas as pd
import requests
from bs4 import BeautifulSoup

from src.collection.collection_utils import add_region_arguments, add_region_column, region_name, region_path
from src.collection.region_config import REGIONS, validate_event_semantics

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
    location_item = item.find( lambda tag: tag.name == "div" and tag.get_text(strip=True) == "Location")
    if not location_item:
        return None
    location_lable = location_item.parent
    locaion_classes = location_lable.find("i").get("class")
    for css_class in locaion_classes:
        if css_class.startswith("mod-"):
            region_classes_raw = css_class
            region_code = css_class.split("-", 1)[1]
            return region_classes_raw, region_code
    return None,None


def get_date(item):
    date_item = item.find(lambda tag: tag.name == "div" and tag.get_text(strip=True) == "Dates")
    if not date_item:
        return None
    date_label = date_item.parent
    date_str = date_label.find("div", class_="value").get_text(strip=True)
    if date_str:
        return date_str
    return None

def split_event_dates(date_str):
    if not date_str:
        return None, None, None
    parts = re.split(r'\s*[–-]\s*|,\s*', date_str) #định dạng date_string đang là: jul 15 - jul 20, 2024
    start_date = parts[0].strip().lower()
    start_month = start_date.split(" ")[0]
    if len(parts) == 1:
        return start_date, None, None
    else:
        end_date = parts[1].strip().lower()
    if end_date.isdigit():
        end_date = f"{start_month} {end_date}"
    if len(parts) == 3:
        year = parts[2].strip()
        return start_date, end_date, year
    return start_date, end_date, None

def convert_date(date, year):
    if not date:
        return None
    date_month, date_day = date.split(" ")
    month = MONTHS[date_month]
    day = date_day.zfill(2)
    return f"{year}-{month}-{day}"



def crawl_one_event(url, event_id, retries=3):
    response = None
    for attempt in range(1, retries + 1):
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            break
        except requests.RequestException as error:
            if attempt == retries:
                print(f"Skipping event {event_id} after {retries} attempts: {error}")
                return None
            time.sleep(2 ** (attempt - 1))
    print("Crawling events from URL:", url)
    print(response.status_code)
    soup = BeautifulSoup(response.text, "html.parser")
    event_item = soup.find("div", class_ = "event-header-main")
    if not event_item:
        return None
    title = event_item.find("h1", class_="event-header-main-title")
    date = get_date(event_item)
    start_date, end_date, year = split_event_dates(date)
    event = {
        "id": event_id,
        "title": title.text.strip() if title else None,
        "date_raw": date,
        "start_date": convert_date(start_date, year) if year else convert_date(start_date, None),
        "end_date": convert_date(end_date, year) if year else convert_date(end_date, None),
        "source_region_raw": get_region(event_item)[0] if get_region(event_item) else None,
        "source_region": get_region(event_item)[1] if get_region(event_item) else None,
        }
    return event

def crawl_events(input_path, region, retries=3):
    df = pd.read_csv(input_path)
    events = []
    for index, row in df.iterrows():
        url = row["url"]
        event_id = row["event_id"]
        event_data = crawl_one_event(url, event_id, retries)
        if event_data:
            events.append(event_data)
    result = add_region_column(pd.DataFrame(events), region)
    validate_event_semantics(result, region, REGIONS[region])
    return result

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Crawl VLR event details for one region")
    add_region_arguments(parser)
    parser.add_argument("--retries", type=int, default=3)
    args = parser.parse_args()
    region = region_name(args.region)
    input_path = region_path(args.raw_dir, region, "events_URL")
    output_path = region_path(args.raw_dir, region, "events")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result = crawl_events(input_path, region, args.retries)
    result.to_csv(output_path, index=False)
    print(f"Saved {len(result)} events to {output_path}")
