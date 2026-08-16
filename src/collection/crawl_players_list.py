from asyncio import events
from pathlib import Path
from datetime import datetime

import re
import pandas as pd
import requests
from bs4 import BeautifulSoup
import time

# reuse a single Session for connection pooling and set a User-Agent
SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "Mozilla/5.0 (compatible; VCTProject/1.0)"})

INPUT_PATH = Path("data/raw/vlr_VCT_EMEA_events_URL.csv")
OUTPUT_PATH = Path("data/raw/players_by_events.csv")

STAT_SELECTION = {
    "player_name" : 'div.st-pl-name.text-of',
    "team" : 'div.st-pl-country',
    "maps" : 'td[data-col = "maps"]',
    "round" : 'td[data-col = "rnd"]',
    "rating" : 'td[data-col = "rating2"]',
    "acs" : 'td[data-col = "acs"]',
    "kd" : 'td[data-col = "kd"]',
    "kast" : 'td[data-col = "kast"]',
    "adr" : 'td[data-col = "adr"]',
    "kpr" : 'td[data-col = "kpr"]',
    "apr" : 'td[data-col = "apr"]',
    "fkfd" : 'td[data-col = "fkfd"]',
    "fkpr" : 'td[data-col = "fbpr"]',
    "fdpr" : 'td[data-col = "fdpr"]',
    "hsp" : 'td[data-col = "hsp"]',
    "clp" : 'td[data-col = "clp"]',
    "k" : 'td[data-col = "k"]',
    "d" : 'td[data-col = "d"]',
    "a" : 'td[data-col = "a"]',
    "fk" : 'td[data-col = "fk"]',
    "fd" : 'td[data-col = "fd"]',
}

def get_player_id(element):
    player_link = element.select_one('a[href^="/player/"]')
    if not player_link:
        return None
    href = player_link.get("href")
    id = re.search(r"/player/(\d+)", href)
    return id.group(1) if id else None

def get_text(element, selector):
    tag = element.select_one(selector)
    if not tag:
        return None
    return tag.text.strip()

def crawl_one_event(url):
    response = SESSION.get(url, timeout=30)
    response.raise_for_status()
    print("Crawling events from URL:", url)
    print(response.status_code)
    soup = BeautifulSoup(response.content, "html.parser")
    player_stats = soup.find_all("tr")
    event_data = []
    for player in player_stats:
        player_data = {stat: get_text(player, selector) for stat, selector in STAT_SELECTION.items()}
        player_data["player_id"] = get_player_id(player)
        event_data.append(player_data)
    return event_data

def crawl_events():
    df = pd.read_csv(INPUT_PATH)
    events = []
    for index, row in df.iterrows():
        event_id = row["event_id"]
        stats_url = (
            f"https://www.vlr.gg/event/stats/{event_id}/"
            "?sort=rating2&dir=desc"
        )
        event_data = crawl_one_event(stats_url)
        if event_data is None:
            print(f"Failed to crawl {stats_url}")
        for player_data in event_data:
            player_data["event_id"] = row["event_id"]
            events.append(player_data)
    time.sleep(0.5)  
    return pd.DataFrame(events)

if __name__ == "__main__":
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    crawl_events().to_csv(OUTPUT_PATH, index=False)









