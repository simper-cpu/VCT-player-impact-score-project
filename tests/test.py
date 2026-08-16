from asyncio import events
from pathlib import Path
from datetime import datetime

import pandas as pd
import requests
from bs4 import BeautifulSoup

URL =  "https://www.vlr.gg/event/stats/2765/?sort=rating2&dir=desc"
INPUT_PATH = Path("data/raw/vlr_VCT_EMEA_events_URL.csv")
OUTPUT_PATH = Path("data/raw/players_test.csv")

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

def get_text(element, selector):
    tag = element.select_one(selector)
    if not tag:
        return None
    return tag.text.strip()


response = requests.get(URL)
response.raise_for_status() 
soup = BeautifulSoup(response.content, "html.parser")
player_stats = soup.find_all("tr")
for player in player_stats:
    player_data = {stat: get_text(player, selector) for stat, selector in STAT_SELECTION.items()}
    print(player_data)
    




















































































































































































































































































































































