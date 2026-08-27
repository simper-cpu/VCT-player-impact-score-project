from asyncio import events
from pathlib import Path
from datetime import datetime

import re
import pandas as pd
import requests
from bs4 import BeautifulSoup
import time

URL = "https://www.vlr.gg/712803/fut-esports-vs-natus-vincere-vct-2026-emea-stage-2-w1"
INPUT_PATH = Path("data/raw/vlr_VCT_EMEA_events_URL.csv")
OUTPUT_PATH = Path("data/raw/vlr_VCT_EMEA_events.csv")


response = requests.get(URL)
response.raise_for_status
print(response.status_code)
soup = BeautifulSoup(response.content, "html.parser")
items = soup.find_all("div", class_ = "ovw-row")
item = items[1]
print(item.find("div", class_ = "ovw-player-name text-of").get_text().strip())
print(item.find("div", class_ ="ovw-player-tag ge-text-light").get_text().strip())
stats = item.find_all("span", class_ = "ovw_cell")
for stat in stats:
    print 