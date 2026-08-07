from asyncio import events
from pathlib import Path
from datetime import datetime

import pandas as pd
import requests
from bs4 import BeautifulSoup

URL =  "https://www.vlr.gg/events/?region=27&tier=60"

OUTPUT_PATH = Path("data/raw/vlr_VCT_EMEA_events_URL_2.csv")


