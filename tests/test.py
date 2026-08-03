from pathlib import Path 
import requests
from bs4 import BeautifulSoup
import pandas as pd


URL = "https://www.vlr.gg/event/2976/vct-2026-emea-stage-2"

response = requests.get(URL, timeout=30)
response.raise_for_status()
print(response.status_code)
soup = BeautifulSoup(response.text, "html.parser")
event_item = soup.find("div", class_ = "event-header-main")


date_item = event_item.find(lambda tag: tag.name == "div" and tag.get_text(strip=True) == "Dates")
date_label = date_item.parent
date = date_label.find("div", class_="value").get_text(strip=True)
print(date.replace(" –", ",").split(", "))
