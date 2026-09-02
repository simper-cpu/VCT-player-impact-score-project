from asyncio import events
from pathlib import Path
from datetime import datetime
from collections import Counter

import re
import pandas as pd
import requests
from bs4 import BeautifulSoup
import time
import logging
import csv

logger = logging.getLogger(__name__)
stats = Counter()

INPUT_PATH = Path("data/raw/EMEA_matches.csv")
OUTPUT_PATH = Path("data/raw/players_stat.csv")
INVALID_HREF_LOG = Path("data/logs/invalid_player_stats.csv")

def get_player_id(player):
    player_link = player.select_one('a[href^="/player/"]')
    if player_link:
        href = player_link.get("href")
        match = re.search(r"/player/(\d+)", href)
        return match.group(1) if match else None
    return None

def get_match_id(match_url):
    match_id = re.search(r"/(\d+)/", match_url)
    return match_id.group(1) if match_id else None

def get_player_stats(player, match_url, map):
    player_data = {}
    player_data["name"] = player.find("div", class_ = "ovw-player-name text-of").get_text().strip()
    player_data["team"] = player.find("div", class_ ="ovw-player-tag ge-text-light").get_text().strip()
    player_data["player_id"] = get_player_id(player)
    player_data["agent"] = player.find("img").get("alt").strip()
    player_data["map"] = map
    player_data["match_id"] = get_match_id(match_url)
    stat_columns = ["rating2", "acs", "kills", "deaths", "assists", "kast", "adr", "hsp", "fb", "fd"]
    for col in stat_columns:
        cell = player.find(attrs={"data-col": col})
        if cell:
            try:
                player_data[f"{col}_t"] = cell.find("span", class_="side mod-t").get_text().strip()
                player_data[f"{col}_ct"] = cell.find("span", class_="side mod-ct").get_text().strip()
                player_data[f"{col}_all"] = cell.find("span", class_="side mod-both").get_text().strip()
            except AttributeError:
                player_data[f"{col}_t"] = None
                player_data[f"{col}_ct"] = None
                player_data[f"{col}_all"] = None
                logger.warning("Cant find player data | match_url = %s", match_url)
        else:
            player_data[f"{col}_t"] = None
            player_data[f"{col}_ct"] = None
            player_data[f"{col}_all"] = None
            logger.warning("Cant find the %s stat | match url = %s", col, match_url)
    return player_data

def get_player(match_url):
    response = requests.get(match_url)
    response.raise_for_status()
    print(f"crawling {match_url}, status code: {response.status_code}")
    match_data = []
    soup = BeautifulSoup(response.content, "html.parser")
    map_list = soup.find_all("div", class_ = "vm-stats-game")
    for map in map_list:
        map_id = map.get("data-game-id")
        if not map_id:
            continue
        if map_id == "all":
            continue
        else:
            map_name_tag = map.find("div", class_ = "map")  
            if map_name_tag:
                map_name = map_name_tag.get_text(separator=" ", strip=True).split()[0]
            else:
                map_name = "Unknown"
        players = map.select("div.ovw-row:not(.mod-head)")
        for player in players:
            match_data.append(get_player_stats(player, match_url, map_name))
    return match_data

def get_all_matches(input_path):
    df = pd.read_csv(input_path)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    already_crawled = set()
    if OUTPUT_PATH.exists():
        existing = pd.read_csv(OUTPUT_PATH)
        already_crawled = set(existing["match_id"].astype(str))
    if OUTPUT_PATH.exists() and OUTPUT_PATH.stat().st_size > 0:
        existing = pd.read_csv(OUTPUT_PATH)
        already_crawled = set(existing["match_id"].astype(str))
    write_header = not OUTPUT_PATH.exists()
    with open(OUTPUT_PATH, "a", newline="", encoding="utf-8") as f:
        writer = None
        for _, row in df.iterrows():
            match_url = row["match_url"]
            match_id = get_match_id(match_url)
            if match_id in already_crawled:
                continue
            try:
                match_data = get_player(match_url)
            except Exception as e:
                logger.warning("Loi khi crawl %s: %s", match_url, e)
                continue  
            if not match_data:
                continue
            if writer is None:
                writer = csv.DictWriter(f, fieldnames=match_data[0].keys())
                if write_header:
                    writer.writeheader()
            for row_data in match_data:
                writer.writerow(row_data)
            f.flush()  
            time.sleep(0.1)

get_all_matches(INPUT_PATH)
    


    


