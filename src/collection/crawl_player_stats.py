import argparse
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

from src.collection.collection_utils import add_region_arguments, region_name, region_path

logger = logging.getLogger(__name__)
stats = Counter()

INVALID_HREF_LOG = Path("data/logs/invalid_player_stats.csv")
REQUEST_TIMEOUT = 15
STAT_COLUMNS = ["rating2", "acs", "kills", "deaths", "assists", "kast", "adr", "hsp", "fb", "fd"]


def _text_or_none(element):
    """Return stripped text from a tag, or None when the tag/value is absent."""
    if element is None:
        return None
    value = element.get_text(strip=True)
    return value or None

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

def get_match_team_ids(soup):
    team_ids = []
    for team_link in soup.select('a.match-header-link[href^="/team/"]'):
        match = re.search(r"/team/(\d+)", team_link.get("href", ""))
        if match and match.group(1) not in team_ids:
            team_ids.append(match.group(1))
    return team_ids[:2]


def assign_team_ids_for_players(players, team_ids):
    team_ids_by_tag = {}
    assigned = []
    last_non_empty_team = None
    first_pass = []

    for idx, player in enumerate(players):
        tag = player.select_one("div.ovw-player-tag")
        team_tag = tag.get_text(strip=True) if tag else ""
        if team_tag:
            last_non_empty_team = team_tag
            if team_tag not in team_ids_by_tag and len(team_ids_by_tag) < len(team_ids):
                team_ids_by_tag[team_tag] = team_ids[len(team_ids_by_tag)]
        first_pass.append((player, team_tag, last_non_empty_team))

    for player, team_tag, last_non_empty_team in first_pass:
        if team_tag and team_tag in team_ids_by_tag:
            assigned_id = team_ids_by_tag[team_tag]
        elif last_non_empty_team and last_non_empty_team in team_ids_by_tag:
            assigned_id = team_ids_by_tag[last_non_empty_team]
        elif team_ids:
            if len(team_ids_by_tag) == 1:
                assigned_id = next(iter(team_ids_by_tag.values()))
            else:
                assigned_id = team_ids[0] if len(assigned) < len(players) / 2 else team_ids[1]
        else:
            assigned_id = None
        assigned.append((player, assigned_id))

    return assigned


def get_player_stats(player, match_url, map, team_id):
    player_data = {}
    player_data["name"] = _text_or_none(player.select_one(".ovw-player-name"))
    player_data["team"] = _text_or_none(player.select_one(".ovw-player-tag"))
    player_data["team_id"] = team_id
    player_data["player_id"] = get_player_id(player)
    agent = player.select_one(".ovw-agents img[alt]") or player.find("img", alt=True)
    player_data["agent"] = agent.get("alt", "").strip() if agent else None
    player_data["map"] = map
    player_data["match_id"] = get_match_id(match_url)
    for col in STAT_COLUMNS:
        cell = player.find(attrs={"data-col": col})
        if cell:
            for side, suffix in (("mod-t", "t"), ("mod-ct", "ct"), ("mod-both", "all")):
                value = _text_or_none(cell.select_one(f"span.side.{side}"))
                player_data[f"{col}_{suffix}"] = value
        else:
            player_data[f"{col}_t"] = None
            player_data[f"{col}_ct"] = None
            player_data[f"{col}_all"] = None
            logger.warning("Cant find the %s stat | match url = %s", col, match_url)
    return player_data

def get_player(match_url):
    response = requests.get(match_url, timeout=REQUEST_TIMEOUT)
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
        team_ids = get_match_team_ids(soup)
        assigned_players = assign_team_ids_for_players(players, team_ids)
        map_data = []
        for player, team_id in assigned_players:
            map_data.append(
                get_player_stats(
                    player,
                    match_url,
                    map_name,
                    team_id,
                )
            )
        # Some VLR pages (notably showmatches) contain player rows but no
        # actual agent or performance values. Do not emit unusable rows.
        if map_data and any(row.get("rating2_all") is not None for row in map_data):
            match_data.extend(map_data)
        elif map_data:
            logger.warning("No completed player stats | match_url = %s | map = %s", match_url, map_name)
    return match_data

def get_all_matches(input_path, output_path, region):
    df = pd.read_csv(input_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    already_crawled = set()
    rewrite_output = False
    if output_path.exists() and output_path.stat().st_size > 0:
        existing_columns = pd.read_csv(output_path, nrows=0).columns
        rewrite_output = "team_id" not in existing_columns
        if not rewrite_output:
            existing = pd.read_csv(output_path, usecols=["match_id"])
            already_crawled = set(existing["match_id"].astype(str))

    temporary_path = output_path.with_suffix(".tmp.csv") if rewrite_output else output_path
    write_header = rewrite_output or not output_path.exists()
    with open(temporary_path, "w" if rewrite_output else "a", newline="", encoding="utf-8") as f:
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
            for row_data in match_data:
                row_data["region"] = region
            if writer is None:
                writer = csv.DictWriter(f, fieldnames=match_data[0].keys())
                if write_header:
                    writer.writeheader()
            for row_data in match_data:
                writer.writerow(row_data)
            f.flush()  
            time.sleep(0.1)

    if rewrite_output:
        temporary_path.replace(output_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Crawl player stats for one region")
    add_region_arguments(parser)
    args = parser.parse_args()
    region = region_name(args.region)
    input_path = region_path(args.raw_dir, region, "matches")
    output_path = region_path(args.raw_dir, region, "players_stat")
    get_all_matches(input_path, output_path, region)
    


    


