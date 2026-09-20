import argparse
import re
import csv
import time
import logging
import pandas as pd
import requests
from bs4 import BeautifulSoup
from pathlib import Path

from src.collection.collection_utils import add_region_arguments, region_name, region_path

# Cấu hình đường dẫn
# Cấu hình request - có timeout và User-Agent để tránh bị treo / bị chặn
REQUEST_TIMEOUT = 15
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    )
}

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

MATCH_HREF_PATTERN = re.compile(r"/(\d+)/")


def get_match_id(url):
    match = MATCH_HREF_PATTERN.search(url)
    return match.group(1) if match else None


def get_team_id(href):
    if href:
        match = re.search(r"/team/(\d+)/", href)
        return match.group(1) if match else None
    return None


def parse_score(tag, match_id=None, map_id=None, default=0):
    """Convert text của 1 thẻ <span> điểm số sang int.
    Nếu text rỗng / không phải số (map chưa đấu, bị hủy...) thì trả về
    default và log cảnh báo, thay vì làm crash cả script."""
    text = tag.get_text(strip=True)
    try:
        return int(text)
    except ValueError:
        logger.warning(
            f"Không parse được điểm số '{text}' (match_id={match_id}, map_id={map_id}) - dùng mặc định {default}"
        )
        return default


def get_map_stats(match_url):
    response = requests.get(match_url, headers=HEADERS, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    print(f"Crawling Map Data: {match_url} | Status: {response.status_code}")
    soup = BeautifulSoup(response.content, "html.parser")
    match_data = []
    match_id = get_match_id(match_url)

    match_header = soup.find("div", class_="match-header-vs")
    if match_header is None:
        raise ValueError(f"Không tìm thấy match-header-vs tại {match_url}")

    team_1_link = match_header.find("a", class_="match-header-link wf-link-hover mod-1")
    team_2_link = match_header.find("a", class_="match-header-link wf-link-hover mod-2")
    if team_1_link is None or team_2_link is None:
        raise ValueError(f"Không tìm thấy link đội (mod-1/mod-2) tại {match_url}")

    team_1_id = get_team_id(team_1_link.get("href"))
    team_2_id = get_team_id(team_2_link.get("href"))

    team_name_tags = match_header.find_all("div", class_="wf-title-med")
    if len(team_name_tags) < 2:
        raise ValueError(
            f"Không tìm đủ tên 2 đội (chỉ thấy {len(team_name_tags)}) tại {match_url}"
        )
    team_1_name = team_name_tags[0].get_text(strip=True)
    team_2_name = team_name_tags[1].get_text(strip=True)

    map_list = soup.find_all("div", class_="vm-stats-game")

    for map_html in map_list:
        map_id = map_html.get("data-game-id")

        if not map_id or map_id == "all":
            continue

        map_data = {
            "match_id": match_id,
            "map_id": map_id,
            "map_name": None,
            "team_pick": None,
            "ct_team": None,
            "t_team": None,
            "team_1": team_1_name,
            "team_1_id": team_1_id,
            "team_2": team_2_name,
            "team_2_id": team_2_id,
            "first_half_ct_win_rounds": 0,
            "first_half_t_win_rounds": 0,
            "second_half_ct_win_rounds": 0,
            "second_half_t_win_rounds": 0,
            "ot_team_ct_win": 0,
            "ot_team_t_win": 0,
            "map_winner": None
        }

        # this block try to identify the ct and t team and their name
        team_tags = map_html.find_all("div", class_="team-name")
        if len(team_tags) < 2:
            logger.warning(
                f"Không đủ team-name trong map_id={map_id} (match_id={match_id}) - bỏ qua map này"
            )
            continue

        team_1_first_half_siblings = team_tags[0].find_next_siblings("span")
        if not team_1_first_half_siblings:
            logger.warning(
                f"Không tìm thấy điểm số cho map_id={map_id} (match_id={match_id}) - bỏ qua map này"
            )
            continue
        team_1_first_half_span = team_1_first_half_siblings[0]

        if "mod-ct" in team_1_first_half_span.get("class", []):
            map_data["ct_team"] = team_1_name
            map_data["t_team"] = team_2_name
        elif "mod-t" in team_1_first_half_span.get("class", []):
            map_data["ct_team"] = team_2_name
            map_data["t_team"] = team_1_name

        # this block gets the score on both halfs
        for team in team_tags:
            if team:
                score_list = team.find_next_siblings("span")
                for score in score_list:
                    score_classes = score.get("class", [])
                    if "mod-ct" in score_classes:
                        if team.get_text(strip=True) == map_data["ct_team"]:
                            map_data["first_half_ct_win_rounds"] = parse_score(score, match_id, map_id)
                        elif team.get_text(strip=True) == map_data["t_team"]:
                            map_data["second_half_ct_win_rounds"] = parse_score(score, match_id, map_id)
                    elif "mod-t" in score_classes:
                        if team.get_text(strip=True) == map_data["ct_team"]:
                            map_data["second_half_t_win_rounds"] = parse_score(score, match_id, map_id)
                        elif team.get_text(strip=True) == map_data["t_team"]:
                            map_data["first_half_t_win_rounds"] = parse_score(score, match_id, map_id)
                    elif "mod-ot" in score_classes:
                        if team.get_text(strip=True) == map_data["ct_team"]:
                            map_data["ot_team_ct_win"] = parse_score(score, match_id, map_id)
                        elif team.get_text(strip=True) == map_data["t_team"]:
                            map_data["ot_team_t_win"] = parse_score(score, match_id, map_id)

        # this block gets the team picked
        map_name_tag = map_html.find("div", class_="map")
        if map_name_tag:
            map_name_span = map_name_tag.find("span", style=lambda v: v and "position: relative" in v)
            if map_name_span:
                # Lấy node text đầu tiên thay vì giả định contents[0] luôn là text thuần
                first_text_node = map_name_span.find(string=True, recursive=True)
                if first_text_node:
                    map_data["map_name"] = first_text_node.strip()
                else:
                    map_data["map_name"] = map_name_span.get_text(strip=True)
            if map_name_tag.find(class_="mod-1"):
                map_data["team_pick"] = team_1_name
            elif map_name_tag.find(class_="mod-2"):
                map_data["team_pick"] = team_2_name
            else:
                map_data["team_pick"] = "Decider"

        # this block gets the map winner
        if map_data["first_half_ct_win_rounds"] + map_data["second_half_t_win_rounds"] + map_data["ot_team_ct_win"] > map_data["first_half_t_win_rounds"] + map_data["second_half_ct_win_rounds"] + map_data["ot_team_t_win"]:
            map_data["map_winner"] = map_data["ct_team"]
        elif map_data["first_half_t_win_rounds"] + map_data["second_half_ct_win_rounds"] + map_data["ot_team_t_win"] > map_data["first_half_ct_win_rounds"] + map_data["second_half_t_win_rounds"] + map_data["ot_team_ct_win"]:
            map_data["map_winner"] = map_data["t_team"]
        else:
            map_data["map_winner"] = "Draw"

        match_data.append(map_data)

    return match_data


def get_all_matches(input_path, output_path, region):
    df = pd.read_csv(input_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    file_has_content = output_path.exists() and output_path.stat().st_size > 0

    already_crawled = set()
    if file_has_content:
        existing = pd.read_csv(output_path)
        already_crawled = set(existing["match_id"].astype(str))

    # Chỉ bỏ qua việc ghi header nếu file đã tồn tại VÀ đã có nội dung
    # (trước đây chỉ check exists() nên file rỗng sẽ không bao giờ có header)
    write_header = not file_has_content

    with open(output_path, "a", newline="", encoding="utf-8") as f:
        writer = None
        for _, row in df.iterrows():
            match_url = row["match_url"]
            match_id = get_match_id(match_url)

            if match_id in already_crawled:
                continue

            try:
                match_data = get_map_stats(match_url)
            except Exception:
                # logger.exception tự động in kèm traceback đầy đủ,
                # giúp biết chính xác lỗi nằm ở đâu thay vì chỉ 1 dòng chung chung
                logger.exception(f"Lỗi khi crawl {match_url}")
                continue

            if not match_data:
                continue
            for row_data in match_data:
                row_data["region"] = region

            if writer is None:
                writer = csv.DictWriter(f, fieldnames=match_data[0].keys())
                if write_header:
                    writer.writeheader()
                    write_header = False

            for row_data in match_data:
                writer.writerow(row_data)

            f.flush()
            time.sleep(0.5)
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Crawl map stats for one region")
    add_region_arguments(parser)
    args = parser.parse_args()
    region = region_name(args.region)
    input_path = region_path(args.raw_dir, region, "matches")
    output_path = region_path(args.raw_dir, region, "match_maps_stat")
    get_all_matches(input_path, output_path, region)