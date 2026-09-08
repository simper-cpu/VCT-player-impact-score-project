import argparse
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUTPUT_PATH = ROOT / "data/processed/player_match_dataset_cleaned.csv"
INITIAL_ELO = 1500.0
ELO_K_FACTOR = 32.0
PLAYER_MAP_KEY = ["player_id", "match_id", "map"]
MATCH_MAP_KEY = ["match_id", "map"]


def parse_percentage_series(series):
    s = series.astype(str).str.strip()
    cleaned = s.str.replace("%", "", regex=False)
    numeric = pd.to_numeric(cleaned, errors="coerce")
    return numeric / 100.0


def clean_player_stats(df):
    df = df.copy()
    df["match_id"] = pd.to_numeric(df["match_id"], errors="coerce")
    df["player_id"] = pd.to_numeric(df["player_id"], errors="coerce")
    df["team_id"] = pd.to_numeric(df["team_id"], errors="coerce")

    # remove rows without team and without match id
    df = df.dropna(subset=["match_id", "team_id"]).copy()

    percentage_cols = [
        "kast_t", "kast_ct", "kast_all",
        "hsp_t", "hsp_ct", "hsp_all",
    ]
    for col in percentage_cols:
        if col in df.columns:
            df[col] = parse_percentage_series(df[col])

    rating_cols = ["rating2_t", "rating2_ct", "rating2_all"]
    for col in rating_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    adr_cols = ["adr_t", "adr_ct", "adr_all"]
    for col in adr_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    numeric_cols = [
        "acs_t", "acs_ct", "acs_all",
        "kills_t", "kills_ct", "kills_all",
        "deaths_t", "deaths_ct", "deaths_all",
        "assists_t", "assists_ct", "assists_all",
        "fb_t", "fb_ct", "fb_all",
        "fd_t", "fd_ct", "fd_all",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.rename(columns={"name": "player_name"})
    df["kda_all"] = (df["kills_all"] + df["assists_all"]) / df["deaths_all"].replace(0, pd.NA)
    df["kda_all"] = pd.to_numeric(df["kda_all"], errors="coerce")

    # The crawler can return the same player/map more than once.  Keeping one
    # deterministic row prevents duplicated targets and double-counted history.
    df = df.drop_duplicates(PLAYER_MAP_KEY, keep="last").copy()

    return df


def add_map_elo_features(dataset):
    """Add pre-map Elo ratings, updating each team once per unique map."""
    dataset = dataset.copy()
    dataset["match_timestamp"] = pd.to_datetime(
        dataset["match_date"].astype(str)
        + " "
        + dataset["match_time"].fillna("").astype(str),
        format="mixed",
        errors="coerce",
    )
    dataset["match_timestamp"] = dataset["match_timestamp"].fillna(
        pd.to_datetime(dataset["match_date"], format="mixed", errors="coerce")
    )

    map_records = (
        dataset[
            [
                "match_id",
                "map",
                "match_timestamp",
                "team_1",
                "team_2",
                "team_1_id",
                "team_2_id",
                "map_winner",
            ]
        ]
        .drop_duplicates(["match_id", "map"])
        .sort_values(["match_timestamp", "match_id", "map"])
    )

    ratings = {}
    map_elo_rows = []

    for record in map_records.itertuples(index=False):
        team_1_id = record.team_1_id
        team_2_id = record.team_2_id
        if pd.isna(team_1_id) or pd.isna(team_2_id):
            continue

        team_1_id = int(team_1_id)
        team_2_id = int(team_2_id)
        team_1_elo = ratings.get(team_1_id, INITIAL_ELO)
        team_2_elo = ratings.get(team_2_id, INITIAL_ELO)
        map_elo_rows.append(
            {
                "match_id": record.match_id,
                "map": record.map,
                "team_1_elo": team_1_elo,
                "team_2_elo": team_2_elo,
            }
        )

        if record.map_winner == record.team_1:
            team_1_score = 1.0
        elif record.map_winner == record.team_2:
            team_1_score = 0.0
        elif str(record.map_winner).strip().lower() == "draw":
            team_1_score = 0.5
        else:
            continue

        expected_team_1 = 1 / (1 + 10 ** ((team_2_elo - team_1_elo) / 400))
        team_1_change = ELO_K_FACTOR * (team_1_score - expected_team_1)
        ratings[team_1_id] = team_1_elo + team_1_change
        ratings[team_2_id] = team_2_elo - team_1_change

    map_elo = pd.DataFrame(map_elo_rows)
    dataset = dataset.merge(map_elo, on=["match_id", "map"], how="left")
    dataset["team_elo"] = dataset["team_1_elo"].where(
        dataset["team_id"] == dataset["team_1_id"], dataset["team_2_elo"]
    )
    dataset["opponent_elo"] = dataset["team_2_elo"].where(
        dataset["team_id"] == dataset["team_1_id"], dataset["team_1_elo"]
    )
    dataset["elo_gap"] = dataset["team_elo"] - dataset["opponent_elo"]
    return dataset.drop(columns=["match_timestamp", "team_1_elo", "team_2_elo"])


def build_feature_dataset(players_path, matches_path, maps_path, output_path=OUTPUT_PATH):
    players = clean_player_stats(pd.read_csv(players_path, low_memory=False))
    matches = pd.read_csv(matches_path, low_memory=False)
    maps_df = pd.read_csv(maps_path, low_memory=False)

    matches["match_id"] = pd.to_numeric(matches["match_id"], errors="coerce")
    maps_df["match_id"] = pd.to_numeric(maps_df["match_id"], errors="coerce")
    maps_df["team_1_id"] = pd.to_numeric(maps_df["team_1_id"], errors="coerce")
    maps_df["team_2_id"] = pd.to_numeric(maps_df["team_2_id"], errors="coerce")

    map_info = maps_df[[
        "match_id",
        "map_name",
        "team_1",
        "team_2",
        "team_1_id",
        "team_2_id",
        "team_pick",
        "map_winner",
        "first_half_ct_win_rounds",
        "first_half_t_win_rounds",
        "second_half_ct_win_rounds",
        "second_half_t_win_rounds",
        "ot_team_ct_win",
        "ot_team_t_win",
    ]].copy()
    map_info = map_info.rename(columns={"map_name": "map"})
    map_info = map_info.drop_duplicates(MATCH_MAP_KEY, keep="last").copy()

    match_columns = ["match_id", "event_id", "match_date", "match_time"]
    for optional_column in ("event_stage", "event_round", "stage", "round"):
        if optional_column in matches.columns:
            match_columns.append(optional_column)
    match_info = matches[match_columns].copy()
    match_info = match_info.rename(columns={"stage": "event_stage", "round": "event_round"})
    match_info = match_info.loc[:, ~match_info.columns.duplicated()]

    dataset = players.merge(
        match_info,
        on="match_id",
        how="left",
    )
    dataset = dataset.merge(map_info, on=["match_id", "map"], how="left")

    dataset["team_1_id"] = pd.to_numeric(dataset["team_1_id"], errors="coerce")
    dataset["team_2_id"] = pd.to_numeric(dataset["team_2_id"], errors="coerce")
    dataset["team_id"] = pd.to_numeric(dataset["team_id"], errors="coerce")

    dataset["opponent_team_id"] = dataset.apply(
        lambda row: row["team_2_id"] if row["team_id"] == row["team_1_id"] else row["team_1_id"],
        axis=1,
    )
    dataset["opponent_team"] = dataset.apply(
        lambda row: row["team_2"] if row["team_id"] == row["team_1_id"] else row["team_1"],
        axis=1,
    )
    dataset = add_map_elo_features(dataset)

    keep_cols = [
        "player_name",
        "region",
        "player_id",
        "team",
        "team_id",
        "opponent_team",
        "opponent_team_id",
        "team_elo",
        "opponent_elo",
        "elo_gap",
        "agent",
        "map",
        "match_id",
        "event_id",
        "event_stage",
        "event_round",
        "match_date",
        "match_time",
        "team_1",
        "team_2",
        "team_pick",
        "map_winner",
        "first_half_ct_win_rounds",
        "first_half_t_win_rounds",
        "second_half_ct_win_rounds",
        "second_half_t_win_rounds",
        "ot_team_ct_win",
        "ot_team_t_win",
        "rating2_all",
        "rating2_t",
        "rating2_ct",
        "acs_all",
        "acs_t",
        "acs_ct",
        "kills_all",
        "kills_t",
        "kills_ct",
        "deaths_all",
        "deaths_t",
        "deaths_ct",
        "assists_all",
        "assists_t",
        "assists_ct",
        "kast_all",
        "kast_t",
        "kast_ct",
        "adr_all",
        "adr_t",
        "adr_ct",
        "hsp_all",
        "hsp_t",
        "hsp_ct",
        "fb_all",
        "fb_t",
        "fb_ct",
        "fd_all",
        "fd_t",
        "fd_ct",
        "kda_all",
    ]

    dataset = dataset[[col for col in keep_cols if col in dataset.columns]]
    dataset = dataset.dropna(subset=["rating2_all", "acs_all", "kills_all", "deaths_all"]).copy()
    dataset = dataset.reset_index(drop=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(output_path, index=False)

    print(f"Clean dataset saved to: {output_path}")
    print(f"Rows: {len(dataset)}")
    print(dataset.isna().sum().sort_values(ascending=False).head(10))

    return dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build the cleaned player-map dataset")
    parser.add_argument("--players", type=Path, default=ROOT / "data/raw/all_regions_players_stat.csv")
    parser.add_argument("--matches", type=Path, default=ROOT / "data/raw/all_regions_matches.csv")
    parser.add_argument("--maps", type=Path, default=ROOT / "data/raw/all_regions_match_maps_stat.csv")
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    args = parser.parse_args()
    build_feature_dataset(args.players, args.matches, args.maps, args.output)
