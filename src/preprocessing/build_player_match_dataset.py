from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PLAYERS_PATH = ROOT / "data/raw/players_stat.csv"
MATCHES_PATH = ROOT / "data/raw/EMEA_matches.csv"
MAPS_PATH = ROOT / "data/raw/match_maps_stat.csv"
OUTPUT_PATH = ROOT / "data/processed/player_match_dataset_cleaned.csv"


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

    ratio_cols = [
        "rating2_t", "rating2_ct", "rating2_all",
        "kast_t", "kast_ct", "kast_all",
        "adr_t", "adr_ct", "adr_all",
        "hsp_t", "hsp_ct", "hsp_all",
    ]
    for col in ratio_cols:
        if col in df.columns:
            df[col] = parse_percentage_series(df[col])

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

    return df


def build_feature_dataset():
    players = clean_player_stats(pd.read_csv(PLAYERS_PATH, low_memory=False))
    matches = pd.read_csv(MATCHES_PATH, low_memory=False)
    maps_df = pd.read_csv(MAPS_PATH, low_memory=False)

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

    dataset = players.merge(
        matches[["match_id", "event_id", "match_date", "match_time"]],
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

    team_summary = (
        dataset.groupby("team_id", as_index=False)
        .agg(
            team_avg_rating=("rating2_all", "mean"),
            team_avg_acs=("acs_all", "mean"),
            team_avg_kda=("kda_all", "mean"),
        )
        .rename(columns={"team_id": "team_id"})
    )

    opp_summary = (
        dataset.groupby("opponent_team_id", as_index=False)
        .agg(
            opp_avg_rating=("rating2_all", "mean"),
            opp_avg_acs=("acs_all", "mean"),
            opp_avg_kda=("kda_all", "mean"),
        )
        .rename(columns={"opponent_team_id": "opponent_team_id"})
    )

    dataset = dataset.merge(team_summary, on="team_id", how="left")
    dataset = dataset.merge(opp_summary, on="opponent_team_id", how="left")

    dataset["opp_rating_gap"] = dataset["team_avg_rating"] - dataset["opp_avg_rating"]
    dataset["opp_acs_gap"] = dataset["team_avg_acs"] - dataset["opp_avg_acs"]
    dataset["opp_kda_gap"] = dataset["team_avg_kda"] - dataset["opp_avg_kda"]

    keep_cols = [
        "player_name",
        "player_id",
        "team",
        "team_id",
        "opponent_team",
        "opponent_team_id",
        "agent",
        "map",
        "match_id",
        "event_id",
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
        "team_avg_rating",
        "team_avg_acs",
        "team_avg_kda",
        "opp_avg_rating",
        "opp_avg_acs",
        "opp_avg_kda",
        "opp_rating_gap",
        "opp_acs_gap",
        "opp_kda_gap",
    ]

    dataset = dataset[[col for col in keep_cols if col in dataset.columns]]
    dataset = dataset.dropna(subset=["rating2_all", "acs_all", "kills_all", "deaths_all"]).copy()
    dataset = dataset.reset_index(drop=True)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(OUTPUT_PATH, index=False)

    print(f"Clean dataset saved to: {OUTPUT_PATH}")
    print(f"Rows: {len(dataset)}")
    print(dataset.isna().sum().sort_values(ascending=False).head(10))

    return dataset


if __name__ == "__main__":
    build_feature_dataset()
