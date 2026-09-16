"""Streamlit VCT performance explorer backed by local CSV data."""

from __future__ import annotations

from datetime import datetime, time
import json
from pathlib import Path

import pandas as pd
import streamlit as st

from src.app_data import (
    CHART_METRICS,
    DEFAULT_DATA_PATH,
    DEFAULT_MODEL_DIR,
    TARGETS,
    aggregate_match_history,
    build_chart_data,
    build_forecast_request,
    filter_completed_history,
    filter_team,
    get_opponent_options,
    get_player_history,
    get_recent_roster,
    get_team_options,
    load_local_dataset,
    run_forecast,
)
from src.models.inference import load_model


st.set_page_config(page_title="VCT Performance Explorer", page_icon="🎯", layout="wide")

PROJECT_ROOT = Path(__file__).resolve().parent
RECOVERY_STATUS_PATH = PROJECT_ROOT / "data" / "recovery" / "region_recovery_20260911" / "recovery_status.json"

if RECOVERY_STATUS_PATH.exists():
    recovery_status = json.loads(RECOVERY_STATUS_PATH.read_text(encoding="utf-8"))
    if recovery_status.get("status") != "verified":
        st.error("Streamlit is paused while region recovery is in progress. Existing merged data/models are not trusted.")
        st.stop()
else:
    recovery_status = {}

ACTIVE_DATA_PATH = Path(recovery_status.get("processed_path", DEFAULT_DATA_PATH))
ACTIVE_MODEL_DIR = Path(recovery_status.get("model_dir", DEFAULT_MODEL_DIR))


@st.cache_data(show_spinner="Loading local VCT data…")
def load_cached_data(path: str, fallback: str, reload_token: int) -> tuple[pd.DataFrame, str]:
    frame, source = load_local_dataset(Path(path), Path(fallback))
    return frame, str(source)


@st.cache_resource(show_spinner="Loading exported model…")
def load_cached_model(target: str, model_dir: str) -> dict:
    return load_model(target, Path(model_dir))


def _display_columns(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    return frame[[column for column in columns if column in frame.columns]].copy()


st.title("🎯 VCT Performance Explorer")
st.caption("Local-data explorer for team history, player form and estimated future-map performance.")

if "reload_token" not in st.session_state:
    st.session_state.reload_token = 0
if st.sidebar.button("Reload local data"):
    st.session_state.reload_token += 1
    st.rerun()

try:
    data, source_path = load_cached_data(
        str(ACTIVE_DATA_PATH),
        str(ACTIVE_DATA_PATH.parent / "player_match_dataset_cleaned.csv"),
        st.session_state.reload_token,
    )
except Exception as exc:  # pragma: no cover - rendered by Streamlit, not unit tests
    st.error(f"Could not load local data: {exc}")
    st.stop()

history = filter_completed_history(data)
team_options = get_team_options(history)
if team_options.empty:
    st.error("No completed teams are available in the local dataset.")
    st.stop()

st.sidebar.caption(f"Data source: `{Path(source_path).name}`")
regions = team_options["region"].dropna().astype(str).unique().tolist()
selected_region = st.sidebar.selectbox("Region", regions)
region_teams = team_options[team_options["region"].eq(selected_region)].reset_index(drop=True)
team_label = st.sidebar.selectbox("Team", region_teams["label"].tolist())
selected_team = region_teams.loc[region_teams["label"].eq(team_label)].iloc[0]
selected_team_id = selected_team["team_id"]
team_history = filter_team(history, selected_team_id, selected_region)
roster = get_recent_roster(team_history, selected_team_id)
match_history = aggregate_match_history(team_history)

tabs = st.tabs(["Team Overview", "Match History", "Players / Player Detail", "Forecast"])

with tabs[0]:
    st.subheader(f"{selected_team['display_name']} · {selected_region}")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Completed maps", len(match_history))
    col2.metric("Completed matches", match_history["match_id"].nunique() if not match_history.empty else 0)
    col3.metric("Recent roster", roster["player_id"].nunique() if not roster.empty else 0)
    col4.metric("Data rows", len(team_history))
    st.markdown("#### Recent roster")
    st.dataframe(
        _display_columns(roster, ["player_name", "player_id", "role", "agent", "last_map_date", "maps_in_window"]),
        use_container_width=True,
        hide_index=True,
    )

with tabs[1]:
    st.subheader("Map-level match history")
    if match_history.empty:
        st.info("No completed map history is available for this team.")
    else:
        dates = match_history["match_date"].dropna()
        date_range = st.date_input(
            "Date range",
            value=(dates.min().date(), dates.max().date()),
            min_value=dates.min().date(),
            max_value=dates.max().date(),
        )
        opponent_values = ["All"] + sorted(match_history["opponent_team"].dropna().astype(str).unique().tolist())
        map_values = ["All"] + sorted(match_history["map"].dropna().astype(str).unique().tolist())
        filter_col1, filter_col2 = st.columns(2)
        selected_opponent = filter_col1.selectbox("Opponent", opponent_values, key="history_opponent")
        selected_map = filter_col2.selectbox("Map", map_values, key="history_map")
        filtered_matches = match_history.copy()
        if isinstance(date_range, tuple) and len(date_range) == 2:
            filtered_matches = filtered_matches[
                filtered_matches["match_date"].dt.date.between(date_range[0], date_range[1])
            ]
        if selected_opponent != "All":
            filtered_matches = filtered_matches[filtered_matches["opponent_team"].eq(selected_opponent)]
        if selected_map != "All":
            filtered_matches = filtered_matches[filtered_matches["map"].eq(selected_map)]
        history_columns = [
            "match_date", "opponent_team", "map", "map_winner", "team_pick", "round_score",
            "event_stage", "event_round", "event_id", "match_id",
        ]
        st.dataframe(_display_columns(filtered_matches, history_columns), use_container_width=True, hide_index=True)

with tabs[2]:
    st.subheader("Player map history")
    if roster.empty:
        st.info("No recent roster is available.")
    else:
        player_labels = {
            f"{row.player_name} (id={int(row.player_id)})": row.player_id
            for row in roster.itertuples()
        }
        selected_player_label = st.selectbox("Player", list(player_labels))
        selected_player_id = player_labels[selected_player_label]
        player_history = get_player_history(team_history, selected_player_id, selected_team_id)
        metric_columns = ["rating2_all", "acs_all", "kda_all", "kills_all", "deaths_all", "assists_all", "adr_all", "kast_all"]
        sample_col, missing_col = st.columns(2)
        sample_col.metric("Completed map sample", len(player_history))
        missing_values = int(player_history[metric_columns].isna().sum().sum()) if not player_history.empty else 0
        missing_col.metric("Missing metric cells", missing_values)

        chart_selection = st.multiselect(
            "Chart metrics (choose 1 or 2)", list(CHART_METRICS), default=["Rating", "ACS"],
        )
        if len(chart_selection) > 2:
            st.error("Please select at most two metrics.")
        elif chart_selection:
            chart_data = build_chart_data(player_history.sort_values("match_date"), chart_selection)
            if not chart_data.empty:
                chart_data["x"] = chart_data.apply(
                    lambda row: f"{row['match_date'].date()} · {row.get('map', '')}", axis=1,
                )
                chart_frame = chart_data.pivot_table(index="x", columns="metric", values="value", aggfunc="mean")
                st.line_chart(chart_frame)
            else:
                st.info("The selected metrics are not present in the player history.")
        player_columns = [
            "match_date", "opponent_team", "map", "agent", "role", "rating2_all", "acs_all", "kda_all",
            "kills_all", "deaths_all", "assists_all", "adr_all", "kast_all", "fb_all", "fd_all",
        ]
        st.dataframe(_display_columns(player_history, player_columns), use_container_width=True, hide_index=True)

with tabs[3]:
    st.subheader("Estimated future-map performance")
    st.caption("Forecasts estimate Rating, ACS and KDA. They are not probabilities of winning.")
    opponents = get_opponent_options(
        team_history,
        all_teams=team_options[team_options["region"].eq(selected_region)],
        exclude_team_id=selected_team_id,
    )
    if opponents.empty or roster.empty:
        st.warning("Forecast requires a team with a recent roster and at least one known opponent.")
    else:
        forecast_col1, forecast_col2 = st.columns(2)
        opponent_label = forecast_col1.selectbox("Opponent (required)", opponents["label"].tolist())
        opponent_row = opponents.loc[opponents["label"].eq(opponent_label)].iloc[0]
        player_labels = {
            f"{row.player_name} (id={int(row.player_id)})": row.player_id
            for row in roster.itertuples()
        }
        selected_players = forecast_col2.multiselect("Players", list(player_labels), default=list(player_labels)[:1])

        all_maps = sorted(team_history["map"].dropna().astype(str).unique().tolist())
        all_agents = sorted(team_history["agent"].dropna().astype(str).unique().tolist())
        scenario_col1, scenario_col2 = st.columns(2)
        map_choice = scenario_col1.selectbox("Map (optional)", ["General forecast"] + all_maps)
        agent_choice = scenario_col2.selectbox("Agent (optional)", ["General forecast"] + all_agents)
        date_col, time_col = st.columns(2)
        prediction_date = date_col.date_input("Prediction date", value=pd.Timestamp.now().date())
        prediction_time = time_col.time_input("Prediction time", value=time(12, 0))
        if map_choice == "General forecast" or agent_choice == "General forecast":
            st.info("General forecast: map and/or agent context is unknown and will use the model fallback path.")
        else:
            st.info("Scenario forecast: the selected map and agent are included as context.")

        if st.button("Run forecast", type="primary"):
            if not selected_players:
                st.error("Select at least one player.")
            else:
                prediction_datetime = pd.Timestamp(datetime.combine(prediction_date, prediction_time))
                history_at_prediction = filter_completed_history(history, cutoff=prediction_datetime)
                try:
                    request = build_forecast_request(
                        history_at_prediction,
                        selected_team_id,
                        opponent_row["opponent_team_id"],
                        [player_labels[label] for label in selected_players],
                        prediction_datetime,
                        None if map_choice == "General forecast" else map_choice,
                        None if agent_choice == "General forecast" else agent_choice,
                        opponent_team_name=opponent_row["opponent_team"],
                    )
                    bundles = {
                        target: load_cached_model(target, str(ACTIVE_MODEL_DIR))
                        for target in TARGETS
                    }
                    predictions = run_forecast(request, history_at_prediction, bundles)
                    st.dataframe(predictions, use_container_width=True, hide_index=True)
                    if predictions["low_confidence"].any():
                        st.warning("At least one row has low confidence because player history is limited or cold-start.")
                except Exception as exc:
                    st.error(f"Forecast could not be generated: {exc}")
