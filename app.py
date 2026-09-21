"""Streamlit VCT performance explorer backed by local CSV data."""

from __future__ import annotations

from datetime import datetime, time
import difflib
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
    get_recent_performance_history,
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


DISPLAY_LABELS = {
    "player_name": "Player", "player_id": "Player ID", "team": "Team", "team_id": "Team ID",
    "region": "Region", "opponent_team": "Opponent", "opponent_team_id": "Opponent ID",
    "match_date": "Date", "match_time": "Time", "map": "Map", "map_winner": "Map Winner",
    "team_pick": "Side Pick", "round_score": "Round Score", "event_stage": "Stage",
    "event_round": "Event Round", "event_id": "Event ID", "match_id": "Match ID",
    "agent": "Agent", "role": "Role", "recent_agents": "Recent Agents",
    "last_map_date": "Last Map", "maps_in_window": "Maps in Window",
    "rating2_all": "Rating", "acs_all": "ACS", "kda_all": "KDA", "kills_all": "Kills",
    "deaths_all": "Deaths", "assists_all": "Assists", "adr_all": "ADR", "kast_all": "KAST",
    "fb_all": "First Kills", "fd_all": "First Deaths", "forecast_map": "Map",
    "forecast_rating2_all": "Forecast Rating", "forecast_acs_all": "Forecast ACS",
    "forecast_kda_all": "Forecast KDA", "avg_rating": "Average Rating",
    "avg_acs": "Average ACS", "avg_kda": "Average KDA", "lineup": "Lineup",
}


def _display_columns(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    result = frame[[column for column in columns if column in frame.columns]].copy()
    return result.rename(columns={column: DISPLAY_LABELS.get(column, column) for column in result.columns})


def _normalise_text(value: object) -> str:
    return " ".join(str(value or "").strip().casefold().split())


def _id_key(value: object) -> str:
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    return str(int(numeric)) if pd.notna(numeric) else str(value)


def _team_display_label(row: pd.Series) -> str:
    full_name = str(row.get("display_name", row.get("opponent_team", "Unknown team"))).strip()
    short_name = str(row.get("team", "")).strip()
    team_id = row.get("team_id", row.get("opponent_team_id"))
    suffix = f" · ID {int(team_id)}" if pd.notna(team_id) else ""
    if short_name and _normalise_text(short_name) != _normalise_text(full_name):
        return f"{full_name} [{short_name}]{suffix}"
    return f"{full_name}{suffix}"


def _player_directory(history: pd.DataFrame, teams: pd.DataFrame) -> pd.DataFrame:
    """Build a latest-player catalog for global search and direct player URLs."""
    work = history.copy()
    work["_timestamp"] = pd.to_datetime(
        work["match_date"].astype("string") + " " + work["match_time"].fillna("").astype("string"),
        format="mixed",
        errors="coerce",
    )
    latest = (
        work.sort_values("_timestamp", ascending=False, kind="stable")
        .drop_duplicates("player_id", keep="first")
    )
    team_catalog = teams.drop_duplicates("team_id").set_index("team_id")
    latest = latest[["player_id", "player_name", "team_id", "team", "region"]].copy()

    def team_label(row: pd.Series) -> str:
        if row["team_id"] in team_catalog.index:
            return _team_display_label(team_catalog.loc[row["team_id"]])
        return _team_display_label(pd.Series({"display_name": row["team"], "team": row["team"], "team_id": row["team_id"]}))

    latest["team_label"] = latest.apply(team_label, axis=1)
    latest["label"] = latest.apply(lambda row: f"{row['player_name']} · {row['team_label']}", axis=1)
    latest["search_text"] = latest.apply(
        lambda row: _normalise_text(f"{row['player_name']} {row['team_label']} {row['team']}"),
        axis=1,
    )
    return latest.reset_index(drop=True)


def _search_players(directory: pd.DataFrame, query: str) -> pd.DataFrame:
    query = _normalise_text(query)
    if not query:
        return directory.iloc[0:0].copy()
    matches = directory[directory["search_text"].str.contains(query, regex=False, na=False)].copy()
    if not matches.empty:
        return matches
    names = directory["player_name"].dropna().astype(str).unique().tolist()
    fuzzy_names = difflib.get_close_matches(query, names, n=10, cutoff=0.55)
    return directory[directory["player_name"].isin(fuzzy_names)].copy()


st.title("VCT Performance Explorer")

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
player_directory = _player_directory(history, team_options)
team_display_lookup = {
    _id_key(row["team_id"]): _team_display_label(row)
    for _, row in team_options.drop_duplicates("team_id").iterrows()
}


def _opponent_ui_label(row: pd.Series) -> str:
    return team_display_lookup.get(
        _id_key(row["opponent_team_id"]),
        f"{row.get('opponent_team', 'Unknown team')} · ID {int(row['opponent_team_id'])}",
    )

st.sidebar.caption(f"Data source: `{Path(source_path).name}`")
regions = team_options["region"].dropna().astype(str).unique().tolist()
selected_region = st.sidebar.selectbox("Region", regions)
region_teams = team_options[team_options["region"].eq(selected_region)].reset_index(drop=True)
team_query = st.sidebar.text_input("Search team", placeholder="Full name or short tag")
if team_query.strip():
    query = _normalise_text(team_query)
    searchable = region_teams[["display_name", "team", "label"]].fillna("").astype(str).agg(" ".join, axis=1)
    matching_teams = region_teams[searchable.map(_normalise_text).str.contains(query, regex=False)]
    if not matching_teams.empty:
        region_teams = matching_teams.reset_index(drop=True)
    else:
        st.sidebar.warning("No matching team found.")
team_labels = [_team_display_label(row) for _, row in region_teams.iterrows()]
team_label = st.sidebar.selectbox("Team", team_labels)
selected_team = region_teams.iloc[team_labels.index(team_label)]
selected_team_id = selected_team["team_id"]
team_history = filter_team(history, selected_team_id, selected_region)
roster = get_recent_roster(team_history, selected_team_id)
match_history = aggregate_match_history(team_history)

player_query_from_url = ""
try:
    player_query_from_url = st.query_params.get("player", "")
except AttributeError:  # pragma: no cover - compatibility with older Streamlit
    pass
player_query = st.sidebar.text_input(
    "Search player",
    value=player_query_from_url,
    placeholder="Player name, full name, or short tag",
    help="You can also open a direct URL such as ?player=Asuna.",
)
player_matches = _search_players(player_directory, player_query)
selected_search_player_id = None
if player_query.strip():
    if player_matches.empty:
        st.sidebar.warning("No matching player found.")
    else:
        search_label = st.sidebar.selectbox(
            "Matching players",
            player_matches["label"].tolist(),
            key="player_search_result",
        )
        selected_search_player_id = player_matches.loc[player_matches["label"].eq(search_label), "player_id"].iloc[0]

tabs = st.tabs([
    "Team Overview", "Match History", "Players / Player Detail",
    "Lineup Reference", "Pre-match Forecast",
])

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
        width="stretch",
        hide_index=True,
    )
    st.markdown("#### Recent performance (last 10 matches)")
    recent_team = get_recent_performance_history(team_history, team_id=selected_team_id, limit=10)
    if recent_team.empty:
        st.info("No recent performance metrics are available.")
    else:
        team_chart_cols = st.columns(3)
        for chart_col, metric in zip(team_chart_cols, ("Rating", "ACS", "KDA")):
            chart_col.caption(metric)
            metric_data = build_chart_data(recent_team, [metric])
            if not metric_data.empty:
                chart_col.line_chart(metric_data.set_index("match_date")["value"])

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
        opponent_filter_rows = (
            match_history[["opponent_team_id", "opponent_team"]]
            .dropna(subset=["opponent_team_id"])
            .drop_duplicates("opponent_team_id")
            .copy()
        )
        opponent_filter_rows["ui_label"] = opponent_filter_rows.apply(_opponent_ui_label, axis=1)
        opponent_lookup = dict(zip(opponent_filter_rows["ui_label"], opponent_filter_rows["opponent_team"]))
        opponent_values = ["All"] + sorted(opponent_lookup)
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
            filtered_matches = filtered_matches[filtered_matches["opponent_team"].eq(opponent_lookup[selected_opponent])]
        if selected_map != "All":
            filtered_matches = filtered_matches[filtered_matches["map"].eq(selected_map)]
        history_columns = [
            "match_date", "opponent_team", "map", "map_winner", "team_pick", "round_score",
            "event_stage", "event_round", "event_id", "match_id",
        ]
        st.dataframe(_display_columns(filtered_matches, history_columns), width="stretch", hide_index=True)

with tabs[2]:
    st.subheader("Player detail")
    search_directory = (
        player_matches
        if selected_search_player_id is not None
        else player_directory[player_directory["team_id"].eq(selected_team_id)].copy()
    )
    if search_directory.empty:
        st.info("No matching player is available.")
    else:
        player_labels = dict(zip(search_directory["label"], search_directory["player_id"]))
        default_index = 0
        if selected_search_player_id is not None:
            default_index = search_directory["player_id"].tolist().index(selected_search_player_id)
        selected_player_label = st.selectbox(
            "Player",
            list(player_labels),
            index=default_index,
            key="player_detail_select",
        )
        selected_player_id = player_labels[selected_player_label]
        player_record = player_directory.loc[player_directory["player_id"].eq(selected_player_id)].iloc[0]
        player_team_id = player_record["team_id"]
        player_history = get_player_history(history, selected_player_id, player_team_id)
        metric_columns = [
            column for column in [
                "rating2_all", "acs_all", "kda_all", "kills_all", "deaths_all", "assists_all",
                "adr_all", "kast_all",
            ] if column in player_history.columns
        ]
        sample_col, missing_col = st.columns(2)
        sample_col.metric("Completed maps", len(player_history))
        missing_values = int(player_history[metric_columns].isna().sum().sum()) if metric_columns else 0
        missing_col.metric("Missing metric cells", missing_values)

        chart_selection = st.multiselect(
            "Chart metrics", list(CHART_METRICS), default=["Rating", "ACS"], max_selections=2,
        )
        if chart_selection:
            chart_data = build_chart_data(player_history.sort_values("match_date"), chart_selection)
            if not chart_data.empty:
                chart_data["x"] = chart_data.apply(
                    lambda row: f"{row['match_date'].date()} - {row.get('map', '')}", axis=1,
                )
                chart_frame = chart_data.pivot_table(index="x", columns="metric", values="value", aggfunc="mean")
                st.line_chart(chart_frame)
            else:
                st.info("The selected metrics are not present in the player history.")
        recent_player = get_recent_performance_history(
            history, player_id=selected_player_id, team_id=player_team_id, limit=10,
        )
        if not recent_player.empty:
            recent_cols = st.columns(3)
            for chart_col, metric in zip(recent_cols, ("Rating", "ACS", "KDA")):
                chart_col.caption(f"{metric} - last 10 matches")
                metric_data = build_chart_data(recent_player, [metric])
                if not metric_data.empty:
                    chart_col.line_chart(metric_data.set_index("match_date")["value"])
        player_columns = [
            "match_date", "opponent_team", "map", "agent", "role", "rating2_all", "acs_all", "kda_all",
            "kills_all", "deaths_all", "assists_all", "adr_all", "kast_all", "fb_all", "fd_all",
        ]
        st.dataframe(_display_columns(player_history, player_columns), width="stretch", hide_index=True)

with tabs[3]:
    st.subheader("Lineup reference")
    reference_opponents = get_opponent_options(
        team_history,
        all_teams=team_options,
        exclude_team_id=selected_team_id,
    )
    if reference_opponents.empty:
        st.info("No opponent team is available in the local dataset.")
    else:
        reference_opponents = reference_opponents.copy()
        reference_opponents["ui_label"] = reference_opponents.apply(_opponent_ui_label, axis=1)
        reference_label = st.selectbox(
            "Opponent team",
            reference_opponents["ui_label"].tolist(),
            key="reference_opponent_team",
        )
        reference_row = reference_opponents.loc[
            reference_opponents["ui_label"].eq(reference_label)
        ].iloc[0]
        reference_roster = get_recent_roster(history, reference_row["opponent_team_id"])
        reference_cols = st.columns(2)
        for column, title, roster_frame in (
            (reference_cols[0], f"Your roster · {selected_team['display_name']}", roster),
            (reference_cols[1], f"Opponent roster - {_opponent_ui_label(reference_row)}", reference_roster),
        ):
            with column:
                st.markdown(f"#### {title}")
                if roster_frame.empty:
                    st.info("No recent roster is available.")
                else:
                    st.dataframe(
                        _display_columns(
                            roster_frame,
                            ["player_name", "role", "recent_agents", "maps_in_window", "last_map_date"],
                        ),
                        width="stretch",
                        hide_index=True,
                    )

with tabs[4]:
    st.subheader("Pre-match lineup forecast")
    opponents = get_opponent_options(
        team_history,
        all_teams=team_options,
        exclude_team_id=selected_team_id,
    )
    if opponents.empty or roster.empty:
        st.warning("Pre-match forecast requires both teams and a recent roster for your team.")
    else:
        opponents = opponents.copy()
        opponents["ui_label"] = opponents.apply(_opponent_ui_label, axis=1)
        opponent_label = st.selectbox(
            "Opponent team (required)",
            opponents["ui_label"].tolist(),
            key="prematch_opponent_team",
        )
        opponent_row = opponents.loc[opponents["ui_label"].eq(opponent_label)].iloc[0]
        opponent_roster = get_recent_roster(history, opponent_row["opponent_team_id"])
        if opponent_roster.empty:
            st.warning(
                "The selected opponent has no recent roster in the local data. "
                "Use Lineup Reference or select another opponent."
            )
            st.stop()

        own_player_labels = {
            f"{row.player_name} (id={int(row.player_id)})": row.player_id
            for row in roster.itertuples()
        }
        opponent_player_labels = {
            f"{row.player_name} (id={int(row.player_id)})": row.player_id
            for row in opponent_roster.itertuples()
        }
        lineup_col1, lineup_col2 = st.columns(2)
        selected_players = lineup_col1.multiselect(
            f"Your roster - {selected_team['display_name']}",
            list(own_player_labels),
            default=list(own_player_labels)[:5],
            max_selections=5,
            key="prematch_own_players",
        )
        selected_opponent_players = lineup_col2.multiselect(
            f"Opponent roster - {_opponent_ui_label(opponent_row)}",
            list(opponent_player_labels),
            default=list(opponent_player_labels)[:5],
            max_selections=5,
            key="prematch_opponent_players",
        )

        all_maps = sorted(history["map"].dropna().astype(str).unique().tolist())
        all_agents = sorted(history["agent"].dropna().astype(str).unique().tolist())
        scenario_col1, scenario_col2 = st.columns(2)
        match_format = scenario_col1.selectbox("Match format", ["Bo1", "Bo3"])
        map_pool = scenario_col2.multiselect(
            "Map pool / ban-pick",
            all_maps,
            default=all_maps[:1] if match_format == "Bo1" else all_maps[:2],
            max_selections=1 if match_format == "Bo1" else 3,
        )
        forecast_maps = map_pool or [None]

        def _agent_selection(
            title: str,
            player_labels: dict[str, object],
            team_id: object,
            key_prefix: str,
        ) -> dict[object, str | None]:
            st.markdown(f"#### {title}")
            selections: dict[object, str | None] = {}
            for player_label, player_id in player_labels.items():
                player_history = get_player_history(history, player_id, team_id)
                player_agents = sorted(player_history["agent"].dropna().astype(str).unique().tolist())
                choices = ["General forecast"] + sorted(set(player_agents).union(all_agents))
                latest_agent = (
                    str(player_history.iloc[0]["agent"])
                    if not player_history.empty and pd.notna(player_history.iloc[0].get("agent"))
                    else "General forecast"
                )
                default_index = choices.index(latest_agent) if latest_agent in choices else 0
                selected_agent = st.selectbox(
                    player_label,
                    choices,
                    index=default_index,
                    key=f"{key_prefix}_{player_id}",
                )
                selections[player_id] = None if selected_agent == "General forecast" else selected_agent
            return selections

        agent_col1, agent_col2 = st.columns(2)
        with agent_col1:
            own_agents = _agent_selection(
                "Your expected agents",
                {label: own_player_labels[label] for label in selected_players},
                selected_team_id,
                "prematch_own_agent",
            )
        with agent_col2:
            opponent_agents = _agent_selection(
                "Opponent expected agents",
                {label: opponent_player_labels[label] for label in selected_opponent_players},
                opponent_row["opponent_team_id"],
                "prematch_opponent_agent",
            )

        date_col, time_col = st.columns(2)
        prediction_date = date_col.date_input("Prediction date", value=pd.Timestamp.now().date())
        prediction_time = time_col.time_input("Prediction time", value=time(12, 0))
        if map_pool:
            st.info(f"Forecast will be generated separately for: {', '.join(map_pool)}.")
        else:
            st.info("No map selected: the model fallback path will provide a general forecast.")

        if st.button("Run pre-match forecast", type="primary", key="run_prematch_forecast"):
            if not selected_players or not selected_opponent_players:
                st.error("Select at least one player for each lineup.")
            else:
                prediction_datetime = pd.Timestamp(datetime.combine(prediction_date, prediction_time))
                history_at_prediction = filter_completed_history(history, cutoff=prediction_datetime)
                try:
                    bundles = {
                        target: load_cached_model(target, str(ACTIVE_MODEL_DIR))
                        for target in TARGETS
                    }
                    own_predictions_by_map = []
                    opponent_predictions_by_map = []
                    for forecast_map in forecast_maps:
                        own_request = build_forecast_request(
                            history_at_prediction,
                            selected_team_id,
                            opponent_row["opponent_team_id"],
                            [own_player_labels[label] for label in selected_players],
                            prediction_datetime,
                            forecast_map,
                            opponent_team_name=opponent_row["opponent_team"],
                            agent_by_player=own_agents,
                        )
                        opponent_request = build_forecast_request(
                            history_at_prediction,
                            opponent_row["opponent_team_id"],
                            selected_team_id,
                            [opponent_player_labels[label] for label in selected_opponent_players],
                            prediction_datetime,
                            forecast_map,
                            opponent_team_name=selected_team["display_name"],
                            agent_by_player=opponent_agents,
                        )
                        own_result = run_forecast(own_request, history_at_prediction, bundles)
                        opponent_result = run_forecast(opponent_request, history_at_prediction, bundles)
                        own_result.insert(0, "forecast_map", forecast_map or "General forecast")
                        opponent_result.insert(0, "forecast_map", forecast_map or "General forecast")
                        own_predictions_by_map.append(own_result)
                        opponent_predictions_by_map.append(opponent_result)

                    own_predictions = pd.concat(own_predictions_by_map, ignore_index=True)
                    opponent_predictions = pd.concat(opponent_predictions_by_map, ignore_index=True)
                    display_columns = [
                        "forecast_map", "player_name", "agent", "forecast_rating2_all",
                        "forecast_acs_all", "forecast_kda_all",
                    ]
                    summary = pd.DataFrame([
                        {
                            "lineup": selected_team["display_name"],
                            "avg_rating": own_predictions["forecast_rating2_all"].mean(),
                            "avg_acs": own_predictions["forecast_acs_all"].mean(),
                            "avg_kda": own_predictions["forecast_kda_all"].mean(),
                        },
                        {
                            "lineup": opponent_row["opponent_team"],
                            "avg_rating": opponent_predictions["forecast_rating2_all"].mean(),
                            "avg_acs": opponent_predictions["forecast_acs_all"].mean(),
                            "avg_kda": opponent_predictions["forecast_kda_all"].mean(),
                        },
                    ])
                    st.markdown("#### Team-level reference from player forecasts")
                    st.dataframe(summary, width="stretch", hide_index=True)
                    st.caption(
                        f"Format: {match_format} - Maps: {', '.join(map_pool) if map_pool else 'General forecast'} - "
                        "Averages summarize player forecasts; they are not win probabilities."
                    )
                    prediction_col1, prediction_col2 = st.columns(2)
                    with prediction_col1:
                        st.markdown(f"#### {selected_team['display_name']} predictions")
                        st.dataframe(
                            _display_columns(own_predictions, display_columns),
                            width="stretch",
                            hide_index=True,
                        )
                    with prediction_col2:
                        st.markdown(f"#### {_opponent_ui_label(opponent_row)} predictions")
                        st.dataframe(
                            _display_columns(opponent_predictions, display_columns),
                            width="stretch",
                            hide_index=True,
                        )
                    if pd.concat([own_predictions, opponent_predictions])["low_confidence"].any():
                        st.warning("At least one row has low confidence because player history is limited or cold-start.")
                except Exception as exc:
                    st.error(f"Forecast could not be generated: {exc}")
