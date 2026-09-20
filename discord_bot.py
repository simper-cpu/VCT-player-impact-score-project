"""Discord Slash Command bot for the local VCT performance models.

The bot intentionally keeps the data/model loading path separate from command
handlers.  CSV data and exported pipelines are loaded once in ``on_ready``;
forecast inference is moved to a worker thread so the Discord event loop stays
responsive while scikit-learn runs on CPU.

Environment variables:

    DISCORD_BOT_TOKEN  Required Discord bot token.
    DISCORD_GUILD_ID   Optional test server ID.  Guild sync is immediate;
                       without it commands are synced globally and can take
                       longer to appear in Discord.

Run from the project root with::

    python discord_bot.py
"""

from __future__ import annotations

import asyncio
import argparse
from dataclasses import dataclass
from datetime import datetime
import difflib
import logging
import os
from pathlib import Path
from typing import Any

try:
    import discord
    from discord import app_commands
except ImportError as exc:  # pragma: no cover - depends on optional runtime dependency
    raise RuntimeError(
        "Discord bot dependencies are missing. Install them with "
        "'pip install -r requirements.txt'."
    ) from exc

import pandas as pd

from src.app_data import (
    DEFAULT_DATA_PATH,
    DEFAULT_MODEL_DIR,
    aggregate_match_history,
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
from src.features.feature_schema import TARGETS, make_match_timestamp
from src.models.inference import load_model


LOGGER = logging.getLogger("vct-discord-bot")
DEFAULT_AVATAR_URL = "https://cdn.discordapp.com/embed/avatars/0.png"


@dataclass
class BotState:
    """Immutable-ish runtime state shared by Slash Command handlers."""

    history: pd.DataFrame
    teams: pd.DataFrame
    players: pd.DataFrame
    models: dict[str, dict[str, Any]]
    data_source: Path


def _normalise(value: object) -> str:
    return " ".join(str(value or "").strip().casefold().split())


def _display(value: object, fallback: str = "Unknown") -> str:
    if value is None or pd.isna(value):
        return fallback
    text = str(value).strip()
    return text if text else fallback


def _format_number(value: object, decimals: int = 2) -> str:
    number = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(number):
        return "N/A"
    return f"{float(number):.{decimals}f}"


def _latest_players(history: pd.DataFrame) -> pd.DataFrame:
    """Build one current/latest row per player for fast command lookup."""
    work = history.copy()
    work["_timestamp"] = make_match_timestamp(work)
    work = work.sort_values("_timestamp", ascending=False, kind="stable")
    columns = [
        column for column in [
            "player_id", "player_name", "team_id", "team", "region", "_timestamp",
        ] if column in work.columns
    ]
    return work[columns].drop_duplicates("player_id", keep="first").reset_index(drop=True)


def load_bot_state(
    data_path: Path = DEFAULT_DATA_PATH,
    model_dir: Path = DEFAULT_MODEL_DIR,
) -> BotState:
    """Load completed history and all exported target models exactly once."""
    data, source = load_local_dataset(data_path, data_path.parent / "player_match_dataset_cleaned.csv")
    history = filter_completed_history(data)
    if history.empty:
        raise RuntimeError("The local dataset contains no completed maps.")

    teams = get_team_options(history)
    if teams.empty:
        raise RuntimeError("The local dataset contains no teams.")

    models = {
        target: load_model(target, model_dir)
        for target in TARGETS
    }
    return BotState(
        history=history,
        teams=teams,
        players=_latest_players(history),
        models=models,
        data_source=source,
    )


def _resolve_team(state: BotState, query: str) -> tuple[pd.Series | None, str | None]:
    """Resolve a team by case-insensitive name, with a close-match suggestion."""
    needle = _normalise(query)
    if not needle:
        return None, None

    name_columns = [column for column in ["display_name", "team", "label"] if column in state.teams]
    exact_mask = pd.Series(False, index=state.teams.index)
    for column in name_columns:
        exact_mask |= state.teams[column].map(_normalise).eq(needle)
    exact = state.teams.loc[exact_mask]
    if not exact.empty:
        return exact.iloc[0], None

    choices: dict[str, int] = {}
    for index, row in state.teams.iterrows():
        for column in name_columns:
            name = _display(row.get(column), "")
            if name:
                choices.setdefault(name, index)
    match = difflib.get_close_matches(query.strip(), list(choices), n=1, cutoff=0.55)
    if match:
        return None, match[0]
    return None, None


def _resolve_player(state: BotState, query: str) -> tuple[pd.Series | None, str | None]:
    """Resolve a player by exact name or return the closest spelling."""
    needle = _normalise(query)
    if not needle:
        return None, None
    names = state.players["player_name"].fillna("").astype(str)
    exact = state.players.loc[names.map(_normalise).eq(needle)]
    if not exact.empty:
        return exact.iloc[0], None

    unique_names = list(dict.fromkeys(name for name in names if name.strip()))
    match = difflib.get_close_matches(query.strip(), unique_names, n=1, cutoff=0.55)
    if match:
        return None, match[0]
    return None, None


def _team_label(row: pd.Series) -> str:
    return _display(row.get("display_name"), _display(row.get("team"), "Unknown team"))


def _team_history_lines(team_history: pd.DataFrame, team_label: str) -> list[str]:
    """Create compact, Embed-safe lines for the latest ten map records."""
    history = aggregate_match_history(team_history).head(10)
    lines = []
    for row in history.itertuples():
        opponent = _display(getattr(row, "opponent_team", None))
        map_name = _display(getattr(row, "map", None))
        winner = _display(getattr(row, "map_winner", None))
        stage = _display(getattr(row, "event_stage", None), "")
        result = "✅" if _normalise(winner) in {_normalise(team_label), _normalise(getattr(row, "team", ""))} else ""
        suffix = f" · {stage}" if stage else ""
        lines.append(f"{result} {map_name} vs {opponent} · winner: {winner}{suffix}".strip())
    return lines or ["No completed match history available."]


def _recent_roster_lines(roster: pd.DataFrame) -> list[str]:
    lines = []
    for row in roster.itertuples():
        name = _display(getattr(row, "player_name", None))
        role = _display(getattr(row, "role", None), "Flex/Unknown")
        agents = _display(getattr(row, "recent_agents", None), "unknown")
        lines.append(f"{name} · {role} · agents: {agents}")
    return lines or ["No recent roster available."]


def _embed(title: str, description: str = "", colour: int = 0xE95D4F) -> discord.Embed:
    return discord.Embed(title=title, description=description, colour=discord.Colour(colour))


def _state_from_interaction(interaction: discord.Interaction) -> BotState | None:
    client = interaction.client
    state = getattr(client, "state", None)
    return state if isinstance(state, BotState) else None


class VCTBot(discord.Client):
    """Discord client that owns one loaded data/model state."""

    def __init__(self) -> None:
        super().__init__(intents=discord.Intents.default())
        self.tree = app_commands.CommandTree(self)
        self.state: BotState | None = None
        self._startup_lock = asyncio.Lock()
        self._commands_synced = False

    async def setup_hook(self) -> None:
        self.tree.add_command(forecast_group)

    async def on_ready(self) -> None:
        async with self._startup_lock:
            if self.state is None:
                LOGGER.info("Loading VCT data and models into memory...")
                self.state = await asyncio.to_thread(load_bot_state)
                LOGGER.info(
                    "Loaded %s rows from %s and %s models",
                    len(self.state.history), self.state.data_source, len(self.state.models),
                )

            if not self._commands_synced:
                guild_id = os.getenv("DISCORD_GUILD_ID", "").strip()
                if guild_id:
                    guild = discord.Object(id=int(guild_id))
                    self.tree.copy_global_to(guild=guild)
                    await self.tree.sync(guild=guild)
                    LOGGER.info("Slash Commands synced to guild %s", guild_id)
                else:
                    await self.tree.sync()
                    LOGGER.info("Slash Commands synced globally")
                self._commands_synced = True

        LOGGER.info("Logged in as %s", self.user)


bot = VCTBot()


@bot.tree.command(name="team", description="Xem roster và 10 map gần nhất của một đội")
@app_commands.describe(team_name="Tên đội, không phân biệt hoa thường")
async def team_command(interaction: discord.Interaction, team_name: str) -> None:
    state = _state_from_interaction(interaction)
    if state is None:
        await interaction.response.send_message("Bot vẫn đang load dữ liệu, thử lại sau vài giây.", ephemeral=True)
        return

    team, suggestion = _resolve_team(state, team_name)
    if team is None:
        message = f"Không tìm thấy team **{team_name}**."
        if suggestion:
            message += f" Có phải bạn muốn tìm **{suggestion}** không?"
        await interaction.response.send_message(message, ephemeral=True)
        return

    team_id = team["team_id"]
    team_history = filter_team(state.history, team_id, _display(team.get("region"), ""))
    roster = get_recent_roster(team_history, team_id)
    embed = _embed(f"{_team_label(team)} · Team overview", f"Region: {_display(team.get('region'))}")
    embed.set_thumbnail(url=DEFAULT_AVATAR_URL)
    embed.add_field(name="Recent roster", value="\n".join(_recent_roster_lines(roster))[:1024], inline=False)
    embed.add_field(
        name="Latest 10 maps",
        value="\n".join(_team_history_lines(team_history, _team_label(team)))[:1024],
        inline=False,
    )
    embed.set_footer(text="Agent hiển thị là lịch sử gần đây, không phải agent chắc chắn của map tới.")
    await interaction.response.send_message(embed=embed)


@bot.tree.command(name="player", description="Xem phong độ 10 map gần nhất của tuyển thủ")
@app_commands.describe(player_name="Tên tuyển thủ; bot hỗ trợ gợi ý nếu gõ sai")
async def player_command(interaction: discord.Interaction, player_name: str) -> None:
    state = _state_from_interaction(interaction)
    if state is None:
        await interaction.response.send_message("Bot vẫn đang load dữ liệu, thử lại sau vài giây.", ephemeral=True)
        return

    player, suggestion = _resolve_player(state, player_name)
    if player is None:
        message = f"Không tìm thấy tuyển thủ **{player_name}**."
        if suggestion:
            message += f" Ý bạn là **{suggestion}** phải không? Hãy dùng lại `/player` với tên đó."
        await interaction.response.send_message(message, ephemeral=True)
        return

    history = get_player_history(state.history, player["player_id"])
    recent = history.head(10)
    latest = recent.iloc[0]
    averages = {
        "Rating": pd.to_numeric(recent["rating2_all"], errors="coerce").mean(),
        "ACS": pd.to_numeric(recent["acs_all"], errors="coerce").mean(),
        "KDA": pd.to_numeric(recent["kda_all"], errors="coerce").mean(),
    }
    embed = _embed(
        f"{_display(latest.get('player_name'))} · Player profile",
        f"Team hiện tại: **{_display(latest.get('team'))}** · Region: {_display(latest.get('region'))}",
        colour=0x3B82F6,
    )
    embed.set_thumbnail(url=DEFAULT_AVATAR_URL)
    embed.add_field(
        name=f"Average · latest {len(recent)} maps",
        value=(
            f"Rating **{_format_number(averages['Rating'])}** · "
            f"ACS **{_format_number(averages['ACS'], 1)}** · "
            f"KDA **{_format_number(averages['KDA'])}**"
        ),
        inline=False,
    )
    recent_lines = [
        f"{_display(row.map)} · {_format_number(row.rating2_all)} / {_format_number(row.acs_all, 1)} / {_format_number(row.kda_all)}"
        for row in recent.itertuples()
    ]
    embed.add_field(
        name="Recent maps · Rating / ACS / KDA",
        value="\n".join(recent_lines)[:1024] or "No recent maps available.",
        inline=False,
    )
    embed.set_footer(text="Các chỉ số là trung bình lịch sử, không phải cam kết kết quả trận đấu.")
    await interaction.response.send_message(embed=embed)


forecast_group = app_commands.Group(name="forecast", description="Dự đoán performance VCT")


@forecast_group.command(name="player", description="Dự đoán phong độ một player khi gặp opponent")
@app_commands.describe(
    player_name="Tên tuyển thủ",
    opponent="Tên team đối thủ",
    agent="Agent dự kiến, không bắt buộc",
    map_name="Map dự kiến, không bắt buộc",
)
@app_commands.rename(map_name="map")
async def forecast_player_command(
    interaction: discord.Interaction,
    player_name: str,
    opponent: str,
    agent: str | None = None,
    map_name: str | None = None,
) -> None:
    state = _state_from_interaction(interaction)
    if state is None:
        await interaction.response.send_message("Bot vẫn đang load dữ liệu, thử lại sau vài giây.", ephemeral=True)
        return

    player, player_suggestion = _resolve_player(state, player_name)
    if player is None:
        message = f"Không tìm thấy tuyển thủ **{player_name}**."
        if player_suggestion:
            message += f" Ý bạn là **{player_suggestion}** phải không?"
        await interaction.response.send_message(message, ephemeral=True)
        return

    opponent_row, opponent_suggestion = _resolve_team(state, opponent)
    if opponent_row is None:
        message = f"Không tìm thấy opponent team **{opponent}**."
        if opponent_suggestion:
            message += f" Có phải bạn muốn tìm **{opponent_suggestion}** không?"
        await interaction.response.send_message(message, ephemeral=True)
        return

    player_team_id = player["team_id"]
    opponent_team_id = opponent_row["team_id"]
    if pd.to_numeric(pd.Series([player_team_id, opponent_team_id]), errors="coerce").nunique() == 1:
        await interaction.response.send_message("Opponent phải khác team hiện tại của player.", ephemeral=True)
        return

    await interaction.response.defer()
    prediction_time = pd.Timestamp(datetime.now().replace(microsecond=0))
    try:
        request = build_forecast_request(
            state.history,
            player_team_id,
            opponent_team_id,
            [player["player_id"]],
            prediction_time,
            map_name=map_name or None,
            agent_name=agent or None,
            opponent_team_name=_team_label(opponent_row),
        )
        predictions = await asyncio.to_thread(
            run_forecast,
            request,
            state.history,
            state.models,
        )
    except Exception as exc:  # pragma: no cover - integration/runtime path
        LOGGER.exception("Forecast failed")
        await interaction.followup.send(f"Không thể chạy forecast: `{exc}`", ephemeral=True)
        return

    row = predictions.iloc[0]
    player_display = _display(row.get("player_name"), player_name)
    team_display = _display(row.get("team"), _display(player.get("team")))
    agent_display = _display(row.get("agent"), agent or "fallback theo role")
    map_display = _display(row.get("map"), map_name or "general forecast")
    embed = _embed(
        f"Forecast · {player_display}",
        f"**{team_display}** gặp **{_team_label(opponent_row)}**\nAgent: **{agent_display}** · Map: **{map_display}**",
        colour=0x22C55E,
    )
    embed.add_field(
        name="Estimated performance",
        value=(
            f"Rating **{_format_number(row.get('forecast_rating2_all'))}**\n"
            f"ACS **{_format_number(row.get('forecast_acs_all'), 1)}**\n"
            f"KDA **{_format_number(row.get('forecast_kda_all'))}**"
        ),
        inline=False,
    )
    if bool(row.get("low_confidence", False)):
        embed.add_field(
            name="Confidence",
            value="⚠️ Low confidence: lịch sử player còn ít hoặc đang ở cold-start.",
            inline=False,
        )
    embed.set_footer(text="Chỉ là dự đoán dựa trên phong độ, không phải tỷ lệ thắng.")
    await interaction.followup.send(embed=embed)


@bot.tree.error
async def on_app_command_error(interaction: discord.Interaction, error: app_commands.AppCommandError) -> None:
    LOGGER.exception("Slash Command error", exc_info=error)
    message = "Có lỗi khi xử lý lệnh. Kiểm tra log của bot để biết chi tiết."
    if interaction.response.is_done():
        await interaction.followup.send(message, ephemeral=True)
    else:
        await interaction.response.send_message(message, ephemeral=True)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description="VCT Discord bot")
    parser.add_argument(
        "--check",
        action="store_true",
        help="Load data/models and run one local forecast without a Discord token",
    )
    args = parser.parse_args()

    if args.check:
        run_offline_check()
        return

    token = os.getenv("DISCORD_BOT_TOKEN", "").strip()
    if not token:
        raise SystemExit(
            "Missing DISCORD_BOT_TOKEN. Set the environment variable before starting the bot."
        )
    bot.run(token)


def run_offline_check() -> None:
    """Validate model loading, command registration, and one forecast locally."""
    state = load_bot_state()
    asyncio.run(bot.setup_hook())
    root_commands = [command.name for command in bot.tree.get_commands()]
    if not {"team", "player", "forecast"}.issubset(root_commands):
        raise RuntimeError(f"Unexpected Slash Commands: {root_commands}")

    player = state.players.iloc[0]
    opponent_options = get_opponent_options(
        state.history,
        all_teams=state.teams,
        exclude_team_id=player["team_id"],
    )
    if opponent_options.empty:
        raise RuntimeError("Offline check could not find an opponent team.")
    opponent = opponent_options.iloc[0]
    request = build_forecast_request(
        state.history,
        player["team_id"],
        opponent["opponent_team_id"],
        [player["player_id"]],
        pd.Timestamp.now().replace(microsecond=0),
        opponent_team_name=opponent["opponent_team"],
    )
    result = run_forecast(request, state.history, state.models)
    row = result.iloc[0]
    print("Offline check passed")
    print(f"Commands: {', '.join(root_commands)} + /forecast player")
    print(f"Data rows: {len(state.history)} | Teams: {len(state.teams)} | Players: {len(state.players)}")
    print(
        f"Forecast: {_display(row.get('player_name'))} vs {_display(row.get('opponent_team'))} | "
        f"Rating={_format_number(row.get('forecast_rating2_all'))}, "
        f"ACS={_format_number(row.get('forecast_acs_all'), 1)}, "
        f"KDA={_format_number(row.get('forecast_kda_all'))}"
    )


if __name__ == "__main__":
    main()
