from bs4 import BeautifulSoup

from src.collection import crawl_player_stats
from src.collection.crawl_player_stats import get_player_stats


def _row(agent=None, rating="1.00"):
    agent_html = f'<div class="ovw-agents"><img alt="{agent}"></div>' if agent else '<div class="ovw-agents"></div>'
    rating_html = f'<span class="side mod-both">{rating}</span>' if rating else '<span class="side mod-both"></span>'
    return BeautifulSoup(
        f'''
        <div class="ovw-row">
            <div class="ovw-player-name text-of">Player</div>
            <div class="ovw-player-tag ge-text-light">TEAM</div>
            <a href="/player/123/player"></a>
            {agent_html}
            <div data-col="rating2">
                <span class="side mod-t">{rating or ""}</span>
                {rating_html}
                <span class="side mod-ct">{rating or ""}</span>
            </div>
        </div>
        ''',
        "html.parser",
    ).select_one(".ovw-row")


def test_player_stats_accepts_rows_without_agent_image():
    data = get_player_stats(
        _row(agent=None, rating=None),
        "https://www.vlr.gg/450589/team-international-vs-team-thailand-champions-tour-2025-masters-bangkok-main-event",
        "Ascent",
        "15315",
    )

    assert data["agent"] is None
    assert data["player_id"] == "123"
    assert data["rating2_all"] is None


def test_get_player_skips_vlr_placeholder_map(monkeypatch):
    class Response:
        status_code = 200
        content = f'''
            <div class="vm-stats-game" data-game-id="1">
                <div class="map">Ascent -</div>
                {str(_row(agent=None, rating=None))}
            </div>
        '''.encode()

        def raise_for_status(self):
            return None

    monkeypatch.setattr(crawl_player_stats.requests, "get", lambda *args, **kwargs: Response())

    assert crawl_player_stats.get_player("https://www.vlr.gg/450589/example") == []
