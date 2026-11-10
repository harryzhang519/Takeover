"""Compatibility layer over ``team_registry`` (single source of truth).

Older modules work in full team names; these helpers route every lookup
through the canonical registry so Odds-API, Kalshi and ESPN always agree.
"""

from __future__ import annotations

from src.data.team_registry import REGISTRY, teams_for, try_resolve_team

SPORT_TO_LEAGUE = {"basketball_nba": "NBA", "americanfootball_nfl": "NFL"}
LEAGUE_TO_KALSHI_GAME_SERIES = {"NBA": "KXNBAGAME", "NFL": "KXNFLGAME"}


def normalize_team(name: str, league: str) -> str | None:
    """Canonical full (Odds-API) name for any alias, or None."""
    if not name or league not in ("NBA", "NFL"):
        return None
    cid = try_resolve_team(name, league)
    return REGISTRY[cid].full_name if cid else None


def teams(league: str) -> list[str]:
    return [t.full_name for t in teams_for(league)]


def abbreviation(full_name: str, league: str) -> str:
    """Kalshi ticker code for a team (what Kalshi event tickers embed)."""
    cid = try_resolve_team(full_name, league)
    if cid is None:
        raise KeyError(full_name)
    return REGISTRY[cid].kalshi_code
