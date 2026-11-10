"""Automated settlement and closing-line-value harvesting.

update_closing_lines()
    For UNSETTLED picks without a close: refreshes sharp snapshots for games
    about to start (live), then, once a game has started, takes the LAST
    sharp (Pinnacle-first) two-way snapshot at or before tipoff, de-vigs it,
    and records closing_american, closing_fair_prob and
    clv_percent = (offered_decimal * closing_fair_prob - 1) * 100.
    If the closing spread/total sits on a different point than the pick, the
    close is recorded but CLV is left NULL (prices on different numbers are
    not comparable).

grade_settled_games()
    Resolves UNSETTLED picks via ESPN's public scoreboard (free), falling
    back to the nba_api / nfl_data_py data packages, then The-Odds-API
    /scores/ (costs quota). Settles through ``Ledger.settle_position`` (WIN /
    LOSS / PUSH, net_units, payout_units, bankroll journal). Cancelled games
    are voided (CANCELLED, stake returned); postponed games stay PENDING.

reconcile_pending_bets()
    Closing lines + grading + resulting bankroll state in one call.

Run ``--scan`` shortly before tipoff so a near-close snapshot exists; the
settler can only use snapshots that were captured.
"""

from __future__ import annotations

import logging
import sqlite3
import sys
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.core import database as db  # noqa: E402
from src.core.config import ODDS_API_BASE, get_setting, load_env  # noqa: E402
from src.core.engine import american_to_decimal, devig_power  # noqa: E402
from src.core.ledger import Ledger, settlement_amounts  # noqa: E402
from src.data.http_client import HttpError, get_json  # noqa: E402
from src.data.market_fetcher import (  # noqa: E402
    SHARP_PRIORITY,
    IngestionError,
    fetch_the_odds_api,
    snapshot_rows,
)
from src.data.teams import SPORT_TO_LEAGUE, normalize_team  # noqa: E402

log = logging.getLogger("breeze.settle")

ESPN_BASE = "https://site.api.espn.com/apis/site/v2/sports"
ESPN_PATHS = {"basketball_nba": "basketball/nba", "americanfootball_nfl": "football/nfl"}
SCOREBOARD_TTL_S = 300
PRE_TIP_REFRESH = timedelta(minutes=30)
ESPN_VOID_STATUSES = frozenset({"STATUS_CANCELED", "STATUS_CANCELLED", "STATUS_FORFEIT"})
ESPN_POSTPONED_STATUSES = frozenset({"STATUS_POSTPONED", "STATUS_SUSPENDED", "STATUS_DELAYED"})
NBA_API_FINAL = 3                     # ScoreboardV2 GAME_STATUS_ID: 1 scheduled, 2 live, 3 final


@dataclass(frozen=True)
class GameScore:
    home: str
    away: str
    home_score: int
    away_score: int
    completed: bool
    cancelled: bool = False         # void: stakes returned (status CANCELLED)
    postponed: bool = False         # left PENDING until it is played or cancelled


def _origin_clause(mock: bool) -> tuple[str, tuple[str, ...]]:
    return ("origin = ?", (db.ORIGIN_MOCK,)) if mock else ("origin = ?", (db.ORIGIN_LIVE,))


# ==========================================================================
# Closing lines
# ==========================================================================
def _refresh_pre_tip_snapshots(conn: sqlite3.Connection, now: datetime) -> int:
    """Live only: capture fresh sharp lines for sports with picks tipping within 30 min."""
    soon = conn.execute(
        "SELECT DISTINCT sport FROM ledger_picks WHERE result = 'UNSETTLED' AND origin = ? "
        "AND closing_fair_prob IS NULL AND commence_time > ? AND commence_time <= ?",
        (db.ORIGIN_LIVE, db.iso(now), db.iso(now + PRE_TIP_REFRESH))).fetchall()
    n = 0
    for row in soon:
        try:
            fetch = fetch_the_odds_api(conn, row["sport"])
        except IngestionError as exc:
            log.warning("pre-tip refresh failed for %s: %s", row["sport"], exc)
            continue
        if not fetch.from_cache:
            n += db.insert_snapshots(conn, snapshot_rows(fetch.events, {}, db.iso(now)))
    return n


def _closing_for(conn: sqlite3.Connection, pick: dict[str, Any]
                 ) -> tuple[int, float, float | None] | None:
    """(closing_american, closing_fair_prob, closing_point) from the last pre-tip sharp pair."""
    market = "h2h" if pick["market_type"] == "kalshi_binary" else pick["market_type"]
    found = db.latest_snapshot_pair(conn, pick["event_id"], market, SHARP_PRIORITY,
                                    pick["commence_time"])
    if not found:
        return None
    _, rows = found
    ours = [r for r in rows if r["selection"] == pick["selection"]]
    theirs = [r for r in rows if r["selection"] != pick["selection"]]
    if len(ours) != 1 or len(theirs) != 1:
        return None
    o, t = ours[0], theirs[0]
    fair = devig_power(american_to_decimal(o["american_odds"]),
                       american_to_decimal(t["american_odds"]))[0]
    return int(o["american_odds"]), fair, o["point"]


def update_closing_lines(conn: sqlite3.Connection, now: datetime | None = None,
                         mock: bool = False) -> dict[str, int]:
    """Record closing sharp lines and CLV for started, unsettled picks."""
    now = now or db.utcnow()
    refreshed = 0 if mock else _refresh_pre_tip_snapshots(conn, now)
    clause, params = _origin_clause(mock)
    picks = db.fetch_picks(conn, f"result = 'UNSETTLED' AND closing_fair_prob IS NULL AND "
                                 f"commence_time <= ? AND {clause}", (db.iso(now), *params))
    recorded = clv = missing = 0
    for p in picks:
        close = _closing_for(conn, p)
        if close is None:
            missing += 1
            continue
        american, fair, point = close
        same_line = p["market_type"] in ("h2h", "kalshi_binary") or point == p["line_point"]
        clv_pct = round((p["offered_decimal"] * fair - 1.0) * 100, 3) if same_line else None
        db.update_pick(conn, p["pick_id"], closing_american=american,
                       closing_fair_prob=round(fair, 6), clv_percent=clv_pct)
        recorded += 1
        clv += clv_pct is not None
    return {"snapshots_refreshed": refreshed, "closes_recorded": recorded,
            "clv_computed": clv, "no_snapshot": missing}


# ==========================================================================
# Scores
# ==========================================================================
def _et_dates(commence: str) -> set[date]:
    """US-Eastern calendar dates a UTC kickoff can fall on (EDT or EST)."""
    t = db.parse_ts(commence)
    return {(t - timedelta(hours=4)).date(), (t - timedelta(hours=5)).date()}


def parse_espn(payload: dict[str, Any]) -> list[GameScore]:
    games = []
    for ev in payload.get("events", []):
        comp = (ev.get("competitions") or [{}])[0]
        teams = {c["homeAway"]: c for c in comp.get("competitors", [])}
        if "home" not in teams or "away" not in teams:
            continue
        try:
            hs, as_ = int(teams["home"].get("score") or 0), int(teams["away"].get("score") or 0)
        except ValueError:
            continue
        stype = (ev.get("status") or {}).get("type") or {}
        name = str(stype.get("name") or "").upper()
        games.append(GameScore(teams["home"]["team"]["displayName"],
                               teams["away"]["team"]["displayName"], hs, as_,
                               bool(stype.get("completed")) and name not in ESPN_VOID_STATUSES,
                               cancelled=name in ESPN_VOID_STATUSES,
                               postponed=name in ESPN_POSTPONED_STATUSES))
    return games


# ---- Data-package fallbacks (nba_api / nfl_data_py; optional, no key) ------
def fetch_nba_api_scores(day: date) -> list[GameScore]:
    """NBA finals for one date from ``nba_api`` ScoreboardV2 ([] if not installed)."""
    try:
        from nba_api.stats.endpoints import scoreboardv2
    except ImportError:
        return []
    data = scoreboardv2.ScoreboardV2(game_date=day.isoformat(), timeout=20).get_normalized_dict()
    lines = {(r["GAME_ID"], r["TEAM_ID"]): r for r in data.get("LineScore", [])}
    games = []
    for g in data.get("GameHeader", []):
        home = lines.get((g["GAME_ID"], g["HOME_TEAM_ID"]))
        away = lines.get((g["GAME_ID"], g["VISITOR_TEAM_ID"]))
        if not home or not away:
            continue
        status = int(g.get("GAME_STATUS_ID") or 0)
        games.append(GameScore(home["TEAM_ABBREVIATION"], away["TEAM_ABBREVIATION"],
                               int(home.get("PTS") or 0), int(away.get("PTS") or 0),
                               status == NBA_API_FINAL))
    return games


def _nfl_season(day: date) -> int:
    return day.year if day.month >= 3 else day.year - 1


def fetch_nfl_data_py_scores(day: date) -> list[GameScore]:
    """NFL finals for one date from ``nfl_data_py.import_schedules`` ([] if not installed)."""
    try:
        import nfl_data_py
    except ImportError:
        return []
    df = nfl_data_py.import_schedules([_nfl_season(day)])
    games = []
    for r in df.to_dict("records"):
        if str(r.get("gameday"))[:10] != day.isoformat():
            continue
        hs, as_ = r.get("home_score"), r.get("away_score")
        done = hs is not None and as_ is not None and hs == hs and as_ == as_      # NaN check
        games.append(GameScore(str(r["home_team"]), str(r["away_team"]),
                               int(hs) if done else 0, int(as_) if done else 0, done))
    return games


PACKAGE_FETCHERS = {"basketball_nba": fetch_nba_api_scores,
                    "americanfootball_nfl": fetch_nfl_data_py_scores}


def fetch_package_scores(sport: str, day: date) -> list[GameScore]:
    fetcher = PACKAGE_FETCHERS.get(sport)
    if fetcher is None:
        return []
    try:
        return fetcher(day)
    except Exception as exc:                                    # package/network failure
        log.warning("%s package scores failed for %s: %s", sport, day, exc)
        return []


def fetch_scores(conn: sqlite3.Connection, sport: str, day: date) -> list[GameScore]:
    """ESPN public scoreboard -> nba_api / nfl_data_py -> The-Odds-API /scores/."""
    try:
        return fetch_espn_scoreboard(conn, sport, day)
    except (HttpError, KeyError) as exc:
        log.warning("ESPN failed (%s); trying data package", exc)
    games = fetch_package_scores(sport, day)
    if games:
        return games
    log.warning("data package empty for %s %s; falling back to Odds-API scores", sport, day)
    return fetch_odds_api_scores(conn, sport)


def fetch_espn_scoreboard(conn: sqlite3.Connection, sport: str, day: date) -> list[GameScore]:
    """ESPN public scoreboard for one date (cached 300s; no key, no quota)."""
    key = f"espn:{sport}:{day:%Y%m%d}"
    payload = db.cache_get(conn, key)
    if payload is None:
        payload, _ = get_json(f"{ESPN_BASE}/{ESPN_PATHS[sport]}/scoreboard",
                              {"dates": f"{day:%Y%m%d}"})
        db.cache_put(conn, key, payload, SCOREBOARD_TTL_S)
    return parse_espn(payload)


def fetch_odds_api_scores(conn: sqlite3.Connection, sport: str) -> list[GameScore]:
    """Fallback: The-Odds-API /scores/?daysFrom=3 (consumes quota)."""
    key = f"oddsscores:{sport}"
    data = db.cache_get(conn, key)
    if data is None:
        api_key = get_setting("ODDS_API_KEY")
        if not api_key:
            return []
        data, hdrs = get_json(f"{ODDS_API_BASE}/sports/{sport}/scores/",
                              {"apiKey": api_key, "daysFrom": 3, "dateFormat": "iso"})
        log.info("The-Odds-API scores %s | x-requests-remaining=%s", sport,
                 hdrs.get("x-requests-remaining"))
        db.cache_put(conn, key, data, SCOREBOARD_TTL_S)
    games = []
    for g in data:
        scores = {s["name"]: s["score"] for s in (g.get("scores") or [])}
        if g["home_team"] in scores and g["away_team"] in scores:
            games.append(GameScore(g["home_team"], g["away_team"], int(scores[g["home_team"]]),
                                   int(scores[g["away_team"]]), bool(g.get("completed"))))
    return games


def grade_pick(pick: dict[str, Any], game: GameScore, league: str) -> str:
    """WIN / LOSS / PUSH for a pick given a final score."""
    home = normalize_team(game.home, league)
    team = normalize_team(pick["selection"], league)
    total = game.home_score + game.away_score
    if pick["market_type"] == "totals":
        diff = total - pick["line_point"] if pick["selection"] == "Over" \
            else pick["line_point"] - total
    else:
        ours, theirs = ((game.home_score, game.away_score) if team == home
                        else (game.away_score, game.home_score))
        diff = ours - theirs + (pick["line_point"] or 0.0
                                if pick["market_type"] == "spreads" else 0.0)
        if pick["market_type"] == "kalshi_binary" and diff == 0:
            return "LOSS"                       # YES on a team that did not win
    return "WIN" if diff > 0 else "PUSH" if diff == 0 else "LOSS"


def net_units_for(result: str, stake: float, offered_decimal: float) -> float:
    return settlement_amounts(result, stake, offered_decimal)[0]


def _match(pick: dict[str, Any], games: list[GameScore], league: str) -> GameScore | None:
    away, _, home = (pick["event_name"] or "").partition(" @ ")
    want = {normalize_team(away, league), normalize_team(home, league)}
    for g in games:
        if {normalize_team(g.home, league), normalize_team(g.away, league)} == want:
            return g
    return None


def grade_settled_games(conn: sqlite3.Connection, now: datetime | None = None,
                        mock: bool = False,
                        mock_scores: list[GameScore] | None = None) -> dict[str, int]:
    """Grade started UNSETTLED picks whose games are final."""
    now = now or db.utcnow()
    clause, params = _origin_clause(mock)
    picks = db.fetch_picks(conn, f"result = 'UNSETTLED' AND commence_time <= ? AND {clause}",
                           (db.iso(now), *params))
    board: dict[tuple[str, date], list[GameScore]] = {}
    ledger = Ledger(conn, db.ORIGIN_MOCK if mock else db.ORIGIN_LIVE)
    graded = not_final = unmatched = cancelled = postponed = 0
    for p in picks:
        league = SPORT_TO_LEAGUE.get(p["sport"] or "", "")
        games: list[GameScore] = []
        if mock:
            games = mock_scores or []
        else:
            for d in sorted(_et_dates(p["commence_time"])):
                if (p["sport"], d) not in board:
                    board[(p["sport"], d)] = fetch_scores(conn, p["sport"], d)
                games += board[(p["sport"], d)]
        game = _match(p, games, league)
        if game is None:
            unmatched += 1
            continue
        if game.cancelled:
            cancelled += ledger.cancel_position(p["pick_id"], "game cancelled", now)
            continue
        if game.postponed:
            postponed += 1
            continue
        if not game.completed:
            not_final += 1
            continue
        graded += ledger.settle_position(p["pick_id"], grade_pick(p, game, league), now)
    return {"graded": graded, "cancelled": cancelled, "postponed": postponed,
            "not_final": not_final, "unmatched": unmatched}


# ==========================================================================
# Mock settlement (offline)
# ==========================================================================
MOCK_FINALS: dict[str, tuple[int, int]] = {         # event_id -> (home, away) final score
    "mock-nba-bos-mil": (112, 108), "mock-nba-den-phx": (115, 110),
    "mock-nba-lal-gsw": (120, 113), "mock-nfl-buf-kc": (24, 27),
    "mock-nfl-det-gb": (20, 23),
}


def run_mock_settlement(conn: sqlite3.Connection) -> dict[str, Any]:
    """Settle MOCK picks offline: synthetic closing snapshots + synthetic finals.

    Closing lines shorten every picked side by 5 cents (a typical sharp
    move toward a +EV position) so CLV is exercised; then all mock games are
    graded as final with fixed scores.
    """
    picks = db.fetch_picks(conn, "result = 'UNSETTLED' AND origin = ?", (db.ORIGIN_MOCK,))
    if not picks:
        return {"closing": {}, "grading": {"graded": 0}}
    rows = []
    for (event_id, market, commence), group in _group_mock(picks).items():
        close_ts = db.iso(db.parse_ts(commence) - timedelta(minutes=5))
        found = db.latest_snapshot_pair(conn, event_id, market, SHARP_PRIORITY, commence)
        if not found:
            continue
        source, pair = found
        picked = {p["selection"] for p in group}
        for r in pair:
            shift = -5 if r["selection"] in picked else 5
            odds = r["american_odds"] + shift
            odds = odds if odds <= -100 or odds >= 100 else (-105 if odds < 0 else 105)
            rows.append({"event_id": event_id, "timestamp": close_ts, "source": source,
                         "market_type": market, "selection": r["selection"], "point": r["point"],
                         "american_odds": odds,
                         "implied_prob": round(1 / american_to_decimal(odds), 6)})
    db.insert_snapshots(conn, rows)
    after = max(db.parse_ts(p["commence_time"]) for p in picks) + timedelta(hours=4)
    closing = update_closing_lines(conn, now=after, mock=True)
    by_event = {p["event_id"]: p for p in picks}
    finals = []
    for event_id, (hs, as_) in MOCK_FINALS.items():
        p = by_event.get(event_id)
        if p:
            away, _, home = p["event_name"].partition(" @ ")
            finals.append(GameScore(home, away, hs, as_, True))
    grading = grade_settled_games(conn, now=after, mock=True, mock_scores=finals)
    return {"closing": closing, "grading": grading}


def _group_mock(picks: list[dict[str, Any]]) -> dict[tuple[str, str, str], list[dict]]:
    out: dict[tuple[str, str, str], list[dict]] = {}
    for p in picks:
        market = "h2h" if p["market_type"] == "kalshi_binary" else p["market_type"]
        out.setdefault((p["event_id"], market, p["commence_time"]), []).append(p)
    return out


def reconcile_pending_bets(conn: sqlite3.Connection, now: datetime | None = None,
                           mock: bool = False,
                           mock_scores: list[GameScore] | None = None) -> dict[str, Any]:
    """Reconcile PENDING positions against finals: closing lines + CLV, grading,
    ledger P/L (credited/debited through ``Ledger``) and the resulting bankroll."""
    origin = db.ORIGIN_MOCK if mock else db.ORIGIN_LIVE
    closing = update_closing_lines(conn, now=now, mock=mock)
    grading = grade_settled_games(conn, now=now, mock=mock, mock_scores=mock_scores)
    return {"closing": closing, "grading": grading,
            "bankroll": Ledger(conn, origin).state().as_dict()}


def run_settlement(conn: sqlite3.Connection, mock: bool = False) -> dict[str, Any]:
    """Entry point used by the CLI."""
    if mock:
        out = run_mock_settlement(conn)
        out["bankroll"] = Ledger(conn, db.ORIGIN_MOCK).state().as_dict()
        return out
    load_env()
    return reconcile_pending_bets(conn)
