"""Player-prop stats ingestion: NBA via ``nba_api``, NFL via nflverse.

``update_prop_stats(conn)`` fills two tables:

    player_game_logs    NBA: every player game this season (LeagueGameLog, ONE
                        call) joined to team logs for final margin and game
                        possessions (FGA - OREB + TOV + 0.44 FTA, both teams
                        averaged). NFL: weekly player stats (targets, shares,
                        carries, yards, attempts).
    team_prop_context   NBA: pace, DRTG and reb/ast/3PM allowed per 100
                        possessions. NFL: plays/game, pass/rush per game,
                        neutral-script pass rate (win prob 0.2-0.8, quarters
                        1-3), yards allowed per pass attempt / per carry and
                        defensive EPA/play.

NBA costs two stats.nba.com calls per refresh (player + team logs) plus the
existing LeagueDashTeamStats-free derivation, so it stays well clear of the
endpoint's rate limits. NFL prefers ``nflreadpy`` (current nflverse loader)
and falls back to ``nfl_data_py``. Both packages are optional: without them
the live prop scan reports "no model" for that league instead of failing.
"""

from __future__ import annotations

import difflib
import logging
import re
import sqlite3
import sys
import unicodedata
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.core import database as db  # noqa: E402
from src.data.team_registry import try_resolve_team  # noqa: E402

log = logging.getLogger("breeze.propstats")

STALE_HOURS = 12.0
NFL_POSITIONS = ("QB", "RB", "WR", "TE")
NBA_LOG_COLS = ("minutes", "pts", "reb", "ast", "fg3m", "fg3a", "fgm", "fga", "ftm", "fta", "tov")
NFL_LOG_COLS = ("targets", "receptions", "rec_yds", "air_yards", "target_share", "air_yards_share",
                "carries", "rush_yds", "pass_att", "completions", "pass_yds")
_SUFFIX = re.compile(r"\b(jr|sr|ii|iii|iv|v)\b")


class PropStatsUnavailable(RuntimeError):
    """A data package is missing or its source could not be reached."""


# ==========================================================================
# Names and seasons
# ==========================================================================
def player_key(name: str) -> str:
    """Accent/punctuation/suffix-insensitive key: 'P.J. Washington Jr.' -> 'pj washington'."""
    s = unicodedata.normalize("NFKD", name or "").encode("ascii", "ignore").decode().lower()
    s = re.sub(r"[.'`]", "", s)
    s = re.sub(r"[^a-z0-9 ]+", " ", s)
    s = _SUFFIX.sub("", s)
    return re.sub(r"\s+", " ", s).strip()


def nba_season_label(day: date) -> str:
    """'2026-27' for any date from August 2026 to July 2027."""
    start = day.year if day.month >= 8 else day.year - 1
    return f"{start}-{(start + 1) % 100:02d}"


def nfl_season(day: date) -> int:
    return day.year if day.month >= 3 else day.year - 1


# ==========================================================================
# NBA (nba_api)
# ==========================================================================
def _nba_rows(mode: str, season: str) -> list[dict[str, Any]]:
    try:
        from nba_api.stats.endpoints import leaguegamelog  # type: ignore
    except ImportError as exc:
        raise PropStatsUnavailable("nba_api not installed (pip install nba_api)") from exc
    try:
        res = leaguegamelog.LeagueGameLog(season=season, player_or_team_abbreviation=mode,
                                          season_type_all_star="Regular Season", timeout=30)
        return list(res.get_normalized_dict()["LeagueGameLog"])
    except Exception as exc:                        # third-party / network
        raise PropStatsUnavailable(f"nba_api LeagueGameLog({mode}) failed: {exc}") from exc


def _poss(row: dict[str, Any]) -> float:
    return (float(row.get("FGA") or 0) - float(row.get("OREB") or 0) + float(row.get("TOV") or 0)
            + 0.44 * float(row.get("FTA") or 0))


def build_nba_tables(player_rows: list[dict[str, Any]], team_rows: list[dict[str, Any]],
                     season: str, fetched_at: str
                     ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """(player_game_logs rows, team_prop_context rows) from LeagueGameLog P and T dicts."""
    by_game: dict[str, list[dict[str, Any]]] = {}
    for t in team_rows:
        by_game.setdefault(str(t["GAME_ID"]), []).append(t)
    game: dict[tuple[str, str], dict[str, Any]] = {}       # (game_id, team abbr) -> context
    agg: dict[str, dict[str, float]] = {}
    for gid, sides in by_game.items():
        if len(sides) != 2:
            continue
        gposs = (_poss(sides[0]) + _poss(sides[1])) / 2.0
        for me, opp in ((sides[0], sides[1]), (sides[1], sides[0])):
            cid = try_resolve_team(str(me["TEAM_ABBREVIATION"]), "NBA")
            ocid = try_resolve_team(str(opp["TEAM_ABBREVIATION"]), "NBA")
            game[(gid, str(me["TEAM_ABBREVIATION"]))] = {
                "cid": cid, "ocid": ocid, "poss": gposs,
                "margin": float(me.get("PTS") or 0) - float(opp.get("PTS") or 0),
                "is_home": " vs. " in str(me.get("MATCHUP", ""))}
            if cid:
                a = agg.setdefault(cid, {"g": 0, "poss": 0.0, "opp_pts": 0.0, "opp_reb": 0.0,
                                         "opp_ast": 0.0, "opp_fg3m": 0.0})
                a["g"] += 1
                a["poss"] += gposs
                for k in ("pts", "reb", "ast", "fg3m"):
                    a[f"opp_{k}"] += float(opp.get(k.upper()) or 0)
    teams = []
    for cid, a in agg.items():
        p = a["poss"]
        teams.append({"canonical_id": cid, "sport": "NBA", "season": season,
                      "updated_at": fetched_at, "games": int(a["g"]), "pace": p / a["g"],
                      "drtg": 100 * a["opp_pts"] / p, "allowed_reb_100": 100 * a["opp_reb"] / p,
                      "allowed_ast_100": 100 * a["opp_ast"] / p,
                      "allowed_fg3m_100": 100 * a["opp_fg3m"] / p})
    logs = []
    for r in player_rows:
        ctx = game.get((str(r["GAME_ID"]), str(r["TEAM_ABBREVIATION"])), {})
        logs.append({
            "sport": "NBA", "player_key": player_key(str(r["PLAYER_NAME"])),
            "game_id": str(r["GAME_ID"]), "player_name": str(r["PLAYER_NAME"]),
            "canonical_id": ctx.get("cid") or try_resolve_team(str(r["TEAM_ABBREVIATION"]), "NBA"),
            "opponent_id": ctx.get("ocid"), "position": None,
            "game_date": str(r["GAME_DATE"])[:10], "season": season,
            "minutes": float(r.get("MIN") or 0), "pts": r.get("PTS"), "reb": r.get("REB"),
            "ast": r.get("AST"), "fg3m": r.get("FG3M"), "fg3a": r.get("FG3A"),
            "fgm": r.get("FGM"), "fga": r.get("FGA"), "ftm": r.get("FTM"), "fta": r.get("FTA"),
            "tov": r.get("TOV"), "team_margin": ctx.get("margin"), "game_poss": ctx.get("poss"),
            "fetched_at": fetched_at})
    return logs, teams


def previous_nba_season(season: str) -> str:
    """'2026-27' -> '2025-26'."""
    start = int(season[:4]) - 1
    return f"{start}-{(start + 1) % 100:02d}"


def _nba_prior_rows(season: str) -> list[dict[str, Any]]:
    """Every player's season totals in ONE LeagueDashPlayerStats call."""
    try:
        from nba_api.stats.endpoints import leaguedashplayerstats  # type: ignore
    except ImportError as exc:
        raise PropStatsUnavailable("nba_api not installed (pip install nba_api)") from exc
    try:
        res = leaguedashplayerstats.LeagueDashPlayerStats(
            season=season, per_mode_detailed="Totals", season_type_all_star="Regular Season",
            timeout=30)
        return list(res.get_normalized_dict()["LeagueDashPlayerStats"])
    except Exception as exc:
        raise PropStatsUnavailable(f"nba_api LeagueDashPlayerStats failed: {exc}") from exc


def build_prior_rows(rows: list[dict[str, Any]], season: str, fetched_at: str
                     ) -> list[dict[str, Any]]:
    return [{"sport": "NBA", "player_key": player_key(str(r["PLAYER_NAME"])), "season": season,
             "player_name": str(r["PLAYER_NAME"]), "gp": r.get("GP"), "min": r.get("MIN"),
             "pts": r.get("PTS"), "reb": r.get("REB"), "ast": r.get("AST"), "fgm": r.get("FGM"),
             "fga": r.get("FGA"), "fg3m": r.get("FG3M"), "fg3a": r.get("FG3A"),
             "ftm": r.get("FTM"), "fta": r.get("FTA"), "fetched_at": fetched_at}
            for r in rows if r.get("PLAYER_NAME")]


_PRIOR_FIELDS = ("sport", "player_key", "season", "player_name", "gp", "min", "pts", "reb",
                 "ast", "fgm", "fga", "fg3m", "fg3a", "ftm", "fta", "fetched_at")


def store_prior_rows(conn: sqlite3.Connection, rows: list[dict[str, Any]]) -> int:
    with conn:
        conn.executemany(
            f"INSERT OR REPLACE INTO player_season_baselines ({', '.join(_PRIOR_FIELDS)}) "
            f"VALUES ({', '.join('?' * len(_PRIOR_FIELDS))})",
            [[r.get(f) for f in _PRIOR_FIELDS] for r in rows])
    return len(rows)


def update_nba_prop_stats(conn: sqlite3.Connection, season: str | None = None) -> dict[str, Any]:
    """Current-season player + team logs (2 calls) and, once per season, the
    prior season's totals (1 call; a completed season never changes)."""
    season = season or nba_season_label(db.utcnow().date())
    now = db.iso(db.utcnow())
    logs, teams = build_nba_tables(_nba_rows("P", season), _nba_rows("T", season), season, now)
    out = {"sport": "NBA", "season": season, **_store(conn, "NBA", logs, teams)}
    prior = previous_nba_season(season)
    have = conn.execute("SELECT COUNT(*) FROM player_season_baselines WHERE sport = 'NBA' AND "
                        "season = ?", (prior,)).fetchone()[0]
    if have:
        out["prior_players"] = int(have)
    else:
        try:
            out["prior_players"] = store_prior_rows(conn, build_prior_rows(
                _nba_prior_rows(prior), prior, now))
        except PropStatsUnavailable as exc:      # optional: small samples fall back to league
            log.warning("prior-season baselines unavailable: %s", exc)
            out["prior_players"], out["prior_error"] = 0, str(exc)
    return out


# ==========================================================================
# NFL (nflreadpy, else nfl_data_py)
# ==========================================================================
_PBP_COLS = ["game_id", "posteam", "defteam", "season_type", "play_type", "wp", "qtr",
             "pass_attempt", "rush_attempt", "yards_gained", "epa"]


def _records(df: Any, cols: Iterable[str]) -> list[dict[str, Any]]:
    cols = [c for c in cols if c in df.columns]
    if hasattr(df, "to_dicts"):                      # polars
        return df.select(cols).to_dicts()
    return df[cols].to_dict("records")              # pandas


def _nfl_frames(season: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]], str]:
    weekly_cols = ["player_display_name", "player_name", "position", "team", "recent_team",
                   "opponent_team", "season", "week", "season_type", "game_id", "targets",
                   "receptions", "receiving_yards", "receiving_air_yards", "target_share",
                   "air_yards_share", "carries", "rushing_yards", "attempts", "completions",
                   "passing_yards"]
    try:
        import nflreadpy as nfl  # type: ignore
        return (_records(nfl.load_player_stats([season]), weekly_cols),
                _records(nfl.load_pbp([season]), _PBP_COLS), "nflreadpy")
    except ImportError:
        pass
    except Exception as exc:
        log.warning("nflreadpy failed (%s); trying nfl_data_py", exc)
    try:
        import nfl_data_py as nfl_old  # type: ignore
    except ImportError as exc:
        raise PropStatsUnavailable("install nflreadpy (or nfl_data_py)") from exc
    try:
        weekly = nfl_old.import_weekly_data([season])
        pbp = nfl_old.import_pbp_data([season], columns=_PBP_COLS, downcast=True)
        return _records(weekly, weekly_cols), _records(pbp, _PBP_COLS), "nfl_data_py"
    except Exception as exc:
        raise PropStatsUnavailable(f"nfl_data_py failed: {exc}") from exc


def build_nfl_tables(weekly: list[dict[str, Any]], pbp: list[dict[str, Any]], season: int,
                     fetched_at: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """(player_game_logs rows, team_prop_context rows) from nflverse weekly + pbp records."""
    off: dict[str, dict[str, Any]] = {}
    dfn: dict[str, dict[str, Any]] = {}
    for p in pbp:
        if p.get("season_type") not in (None, "REG", "POST"):
            continue
        is_pass, is_rush = int(p.get("pass_attempt") or 0) == 1, int(p.get("rush_attempt") or 0) == 1
        if not (is_pass or is_rush) or not p.get("posteam") or not p.get("defteam"):
            continue
        wp, qtr = p.get("wp"), p.get("qtr")
        neutral = wp is not None and 0.2 <= float(wp) <= 0.8 and int(qtr or 5) <= 3
        yds, epa = float(p.get("yards_gained") or 0), float(p.get("epa") or 0)
        o = off.setdefault(p["posteam"], {"games": set(), "plays": 0, "pass": 0, "rush": 0,
                                          "n_neutral": 0, "n_neutral_pass": 0})
        o["games"].add(p.get("game_id"))
        o["plays"] += 1
        o["pass" if is_pass else "rush"] += 1
        if neutral:
            o["n_neutral"] += 1
            o["n_neutral_pass"] += is_pass
        d = dfn.setdefault(p["defteam"], {"games": set(), "plays": 0, "pass": 0, "rush": 0,
                                          "pass_yds": 0.0, "rush_yds": 0.0, "epa": 0.0,
                                          "pass_epa": 0.0, "rush_epa": 0.0})
        d["games"].add(p.get("game_id"))
        d["plays"] += 1
        d["epa"] += epa
        if is_pass:
            d["pass"] += 1
            d["pass_yds"] += yds
            d["pass_epa"] += epa
        else:
            d["rush"] += 1
            d["rush_yds"] += yds
            d["rush_epa"] += epa
    teams = []
    for abbr, o in off.items():
        cid = try_resolve_team(str(abbr), "NFL")
        d = dfn.get(abbr)
        if not cid or not d:
            continue
        g, dg = max(len(o["games"]), 1), max(len(d["games"]), 1)
        teams.append({
            "canonical_id": cid, "sport": "NFL", "season": str(season), "updated_at": fetched_at,
            "games": len(o["games"]), "plays_pg": o["plays"] / g, "def_plays_pg": d["plays"] / dg,
            "pass_pg": o["pass"] / g, "rush_pg": o["rush"] / g,
            "neutral_pass_rate": (o["n_neutral_pass"] / o["n_neutral"]) if o["n_neutral"] else None,
            "def_pass_ypa": d["pass_yds"] / d["pass"] if d["pass"] else None,
            "def_rush_ypc": d["rush_yds"] / d["rush"] if d["rush"] else None,
            "def_epa_play": d["epa"] / d["plays"] if d["plays"] else None,
            "def_pass_epa": d["pass_epa"] / d["pass"] if d["pass"] else None,
            "def_rush_epa": d["rush_epa"] / d["rush"] if d["rush"] else None})
    logs = []
    for r in weekly:
        if r.get("season_type") not in (None, "REG", "POST") or \
                str(r.get("position") or "") not in NFL_POSITIONS:
            continue
        name = str(r.get("player_display_name") or r.get("player_name") or "")
        team_abbr = str(r.get("team") or r.get("recent_team") or "")
        week = int(r.get("week") or 0)
        logs.append({
            "sport": "NFL", "player_key": player_key(name),
            "game_id": str(r.get("game_id") or f"{season}_{week:02d}_{team_abbr}"),
            "player_name": name, "canonical_id": try_resolve_team(team_abbr, "NFL"),
            "opponent_id": try_resolve_team(str(r.get("opponent_team") or ""), "NFL"),
            "position": r.get("position"), "game_date": f"{season}-W{week:02d}",
            "season": str(season), "targets": r.get("targets"), "receptions": r.get("receptions"),
            "rec_yds": r.get("receiving_yards"), "air_yards": r.get("receiving_air_yards"),
            "target_share": r.get("target_share"), "air_yards_share": r.get("air_yards_share"),
            "carries": r.get("carries"), "rush_yds": r.get("rushing_yards"),
            "pass_att": r.get("attempts"), "completions": r.get("completions"),
            "pass_yds": r.get("passing_yards"), "fetched_at": fetched_at})
    return logs, teams


def update_nfl_prop_stats(conn: sqlite3.Connection, season: int | None = None) -> dict[str, Any]:
    season = season or nfl_season(db.utcnow().date())
    weekly, pbp, source = _nfl_frames(season)
    logs, teams = build_nfl_tables(weekly, pbp, season, db.iso(db.utcnow()))
    return {"sport": "NFL", "season": season, "source": source, **_store(conn, "NFL", logs, teams)}


# ==========================================================================
# Storage
# ==========================================================================
_LOG_FIELDS = ("sport", "player_key", "game_id", "player_name", "canonical_id", "opponent_id",
               "position", "game_date", "season", *NBA_LOG_COLS, "team_margin", "game_poss",
               *NFL_LOG_COLS, "fetched_at")
_TEAM_FIELDS = ("canonical_id", "sport", "season", "updated_at", "games", "pace", "drtg",
                "allowed_reb_100", "allowed_ast_100", "allowed_fg3m_100", "plays_pg",
                "def_plays_pg", "pass_pg", "rush_pg", "neutral_pass_rate", "def_pass_ypa",
                "def_rush_ypc", "def_epa_play", "def_pass_epa", "def_rush_epa")


def _store(conn: sqlite3.Connection, sport: str, logs: list[dict[str, Any]],
           teams: list[dict[str, Any]]) -> dict[str, int]:
    with conn:
        conn.executemany(
            f"INSERT OR REPLACE INTO player_game_logs ({', '.join(_LOG_FIELDS)}) "
            f"VALUES ({', '.join('?' * len(_LOG_FIELDS))})",
            [[r.get(f) for f in _LOG_FIELDS] for r in logs])
        conn.executemany(
            f"INSERT OR REPLACE INTO team_prop_context ({', '.join(_TEAM_FIELDS)}) "
            f"VALUES ({', '.join('?' * len(_TEAM_FIELDS))})",
            [[t.get(f) for f in _TEAM_FIELDS] for t in teams])
    return {"player_rows": len(logs), "teams": len(teams),
            "players": len({r["player_key"] for r in logs})}


def update_prop_stats(conn: sqlite3.Connection, leagues: Iterable[str] = ("NBA", "NFL")
                      ) -> list[dict[str, Any]]:
    """Refresh each league; a missing package or outage is reported, never raised."""
    out = []
    for lg in leagues:
        fn = update_nba_prop_stats if lg == "NBA" else update_nfl_prop_stats
        try:
            out.append({**fn(conn), "error": None})
        except PropStatsUnavailable as exc:
            log.warning("%s prop stats unavailable: %s", lg, exc)
            out.append({"sport": lg, "error": str(exc), "player_rows": 0, "teams": 0,
                        "players": 0})
    return out


def stats_age_hours(conn: sqlite3.Connection, sport: str) -> float | None:
    row = conn.execute("SELECT MAX(updated_at) FROM team_prop_context WHERE sport = ?",
                       (sport,)).fetchone()
    if not row or not row[0]:
        return None
    return (db.utcnow() - db.parse_ts(row[0])).total_seconds() / 3600.0


def ensure_fresh(conn: sqlite3.Connection, leagues: Iterable[str],
                 max_age_h: float = STALE_HOURS) -> list[dict[str, Any]]:
    """Refresh only leagues whose context is missing or older than ``max_age_h``."""
    stale = [lg for lg in leagues if (stats_age_hours(conn, lg) or 1e9) > max_age_h]
    return update_prop_stats(conn, stale) if stale else []


# ==========================================================================
# Loaders (used by the prop scanner)
# ==========================================================================
@dataclass
class PlayerHistory:
    name: str
    key: str
    sport: str
    team_id: str | None
    position: str | None
    logs: list[dict[str, Any]]                  # most recent first
    season_logs: list[dict[str, Any]] = field(default_factory=list)
    prior: dict[str, Any] | None = None         # NBA prior-season totals, if known


def load_player_history(conn: sqlite3.Connection, sport: str, name: str,
                        team_ids: Iterable[str]) -> PlayerHistory | None:
    """History for an Odds-API player name, restricted to the event's two teams.

    Matching, among players whose latest team is in ``team_ids``:
      1. exact normalized name;
      2. same last name and one first name a prefix of the other, if unique
         ('Nic Claxton' -> 'Nicolas Claxton');
      3. difflib ratio >= 0.88 (typos / transliterations).
    """
    teams = tuple(t for t in team_ids if t)
    if not teams:
        return None
    marks = ",".join("?" * len(teams))
    key = player_key(name)
    candidates = {r[0]: r[1] for r in conn.execute(
        f"SELECT player_key, canonical_id FROM player_game_logs p WHERE sport = ? AND "
        f"game_date = (SELECT MAX(game_date) FROM player_game_logs q WHERE q.sport = p.sport "
        f"AND q.player_key = p.player_key) AND canonical_id IN ({marks})", (sport, *teams))}
    if key not in candidates:
        key = _nickname_match(key, candidates) or next(
            iter(difflib.get_close_matches(key, list(candidates), n=1, cutoff=0.88)), None)
        if key is None:
            return None
    rows = [dict(r) for r in conn.execute(
        "SELECT * FROM player_game_logs WHERE sport = ? AND player_key = ? "
        "ORDER BY game_date DESC", (sport, key))]
    if not rows:
        return None
    season = rows[0]["season"]
    current = [r for r in rows if r["season"] == season]
    prior = load_prior_season(conn, sport, key, previous_nba_season(season)) \
        if sport == "NBA" and season and re.fullmatch(r"\d{4}-\d{2}", season) else None
    return PlayerHistory(rows[0]["player_name"], key, sport, rows[0]["canonical_id"],
                         next((r["position"] for r in rows if r["position"]), None),
                         current, current, prior)


def load_prior_season(conn: sqlite3.Connection, sport: str, key: str, season: str
                      ) -> dict[str, Any] | None:
    row = conn.execute("SELECT * FROM player_season_baselines WHERE sport = ? AND "
                       "player_key = ? AND season = ?", (sport, key, season)).fetchone()
    return dict(row) if row and (row["min"] or 0) > 0 else None


def teammate_recently_active(conn: sqlite3.Connection, sport: str, team_id: str | None,
                             name: str, last_n: int = 3) -> bool | None:
    """Did ``name`` play in any of the team's last ``last_n`` logged games?

    True  -> his absence is NEW, so the player's logs do not yet reflect it.
    False -> he has been out a while; the logs already carry the extra usage.
    None  -> no history for him at all (name mismatch / never played): unknown.
    """
    if not team_id:
        return None
    key = player_key(name)
    if not conn.execute("SELECT 1 FROM player_game_logs WHERE sport = ? AND player_key = ? "
                        "LIMIT 1", (sport, key)).fetchone():
        return None
    games = [r[0] for r in conn.execute(
        "SELECT game_id FROM player_game_logs WHERE sport = ? AND canonical_id = ? "
        "GROUP BY game_id ORDER BY MAX(game_date) DESC LIMIT ?", (sport, team_id, last_n))]
    if not games:
        return None
    marks = ",".join("?" * len(games))
    return conn.execute(
        f"SELECT 1 FROM player_game_logs WHERE sport = ? AND player_key = ? AND game_id IN "
        f"({marks}) AND IFNULL(minutes, 1) > 0 LIMIT 1", (sport, key, *games)).fetchone() \
        is not None


def _nickname_match(key: str, candidates: Iterable[str]) -> str | None:
    parts = key.split()
    if len(parts) < 2 or len(parts[0]) < 2:
        return None
    first, last = parts[0], " ".join(parts[1:])
    hits = []
    for c in candidates:
        cp = c.split()
        if len(cp) >= 2 and " ".join(cp[1:]) == last and \
                (cp[0].startswith(first) or first.startswith(cp[0])):
            hits.append(c)
    return hits[0] if len(hits) == 1 else None


def load_team_context(conn: sqlite3.Connection, canonical_id: str | None) -> dict[str, Any]:
    if not canonical_id:
        return {}
    row = conn.execute("SELECT * FROM team_prop_context WHERE canonical_id = ?",
                       (canonical_id,)).fetchone()
    return dict(row) if row else {}


def league_context(conn: sqlite3.Connection, sport: str) -> dict[str, Any]:
    cols = ("pace", "drtg", "allowed_reb_100", "allowed_ast_100", "allowed_fg3m_100",
            "plays_pg", "def_plays_pg", "neutral_pass_rate", "def_pass_ypa", "def_rush_ypc")
    row = conn.execute(f"SELECT {', '.join(f'AVG({c}) AS {c}' for c in cols)} FROM "
                       f"team_prop_context WHERE sport = ?", (sport,)).fetchone()
    return {c: row[c] for c in cols} if row else {}


def player_injury(conn: sqlite3.Connection, canonical_id: str | None, name: str
                  ) -> dict[str, Any] | None:
    """The player's own row in injury_reports (from --update-stats), if any."""
    if not canonical_id:
        return None
    key = player_key(name)
    for r in conn.execute("SELECT player, position, status, is_key, key_reason FROM "
                          "injury_reports WHERE canonical_id = ?", (canonical_id,)):
        if player_key(r["player"]) == key:
            return dict(r)
    return None


def key_teammate_injuries(conn: sqlite3.Connection, canonical_id: str | None, name: str
                          ) -> list[dict[str, Any]]:
    """Key teammates (is_key = 1) not cleared to play, excluding the player."""
    if not canonical_id:
        return []
    key = player_key(name)
    return [dict(r) for r in conn.execute(
        "SELECT player, position, status FROM injury_reports WHERE canonical_id = ? AND "
        "is_key = 1 AND lower(status) != 'active'", (canonical_id,))
        if player_key(r["player"]) != key]
