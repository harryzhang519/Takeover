"""Live team stats, schedules and injuries from ESPN's public JSON (no API key).

``update_stats(conn)`` refreshes, for all 30 NBA and 32 NFL teams:
    team_schedule   every game (date, opponent, home/away, score) -> rest + SRS
    team_ratings    record, splits, PF/PA, Pythagorean, SRS, and
                    NBA: ORTG / DRTG / Net / Adj Net / Pace / eFG%
                    NFL: yards per play, turnover margin (+ optional EPA hook)
    injury_reports  ESPN injury feed, with is_key flagged for NBA starters
                    (from the last box score) and NFL starting QBs

Derivations (ESPN does not publish efficiency ratings directly):
    possessions/game = FGA - OREB + TOV + 0.44 * FTA        (team-side estimate)
    ORTG = 100 * PF / poss,  DRTG = 100 * PA / poss,  Pace = poss
    eFG% = (FGM + 0.5 * 3PM) / FGA
    Pythagorean = PF^x / (PF^x + PA^x), x = 14 (NBA), 2.37 (NFL)
    SRS: iterative opponent- and venue-adjusted average margin (mean 0)
    Adj Net (NBA) = SRS * 100 / Pace
Optional enrichments, used only when installed: ``nba_api`` (official
advanced ratings) and ``nfl_data_py`` (EPA/play). Neither is required.

Requests run on a small thread pool; every payload is cached in api_cache.
"""

from __future__ import annotations

import logging
import sqlite3
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.core import database as db  # noqa: E402
from src.core.matchup_model import HOME_ADVANTAGE, QUESTIONABLE_STATUSES  # noqa: E402
from src.data.http_client import HttpError, get_json  # noqa: E402
from src.data.team_registry import (  # noqa: E402
    REGISTRY,
    by_espn_id,
    sync_registry,
    teams_for,
    try_resolve_team,
)

log = logging.getLogger("breeze.stats")

ESPN_BASE = "https://site.api.espn.com/apis/site/v2/sports"
ESPN_PATH = {"NBA": "basketball/nba", "NFL": "football/nfl"}
TEAM_TTL_S = 6 * 3600
INJURY_TTL_S = 30 * 60
SUMMARY_TTL_S = 7 * 24 * 3600           # completed box scores do not change
PYTHAG_EXP = {"NBA": 14.0, "NFL": 2.37}
SRS_ITERATIONS = 300
WORKERS = 8


# ==========================================================================
# Fetch helpers
# ==========================================================================
def _fetch_many(conn: sqlite3.Connection, reqs: list[tuple[str, str, dict | None]],
                ttl: int) -> dict[str, Any]:
    """Fetch (cache_key, url, params) triples; cache hits skip the network.

    Network calls run in a thread pool; all SQLite access stays on this thread.
    Failed requests map to None.
    """
    out: dict[str, Any] = {}
    misses = []
    for key, url, params in reqs:
        hit = db.cache_get(conn, key)
        if hit is not None:
            out[key] = hit
        else:
            misses.append((key, url, params))

    def one(req: tuple[str, str, dict | None]) -> tuple[str, Any]:
        key, url, params = req
        try:
            return key, get_json(url, params, retries=2)[0]
        except HttpError as exc:
            log.warning("ESPN fetch failed (%s): %s", key, exc)
            return key, None

    if misses:
        with ThreadPoolExecutor(max_workers=WORKERS) as pool:
            for key, payload in pool.map(one, misses):
                out[key] = payload
                if payload is not None:
                    db.cache_put(conn, key, payload, ttl)
    return out


def _num(v: Any) -> float | None:
    if isinstance(v, dict):
        v = v.get("value", v.get("displayValue"))
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def flatten_stats(payload: dict | None) -> dict[str, float]:
    """``{"category.stat": value}`` plus unqualified names (first occurrence wins)."""
    flat: dict[str, float] = {}
    cats = ((payload or {}).get("results") or {}).get("stats", {}).get("categories", [])
    for cat in cats:
        for st in cat.get("stats", []):
            val = _num(st.get("value"))
            if val is None:
                continue
            flat[f"{cat['name']}.{st['name']}"] = val
            flat.setdefault(st["name"], val)
    return flat


def parse_record(team_payload: dict | None) -> dict[str, Any]:
    items = ((team_payload or {}).get("team") or {}).get("record", {}).get("items", [])
    out: dict[str, Any] = {}
    for it in items:
        stats = {s["name"]: s.get("value") for s in it.get("stats", [])}
        if it.get("type") == "total":
            out.update(stats)
            out["summary"] = it.get("summary")
        elif it.get("type") == "home":
            out["home_record"] = it.get("summary")
        elif it.get("type") in ("road", "away"):
            out["away_record"] = it.get("summary")
    return out


def parse_schedule(payload: dict | None, sport: str, cid: str) -> list[dict[str, Any]]:
    """Schedule rows for one team (completed and upcoming)."""
    me = REGISTRY[cid]
    rows = []
    for ev in (payload or {}).get("events", []):
        comp = (ev.get("competitions") or [{}])[0]
        sides = comp.get("competitors", [])
        ours = next((c for c in sides if str(c.get("team", {}).get("id")) == me.espn_id), None)
        theirs = next((c for c in sides if c is not ours), None)
        if not ours or not theirs:
            continue
        opp = by_espn_id(sport, theirs["team"]["id"])
        completed = bool(comp.get("status", {}).get("type", {}).get("completed"))
        rows.append({
            "canonical_id": cid, "espn_event_id": str(ev["id"]),
            "game_time": db.iso(db.parse_ts(ev["date"])),
            "opponent_id": opp.canonical_id if opp else None,
            "is_home": int(ours.get("homeAway") == "home"), "completed": int(completed),
            "points_for": _num(ours.get("score")) if completed else None,
            "points_against": _num(theirs.get("score")) if completed else None,
        })
    return rows


# ==========================================================================
# Ratings math
# ==========================================================================
def pythagorean(pf: float, pa: float, exponent: float) -> float | None:
    if pf <= 0 or pa <= 0:
        return None
    return pf ** exponent / (pf ** exponent + pa ** exponent)


def compute_srs(schedule: list[dict[str, Any]], hca: float,
                iterations: int = SRS_ITERATIONS) -> dict[str, float]:
    """Simple Rating System: rating = avg venue-neutral margin + avg opponent rating.

    Solved by fixed-point iteration and centred to mean 0 each pass.
    """
    games: dict[str, list[tuple[str, float]]] = {}
    seen: set[tuple[str, str]] = set()
    for r in schedule:
        if not r["completed"] or r["opponent_id"] is None or r["points_for"] is None:
            continue
        if (r["canonical_id"], r["espn_event_id"]) in seen:
            continue
        seen.add((r["canonical_id"], r["espn_event_id"]))
        margin = r["points_for"] - r["points_against"] - (hca if r["is_home"] else -hca)
        games.setdefault(r["canonical_id"], []).append((r["opponent_id"], margin))
    rating = {t: sum(m for _, m in g) / len(g) for t, g in games.items()}
    for _ in range(iterations):
        new = {t: sum(m + rating.get(o, 0.0) for o, m in g) / len(g) for t, g in games.items()}
        mean = sum(new.values()) / len(new) if new else 0.0
        rating = {t: v - mean for t, v in new.items()}
    return rating


def nba_efficiency(stats: dict[str, float], pf: float | None, pa: float | None
                   ) -> dict[str, float | None]:
    fga, oreb = stats.get("offensive.avgFieldGoalsAttempted"), stats.get("offensive.avgOffensiveRebounds")
    tov, fta = stats.get("offensive.avgTurnovers"), stats.get("offensive.avgFreeThrowsAttempted")
    fgm, tpm = stats.get("offensive.avgFieldGoalsMade"), stats.get("offensive.avgThreePointFieldGoalsMade")
    if None in (fga, oreb, tov, fta) or not pf or not pa:
        return {"ortg": None, "drtg": None, "net_rtg": None, "pace": None, "efg_pct": None}
    poss = fga - oreb + tov + 0.44 * fta                    # type: ignore[operator]
    ortg, drtg = 100 * pf / poss, 100 * pa / poss
    efg = (fgm + 0.5 * tpm) / fga if fgm is not None and tpm is not None and fga else None
    return {"ortg": ortg, "drtg": drtg, "net_rtg": ortg - drtg, "pace": poss, "efg_pct": efg}


def nfl_efficiency(stats: dict[str, float]) -> dict[str, float | None]:
    yards = stats.get("rushing.totalYards") or stats.get("totalYards")
    plays = stats.get("rushing.totalOffensivePlays") or stats.get("totalOffensivePlays")
    return {"yards_per_play": yards / plays if yards and plays else None,
            "turnover_margin": stats.get("miscellaneous.turnOverDifferential"),
            "points_per_drive": None}


def _nba_api_overrides(season_label: str | None) -> dict[str, dict[str, float]]:
    """Official advanced ratings via ``nba_api`` when installed; {} otherwise."""
    try:
        from nba_api.stats.endpoints import leaguedashteamstats  # type: ignore
    except ImportError:
        return {}
    try:
        df = leaguedashteamstats.LeagueDashTeamStats(
            measure_type_detailed_defense="Advanced", season=season_label,
            timeout=20).get_data_frames()[0]
    except Exception as exc:        # third-party; never let it break the harvest
        log.warning("nba_api unavailable (%s); using ESPN-derived ratings", exc)
        return {}
    out = {}
    for _, row in df.iterrows():
        cid = try_resolve_team(str(row["TEAM_NAME"]), "NBA")
        if cid:
            out[cid] = {"ortg": float(row["OFF_RATING"]), "drtg": float(row["DEF_RATING"]),
                        "net_rtg": float(row["NET_RATING"]), "pace": float(row["PACE"]),
                        "efg_pct": float(row["EFG_PCT"])}
    return out


def _nfl_epa_hook(season: int | None) -> dict[str, float]:
    """Net EPA/play (offense minus defense) via ``nfl_data_py`` when installed."""
    try:
        import nfl_data_py as nfl  # type: ignore
    except ImportError:
        return {}
    try:
        pbp = nfl.import_pbp_data([season], columns=["posteam", "defteam", "epa", "play_type"],
                                  downcast=True)
        pbp = pbp[pbp["play_type"].isin(["pass", "run"])]
        off = pbp.groupby("posteam")["epa"].mean()
        dfn = pbp.groupby("defteam")["epa"].mean()
    except Exception as exc:
        log.warning("nfl_data_py EPA unavailable (%s)", exc)
        return {}
    out = {}
    for abbr in off.index:
        cid = try_resolve_team(str(abbr), "NFL")
        if cid and abbr in dfn.index:
            out[cid] = float(off[abbr] - dfn[abbr])
    return out


# ==========================================================================
# Injuries
# ==========================================================================
def _starters_from_summary(payload: dict | None, sport: str) -> dict[str, set[str]]:
    """{canonical_id: {key player names}} from one box score.

    NBA: players flagged ``starter``. NFL: the first passer (starting QB).
    """
    out: dict[str, set[str]] = {}
    for side in ((payload or {}).get("boxscore") or {}).get("players", []):
        t = by_espn_id(sport, side.get("team", {}).get("id", ""))
        if not t:
            continue
        names: set[str] = set()
        for cat in side.get("statistics", []):
            athletes = cat.get("athletes", [])
            if sport == "NBA":
                names |= {a["athlete"]["displayName"] for a in athletes if a.get("starter")}
            elif cat.get("name") == "passing" and athletes:
                names.add(athletes[0]["athlete"]["displayName"])
        out[t.canonical_id] = names
    return out


def parse_injuries(payload: dict | None, sport: str, key_players: dict[str, set[str]],
                   fetched_at: str) -> list[dict[str, Any]]:
    rows = []
    for team_block in (payload or {}).get("injuries", []):
        t = by_espn_id(sport, team_block.get("id", "")) or (
            REGISTRY.get(try_resolve_team(team_block.get("displayName"), sport) or ""))
        if not t:
            continue
        starters = key_players.get(t.canonical_id, set())
        for inj in team_block.get("injuries", []):
            if (inj.get("status") or "").lower() == "active":
                continue                        # cleared to play; nothing to model
            ath = inj.get("athlete") or {}
            name = ath.get("displayName") or "?"
            pos = (ath.get("position") or {}).get("abbreviation")
            reason = None
            if name in starters:
                reason = "starter in last game" if sport == "NBA" else "starting QB in last game"
            elif sport == "NFL" and pos == "QB" and not starters:
                reason = "QB (no box score to confirm starter)"
            rows.append({"canonical_id": t.canonical_id, "sport": sport, "player": name,
                         "position": pos, "status": inj.get("status") or "Unknown",
                         "is_key": int(reason is not None), "key_reason": reason,
                         "detail": inj.get("shortComment"), "reported_at": inj.get("date"),
                         "fetched_at": fetched_at})
    return rows


# ==========================================================================
# Orchestration
# ==========================================================================
def update_sport(conn: sqlite3.Connection, sport: str) -> dict[str, Any]:
    """Refresh schedule, ratings and injuries for one league."""
    base = f"{ESPN_BASE}/{ESPN_PATH[sport]}"
    tlist = teams_for(sport)
    reqs: list[tuple[str, str, dict | None]] = []
    for t in tlist:
        reqs += [(f"espn:team:{t.canonical_id}", f"{base}/teams/{t.espn_id}", None),
                 (f"espn:stats:{t.canonical_id}", f"{base}/teams/{t.espn_id}/statistics", None),
                 (f"espn:sched:{t.canonical_id}", f"{base}/teams/{t.espn_id}/schedule", None)]
    payloads = _fetch_many(conn, reqs, TEAM_TTL_S)

    schedule: list[dict[str, Any]] = []
    for t in tlist:
        schedule += parse_schedule(payloads.get(f"espn:sched:{t.canonical_id}"), sport,
                                   t.canonical_id)
    with conn:
        conn.execute("DELETE FROM team_schedule WHERE canonical_id LIKE ?", (f"{sport}_%",))
        conn.executemany(
            "INSERT OR REPLACE INTO team_schedule VALUES (:canonical_id, :espn_event_id, "
            ":game_time, :opponent_id, :is_home, :completed, :points_for, :points_against)",
            schedule)
    srs = compute_srs(schedule, HOME_ADVANTAGE[sport])

    season_payload = payloads.get(f"espn:stats:{tlist[0].canonical_id}") or {}
    season = (season_payload.get("season") or {}).get("year")
    overrides: dict[str, Any] = {}
    if sport == "NBA" and season:
        overrides = _nba_api_overrides(f"{int(season) - 1}-{str(season)[-2:]}")
    epa = _nfl_epa_hook(int(season)) if sport == "NFL" and season else {}

    now = db.iso(db.utcnow())
    ratings = []
    for t in tlist:
        rec = parse_record(payloads.get(f"espn:team:{t.canonical_id}"))
        stats = flatten_stats(payloads.get(f"espn:stats:{t.canonical_id}"))
        games = int(rec.get("gamesPlayed") or 0)
        pf, pa = _num(rec.get("avgPointsFor")), _num(rec.get("avgPointsAgainst"))
        row: dict[str, Any] = {
            "canonical_id": t.canonical_id, "sport": sport, "season": str(season or ""),
            "updated_at": now, "source": "espn", "games": games,
            "wins": int(rec.get("wins") or 0), "losses": int(rec.get("losses") or 0),
            "home_record": rec.get("home_record"), "away_record": rec.get("away_record"),
            "points_for_pg": pf, "points_against_pg": pa,
            "point_diff_pg": (pf - pa) if pf is not None and pa is not None else None,
            "pythag_win_pct": pythagorean(pf, pa, PYTHAG_EXP[sport]) if pf and pa else None,
            "srs": srs.get(t.canonical_id), "ortg": None, "drtg": None, "net_rtg": None,
            "adj_net_rtg": None, "pace": None, "efg_pct": None, "yards_per_play": None,
            "points_per_drive": None, "turnover_margin": None,
            "epa_per_play": epa.get(t.canonical_id),
        }
        if sport == "NBA":
            row.update(nba_efficiency(stats, pf, pa))
            if t.canonical_id in overrides:
                row.update(overrides[t.canonical_id])
                row["source"] = "nba_api+espn"
            if row["pace"] and row["srs"] is not None:
                row["adj_net_rtg"] = row["srs"] * 100.0 / row["pace"]
        else:
            row.update(nfl_efficiency(stats))
            if t.canonical_id in epa:
                row["source"] = "espn+nfl_data_py"
        ratings.append(row)
    cols = list(ratings[0])
    with conn:
        conn.executemany(f"INSERT OR REPLACE INTO team_ratings ({', '.join(cols)}) VALUES "
                         f"({', '.join(':' + c for c in cols)})", ratings)

    # Key players from each team's most recent completed box score.
    last_game: dict[str, str] = {}
    for r in sorted(schedule, key=lambda r: r["game_time"]):
        if r["completed"]:
            last_game[r["canonical_id"]] = r["espn_event_id"]
    summaries = _fetch_many(conn, [(f"espn:summary:{sport}:{e}", f"{base}/summary", {"event": e})
                                   for e in sorted(set(last_game.values()))], SUMMARY_TTL_S)
    key_players: dict[str, set[str]] = {}
    for cid, ev in last_game.items():
        found = _starters_from_summary(summaries.get(f"espn:summary:{sport}:{ev}"), sport)
        key_players[cid] = found.get(cid, set())

    inj_payload = _fetch_many(conn, [(f"espn:injuries:{sport}", f"{base}/injuries", None)],
                              INJURY_TTL_S).get(f"espn:injuries:{sport}")
    injuries = parse_injuries(inj_payload, sport, key_players, now)
    with conn:
        conn.execute("DELETE FROM injury_reports WHERE sport = ?", (sport,))
        conn.executemany(
            "INSERT INTO injury_reports (canonical_id, sport, player, position, status, is_key,"
            " key_reason, detail, reported_at, fetched_at) VALUES (:canonical_id, :sport,"
            " :player, :position, :status, :is_key, :key_reason, :detail, :reported_at,"
            " :fetched_at)", injuries)

    rated = sum(1 for r in ratings if r["games"])
    return {"sport": sport, "teams": len(tlist), "teams_with_games": rated,
            "schedule_rows": len(schedule), "injuries": len(injuries),
            "key_questionable": sum(1 for i in injuries if i["is_key"]
                                    and i["status"].lower() in QUESTIONABLE_STATUSES),
            "season": season, "source": "nba_api+espn" if overrides else
            ("espn+nfl_data_py" if epa else "espn"),
            "failed_requests": sum(1 for v in payloads.values() if v is None)}


def update_stats(conn: sqlite3.Connection, sports: tuple[str, ...] = ("NBA", "NFL")
                 ) -> list[dict[str, Any]]:
    """Refresh every league in ``sports``; registry is synced first."""
    sync_registry(conn)
    return [update_sport(conn, s) for s in sports]
