"""Discord capper write-ups for APPROVED picks, grounded in our own data.

    gather_context(conn, pick)   -> WriteupContext: real facts from ledger_picks (incl. the
                                    pick's analysis), team_schedule, injury_reports,
                                    team_ratings, team_prop_context, player_game_logs,
                                    market_snapshots and the scan JSON files
    fallback_writeup(ctx)        -> deterministic capper-voice blurb (always available)
    generate_writeup(conn, pick) -> (text, source): local Ollama first, validated, else fallback
    attach_writeup(conn, id)     -> generate + store on an APPROVED, signed-off pick

Every fact is a finished sentence in sports talk, built only from numbers we
hold. The fallback stitches the strongest few together; Ollama gets the same
facts plus three voice examples and its reply is rejected (fallback used) if
it uses a number that is not in the facts, mentions the math (model, Kelly,
probabilities, percentages), runs too long or too short. Em dashes are
rewritten to commas. Every write-up ends with @everyone.

Ollama settings (environment):
    BREEZE_WRITEUP_LLM      "off" disables Ollama (template only)
    BREEZE_OLLAMA_URL       default http://localhost:11434
    BREEZE_OLLAMA_MODEL     default qwen2.5-coder:7b
    BREEZE_OLLAMA_TIMEOUT   seconds, default 4
After a failed call Ollama is skipped for 60 s, so a batch of approvals with
Ollama down costs one timeout, not one per pick.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import re
import sqlite3
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable
from zoneinfo import ZoneInfo

from src.core import database as db
from src.core.engine import HITL_APPROVED, american_to_decimal
from src.core.prop_model import PROP_MARKETS, is_prop_market
from src.data.team_registry import REGISTRY, try_resolve_team
from src.data.teams import SPORT_TO_LEAGUE
from templates.discord_templates import post_selection, team_nickname

log = logging.getLogger("breeze.writeup")

ROOT = Path(__file__).resolve().parents[1]
SCAN_PATHS = (ROOT / "OUTBOX" / "market_scans" / "live_candidates.json",
              ROOT / "OUTBOX" / "market_scans" / "prop_candidates.json")
EASTERN = ZoneInfo("America/New_York")
SOURCE_OLLAMA, SOURCE_TEMPLATE, SOURCE_EDITED = "ollama", "template", "edited"
MAX_FACTS = 3                      # + closer = at most 4 sentences
MAX_CHARS = 700
OLLAMA_COOLDOWN_S = 60.0

STYLE_EXAMPLES = (
    "Hawks give up the 3rd most pts to his position and 6th most 3's to his position as well. "
    "This game is also the highest slated total at 245.5 pts (juiced to the over) which makes "
    "sense because the Bulls and Hawks are #1 and #2 in pace in the entire league. Matas should "
    "be the primary scorer and as long as he hits his shots he should easily soar over this "
    "line. @everyone",
    "@everyone Yes, this is the public side and yes its okay. The public can win sometimes too. "
    "In fact the public has been crushed so badly this season, they're almost due. Yes, you'd "
    "normally play on a team with a backup QB like the Ravens but NOT when he's a 3rd string "
    "emergency QB that was never expected to see the field. Not on a short week when he hasn't "
    "had time to learn the playbook. This Jaguars team is really really good.",
    "Tough spot for Arizona coming home after coming up short at Texas Tech, hangover/let down "
    "spot is in effect. And next week they have a date with undefeated Utah so that makes this "
    "a sandwich spot. Meanwhile TCU won a game they probably didn't deserve to win, usually we "
    "get a focused effort the following game after a win like that. @everyone",
)

# Words that give the math away. Checked on every generated blurb (Ollama or template).
BANNED = re.compile(
    r"algorithm|\bmodel(s|ed|ing)?\b|kelly|baker|mchale|shrink|probabilit|expected value|"
    r"\bEV\b|\bAI\b|simulat|%|percent", re.I)
DASHES = re.compile(r"\s*[—–]\s*")
NUMBER = re.compile(r"\d+(?:\.\d+)?")

STAT_WORDS = {"pts": "points", "reb": "rebounds", "ast": "assists", "fg3m": "threes",
              "pass_yds": "passing yards", "rush_yds": "rushing yards",
              "rec_yds": "receiving yards", "receptions": "catches"}
# market -> (team_prop_context column, higher value = more allowed?, what it measures)
DEFENSE = {
    "player_points": ("drtg", True, "defense in the league by points allowed per 100 possessions"),
    "player_rebounds": ("allowed_reb_100", True, "rebounds per 100 possessions"),
    "player_assists": ("allowed_ast_100", True, "assists per 100 possessions"),
    "player_threes": ("allowed_fg3m_100", True, "threes per 100 possessions"),
    "player_pass_yds": ("def_pass_ypa", True, "yards per pass attempt"),
    "player_reception_yds": ("def_pass_ypa", True, "yards per pass attempt"),
    "player_receptions": ("def_pass_ypa", True, "yards per pass attempt"),
    "player_rush_yds": ("def_rush_ypc", True, "yards per carry"),
}
BLOWOUT = {"NBA": 15, "NFL": 17}
CLOSE = 3


# ==========================================================================
# Context
# ==========================================================================
@dataclass
class Fact:
    tag: str
    text: str            # one finished sentence; {NAME} marks a player mention
    weight: int


@dataclass
class WriteupContext:
    pick: dict[str, Any]
    league: str
    kind: str                                   # game | prop
    side: str | None                            # Over / Under for props and totals
    team: str | None = None                     # canonical id of the side we back
    opp: str | None = None
    home: str | None = None
    away: str | None = None
    player: str | None = None
    facts: list[Fact] = field(default_factory=list)

    def tags(self) -> set[str]:
        return {f.tag for f in self.facts}

    def header(self) -> str:
        p = self.pick
        return (f"{post_selection({**p, 'league': self.league})} at "
                f"{int(p['offered_american']):+d} ({p.get('bookmaker') or 'book'})")


def ordinal(n: int) -> str:
    suffix = "th" if 10 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def _num(x: float) -> str:
    return f"{x:g}" if float(x).is_integer() else f"{x:.1f}"


def _nick(cid: str | None) -> str:
    if not cid or cid not in REGISTRY:
        return ""
    t = REGISTRY[cid]
    return team_nickname(t.full_name, t.sport)


def _cid(name: str | None, league: str) -> str | None:
    if not name or league not in ("NBA", "NFL"):
        return None
    return try_resolve_team(name, league)


def _analysis(p: dict[str, Any]) -> dict[str, Any]:
    try:
        a = json.loads(p.get("analysis") or "{}")
    except (TypeError, ValueError):
        return {}
    return a if isinstance(a, dict) else {}


def _rank(conn: sqlite3.Connection, table: str, league: str, col: str, cid: str | None,
          highest_first: bool = True) -> tuple[int, int] | None:
    """(rank, teams) of ``cid`` on ``col`` within the league; rank 1 = highest value."""
    if not cid:
        return None
    rows = conn.execute(f"SELECT canonical_id, {col} AS v FROM {table} WHERE sport = ? "
                        f"AND {col} IS NOT NULL", (league,)).fetchall()
    order = sorted(rows, key=lambda r: r["v"], reverse=highest_first)
    ids = [r["canonical_id"] for r in order]
    return (ids.index(cid) + 1, len(ids)) if cid in ids else None


def _rating_col(league: str) -> str:
    return "adj_net_rtg" if league == "NBA" else "srs"


def _rating_name(league: str) -> str:
    return "net rating" if league == "NBA" else "SRS"


def _et_date(iso: str):
    return db.parse_ts(iso).astimezone(EASTERN).date()


@dataclass
class Spot:
    prev: sqlite3.Row | None
    next: sqlite3.Row | None
    days_since_prev: int | None              # calendar days (ET) since the previous game


def _spot(conn: sqlite3.Connection, cid: str | None, commence: str | None) -> Spot | None:
    if not cid or not commence:
        return None
    t = db.parse_ts(commence)
    lo, hi = db.iso(t - timedelta(hours=6)), db.iso(t + timedelta(hours=6))
    prev = conn.execute("SELECT * FROM team_schedule WHERE canonical_id = ? AND completed = 1 "
                        "AND game_time < ? AND points_for IS NOT NULL ORDER BY game_time DESC "
                        "LIMIT 1", (cid, lo)).fetchone()
    nxt = conn.execute("SELECT * FROM team_schedule WHERE canonical_id = ? AND game_time > ? "
                       "ORDER BY game_time LIMIT 1", (cid, hi)).fetchone()
    days = None if prev is None else (_et_date(commence) - _et_date(prev["game_time"])).days
    return Spot(prev, nxt, days)


def _key_absences(conn: sqlite3.Connection, cid: str | None, league: str,
                  statuses: tuple[str, ...] = ("out", "doubtful")) -> list[sqlite3.Row]:
    if not cid:
        return []
    marks = ",".join("?" * len(statuses))
    rows = conn.execute(
        f"SELECT player, position, status FROM injury_reports WHERE canonical_id = ? "
        f"AND is_key = 1 AND lower(status) IN ({marks}) "
        f"AND id IN (SELECT MAX(id) FROM injury_reports WHERE canonical_id = ? GROUP BY player)",
        (cid, *statuses, cid)).fetchall()
    qb_first = league == "NFL"
    return sorted(rows, key=lambda r: (not (qb_first and (r["position"] or "") == "QB"),
                                       r["player"]))


def _latest_total(conn: sqlite3.Connection, event_id: str) -> tuple[float, int, int] | None:
    """(point, over odds, under odds) from the newest totals snapshot, sharp book first."""
    for src in ("pinnacle", "draftkings", "fanduel", "betmgm", None):
        cond, params = ("AND source = ?", (src,)) if src else ("", ())
        row = conn.execute(f"SELECT source, timestamp FROM market_snapshots WHERE event_id = ? "
                           f"AND market_type = 'totals' {cond} ORDER BY timestamp DESC LIMIT 1",
                           (event_id, *params)).fetchone()
        if row is None:
            continue
        pair = {r["selection"]: r for r in conn.execute(
            "SELECT selection, point, american_odds FROM market_snapshots WHERE event_id = ? "
            "AND market_type = 'totals' AND source = ? AND timestamp = ?",
            (event_id, row["source"], row["timestamp"]))}
        if "Over" in pair and "Under" in pair and pair["Over"]["point"] is not None:
            return (float(pair["Over"]["point"]), int(pair["Over"]["american_odds"]),
                    int(pair["Under"]["american_odds"]))
    return None


def _sharp_points(conn: sqlite3.Connection, p: dict[str, Any]) -> tuple[float, float] | None:
    """(opening, latest) sharp point for the pick's own selection (spreads / totals)."""
    rows = conn.execute(
        "SELECT point FROM market_snapshots WHERE event_id = ? AND market_type = ? "
        "AND selection = ? AND source = 'pinnacle' AND point IS NOT NULL ORDER BY timestamp, id",
        (p["event_id"], p["market_type"], p["selection"])).fetchall()
    if len(rows) < 2:
        return None
    return float(rows[0]["point"]), float(rows[-1]["point"])


def _scan_entry(pick_id: str, paths: tuple[Path, ...]) -> dict[str, Any]:
    for path in paths:
        try:
            data = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        for item in (data.get("opportunities") or []) + (data.get("props") or []):
            if isinstance(item, dict) and item.get("pick_id") == pick_id:
                return item
    return {}


def _recent_form(conn: sqlite3.Connection, league: str, player: str, stat: str,
                 limit: int = 10) -> list[float]:
    from src.data.prop_stats_fetcher import player_key
    rows = conn.execute(
        f"SELECT {stat} AS v FROM player_game_logs WHERE sport = ? AND player_key = ? "
        f"AND {stat} IS NOT NULL AND (minutes IS NULL OR minutes > 0) "
        f"ORDER BY game_date DESC LIMIT ?", (league, player_key(player), limit)).fetchall()
    return [float(r["v"]) for r in rows]


# ---- fact builders --------------------------------------------------------
def _situational_facts(conn: sqlite3.Connection, ctx: WriteupContext) -> list[Fact]:
    """Rest, short week, hangover / letdown / bounce-back, lookahead and sandwich spots."""
    p, lg, out = ctx.pick, ctx.league, []
    if not ctx.team or not ctx.opp:
        return out
    us, them = _spot(conn, ctx.team, p.get("commence_time")), _spot(conn, ctx.opp,
                                                                    p.get("commence_time"))
    T, O = _nick(ctx.team), _nick(ctx.opp)
    if us and them and them.days_since_prev is not None:
        if lg == "NBA" and them.days_since_prev == 1 and (us.days_since_prev or 9) > 1:
            out.append(Fact("opp_b2b", f"The {O} are on the second night of a back-to-back "
                                       f"while the {T} come in rested.", 80))
        if lg == "NFL" and them.days_since_prev <= 4 and (us.days_since_prev or 9) > 4:
            out.append(Fact("short_week", f"The {O} are on a short week, just "
                                          f"{them.days_since_prev} days after playing the "
                                          f"{_nick(them.prev['opponent_id'])}.", 80))
    if them and them.prev is not None:
        pf, pa = int(them.prev["points_for"]), int(them.prev["points_against"])
        foe = _nick(them.prev["opponent_id"])
        if -CLOSE <= pf - pa < 0:
            out.append(Fact("hangover", f"The {O} are coming off a {pf}-{pa} heartbreaker "
                                        f"against the {foe}, hangover spot is in effect.", 70))
        elif pf - pa >= BLOWOUT[lg]:
            out.append(Fact("letdown", f"The {O} are coming off a {pf}-{pa} beatdown of the "
                                       f"{foe}, that's a classic letdown spot.", 55))
    if us and us.prev is not None:
        pf, pa = int(us.prev["points_for"]), int(us.prev["points_against"])
        if pa - pf >= BLOWOUT[lg]:
            out.append(Fact("bounce_back", f"The {T} got embarrassed {pf}-{pa} by the "
                                           f"{_nick(us.prev['opponent_id'])} last time out, "
                                           f"expect a focused effort here.", 55))
    if them and them.next is not None:
        col = _rating_col(lg)
        nxt = _rank(conn, "team_ratings", lg, col, them.next["opponent_id"])
        prv = (_rank(conn, "team_ratings", lg, col, them.prev["opponent_id"])
               if them.prev is not None else None)
        if nxt and nxt[0] <= 5:
            n_name = _nick(them.next["opponent_id"])
            if prv and prv[0] <= 5:
                out.append(Fact("sandwich", f"The {O} just played the "
                                            f"{_nick(them.prev['opponent_id'])} and have the "
                                            f"{n_name} up next, two top 5 teams, so this is a "
                                            f"sandwich spot.", 65))
            else:
                out.append(Fact("lookahead", f"The {O} have the {n_name} up next, "
                                             f"{ordinal(nxt[0])} in {_rating_name(lg)}, "
                                             f"so this is a lookahead spot.", 60))
    return out


def _injury_facts(conn: sqlite3.Connection, ctx: WriteupContext) -> list[Fact]:
    out: list[Fact] = []
    if ctx.kind == "game" and ctx.opp:
        rows = _key_absences(conn, ctx.opp, ctx.league)[:2]
        if rows:
            O = _nick(ctx.opp)
            if ctx.league == "NFL" and (rows[0]["position"] or "") == "QB":
                out.append(Fact("opp_qb_out", f"The {O} will be without their QB "
                                              f"{rows[0]['player']}.", 90))
            else:
                names = " and ".join(r["player"] for r in rows)
                out.append(Fact("opp_injuries", f"The {O} are down {names}.", 75))
    if ctx.kind == "prop" and ctx.side == "Over":
        reason = (_analysis(ctx.pick).get("details") or {}).get("usage_boost_reason") or ""
        m = re.search(r"key teammate (.+?) newly Out", reason)
        if m:
            out.append(Fact("usage", f"With {m.group(1)} ruled out, there are more touches "
                                     f"to go around for {{NAME}}.", 75))
    return out


def _pace_facts(conn: sqlite3.Connection, ctx: WriteupContext) -> list[Fact]:
    if ctx.side not in ("Over", "Under"):
        return []
    a, b = (ctx.team, ctx.opp) if ctx.kind == "prop" else (ctx.away, ctx.home)
    table, col = (("team_ratings", "pace") if ctx.league == "NBA"
                  else ("team_prop_context", "plays_pg"))
    ra, rb = _rank(conn, table, ctx.league, col, a), _rank(conn, table, ctx.league, col, b)
    if not ra or not rb:
        return []
    word = "pace" if ctx.league == "NBA" else "plays per game"
    lo, hi = sorted((ra[0], rb[0]))
    if ctx.side == "Over" and (hi <= 12 or lo <= 5):
        return [Fact("pace", f"The {_nick(a)} and {_nick(b)} are #{ra[0]} and #{rb[0]} in "
                             f"{word}, this game should have plenty of possessions.", 55)]
    n = ra[1]
    if ctx.side == "Under" and lo >= n - 11:
        return [Fact("pace", f"Both of these teams grind it out, the {_nick(a)} are #{ra[0]} "
                             f"and the {_nick(b)} are #{rb[0]} in {word}.", 55)]
    return []


def _total_fact(conn: sqlite3.Connection, ctx: WriteupContext) -> list[Fact]:
    if ctx.side not in ("Over", "Under"):
        return []
    tot = _latest_total(conn, ctx.pick["event_id"])
    if tot is None:
        return []
    pt, over, under = tot
    if ctx.side == "Over" and american_to_decimal(over) < american_to_decimal(under):
        return [Fact("total", f"The game total is {_num(pt)} and it's juiced to the over.", 40)]
    if ctx.side == "Under" and american_to_decimal(under) < american_to_decimal(over):
        return [Fact("total", f"The game total is {_num(pt)} and it's juiced to the under.", 40)]
    return []


def _prop_facts(conn: sqlite3.Connection, ctx: WriteupContext) -> list[Fact]:
    p, out = ctx.pick, []
    mt, line = p["market_type"], p.get("line_point")
    if line is None or not ctx.player:
        return out
    line = float(line)
    over = ctx.side == "Over"
    spec = DEFENSE.get(mt)
    if spec and ctx.opp:
        col, high_allows, what = spec
        r = _rank(conn, "team_prop_context", ctx.league, col, ctx.opp, highest_first=high_allows)
        O = _nick(ctx.opp)
        if r:
            rank_most, n = r
            rank_fewest = n - rank_most + 1
            if mt == "player_points":
                if over and rank_most <= 10:
                    out.append(Fact("defense", f"The {O} have the {ordinal(rank_most)} worst "
                                               f"{what}.", 70))
                elif not over and rank_fewest <= 10:
                    out.append(Fact("defense", f"The {O} have the {ordinal(rank_fewest)} best "
                                               f"{what}.", 70))
            elif over and rank_most <= 10:
                out.append(Fact("defense", f"The {O} give up the {ordinal(rank_most)} most "
                                           f"{what} in the league.", 70))
            elif not over and rank_fewest <= 10:
                out.append(Fact("defense", f"The {O} allow the {ordinal(rank_fewest)} fewest "
                                           f"{what} in the league.", 70))
    stat = PROP_MARKETS[mt].stat
    games = _recent_form(conn, ctx.league, ctx.player, stat)
    if len(games) >= 4:
        hits = sum((g > line) if over else (g < line) for g in games)
        if hits >= math.ceil(len(games) / 2):
            avg = sum(games) / len(games)
            span = "games" if ctx.league == "NBA" else "weeks"
            out.append(Fact("form", f"{{NAME}} is averaging {avg:.1f} {STAT_WORDS[stat]} over "
                                    f"the last {len(games)} {span} and has gone "
                                    f"{'over' if over else 'under'} {_num(line)} in {hits} of "
                                    f"them.", 65))
    a = _analysis(p)
    proj = a.get("projection")
    if proj is not None and ((over and proj > line) or (not over and proj < line)):
        word = "clear of" if over else "under"
        out.append(Fact("projection", f"I have {{NAME}} projected for {float(proj):.1f}, "
                                      f"well {word} this {_num(line)} line.", 60))
    minutes = (a.get("details") or {}).get("minutes")
    if over and ctx.league == "NBA" and minutes and minutes >= 28:
        out.append(Fact("minutes", f"{{NAME}} is playing {round(float(minutes))} minutes a "
                                   f"night, the volume is there.", 30))
    return out


def _game_facts(conn: sqlite3.Connection, ctx: WriteupContext) -> list[Fact]:
    p, lg, out = ctx.pick, ctx.league, []
    mt = p["market_type"]
    if ctx.team and ctx.opp:
        col = _rating_col(lg)
        rt, ro = (_rank(conn, "team_ratings", lg, col, ctx.team),
                  _rank(conn, "team_ratings", lg, col, ctx.opp))
        if rt and ro and ro[0] - rt[0] >= 6:
            out.append(Fact("ratings", f"The {_nick(ctx.team)} are {ordinal(rt[0])} in "
                                       f"{_rating_name(lg)} and the {_nick(ctx.opp)} are "
                                       f"{ordinal(ro[0])}.", 50))
    a = _analysis(p)
    margin, line = a.get("selection_margin"), p.get("line_point")
    if mt == "spreads" and margin is not None and line is not None:
        m, ln, T = float(margin), float(line), _nick(ctx.team)
        if ln < 0 and m > -ln:
            out.append(Fact("projection", f"I have the {T} winning this by {m:.1f} and we're "
                                          f"only laying {_num(-ln)}.", 45))
        elif ln > 0 and m >= 0:
            out.append(Fact("projection", f"I have the {T} winning this outright and we're "
                                          f"getting {_num(ln)} points.", 45))
        elif ln > 0 and -m < ln:
            out.append(Fact("projection", f"I only have the {T} losing by {abs(m):.1f} and "
                                          f"we're getting {_num(ln)}.", 45))
    off, bench = p.get("offered_american"), p.get("benchmark_american")
    if mt in ("h2h", "kalshi_binary") and off is not None and bench is not None and \
            american_to_decimal(int(off)) > american_to_decimal(int(bench)):
        src = p.get("benchmark_source") or "The sharp book"
        out.append(Fact("price", f"{src} has this at {int(bench):+d} and we're getting "
                                 f"{int(off):+d} at {p.get('bookmaker') or 'the book'}, "
                                 f"that's a better number than the sharp market.", 35))
    if mt in ("spreads", "totals") and line is not None:
        moved = _sharp_points(conn, p)
        if moved:
            first, last = moved
            ours = float(line)
            # spreads / unders: the sharp number dropped and ours is still the better side of it;
            # overs: the sharp total rose and ours is still below it
            if mt == "spreads" or ctx.side == "Under":
                toward = last < first and ours > last
            else:
                toward = last > first and ours < last
            if toward:
                fmt = (lambda v: f"{v:+g}") if mt == "spreads" else _num
                out.append(Fact("line_move", f"This opened at {fmt(first)} and the sharp books "
                                             f"have moved it to {fmt(last)}, we're still "
                                             f"getting {fmt(ours)} at "
                                             f"{p.get('bookmaker') or 'the book'}.", 50))
    kalshi = p.get("kalshi_mid_prob")
    entry = ctx.pick.get("_scan") or {}
    if kalshi is None and entry.get("exchange_probability") is not None:
        kalshi = entry["exchange_probability"]
    on_kalshi = (p.get("bookmaker") or "").lower() == "kalshi"
    if mt in ("h2h", "kalshi_binary") and kalshi and ctx.team and not on_kalshi and \
            0.05 <= float(kalshi) <= 0.95:
        out.append(Fact("exchange", f"Kalshi has the {_nick(ctx.team)} trading at "
                                    f"{round(float(kalshi) * 100)} cents.", 30))
    return out


def gather_context(conn: sqlite3.Connection, pick: dict[str, Any],
                   scan_paths: tuple[Path, ...] = SCAN_PATHS) -> WriteupContext:
    """Collect grounded facts for one pick. Missing data just means fewer facts."""
    p = dict(pick)
    league = SPORT_TO_LEAGUE.get(p.get("sport") or "") or \
        (PROP_MARKETS[p["market_type"]].league if is_prop_market(p.get("market_type")) else "")
    prop = is_prop_market(p.get("market_type"))
    sel = p.get("selection") or ""
    side = (sel.rpartition(" ")[2] if prop else
            sel if p.get("market_type") == "totals" else None)
    away_name, _, home_name = (p.get("event_name") or "").partition(" @ ")
    ctx = WriteupContext(p, league, "prop" if prop else "game",
                         side if side in ("Over", "Under") else None,
                         away=_cid(away_name, league), home=_cid(home_name, league))
    p["_scan"] = _scan_entry(p["pick_id"], scan_paths)
    if prop:
        a = _analysis(p)
        ctx.player = sel.rpartition(" ")[0] or None
        ctx.team, ctx.opp = a.get("team_id"), a.get("opponent_id")
    elif p.get("market_type") != "totals":
        ctx.team = _cid(sel, league)
        ctx.opp = ctx.home if ctx.team == ctx.away else ctx.away if ctx.team == ctx.home else None
    builders: list[Callable[[sqlite3.Connection, WriteupContext], list[Fact]]] = [
        _injury_facts, _pace_facts, _total_fact,
        *([_prop_facts] if prop else [_situational_facts, _game_facts])]
    for build in builders:
        try:
            ctx.facts += build(conn, ctx)
        except (sqlite3.Error, KeyError, TypeError, ValueError) as exc:
            log.warning("writeup facts %s failed for %s: %s", build.__name__, p["pick_id"], exc)
    ctx.facts.sort(key=lambda f: -f.weight)
    return ctx


# ==========================================================================
# Template (fallback) voice
# ==========================================================================
def _choice(pick_id: str, options: list[str], salt: str = "") -> str:
    h = int(hashlib.sha1((pick_id + salt).encode()).hexdigest(), 16)
    return options[h % len(options)]


def _closer(ctx: WriteupContext) -> str:
    p, mt = ctx.pick, ctx.pick.get("market_type")
    if ctx.kind == "prop":
        last = (ctx.player or "").split(" ")[-1] or "This one"
        opts = ([f"{last} should clear this number.", f"Give me the over on {last}.",
                 f"{last} goes over this one."] if ctx.side == "Over" else
                [f"{last} stays under this number.", f"Give me the under on {last}.",
                 f"{last} comes in under this one."])
    elif mt == "totals":
        opts = (["Give me the over.", "This one goes over.", "Points are coming in this one."]
                if ctx.side == "Over" else
                ["Give me the under.", "This one stays under.", "Expect a grind."])
    else:
        T = _nick(ctx.team) or (p.get("selection") or "this side")
        line = p.get("line_point")
        if mt == "spreads" and line is not None and float(line) < 0:
            opts = [f"Lay it with the {T}.", f"The {T} roll here.", f"Give me the {T}."]
        elif mt == "spreads":
            opts = [f"Take the points with the {T}.", f"The {T} keep this close.",
                    f"Give me the {T} and the points."]
        else:
            opts = [f"The {T} get it done.", f"Riding the {T} here.", f"Give me the {T}."]
    return _choice(p["pick_id"], opts, "closer")


def _name_player(sentences: list[str], player: str | None) -> list[str]:
    """First mention gets the full name, later ones the last name (no pronouns)."""
    full = player or "This player"
    last = full.split(" ")[-1]
    out, named = [], False
    for s in sentences:
        if "{NAME}" in s:
            s = s.replace("{NAME}", last if named else full)
            named = True
        out.append(s)
    return out


def fallback_writeup(ctx: WriteupContext) -> str:
    """Deterministic 2-5 sentence blurb from the strongest facts, ending with @everyone."""
    picked = ctx.facts[:MAX_FACTS]
    sentences = [f.text for f in picked]
    if not sentences:
        p = ctx.pick
        sentences = [f"{p.get('bookmaker') or 'The book'} is hanging "
                     f"{int(p['offered_american']):+d} on this and that's a number we like."]
    sentences = _name_player(sentences, ctx.player)
    return clean(" ".join(sentences + [_closer(ctx)]))


# ==========================================================================
# Ollama
# ==========================================================================
_llm_lock = threading.Lock()
_llm_down_until = 0.0


def _settings() -> tuple[bool, str, str, float]:
    enabled = os.environ.get("BREEZE_WRITEUP_LLM", "on").strip().lower() not in ("0", "off",
                                                                                "false", "no")
    url = os.environ.get("BREEZE_OLLAMA_URL", "http://localhost:11434").rstrip("/")
    model = os.environ.get("BREEZE_OLLAMA_MODEL", "qwen2.5-coder:7b")
    try:
        timeout = float(os.environ.get("BREEZE_OLLAMA_TIMEOUT", "4"))
    except ValueError:
        timeout = 4.0
    return enabled, url, model, timeout


def reset_ollama_cooldown() -> None:
    global _llm_down_until
    with _llm_lock:
        _llm_down_until = 0.0


def _post_json(url: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
    req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"),
                                 headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def ollama_generate(prompt: str) -> str | None:
    """Raw Ollama completion, or None when disabled, cooling down, offline or timed out."""
    global _llm_down_until
    enabled, url, model, timeout = _settings()
    if not enabled:
        return None
    with _llm_lock:
        if time.monotonic() < _llm_down_until:
            return None
    try:
        out = _post_json(f"{url}/api/generate",
                         {"model": model, "prompt": prompt, "stream": False,
                          "options": {"temperature": 0.7, "num_predict": 220}}, timeout)
        text = out.get("response")
        return text if isinstance(text, str) else None
    except (urllib.error.URLError, TimeoutError, OSError, ValueError) as exc:
        log.info("Ollama unavailable (%s); using the template write-up", exc)
        with _llm_lock:
            _llm_down_until = time.monotonic() + OLLAMA_COOLDOWN_S
        return None


def build_prompt(ctx: WriteupContext) -> str:
    facts = _name_player([f.text for f in ctx.facts], ctx.player) or ["(no extra facts)"]
    examples = "\n".join(f"Example {i}: {e}" for i, e in enumerate(STYLE_EXAMPLES, 1))
    return (
        "You write short Discord posts for a sports betting group in the voice of a sharp, "
        "confident capper. Write 2 to 5 sentences about the pick below.\n"
        "Rules:\n"
        "- Use ONLY the facts listed. Every number you write must appear in the facts. Do not "
        "invent stats, records, injuries, streaks or storylines.\n"
        "- Never mention algorithms, models, Kelly, probabilities or percentages.\n"
        "- No em dashes. Short, punchy, conversational sentences.\n"
        "- Refer to players by name, not pronouns.\n"
        "- End with @everyone.\n\n"
        f"Voice examples (style only, their facts do not apply):\n{examples}\n\n"
        f"PICK: {ctx.header()}\n"
        "FACTS:\n" + "\n".join(f"- {f}" for f in facts) + "\n\n"
        "Write the post now. Output only the post text."
    )


# ==========================================================================
# Cleaning and validation
# ==========================================================================
def clean(text: str) -> str:
    """No em/en dashes, single spaces, exactly one trailing @everyone."""
    t = text.strip().strip('"').strip("`").strip()
    t = re.sub(r"^(post|discord post)\s*:\s*", "", t, flags=re.I)
    t = DASHES.sub(", ", t)
    t = re.sub(r"\s*@everyone\s*", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    t = re.sub(r"\s+([,.!?])", r"\1", t).replace(",,", ",")
    if t and t[-1] not in ".!?":
        t += "."
    return f"{t} @everyone"


def _norm_num(s: str) -> str:
    return f"{float(s):g}"


def allowed_numbers(ctx: WriteupContext) -> set[str]:
    p = ctx.pick
    blob = " ".join(f.text for f in ctx.facts) + " " + ctx.header() + " " + \
        f"{p.get('line_point') or ''} {float(p.get('kelly_units') or 0):.2f}"
    return {_norm_num(n) for n in NUMBER.findall(blob)}


def sentence_count(text: str) -> int:
    body = text.replace("@everyone", " ").strip()
    return len([s for s in re.split(r"(?<=[.!?])\s+", body) if re.search(r"\w", s)])


def validate(text: str, ctx: WriteupContext) -> tuple[str | None, str]:
    """(cleaned text, "") if the write-up passes every rule, else (None, reason)."""
    if not text or not text.strip():
        return None, "empty"
    t = clean(text)
    if BANNED.search(t):
        return None, f"mentions the math ({BANNED.search(t).group(0)!r})"   # type: ignore[union-attr]
    stray = {_norm_num(n) for n in NUMBER.findall(t)} - allowed_numbers(ctx)
    if stray:
        return None, f"numbers not in the facts: {sorted(stray)}"
    n = sentence_count(t)
    if not 2 <= n <= 5:
        return None, f"{n} sentences"
    if len(t) > MAX_CHARS:
        return None, "too long"
    return t, ""


# ==========================================================================
# Public entry points
# ==========================================================================
Llm = Callable[[str], "str | None"]


def generate_writeup(conn: sqlite3.Connection, pick: dict[str, Any], llm: Llm | None = None,
                     scan_paths: tuple[Path, ...] = SCAN_PATHS) -> tuple[str, str]:
    """(write-up, source). Ollama when it answers with a valid post, else the template."""
    ctx = gather_context(conn, pick, scan_paths)
    raw = (llm or ollama_generate)(build_prompt(ctx))
    if raw:
        text, why = validate(raw, ctx)
        if text:
            return text, SOURCE_OLLAMA
        log.info("Ollama write-up for %s rejected (%s); using the template", pick["pick_id"], why)
    return fallback_writeup(ctx), SOURCE_TEMPLATE


def attach_writeup(conn: sqlite3.Connection, pick_id: str, llm: Llm | None = None,
                   scan_paths: tuple[Path, ...] = SCAN_PATHS) -> dict[str, Any]:
    """Generate and store the write-up for an APPROVED, signed-off pick (never anything else)."""
    p = db.get_pick(conn, pick_id)
    if p is None:
        raise KeyError(f"unknown pick {pick_id}")
    if p["hitl_status"] != HITL_APPROVED or not p["signoff_at"]:
        raise ValueError(f"{pick_id} is {p['hitl_status']} without a Captain HZ sign-off; "
                         f"write-ups are for approved picks only")
    text, source = generate_writeup(conn, p, llm, scan_paths)
    db.update_pick(conn, pick_id, discord_writeup=text, writeup_source=source)
    return db.get_pick(conn, pick_id)  # type: ignore[return-value]


def save_custom_writeup(conn: sqlite3.Connection, pick_id: str, text: str) -> dict[str, Any]:
    """Store Captain HZ's own edit (kept as written, apart from trimming)."""
    p = db.get_pick(conn, pick_id)
    if p is None:
        raise KeyError(f"unknown pick {pick_id}")
    if p["hitl_status"] != HITL_APPROVED or not p["signoff_at"]:
        raise ValueError(f"{pick_id} is not an approved pick; nothing to edit")
    body = (text or "").strip()
    if not body:
        raise ValueError("write-up text is empty")
    if len(body) > 2000:
        raise ValueError("write-up is longer than 2000 characters")
    db.update_pick(conn, pick_id, discord_writeup=body, writeup_source=SOURCE_EDITED)
    return db.get_pick(conn, pick_id)  # type: ignore[return-value]
