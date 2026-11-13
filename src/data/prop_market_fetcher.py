"""Player-prop mispricing scanner: model projection vs de-vigged market.

``run_prop_scan(conn, sports, mock)``:

    slate     live: GET /v4/sports/{sport}/events (no quota cost), keep games
              on TODAY's US-Eastern slate that start within PROP_LOOKAHEAD_HOURS
              (6h) and have not started, cap at PROP_MAX_EVENTS. Mock: the
              synthetic slate, no date/lookahead gate.
    odds      live: GET /v4/sports/{sport}/events/{id}/odds?markets=<props>,
              cached 15 min per event; stops early when the last seen
              x-requests-remaining is below PROP_QUOTA_FLOOR.
    market    per (player, market, line): Pinnacle's pair power de-vigged, else
              the power de-vig consensus of the OTHER books quoting both sides.
    model     src.core.prop_model projection + distribution -> P(side | action)
    gate      market EV > 0 AND z >= 0.65 AND shrunk EV >= 4% (PROP_DIVERGENCE_MODE
              all; "any" = sigma OR shrunk EV), -130 floor, player not Out/Doubtful
    absences  NBA: +4% volume per key teammate newly Out (max +8%)
    sizing    flat official 1.00u per qualified prop (MIN_PROP_OFFICIAL_UNITS =
              MAX_PROP_UNITS = 1.0, the Discord standard play); raw Kelly from
              risk.size_bet_under_uncertainty (min of quarter-Kelly and
              Baker-McHale) is kept as diagnostic metadata only. Props draw on
              their own 2.0u per-event bucket, independent of the game-line
              bucket, under a 4.0u combined single-game ceiling.
    output    ledger_picks (always PENDING_CAPTAIN_HZ), market_snapshots,
              OUTBOX/market_scans/prop_candidates.json

Prop picks are stored with ``selection = "<Player> Over|Under"`` and
``line_point = <line>``; the settler grades them from box scores.
"""

from __future__ import annotations

import json
import logging
import math
import random
import sqlite3
import sys
from dataclasses import dataclass, field, fields
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Protocol

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.core import database as db  # noqa: E402
from src.core.config import ODDS_API_BASE, bankroll_units, get_setting, load_env  # noqa: E402
from src.core.engine import (  # noqa: E402
    HITL_PENDING,
    HITL_REJECTED,
    american_to_decimal,
    decimal_to_american,
    devig_power,
    validate_odds_floor,
)
from src.core.ledger import (  # noqa: E402
    BUCKET_PROP,
    MAX_EVENT_UNITS,
    MAX_PROP_EVENT_UNITS,
    Ledger,
)
from src.core.prop_model import (  # noqa: E402
    DEFAULT_DIVERGENCE_MODE,
    DIST_GAMMA,
    MARKETS_BY_LEAGUE,
    MAX_PROP_UNITS,
    MIN_PROP_OFFICIAL_UNITS,
    PACE_HARMONIC,
    PROP_MARKETS,
    PROP_MIN_SHRUNK_EV,
    PROP_MODEL_WEIGHT,
    PROP_SIGMA_THRESHOLD,
    Projection,
    absence_boost,
    apply_usage_boost,
    evaluate_side,
    official_prop_units,
    project_nba,
    project_nfl,
)
from src.core.risk import size_bet_under_uncertainty  # noqa: E402
from src.data.http_client import HttpError, get_json  # noqa: E402
from src.data.market_fetcher import (  # noqa: E402
    DEFAULT_BOOKMAKERS,
    RETAIL_BOOKS,
    SHARP_BOOKS,
    SHARP_PRIORITY,
    build_mock_feeds,
    make_pick_id,
)
from src.data.prop_stats_fetcher import (  # noqa: E402
    PlayerHistory,
    ensure_fresh,
    key_teammate_injuries,
    league_context,
    load_player_history,
    load_team_context,
    player_injury,
    player_key,
    teammate_recently_active,
)
from src.data.team_registry import try_resolve_team  # noqa: E402
from src.data.teams import SPORT_TO_LEAGUE  # noqa: E402

log = logging.getLogger("breeze.props")

PROP_OUTPUT_PATH = ROOT / "OUTBOX" / "market_scans" / "prop_candidates.json"
PROP_CACHE_TTL_S = 900
EVENTS_CACHE_TTL_S = 300
QUOTA_CACHE_KEY = "odds:quota_remaining"
DEFAULT_QUOTA_FLOOR = 40           # free tier starts at 500 credits / month
DEFAULT_MAX_EVENTS = 12
DEFAULT_LOOKAHEAD_HOURS = 6.0
OUT_STATUSES = frozenset({"out", "doubtful", "injured reserve", "ir", "suspended"})
QUESTIONABLE = frozenset({"questionable", "day-to-day", "game-time decision", "gtd"})
PROP_HITL_REASON = "Prop model: Captain HZ sign-off required until prop CLV history exists"
SIDES = ("Over", "Under")


# ==========================================================================
# Configuration
# ==========================================================================
@dataclass(frozen=True)
class PropConfig:
    sigma_threshold: float = PROP_SIGMA_THRESHOLD
    min_shrunk_ev: float = PROP_MIN_SHRUNK_EV
    divergence_mode: str = DEFAULT_DIVERGENCE_MODE
    pace_method: str = PACE_HARMONIC
    continuous_dist: str = DIST_GAMMA
    quota_floor: int = DEFAULT_QUOTA_FLOOR
    max_events: int = DEFAULT_MAX_EVENTS
    model_weight: float = PROP_MODEL_WEIGHT
    lookahead_hours: float = DEFAULT_LOOKAHEAD_HOURS

    @classmethod
    def from_env(cls) -> "PropConfig":
        """PROP_SIGMA_THRESHOLD, PROP_MIN_SHRUNK_EV, PROP_DIVERGENCE_MODE (any|all),
        PROP_PACE_METHOD (harmonic|multiplicative), PROP_CONTINUOUS_DIST
        (gamma|skew_normal), PROP_QUOTA_FLOOR, PROP_MAX_EVENTS, PROP_MODEL_WEIGHT,
        PROP_LOOKAHEAD_HOURS."""
        d = cls()
        num = lambda name, dflt, typ: typ(get_setting(name) or dflt)  # noqa: E731
        return cls(num("PROP_SIGMA_THRESHOLD", d.sigma_threshold, float),
                   num("PROP_MIN_SHRUNK_EV", d.min_shrunk_ev, float),
                   get_setting("PROP_DIVERGENCE_MODE") or d.divergence_mode,
                   get_setting("PROP_PACE_METHOD") or d.pace_method,
                   get_setting("PROP_CONTINUOUS_DIST") or d.continuous_dist,
                   num("PROP_QUOTA_FLOOR", d.quota_floor, int),
                   num("PROP_MAX_EVENTS", d.max_events, int),
                   num("PROP_MODEL_WEIGHT", d.model_weight, float),
                   num("PROP_LOOKAHEAD_HOURS", d.lookahead_hours, float))


@dataclass
class PropScanStats:
    events_listed: int = 0
    events_on_slate: int = 0
    events_queried: int = 0
    events_from_cache: int = 0
    prop_quotes: int = 0
    player_lines: int = 0
    players_modeled: int = 0
    no_model: int = 0
    no_benchmark: int = 0
    floor_rejected: int = 0
    negative_market_ev: int = 0
    no_divergence: int = 0
    injury_out: int = 0
    event_cap_rejected: int = 0
    candidates: int = 0
    pending_hitl: int = 0
    injury_flagged: int = 0
    ledger_inserted: int = 0
    snapshots: int = 0
    fetch_errors: int = 0
    quota_guard_tripped: bool = False
    slate_gate: str = "today_et"
    quota: dict[str, str | None] = field(default_factory=dict)


# ==========================================================================
# US-Eastern slate date (no tzdata dependency)
# ==========================================================================
def _nth_sunday(year: int, month: int, n: int) -> date:
    d = date(year, month, 1)
    d += timedelta(days=(6 - d.weekday()) % 7)
    return d + timedelta(weeks=n - 1)


def eastern_offset_hours(t: datetime) -> int:
    """-4 during US daylight time (2nd Sun Mar 07:00Z .. 1st Sun Nov 06:00Z), else -5."""
    t = t.astimezone(timezone.utc)
    start = datetime.combine(_nth_sunday(t.year, 3, 2), datetime.min.time(), timezone.utc) \
        + timedelta(hours=7)
    end = datetime.combine(_nth_sunday(t.year, 11, 1), datetime.min.time(), timezone.utc) \
        + timedelta(hours=6)
    return -4 if start <= t < end else -5


def eastern_date(t: datetime) -> date:
    return (t.astimezone(timezone.utc) + timedelta(hours=eastern_offset_hours(t))).date()


def slate_events(events: list[dict[str, Any]], now: datetime, max_events: int,
                 gate: bool = True, lookahead_hours: float | None = DEFAULT_LOOKAHEAD_HOURS
                 ) -> list[dict[str, Any]]:
    """Not-started games (soonest first), capped at ``max_events``.

    With ``gate``: only today's ET slate, and only games starting within
    ``lookahead_hours`` (None = no lookahead limit).
    """
    today = eastern_date(now)
    horizon = None if lookahead_hours is None else now + timedelta(hours=lookahead_hours)
    out = []
    for ev in events:
        t = db.parse_ts(ev["commence_time"])
        if t <= now:
            continue
        if gate and (eastern_date(t) != today or (horizon is not None and t > horizon)):
            continue
        out.append(ev)
    return sorted(out, key=lambda e: e["commence_time"])[:max_events]


# ==========================================================================
# The-Odds-API
# ==========================================================================
def _quota_int(v: Any) -> int | None:
    try:
        return int(float(v))
    except (TypeError, ValueError):
        return None


def known_quota(conn: sqlite3.Connection) -> int | None:
    return _quota_int(db.cache_get(conn, QUOTA_CACHE_KEY))


def _record_quota(conn: sqlite3.Connection, remaining: Any) -> None:
    if _quota_int(remaining) is not None:
        db.cache_put(conn, QUOTA_CACHE_KEY, _quota_int(remaining), 24 * 3600)


def fetch_events(conn: sqlite3.Connection, sport: str, api_key: str) -> list[dict[str, Any]]:
    """GET /v4/sports/{sport}/events (documented as not counting against quota)."""
    key = f"events:{sport}"
    cached = db.cache_get(conn, key)
    if cached is not None:
        return cached
    events, hdrs = get_json(f"{ODDS_API_BASE}/sports/{sport}/events/",
                            {"apiKey": api_key, "dateFormat": "iso"})
    log.info("The-Odds-API %s events: %d | x-requests-last=%s (expected 0)", sport, len(events),
             hdrs.get("x-requests-last"))
    _record_quota(conn, hdrs.get("x-requests-remaining"))
    db.cache_put(conn, key, events, EVENTS_CACHE_TTL_S)
    return events


@dataclass
class PropFetch:
    event: dict[str, Any]
    from_cache: bool
    quota_remaining: str | None = None


def fetch_event_props(conn: sqlite3.Connection, sport: str, event_id: str, markets: str,
                      api_key: str, bookmakers: str) -> PropFetch:
    """GET /v4/sports/{sport}/events/{id}/odds for prop markets, cached 15 minutes."""
    key = f"props:{sport}:{event_id}:{markets}:{bookmakers}"
    cached = db.cache_get(conn, key)
    if cached is not None:
        return PropFetch(cached, True)
    event, hdrs = get_json(f"{ODDS_API_BASE}/sports/{sport}/events/{event_id}/odds/",
                           {"apiKey": api_key, "markets": markets, "oddsFormat": "american",
                            "dateFormat": "iso", "bookmakers": bookmakers})
    _record_quota(conn, hdrs.get("x-requests-remaining"))
    db.cache_put(conn, key, event, PROP_CACHE_TTL_S)
    log.info("The-Odds-API props %s %s | x-requests-remaining=%s x-requests-last=%s", sport,
             event_id, hdrs.get("x-requests-remaining"), hdrs.get("x-requests-last"))
    return PropFetch(event, False, hdrs.get("x-requests-remaining"))


@dataclass(frozen=True)
class PropQuote:
    book: str
    market: str
    player: str
    side: str
    point: float
    price: int


def parse_prop_quotes(event: dict[str, Any]) -> list[PropQuote]:
    """Flatten bookmakers -> markets -> outcomes (name Over/Under, description = player)."""
    out = []
    for bm in event.get("bookmakers", []) or []:
        for m in bm.get("markets", []) or []:
            if m.get("key") not in PROP_MARKETS:
                continue
            for o in m.get("outcomes", []) or []:
                side, player, point = o.get("name"), o.get("description"), o.get("point")
                if side not in SIDES or not player or point is None or o.get("price") is None:
                    continue
                out.append(PropQuote(bm["key"], m["key"], str(player), side, float(point),
                                     int(o["price"])))
    return out


def prop_snapshot_rows(event_id: str, quotes: list[PropQuote], ts: str) -> list[dict[str, Any]]:
    return [{"event_id": event_id, "timestamp": ts, "source": q.book, "market_type": q.market,
             "selection": f"{q.player} {q.side}", "point": q.point, "american_odds": q.price,
             "implied_prob": round(1.0 / american_to_decimal(q.price), 6)} for q in quotes]


# ==========================================================================
# Market benchmark
# ==========================================================================
def market_fair(pairs: dict[str, dict[str, int]], side: str, exclude_book: str
                ) -> tuple[float, str, int | None] | None:
    """(fair P(side), source label, Pinnacle price) for one player line.

    ``pairs`` maps book -> {"Over": price, "Under": price}. Pinnacle's two-way
    pair is preferred; otherwise the mean power de-vig of every OTHER book
    quoting both sides (the offering book never benchmarks itself).
    """
    other = "Under" if side == "Over" else "Over"
    for sharp in SHARP_PRIORITY:
        pr = pairs.get(sharp)
        if pr and side in pr and other in pr:
            p = devig_power(american_to_decimal(pr[side]), american_to_decimal(pr[other]))[0]
            return p, SHARP_BOOKS[sharp], pr[side]
    probs = [devig_power(american_to_decimal(pr[side]), american_to_decimal(pr[other]))[0]
             for book, pr in pairs.items()
             if book != exclude_book and book not in SHARP_BOOKS and side in pr and other in pr]
    if not probs:
        return None
    return sum(probs) / len(probs), f"Consensus ({len(probs)} book{'s' * (len(probs) > 1)})", None


# ==========================================================================
# Model sources (live DB or mock fixture)
# ==========================================================================
class ModelSource(Protocol):
    def history(self, sport: str, name: str, team_ids: tuple[str, str]
                ) -> PlayerHistory | None: ...
    def team(self, canonical_id: str | None) -> dict[str, Any]: ...
    def league(self, sport: str) -> dict[str, Any]: ...
    def injury(self, canonical_id: str | None, name: str) -> dict[str, Any] | None: ...
    def teammates(self, canonical_id: str | None, name: str) -> list[dict[str, Any]]: ...
    def recently_active(self, sport: str, canonical_id: str | None, name: str
                        ) -> bool | None: ...


class LiveModelSource:
    def __init__(self, conn: sqlite3.Connection) -> None:
        self.conn = conn
        self._league: dict[str, dict[str, Any]] = {}

    def history(self, sport, name, team_ids):
        return load_player_history(self.conn, sport, name, team_ids)

    def team(self, canonical_id):
        return load_team_context(self.conn, canonical_id)

    def league(self, sport):
        if sport not in self._league:
            self._league[sport] = league_context(self.conn, sport)
        return self._league[sport]

    def injury(self, canonical_id, name):
        return player_injury(self.conn, canonical_id, name)

    def teammates(self, canonical_id, name):
        return key_teammate_injuries(self.conn, canonical_id, name)

    def recently_active(self, sport, canonical_id, name):
        return teammate_recently_active(self.conn, sport, canonical_id, name)


def project_player(src: ModelSource, league: str, hist: PlayerHistory, market: str,
                   home_id: str | None, away_id: str | None, cfg: PropConfig
                   ) -> tuple[Projection, str | None]:
    """(projection, opponent canonical id) or raise ValueError when unmodelable."""
    if hist.team_id not in (home_id, away_id):
        raise ValueError(f"{hist.name} last played for {hist.team_id}, not in this game")
    opp_id = away_id if hist.team_id == home_id else home_id
    pm = PROP_MARKETS[market]
    lg = src.league(league)
    if not lg or not lg.get("pace" if league == "NBA" else "plays_pg"):
        raise ValueError(f"no {league} league context (run --update-stats --props)")
    if league == "NBA":
        proj = project_nba(hist.logs, hist.season_logs, pm.stat, src.team(hist.team_id),
                           src.team(opp_id), lg, cfg.pace_method, hist.prior)
        # Only NEW absences: a teammate out for weeks is already in the player's logs.
        newly_out = [t["player"] for t in src.teammates(hist.team_id, hist.name)
                     if str(t["status"]).lower() == "out"
                     and src.recently_active(league, hist.team_id, t["player"])]
        boost = absence_boost(len(newly_out))
        if boost:
            proj = apply_usage_boost(proj, boost, f"Usage +{boost * 100:g}% (key teammate "
                                                  f"{', '.join(newly_out)} newly Out)")
        return proj, opp_id
    # NFL key injuries are starting QBs: a QB absence cuts receiver volume, so no
    # bump is applied there.
    return project_nfl(hist.logs, pm.stat, hist.position or "WR", src.team(hist.team_id),
                       src.team(opp_id), lg, cfg.pace_method), opp_id


# ==========================================================================
# Candidate building
# ==========================================================================
def _label(player: str, side: str, line: float, market: str) -> str:
    return f"{player} {side} {line:g} {PROP_MARKETS[market].label}"


def _injury_notes(src: ModelSource, team_id: str | None, player: str
                  ) -> tuple[str | None, list[str]]:
    """(blocking reason or None, flag notes) from the injury report."""
    own = src.injury(team_id, player)
    notes = []
    if own:
        status = str(own["status"]).lower()
        if status in OUT_STATUSES:
            return f"Injury: {player} listed {own['status']}", []
        if status in QUESTIONABLE:
            notes.append(f"Injury: {player} {own['status']}")
    for t in src.teammates(team_id, player):
        notes.append(f"Injury: key teammate {t['player']} ({t['status']})")
    return None, notes


def build_event_candidates(ev: dict[str, Any], sport: str, quotes: list[PropQuote],
                           src: ModelSource, cfg: PropConfig, stats: PropScanStats,
                           rejections: list[dict[str, Any]]) -> list[dict[str, Any]]:
    league = SPORT_TO_LEAGUE[sport]
    event_name = f"{ev['away_team']} @ {ev['home_team']}"
    home_id = try_resolve_team(ev["home_team"], league)
    away_id = try_resolve_team(ev["away_team"], league)
    grouped: dict[tuple[str, str, float], dict[str, dict[str, int]]] = {}
    for q in quotes:
        grouped.setdefault((q.market, q.player, q.point), {}).setdefault(q.book, {})[q.side] = \
            q.price
    out: list[dict[str, Any]] = []
    hist_cache: dict[str, PlayerHistory | None] = {}
    for (market, player, line), pairs in sorted(grouped.items()):
        stats.player_lines += 1
        reject = lambda side, odds, reason: rejections.append({  # noqa: E731
            "event": event_name, "selection": _label(player, side, line, market),
            "offered_american": odds, "reason": reason})
        retail = {side: max(((pr[side], b) for b, pr in pairs.items()
                             if b in RETAIL_BOOKS and side in pr),
                            key=lambda t: american_to_decimal(t[0]), default=None)
                  for side in SIDES}
        if not any(retail.values()):
            continue                                           # sharp-only line
        k = player_key(player)
        if k not in hist_cache:
            hist_cache[k] = src.history(league, player, (home_id, away_id))
        hist = hist_cache[k]
        if hist is None:
            stats.no_model += 1
            continue                                           # unknown player: not a rejection
        block, notes = _injury_notes(src, hist.team_id, player)
        if block:
            stats.injury_out += 1
            side0 = next(s for s in SIDES if retail[s])
            reject(side0, retail[side0][0], block)
            continue
        try:
            proj, opp_id = project_player(src, league, hist, market, home_id, away_id, cfg)
        except (ValueError, KeyError, ZeroDivisionError) as exc:
            stats.no_model += 1
            log.debug("no projection for %s %s: %s", player, market, exc)
            continue
        stats.players_modeled += 1
        best_fail: tuple[float, str, int, str] | None = None
        for side in SIDES:
            if not retail[side]:
                continue
            price, book = retail[side]
            if not validate_odds_floor(price):
                stats.floor_rejected += 1
                reject(side, price, f"Best retail {price:+d} worse than -130 floor")
                continue
            bench = market_fair(pairs, side, book)
            if bench is None:
                stats.no_benchmark += 1
                reject(side, price, "No market benchmark (no Pinnacle pair or second book)")
                continue
            mkt_p, bench_src, pin_price = bench
            dec = american_to_decimal(price)
            e = evaluate_side(proj, line, side, dec, mkt_p, cfg.continuous_dist,
                              cfg.sigma_threshold, cfg.min_shrunk_ev, cfg.divergence_mode,
                              cfg.model_weight)
            if not e.qualified:
                score = e.z
                if best_fail is None or score > best_fail[0]:
                    best_fail = (score, side, price, "; ".join(e.reasons))
                continue
            out.append({
                "event_id": ev["id"], "event": event_name, "sport": sport, "league": league,
                "commence_time": ev["commence_time"], "player": hist.name if hist.name else player,
                "book_player": player, "team_id": hist.team_id, "opponent_id": opp_id,
                "market_type": market, "side": side, "line": line, "price": price, "book": book,
                "dec": dec, "proj": proj, "eval": e, "bench_src": bench_src,
                "pin_price": pin_price, "notes": notes,
                "selection": f"{player} {side}"})
        if best_fail and not any(c["book_player"] == player and c["market_type"] == market
                                 and c["line"] == line for c in out):
            _, side, price, reason = best_fail
            if "Market EV" in reason:
                stats.negative_market_ev += 1
            else:
                stats.no_divergence += 1
            reject(side, price, reason)
    return out


def _finalize(conn: sqlite3.Connection, raw: list[dict[str, Any]], stats: PropScanStats,
              rejections: list[dict[str, Any]], bankroll: float, origin: str, now_iso: str
              ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Flat 1.0u sizing under the prop event bucket (highest shrunk EV first).

    Returns (json props, ledger rows). Every gate-passing prop is an official
    ``official_prop_units`` play (1.00u); the risk-sized Kelly stake is only
    recorded as ``raw_kelly_units``. A prop is refused, never trimmed, when its
    event's prop bucket (2.0u) or the combined single-game ceiling (4.0u) has
    less than a full official unit left, so game-line exposure never starves it.
    """
    props, rows = [], []
    by_event: dict[str, list[dict[str, Any]]] = {}
    for c in raw:
        c["pick_id"] = make_pick_id(c["event_id"], c["market_type"], c["selection"], c["line"])
        by_event.setdefault(c["event_id"], []).append(c)
    for event_id, cands in by_event.items():
        cands.sort(key=lambda c: -c["eval"].shrunk_ev)
        scan_ids = [c["pick_id"] for c in cands]
        allocated = 0.0
        for c in cands:
            e, proj = c["eval"], c["proj"]
            label = _label(c["book_player"], c["side"], c["line"], c["market_type"])

            def refuse(reason: str, c: dict[str, Any] = c, label: str = label) -> None:
                rejections.append({"event": c["event"], "selection": label,
                                   "offered_american": c["price"], "reason": reason})

            # Diagnostic Kelly, sized as if the event were empty (caps are applied below).
            sized = size_bet_under_uncertainty(e.fair_prob, c["dec"], e.model_prob_sigma,
                                               bankroll_units=bankroll)
            if sized["status"] == HITL_REJECTED and sized["reason"] != "Stake rounds to zero":
                refuse(sized["reason"])             # non-positive EV / floor / not a single
                continue
            raw_units = round(min(sized["quarter_kelly_fraction"],
                                  sized["baker_mchale_fraction"]) * bankroll, 4)
            units = official_prop_units(raw_units)
            prop_open = db.open_event_exposure(conn, event_id, exclude=scan_ids,
                                               bucket=BUCKET_PROP) + allocated
            total_open = db.open_event_exposure(conn, event_id, exclude=scan_ids) + allocated
            if prop_open + units > MAX_PROP_EVENT_UNITS + 1e-9:
                stats.event_cap_rejected += 1
                refuse(f"Event exposure cap reached: prop bucket {prop_open:.2f}u of "
                       f"{MAX_PROP_EVENT_UNITS:.2f}u open; official props are {units:.2f}u flat")
                continue
            if total_open + units > MAX_EVENT_UNITS + 1e-9:
                stats.event_cap_rejected += 1
                refuse(f"Event exposure cap reached: {total_open:.2f}u of "
                       f"{MAX_EVENT_UNITS:.2f}u combined single-game ceiling open")
                continue
            cap_note = (f"Official {units:.2f}u standard play (raw Kelly {raw_units:.2f}u)"
                        if abs(units - raw_units) > 0.005 else None)
            allocated += units
            stats.candidates += 1
            stats.pending_hitl += 1
            stats.injury_flagged += bool(c["notes"])
            boost_note = proj.details.get("usage_boost_reason")
            reasons = [PROP_HITL_REASON, *c["notes"], *([boost_note] if boost_note else []),
                       *([cap_note] if cap_note else [])]
            sizing = {"method": "flat_official", "official_units": units,
                      "raw_kelly_units": raw_units,
                      "full_kelly_fraction": round(sized["raw_kelly"], 6),
                      "quarter_kelly_fraction": round(sized["quarter_kelly_fraction"], 6),
                      "baker_mchale_fraction": round(sized["baker_mchale_fraction"], 6),
                      "bankroll_units": bankroll}
            pm = PROP_MARKETS[c["market_type"]]
            analysis = {"projection": round(proj.mean, 3), "sd": round(proj.sd, 3),
                        "kind": proj.kind, "n_eff": round(proj.n_eff, 2),
                        "se_mean": round(proj.se_mean, 3), "details": proj.details,
                        "z": round(e.z, 3), "triggers": e.triggers,
                        "market_benchmark": c["bench_src"], "team_id": c["team_id"],
                        "opponent_id": c["opponent_id"], "sizing": sizing}
            rows.append({
                "pick_id": c["pick_id"], "created_at": now_iso, "commence_time": c["commence_time"],
                "sport": c["sport"], "event_id": event_id, "event_name": c["event"],
                "market_type": c["market_type"], "selection": c["selection"],
                "line_point": c["line"], "bookmaker": RETAIL_BOOKS.get(c["book"], c["book"]),
                "offered_american": c["price"], "offered_decimal": round(c["dec"], 6),
                "benchmark_source": c["bench_src"], "benchmark_american": c["pin_price"],
                "fair_prob": round(e.fair_prob, 6), "ev_percent": round(e.shrunk_ev * 100, 3),
                "kelly_units": units, "hitl_status": HITL_PENDING,
                "hitl_reason": "; ".join(reasons), "result": "UNSETTLED", "origin": origin,
                "model_prob": round(e.model_prob, 6),
                "analysis": json.dumps(analysis, separators=(",", ":"))})
            props.append({
                "pick_id": c["pick_id"], "event": c["event"], "event_id": event_id,
                "sport": c["sport"], "league": c["league"], "commence_time": c["commence_time"],
                "player": c["book_player"], "team": c["team_id"], "opponent": c["opponent_id"],
                "market_type": c["market_type"], "stat_label": pm.label, "side": c["side"],
                "line": c["line"], "selection": label, "bet_type": "single",
                "model_projection": round(proj.mean, 2),
                "delta": round(proj.mean - c["line"], 2), "z_score": round(e.z, 3),
                "model_sd": round(proj.sd, 3),
                "distribution": ("negative_binomial" if proj.var > proj.mean * 1.0001
                                 else "poisson") if proj.kind == "count" else c["dist"],
                "model_raw_probability": round(e.model_prob, 6),
                "model_blend_probability": round(e.blend_prob, 6),
                "model_fair_probability": round(e.fair_prob, 6),
                "market_fair_probability": round(e.market_prob, 6),
                "market_benchmark": c["bench_src"], "push_probability": round(e.p_push, 6),
                "offered_american": c["price"],
                "retail_book": RETAIL_BOOKS.get(c["book"], c["book"]),
                "fair_american": decimal_to_american(1.0 / e.fair_prob),
                "ev_percent": round(e.shrunk_ev * 100, 2),
                "market_ev_percent": round(e.market_ev * 100, 2),
                "model_ev_percent": round(e.model_ev * 100, 2),
                "shrinkage_factor": round(e.shrinkage, 4),
                "estimation_sigma": round(e.model_prob_sigma, 5),
                "official_units": units, "raw_kelly_units": raw_units, "sizing": sizing,
                "hitl_status": HITL_PENDING,
                "hitl_reason": "; ".join(reasons), "injury_flag": bool(c["notes"]),
                "triggers": e.triggers, "projection": proj.details})
    return props, rows


# ==========================================================================
# Orchestration
# ==========================================================================
def run_prop_scan(conn: sqlite3.Connection, sports: tuple[str, ...] = ("basketball_nba",),
                  mock: bool = False, output_path: Path = PROP_OUTPUT_PATH,
                  now: datetime | None = None, cfg: PropConfig | None = None,
                  refresh_stats: bool = True) -> dict[str, Any]:
    """Fetch (or mock) prop lines, price against the model, persist, write the JSON."""
    now = now or db.utcnow()
    now_iso = db.iso(now)
    load_env()
    cfg = cfg or PropConfig.from_env()
    bankroll = bankroll_units()
    origin = db.ORIGIN_MOCK if mock else db.ORIGIN_LIVE
    stats = PropScanStats(slate_gate="bypassed (mock)" if mock
                          else f"today_et, start <= {cfg.lookahead_hours:g}h")
    rejections: list[dict[str, Any]] = []
    raw: list[dict[str, Any]] = []
    leagues = [SPORT_TO_LEAGUE[s] for s in sports if s in SPORT_TO_LEAGUE]
    unknown = [s for s in sports if s not in SPORT_TO_LEAGUE]
    if unknown:
        raise ValueError(f"Unsupported sport(s) {unknown}; known: {list(SPORT_TO_LEAGUE)}")
    stats_refresh: list[dict[str, Any]] = []

    if mock:
        slate_by_sport, src = build_mock_prop_slate(now)
    else:
        src = LiveModelSource(conn)
        if refresh_stats:
            stats_refresh = ensure_fresh(conn, leagues)
        api_key = get_setting("ODDS_API_KEY")
        books = get_setting("ODDS_API_BOOKMAKERS", DEFAULT_BOOKMAKERS) or DEFAULT_BOOKMAKERS
        if not api_key:
            raise ValueError("ODDS_API_KEY missing. Run --setup-env or use --mock.")

    for sport in sports:
        league = SPORT_TO_LEAGUE[sport]
        if mock:
            events = slate_by_sport.get(sport, [])
            stats.events_listed += len(events)
            todo = slate_events(events, now, cfg.max_events, gate=False)
            stats.quota[sport] = "mock"
        else:
            try:
                listed = fetch_events(conn, sport, api_key)
            except HttpError as exc:
                log.warning("events list failed for %s: %s", sport, exc)
                stats.fetch_errors += 1
                stats.quota[sport] = "error"
                continue
            stats.events_listed += len(listed)
            todo = slate_events(listed, now, cfg.max_events,
                                lookahead_hours=cfg.lookahead_hours)
        stats.events_on_slate += len(todo)
        queried_here = 0
        for ev in todo:
            if mock:
                event = ev
            else:
                remaining = known_quota(conn)
                if remaining is not None and remaining < cfg.quota_floor:
                    stats.quota_guard_tripped = True
                    log.warning("prop scan stopped: x-requests-remaining %s < floor %s",
                                remaining, cfg.quota_floor)
                    break
                try:
                    fetch = fetch_event_props(conn, sport, ev["id"],
                                              ",".join(MARKETS_BY_LEAGUE[league]), api_key, books)
                except HttpError as exc:
                    log.warning("prop odds failed for %s: %s", ev["id"], exc)
                    stats.fetch_errors += 1
                    continue
                event = {**ev, **fetch.event}
                stats.events_from_cache += fetch.from_cache
                if not fetch.from_cache:
                    stats.quota[sport] = fetch.quota_remaining
            stats.events_queried += 1
            queried_here += 1
            quotes = parse_prop_quotes(event)
            stats.prop_quotes += len(quotes)
            if mock or not fetch.from_cache:
                stats.snapshots += db.insert_snapshots(
                    conn, prop_snapshot_rows(event["id"], quotes, now_iso))
            for c in build_event_candidates(event, sport, quotes, src, cfg, stats, rejections):
                c["dist"] = cfg.continuous_dist
                raw.append(c)
        if not mock and sport not in stats.quota:       # nothing fetched over the network
            stats.quota[sport] = (f"cache (last seen {known_quota(conn)})" if queried_here
                                  else "quota floor reached" if stats.quota_guard_tripped
                                  else "no games on today's slate")

    props, rows = _finalize(conn, raw, stats, rejections, bankroll, origin, now_iso)
    stats.ledger_inserted, refused = Ledger(conn, origin).record_positions(rows)
    if refused:
        bad = dict(refused)
        for p in props:
            if p["pick_id"] in bad:
                rejections.append({"event": p["event"], "selection": p["selection"],
                                   "offered_american": p["offered_american"],
                                   "reason": f"Ledger refused: {bad[p['pick_id']]}"})
        props = [p for p in props if p["pick_id"] not in bad]
        stats.candidates = len(props)
    props.sort(key=lambda p: (-p["ev_percent"], p["event"]))
    scan = {
        "timestamp": now_iso, "slate_date": eastern_date(now).isoformat(),
        "operating_authority": "Captain HZ", "mode": "MOCK" if mock else "LIVE",
        "bankroll_units": bankroll,
        "config": {"sigma_threshold": cfg.sigma_threshold,
                   "min_shrunk_ev_percent": round(cfg.min_shrunk_ev * 100, 4),
                   "divergence_mode": cfg.divergence_mode, "pace_method": cfg.pace_method,
                   "continuous_distribution": cfg.continuous_dist,
                   "quota_floor": cfg.quota_floor, "max_events": cfg.max_events,
                   "model_weight": cfg.model_weight,
                   "lookahead_hours": cfg.lookahead_hours,
                   "official_prop_units": MIN_PROP_OFFICIAL_UNITS,
                   "max_prop_units": MAX_PROP_UNITS,
                   "prop_event_cap_units": MAX_PROP_EVENT_UNITS,
                   "event_total_cap_units": MAX_EVENT_UNITS},
        "props": props, "rejections": rejections,
        "ingestion": {"sources": ["The-Odds-API (event props)", "nba_api", "nflverse"],
                      "sports": list(sports), "stats": {f.name: getattr(stats, f.name)
                                                        for f in fields(stats)},
                      "stats_refresh": stats_refresh},
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(scan, indent=2) + "\n", encoding="utf-8")
    return scan


# ==========================================================================
# Mock slate (synthetic players; exact Odds-API event-odds wire format)
# ==========================================================================
# Fictional "<TEAM> Mock <Role>" players so synthetic stats are never
# attributed to real people. Lines are placed RELATIVE to the model's own
# projection, so each scenario keeps exercising its branch if constants move.

MOCK_INJURIES: dict[str, list[dict[str, Any]]] = {
    "NBA_LAL": [{"player": "LAL Mock Starter", "position": "F", "status": "Questionable",
                 "is_key": 1, "key_reason": "starter in last game"}],
    "NBA_GSW": [{"player": "GSW Mock Star", "position": "G", "status": "Out", "is_key": 1,
                 "key_reason": "starter in last game"}],
}

_NBA_TEAMS = {"NBA_BOS": (99.0, 112.0), "NBA_MIL": (100.5, 116.0), "NBA_DEN": (98.0, 114.5),
              "NBA_PHX": (99.5, 117.0), "NBA_LAL": (101.0, 115.5), "NBA_GSW": (100.0, 113.5)}
_NFL_TEAMS = {"NFL_BUF": (64.0, 63.0, 6.0, 4.1), "NFL_KC": (63.0, 61.0, 6.6, 4.6),
              "NFL_DET": (65.0, 62.0, 6.4, 4.0), "NFL_GB": (62.0, 64.0, 6.8, 4.5)}


def _mock_nba_logs(rng: random.Random, minutes: float, pts36: float, reb36: float, ast36: float,
                   fg3a36: float, fg3p: float, n: int = 16,
                   hot: list[tuple[int, int, int]] | None = None) -> list[dict[str, Any]]:
    """Game logs MOST RECENT FIRST. ``hot`` overrides the latest games' (pts, fg3m, fg3a)."""
    logs = []
    for i in range(n):
        m = max(18.0, rng.gauss(minutes, 2.5))
        f = m / 36.0
        fg3a = max(0, round(rng.gauss(fg3a36 * f, 1.2)))
        fg3m = sum(rng.random() < fg3p for _ in range(fg3a))
        fta = max(0, round(rng.gauss(4.0 * f, 1.0)))
        ftm = round(fta * 0.8)
        pts_target = max(0.0, rng.gauss(pts36 * f, 3.0))
        fg2m = max(0, round((pts_target - 3 * fg3m - ftm) / 2))
        fg2a = max(fg2m, round(fg2m / 0.54))
        logs.append({"minutes": round(m, 1), "fgm": fg2m + fg3m, "fga": fg2a + fg3a,
                     "fg3m": fg3m, "fg3a": fg3a, "ftm": ftm, "fta": fta,
                     "pts": 2 * fg2m + 3 * fg3m + ftm,
                     "reb": max(0, round(rng.gauss(reb36 * f, 1.8))),
                     "ast": max(0, round(rng.gauss(ast36 * f, 1.5))), "tov": 2,
                     "team_margin": rng.choice([-12, -6, -3, 2, 5, 9, 14, 22]),
                     "game_poss": round(rng.gauss(99.5, 2.5), 1),
                     "game_date": (date(2026, 11, 10) - timedelta(days=2 * i)).isoformat()})
    for i, (pts, fg3m, fg3a) in enumerate(hot or []):
        g = logs[i]
        g.update({"fg3m": fg3m, "fg3a": fg3a, "pts": pts,
                  "fgm": (pts - 3 * fg3m - g["ftm"]) // 2 + fg3m,
                  "fga": max(g["fga"], (pts - 3 * fg3m - g["ftm"]) // 2 + fg3a + 4)})
    return logs


def _mock_nfl_weeks(rng: random.Random, pos: str, tshare: float, ypt: float, catch: float,
                    carries: float, ypc: float, att: float, ypa: float, n: int = 9
                    ) -> list[dict[str, Any]]:
    weeks = []
    for i in range(n):
        tg = max(0, round(rng.gauss(tshare * 35.0, 2.0))) if tshare else 0
        rec = sum(rng.random() < catch for _ in range(tg))
        car = max(0, round(rng.gauss(carries, 3.0))) if carries else 0
        pa = max(0, round(rng.gauss(att, 4.0))) if att else 0
        weeks.append({"targets": tg, "receptions": rec, "target_share": tg / 35.0,
                      "air_yards_share": tg / 35.0 * 1.1,
                      "rec_yds": max(0, round(rng.gauss(tg * ypt, 15.0))) if tg else 0,
                      "carries": car, "rush_yds": max(0, round(rng.gauss(car * ypc, 12.0))),
                      "pass_att": pa, "completions": round(pa * 0.65),
                      "pass_yds": max(0, round(rng.gauss(pa * ypa, 30.0))),
                      "game_date": f"2026-W{10 - i:02d}"})
    return weeks


class MockModelSource:
    """Fixture-backed ModelSource with deterministic synthetic histories."""

    def __init__(self) -> None:
        self.players: dict[str, PlayerHistory] = {}
        self.teams: dict[str, dict[str, Any]] = {}
        for cid, (pace, drtg) in _NBA_TEAMS.items():
            self.teams[cid] = {"pace": pace, "drtg": drtg, "games": 14, "allowed_reb_100": 44.0,
                               "allowed_ast_100": 26.5, "allowed_fg3m_100": 13.0}
        for cid, (plays, dplays, ypa, ypc) in _NFL_TEAMS.items():
            self.teams[cid] = {"plays_pg": plays, "def_plays_pg": dplays, "pass_pg": 36.0,
                               "rush_pg": 26.0, "neutral_pass_rate": 0.57, "def_pass_ypa": ypa,
                               "def_rush_ypc": ypc, "games": 9}
        self.leagues = {
            "NBA": {"pace": 99.6, "drtg": 115.0, "allowed_reb_100": 44.0,
                    "allowed_ast_100": 26.5, "allowed_fg3m_100": 13.0},
            "NFL": {"plays_pg": 63.0, "def_plays_pg": 63.0, "neutral_pass_rate": 0.57,
                    "def_pass_ypa": 6.4, "def_rush_ypc": 4.3}}
        nba = [  # name, team, minutes, pts36, reb36, ast36, 3PA36, 3P%, hot streak
            ("BOS Mock Wing", "NBA_BOS", 35, 24, 6, 4, 8, 0.36, [(38, 7, 11), (41, 8, 12),
                                                                 (36, 6, 10)]),
            ("MIL Mock Guard", "NBA_MIL", 33, 20, 4, 6, 9, 0.38, None),
            ("DEN Mock Center", "NBA_DEN", 34, 18, 12, 5, 1, 0.30, None),
            ("PHX Mock Forward", "NBA_PHX", 32, 16, 7, 7, 4, 0.35, None),
            ("PHX Mock Guard", "NBA_PHX", 34, 25, 4, 5, 7, 0.37, None),
            ("LAL Mock Starter", "NBA_LAL", 35, 26, 7, 6, 6, 0.36, None),
            ("GSW Mock Star", "NBA_GSW", 34, 28, 5, 6, 11, 0.40, None),
            ("GSW Mock Wing", "NBA_GSW", 32, 19, 5, 3, 7, 0.37, None),
        ]
        for i, (name, cid, mins, p, r, a, f3, f3p, hot) in enumerate(nba):
            logs = _mock_nba_logs(random.Random(100 + i), mins, p, r, a, f3, f3p, hot=hot)
            self.players[player_key(name)] = PlayerHistory(name, player_key(name), "NBA", cid,
                                                           None, logs, logs)
        nfl = [  # name, team, pos, tshare, ypt, catch, carries, ypc, att, ypa
            ("BUF Mock WR1", "NFL_BUF", "WR", 0.27, 9.0, 0.66, 0, 0, 0, 0),
            ("KC Mock QB", "NFL_KC", "QB", 0, 0, 0, 3, 5.0, 36, 7.1),
            ("DET Mock RB", "NFL_DET", "RB", 0.08, 6.0, 0.80, 15, 4.2, 0, 0),
            ("GB Mock TE", "NFL_GB", "TE", 0.16, 7.5, 0.72, 0, 0, 0, 0),
        ]
        for i, (name, cid, pos, ts, ypt, ct, car, ypc, att, ypa) in enumerate(nfl):
            weeks = _mock_nfl_weeks(random.Random(200 + i), pos, ts, ypt, ct, car, ypc, att, ypa)
            self.players[player_key(name)] = PlayerHistory(name, player_key(name), "NFL", cid,
                                                           pos, weeks, weeks)

    def history(self, sport, name, team_ids):
        h = self.players.get(player_key(name))
        return h if h and h.sport == sport and h.team_id in team_ids else None

    def team(self, canonical_id):
        return self.teams.get(canonical_id or "", {})

    def league(self, sport):
        return self.leagues.get(sport, {})

    def injury(self, canonical_id, name):
        return next((r for r in MOCK_INJURIES.get(canonical_id or "", [])
                     if player_key(r["player"]) == player_key(name)), None)

    def teammates(self, canonical_id, name):
        return [r for r in MOCK_INJURIES.get(canonical_id or "", [])
                if r["is_key"] and player_key(r["player"]) != player_key(name)]

    def recently_active(self, sport, canonical_id, name):
        h = self.players.get(player_key(name))      # mock histories run to the last game
        return None if h is None else h.team_id == canonical_id


def _half_line(x: float) -> float:
    return math.floor(x) + 0.5


# (event, player, market, line offset vs projection, {book: (over, under)})
_MOCK_SCENARIOS: list[tuple[str, str, str, float, dict[str, tuple[int, int]]]] = [
    # Hot 3-game streak inflates the line: model regresses -> Under (sigma trigger).
    ("mock-nba-bos-mil", "BOS Mock Wing", "player_points", 5.6,
     {"pinnacle": (-110, -110), "draftkings": (-125, 105), "fanduel": (-110, -110)}),
    # Count prop (NB): 3PM Over at a soft retail price (shrunk-EV trigger).
    ("mock-nba-bos-mil", "MIL Mock Guard", "player_threes", -0.9,
     {"pinnacle": (-102, -118), "draftkings": (118, -145), "fanduel": (-105, -115)}),
    # Line at the projection: market EV > 0 but no model divergence.
    ("mock-nba-den-phx", "DEN Mock Center", "player_rebounds", 0.0,
     {"pinnacle": (-110, -110), "draftkings": (104, -128), "fanduel": (-110, -110)}),
    # Model likes Over but every retail price is worse than -130.
    ("mock-nba-den-phx", "PHX Mock Forward", "player_assists", -1.5,
     {"pinnacle": (-140, 120), "draftkings": (-145, 115), "fanduel": (-150, 120)}),
    # Model likes Over but retail is worse than the market (negative market EV).
    ("mock-nba-den-phx", "PHX Mock Guard", "player_points", -3.5,
     {"pinnacle": (-105, -115), "draftkings": (-125, 105), "fanduel": (-128, 104)}),
    # Questionable player himself: flagged with [INJURY].
    ("mock-nba-lal-gsw", "LAL Mock Starter", "player_points", -3.8,
     {"pinnacle": (-108, -112), "draftkings": (110, -130), "fanduel": (-108, -112)}),
    # Listed Out: never priced.
    ("mock-nba-lal-gsw", "GSW Mock Star", "player_points", 0.0,
     {"pinnacle": (-110, -110), "draftkings": (-110, -110), "fanduel": (-110, -110)}),
    # Key teammate Out: flagged with [INJURY]; shares the LAL-GSW 2.0u event cap.
    ("mock-nba-lal-gsw", "GSW Mock Wing", "player_points", -3.9,
     {"pinnacle": (-108, -112), "draftkings": (112, -135), "fanduel": (-105, -115)}),
    # NFL receiving yards (Gamma) and receptions (NB) on one event: event cap trims.
    ("mock-nfl-buf-kc", "BUF Mock WR1", "player_reception_yds", -14.0,
     {"pinnacle": (-108, -112), "draftkings": (108, -130), "fanduel": (-110, -110)}),
    ("mock-nfl-buf-kc", "BUF Mock WR1", "player_receptions", -1.2,
     {"pinnacle": (-115, -105), "draftkings": (105, -128), "fanduel": (-110, -110)}),
    # QB passing yards at the projection: no divergence.
    ("mock-nfl-buf-kc", "KC Mock QB", "player_pass_yds", 0.0,
     {"pinnacle": (-110, -110), "draftkings": (105, -125), "fanduel": (-110, -110)}),
    # RB rushing yards: line far above projection -> Under.
    ("mock-nfl-det-gb", "DET Mock RB", "player_rush_yds", 15.0,
     {"pinnacle": (-105, -115), "draftkings": (-120, 100), "fanduel": (-110, -110)}),
    # Single book, no Pinnacle: no market benchmark.
    ("mock-nfl-det-gb", "GB Mock TE", "player_receptions", -1.5,
     {"draftkings": (110, -140)}),
]

# Final box-score lines for mock settlement: (event_id, player) -> stats.
MOCK_PLAYER_FINALS: dict[tuple[str, str], dict[str, Any]] = {
    ("mock-nba-bos-mil", "BOS Mock Wing"): {"played": True, "pts": 24, "reb": 6, "ast": 4,
                                            "fg3m": 2},
    ("mock-nba-bos-mil", "MIL Mock Guard"): {"played": True, "pts": 19, "reb": 3, "ast": 7,
                                             "fg3m": 3},
    ("mock-nba-lal-gsw", "LAL Mock Starter"): {"played": True, "pts": 31, "reb": 8, "ast": 5,
                                               "fg3m": 3},
    ("mock-nba-lal-gsw", "GSW Mock Wing"): {"played": True, "pts": 15, "reb": 5, "ast": 3,
                                            "fg3m": 1},
    ("mock-nba-lal-gsw", "GSW Mock Star"): {"played": False},
    ("mock-nfl-buf-kc", "BUF Mock WR1"): {"played": True, "rec_yds": 88, "receptions": 7},
    ("mock-nfl-det-gb", "DET Mock RB"): {"played": True, "rush_yds": 51, "receptions": 2},
}


def build_mock_prop_slate(now: datetime) -> tuple[dict[str, list[dict[str, Any]]],
                                                  MockModelSource]:
    """(events with prop bookmakers per sport, model source) for the mock scan."""
    src = MockModelSource()
    game_events = {e["id"]: e for evs in build_mock_feeds(now)[0].values() for e in evs}
    cfg = PropConfig()
    by_event: dict[str, dict[str, Any]] = {}
    for event_id, player, market, offset, books in _MOCK_SCENARIOS:
        g = game_events[event_id]
        league = SPORT_TO_LEAGUE[g["sport_key"]]
        hist = src.players[player_key(player)]
        home = try_resolve_team(g["home_team"], league)
        away = try_resolve_team(g["away_team"], league)
        proj, _ = project_player(src, league, hist, market, home, away, cfg)
        line = _half_line(proj.mean + offset)
        ev = by_event.setdefault(event_id, {
            "id": event_id, "sport_key": g["sport_key"], "sport_title": league,
            "commence_time": g["commence_time"], "home_team": g["home_team"],
            "away_team": g["away_team"], "bookmakers": []})
        for book, (over, under) in books.items():
            bm = next((b for b in ev["bookmakers"] if b["key"] == book), None)
            if bm is None:
                bm = {"key": book, "title": {**SHARP_BOOKS, **RETAIL_BOOKS}.get(book, book),
                      "last_update": db.iso(now), "markets": []}
                ev["bookmakers"].append(bm)
            mk = next((m for m in bm["markets"] if m["key"] == market), None)
            if mk is None:
                mk = {"key": market, "last_update": db.iso(now), "outcomes": []}
                bm["markets"].append(mk)
            mk["outcomes"] += [{"name": "Over", "description": player, "price": over,
                                "point": line},
                               {"name": "Under", "description": player, "price": under,
                                "point": line}]
    out: dict[str, list[dict[str, Any]]] = {}
    for ev in by_event.values():
        out.setdefault(ev["sport_key"], []).append(ev)
    return out, src
