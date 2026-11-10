"""Live ingestion and entity resolution: The-Odds-API (books) + Kalshi (exchange).

Pipeline (one call: ``run_scan``):
    fetch_the_odds_api    -> events, SQLite-cached 300s, quota headers logged
    fetch_kalshi_markets  -> YES bid/ask/mid/volume + $ depth at best bid/ask,
                             optional signed requests
    align_kalshi          -> canonical team mapping onto Odds-API events
    snapshots             -> every sharp/retail/Kalshi price into market_snapshots
    build_candidates      -> exact point match, -130 floor, de-vig, consensus,
                             EV >= 1.5%, HITL gates, event-capped quarter-Kelly;
                             Kalshi probability is liquidity-filtered, and a
                             lead-lag EXCHANGE_CANARY_SIGNAL is flagged when
                             sharp/exchange sources move while retail is static
    export                -> ledger_picks + OUTBOX/market_scans/live_candidates.json

``mock=True`` replays synthetic payloads in each API's exact wire format, so
the same parsing, alignment and pricing code runs offline.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import sqlite3
import sys
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.core import database as db  # noqa: E402
from src.core.config import (  # noqa: E402
    ODDS_API_BASE,
    bankroll_units,
    get_setting,
    kalshi_base_url,
    kalshi_headers_for,
    load_env,
)
from src.core.engine import (  # noqa: E402
    HITL_PENDING,
    HITL_REJECTED,
    LIQ_ILLIQUID,
    LIQ_ONE_SIDED,
    american_to_decimal,
    calculate_ev,
    compute_liquidity_weighted_prob,
    decimal_to_american,
    devig_odds_ratio,
    devig_power,
    kalshi_net_odds,
    raw_quarter_kelly_units,
    size_position,
    validate_odds_floor,
)
from src.core.ledger import Ledger  # noqa: E402
from src.core.matchup_model import (  # noqa: E402
    GameContext,
    build_game_context,
    compute_bayesian_true_prob,
    model_gate_reasons,
)
from src.data.http_client import HttpError, get_json  # noqa: E402
from src.data.teams import (  # noqa: E402
    LEAGUE_TO_KALSHI_GAME_SERIES,
    SPORT_TO_LEAGUE,
    abbreviation,
    normalize_team,
)

log = logging.getLogger("breeze.ingest")

OUTPUT_PATH = ROOT / "OUTBOX" / "market_scans" / "live_candidates.json"

SHARP_BOOKS: dict[str, str] = {"pinnacle": "Pinnacle", "circa": "Circa", "circasports": "Circa"}
SHARP_PRIORITY: tuple[str, ...] = ("pinnacle", "circa", "circasports")
RETAIL_BOOKS: dict[str, str] = {"draftkings": "DraftKings", "fanduel": "FanDuel",
                                "betmgm": "BetMGM", "williamhill_us": "Caesars"}
DEFAULT_BOOKMAKERS = "pinnacle,draftkings,fanduel,betmgm,williamhill_us"
DEFAULT_MARKETS = "h2h,spreads,totals"
BOOK_MARKETS: tuple[str, ...] = ("h2h", "spreads", "totals")

ODDS_CACHE_TTL_S = 300
KALSHI_CACHE_TTL_S = 60
MIN_EV_PERCENT = 1.5
DIVERGENCE_LIMIT = 0.04
LINE_MOVE_LIMIT = 1.5
LINE_MOVE_WINDOW = timedelta(hours=2)

DIVERGENCE_REASON = "Pinnacle/Kalshi divergence > 4%"

# Exchange liquidity / lead-lag canary
ONE_SIDED_HITL_LIQ_USD = 500.0       # one-sided synthetic line below this -> Captain HZ
CANARY_MIN_LIQ_USD = 300.0           # one-sided exchange depth that counts as a signal
CANARY_MIN_MOVE = 0.01               # sharp/exchange implied-prob move toward our side
CANARY_RETAIL_STATIC = 0.005         # retail implied-prob change still counted as static
CANARY_MIN_SOURCES = 2
CANARY_FLAG = "EXCHANGE_CANARY_SIGNAL"
LIQ_UNSIZED = "MID_UNSIZED"          # no depth published: legacy mid, unfiltered


class IngestionError(RuntimeError):
    """A live source could not be fetched."""


# ==========================================================================
# The-Odds-API
# ==========================================================================
@dataclass
class OddsFetch:
    sport: str
    events: list[dict[str, Any]]
    from_cache: bool
    quota_remaining: str | None = None
    quota_used: str | None = None


def fetch_the_odds_api(conn: sqlite3.Connection, sport: str = "basketball_nba",
                       markets: str = DEFAULT_MARKETS, api_key: str | None = None,
                       bookmakers: str | None = None) -> OddsFetch:
    """GET /v4/sports/{sport}/odds/ (American odds), cached in SQLite for 300s.

    Requests named bookmakers (sharp + retail) so quota cost is one region
    per market. Logs ``x-requests-remaining`` / ``x-requests-used``.
    """
    books = bookmakers or get_setting("ODDS_API_BOOKMAKERS", DEFAULT_BOOKMAKERS)
    key = f"odds:{sport}:{markets}:{books}"
    cached = db.cache_get(conn, key)
    if cached is not None:
        log.info("The-Odds-API %s: cache hit (%d events, no quota used)", sport, len(cached))
        return OddsFetch(sport, cached, True)
    api_key = api_key or get_setting("ODDS_API_KEY")
    if not api_key:
        raise IngestionError("ODDS_API_KEY missing. Run --setup-env or use --mock.")
    params = {"apiKey": api_key, "markets": markets, "oddsFormat": "american",
              "dateFormat": "iso", "bookmakers": books}
    try:
        events, hdrs = get_json(f"{ODDS_API_BASE}/sports/{sport}/odds/", params)
    except HttpError as exc:
        raise IngestionError(str(exc)) from None
    db.cache_put(conn, key, events, ODDS_CACHE_TTL_S)
    fetch = OddsFetch(sport, events, False, hdrs.get("x-requests-remaining"),
                      hdrs.get("x-requests-used"))
    log.info("The-Odds-API %s: %d events | x-requests-remaining=%s x-requests-used=%s "
             "x-requests-last=%s", sport, len(events), fetch.quota_remaining, fetch.quota_used,
             hdrs.get("x-requests-last"))
    return fetch


def split_books(event: dict[str, Any]) -> tuple[dict[str, dict], dict[str, dict]]:
    """(sharp, retail) bookmakers of an event, keyed by book key."""
    sharp, retail = {}, {}
    for bm in event.get("bookmakers", []):
        if bm["key"] in SHARP_BOOKS:
            sharp[bm["key"]] = bm
        elif bm["key"] in RETAIL_BOOKS:
            retail[bm["key"]] = bm
    return sharp, retail


def _market(bookmaker: dict, key: str) -> list[dict] | None:
    for m in bookmaker.get("markets", []):
        if m["key"] == key:
            return m["outcomes"]
    return None


# ==========================================================================
# Kalshi
# ==========================================================================
@dataclass
class KalshiQuote:
    ticker: str
    event_ticker: str
    team_text: str
    yes_bid: float | None
    yes_ask: float | None
    mid_probability: float
    volume: float
    yes_bid_size_usd: float | None = None      # $ resting at the best YES bid
    yes_ask_size_usd: float | None = None      # $ to lift the best YES ask


def _price(market: dict, name: str) -> float | None:
    """Price as 0.00-1.00: ``<name>_dollars`` string, else legacy integer cents."""
    dollars = market.get(f"{name}_dollars")
    if dollars not in (None, ""):
        return float(dollars)
    cents = market.get(name)
    return None if cents is None else float(cents) / 100.0


def _depth_usd(market: dict, name: str, price: float | None) -> float | None:
    """Dollar depth at the best ``name`` (yes_bid / yes_ask) = contracts x price.

    Kalshi's Market object publishes ``<name>_size_fp`` (fixed-point contract
    count string); a legacy integer ``<name>_size`` is accepted too. None when
    the feed carries no size at all.
    """
    raw = market.get(f"{name}_size_fp")
    if raw in (None, ""):
        raw = market.get(f"{name}_size")
    if raw in (None, ""):
        return None
    return round(float(raw) * price, 2) if price else 0.0


def parse_kalshi_market(market: dict) -> KalshiQuote | None:
    """Bid/ask/mid/volume/$ depth for one market; None when it has no usable price."""
    bid, ask = _price(market, "yes_bid"), _price(market, "yes_ask")
    mid = (bid + ask) / 2.0 if bid and ask else (_price(market, "last_price") or 0.0)
    if not 0.0 < mid < 1.0:
        return None
    vol = market.get("volume_fp")
    volume = float(vol) if vol not in (None, "") else float(market.get("volume") or 0)
    return KalshiQuote(market["ticker"], market.get("event_ticker", ""),
                       market.get("yes_sub_title") or market.get("title", ""),
                       bid, ask, round(mid, 4), volume,
                       _depth_usd(market, "yes_bid", bid), _depth_usd(market, "yes_ask", ask))


def exchange_probability(quote: KalshiQuote | None) -> tuple[float | None, str | None]:
    """Liquidity-filtered YES probability: ``(prob, method)``.

    No depth published -> legacy mid (``MID_UNSIZED``). Otherwise the engine's
    filter decides: ILLIQUID_NOISE (prob None), ONE_SIDED_SYNTHETIC,
    SYNTHETIC_INVALID or TWO_SIDED_LIQUID. A missing price counts as zero depth.
    """
    if quote is None:
        return None, None
    if quote.yes_bid_size_usd is None and quote.yes_ask_size_usd is None:
        return quote.mid_probability, LIQ_UNSIZED
    bid, ask = quote.yes_bid or 0.0, quote.yes_ask or 0.0
    bid_liq = (quote.yes_bid_size_usd or 0.0) if 0.0 < bid < 1.0 else 0.0
    ask_liq = (quote.yes_ask_size_usd or 0.0) if 0.0 < ask < 1.0 else 0.0
    return compute_liquidity_weighted_prob(bid, ask, bid_liq, ask_liq)


def one_sided_depth(quote: KalshiQuote) -> float:
    """Depth of the credible side of a one-sided book."""
    return max(quote.yes_bid_size_usd or 0.0, quote.yes_ask_size_usd or 0.0)


def fetch_kalshi_markets(conn: sqlite3.Connection, series_ticker: str = "KXNBAGAME",
                         status: str = "open", max_pages: int = 10) -> list[KalshiQuote]:
    """GET {base}/markets for a series (paginated, cached 60s, signed if a key is set)."""
    base = kalshi_base_url()
    key = f"kalshi:{base}:{series_ticker}:{status}"
    raw = db.cache_get(conn, key)
    if raw is None:
        raw, cursor = [], None
        path = "/" + base.split("://", 1)[1].split("/", 1)[1] + "/markets"
        for _ in range(max_pages):
            params: dict[str, Any] = {"series_ticker": series_ticker, "status": status,
                                      "limit": 200}
            if cursor:
                params["cursor"] = cursor
            try:
                data, _ = get_json(f"{base}/markets", params, headers=kalshi_headers_for(path))
            except HttpError as exc:
                raise IngestionError(str(exc)) from None
            raw += data.get("markets", [])
            cursor = data.get("cursor")
            if not cursor:
                break
        db.cache_put(conn, key, raw, KALSHI_CACHE_TTL_S)
    quotes = [q for q in map(parse_kalshi_market, raw) if q]
    log.info("Kalshi %s: %d markets, %d priced", series_ticker, len(raw), len(quotes))
    return quotes


# ==========================================================================
# Entity resolution
# ==========================================================================
_TICKER_DATE = re.compile(r"-(\d{2})([A-Z]{3})(\d{2})")
_MONTHS = {m: i for i, m in enumerate(["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG",
                                       "SEP", "OCT", "NOV", "DEC"], 1)}


def _ticker_date(event_ticker: str) -> date | None:
    m = _TICKER_DATE.search(event_ticker)
    if not m or m.group(2) not in _MONTHS:
        return None
    return date(2000 + int(m.group(1)), _MONTHS[m.group(2)], int(m.group(3)))


def align_kalshi(events: list[dict], quotes: list[KalshiQuote], league: str
                 ) -> tuple[dict[tuple[str, str], KalshiQuote], int]:
    """Map Kalshi game markets to (odds_event_id, canonical_team).

    A Kalshi event matches when both of its resolved teams (ticker suffix or
    ``yes_sub_title``) equal the Odds-API home/away pair and the ticker date
    is within a day of commence time (Kalshi uses US local dates).
    """
    index: dict[tuple[frozenset[str], date], str] = {}
    for ev in events:
        pair = frozenset(filter(None, (normalize_team(ev["home_team"], league),
                                       normalize_team(ev["away_team"], league))))
        start = db.parse_ts(ev["commence_time"]).date()
        for delta in (-1, 0, 1):
            index[(pair, start + timedelta(days=delta))] = ev["id"]

    groups: dict[str, list[KalshiQuote]] = {}
    for q in quotes:
        groups.setdefault(q.event_ticker, []).append(q)
    aligned: dict[tuple[str, str], KalshiQuote] = {}
    unmatched = 0
    for ev_ticker, group in groups.items():
        resolved = {q.ticker: (normalize_team(q.ticker.rsplit("-", 1)[-1], league)
                               or normalize_team(q.team_text, league)) for q in group}
        pair = frozenset(t for t in resolved.values() if t)
        tdate = _ticker_date(ev_ticker)
        event_id = index.get((pair, tdate)) if len(pair) == 2 and tdate else None
        if not event_id:
            unmatched += len(group)
            continue
        for q in group:
            aligned[(event_id, resolved[q.ticker])] = q   # type: ignore[index]
    return aligned, unmatched


# ==========================================================================
# Snapshots
# ==========================================================================
def snapshot_rows(events: list[dict], aligned: dict[tuple[str, str], KalshiQuote],
                  ts: str) -> list[dict[str, Any]]:
    """market_snapshots rows for every sharp/retail outcome and matched Kalshi mid."""
    rows = []
    for ev in events:
        for bm in ev.get("bookmakers", []):
            if bm["key"] not in SHARP_BOOKS and bm["key"] not in RETAIL_BOOKS:
                continue
            for m in bm.get("markets", []):
                for o in m["outcomes"]:
                    rows.append({"event_id": ev["id"], "timestamp": ts, "source": bm["key"],
                                 "market_type": m["key"], "selection": o["name"],
                                 "point": o.get("point"), "american_odds": int(o["price"]),
                                 "implied_prob": round(1.0 / american_to_decimal(o["price"]), 6)})
    for (event_id, team), q in aligned.items():
        rows.append({"event_id": event_id, "timestamp": ts, "source": "kalshi",
                     "market_type": "h2h", "selection": team, "point": None,
                     "american_odds": decimal_to_american(1.0 / q.mid_probability),
                     "implied_prob": q.mid_probability})
    return rows


# ==========================================================================
# Candidates
# ==========================================================================
@dataclass
class ScanStats:
    events: int = 0
    retail_quotes: int = 0
    point_mismatch: int = 0
    floor_rejected: int = 0
    no_benchmark: int = 0
    below_min_ev: int = 0
    event_cap_rejected: int = 0
    candidates: int = 0
    pending_hitl: int = 0
    ledger_inserted: int = 0
    kalshi_markets: int = 0
    kalshi_matched: int = 0
    kalshi_unmatched: int = 0
    snapshots: int = 0
    model_games: int = 0
    quota: dict[str, str | None] = field(default_factory=dict)


def make_pick_id(event_id: str, market_type: str, selection: str, point: float | None) -> str:
    """Stable id: one pick per (event, market, selection, point), whatever the book."""
    raw = f"{event_id}|{market_type}|{selection}|{'' if point is None else point}"
    return "PK-" + hashlib.sha1(raw.encode()).hexdigest()[:12]


def display_selection(market_type: str, selection: str, point: float | None) -> str:
    if market_type == "h2h":
        return f"{selection} ML"
    if market_type == "spreads":
        return f"{selection} {point:+g}"
    if market_type == "totals":
        return f"{selection} {point:g}"
    return f"{selection} YES (Kalshi)"


def _line_move(conn: sqlite3.Connection, ev: dict, mkey: str, name: str, point: float,
               now: datetime) -> float | None:
    """Sharp line move in points if tipoff is within 2h, else None."""
    to_tip = db.parse_ts(ev["commence_time"]) - now
    if not timedelta(0) <= to_tip <= LINE_MOVE_WINDOW:
        return None
    first = db.earliest_point(conn, ev["id"], mkey, name, SHARP_PRIORITY)
    return None if first is None else round(point - first, 2)


def _prior_implied(conn: sqlite3.Connection, event_id: str, mkey: str, source: str,
                   selection: str, point: float | None, before: str) -> float | None:
    """Latest implied probability for a selection at ``source`` strictly before ``before``."""
    row = conn.execute(
        "SELECT implied_prob FROM market_snapshots WHERE event_id = ? AND market_type = ?"
        " AND source = ? AND selection = ? AND IFNULL(point, -9999) = IFNULL(?, -9999)"
        " AND timestamp < ? ORDER BY timestamp DESC, id DESC LIMIT 1",
        (event_id, mkey, source, selection, point, before)).fetchone()
    return None if row is None or row["implied_prob"] is None else float(row["implied_prob"])


def _outcome(bm: dict, mkey: str, name: str, point: float | None) -> dict | None:
    return next((o for o in _market(bm, mkey) or [] if o["name"] == name
                 and (mkey == "h2h" or o.get("point") == point)), None)


def detect_exchange_canary(conn: sqlite3.Connection, ev: dict, mkey: str, name: str,
                           point: float | None, team: str | None, quote: KalshiQuote | None,
                           opp_quote: KalshiQuote | None, now_iso: str) -> list[str]:
    """Lead-lag canary: evidence strings, or [] when there is no signal.

    A source leans our way when:
        * a sharp book's implied probability for ``name`` rose >= 1 pt since
          its previous snapshot; or
        * Kalshi (moneyline) shows >= $300 one-sided depth pushing toward
          ``name`` (bids on our YES with a thin ask, or offers on the
          opponent's YES with a thin bid), or its mid rose >= 1 pt.
    Signal = >= 2 distinct sources leaning our way while every retail book
    with history is static (< 0.5 pt implied change, and at least one such book).
    """
    sharp, retail = split_books(ev)
    evidence: list[str] = []
    seen: set[str] = set()
    for bkey, bm in sharp.items():
        label = SHARP_BOOKS[bkey]
        o = _outcome(bm, mkey, name, point)
        if label in seen or o is None:
            continue
        prior = _prior_implied(conn, ev["id"], mkey, bkey, name, point, now_iso)
        now_p = 1.0 / american_to_decimal(o["price"])
        if prior is not None and now_p - prior >= CANARY_MIN_MOVE:
            seen.add(label)
            evidence.append(f"{label} {prior * 100:.1f}%->{now_p * 100:.1f}%")

    ex: str | None = None
    if quote is not None:
        bid_l, ask_l = quote.yes_bid_size_usd or 0.0, quote.yes_ask_size_usd or 0.0
        prior = _prior_implied(conn, ev["id"], "h2h", "kalshi", team or name, None, now_iso)
        if bid_l >= CANARY_MIN_LIQ_USD and ask_l < CANARY_MIN_LIQ_USD:
            ex = f"Kalshi ${bid_l:,.0f} YES bid vs ${ask_l:,.0f} ask"
        elif prior is not None and quote.mid_probability - prior >= CANARY_MIN_MOVE:
            ex = f"Kalshi mid {prior * 100:.1f}%->{quote.mid_probability * 100:.1f}%"
    if ex is None and opp_quote is not None:
        bid_l, ask_l = opp_quote.yes_bid_size_usd or 0.0, opp_quote.yes_ask_size_usd or 0.0
        if ask_l >= CANARY_MIN_LIQ_USD and bid_l < CANARY_MIN_LIQ_USD:
            ex = f"Kalshi ${ask_l:,.0f} offered on opponent YES vs ${bid_l:,.0f} bid"
    if ex:
        evidence.append(ex)
    if len(evidence) < CANARY_MIN_SOURCES:
        return []

    static = 0
    for bkey, bm in retail.items():
        o = _outcome(bm, mkey, name, point)
        prior = (_prior_implied(conn, ev["id"], mkey, bkey, name, point, now_iso)
                 if o is not None else None)
        if prior is None:
            continue
        if abs(1.0 / american_to_decimal(o["price"]) - prior) >= CANARY_RETAIL_STATIC:
            return []
        static += 1
    if not static:
        return []
    return [*evidence, f"{static} retail book(s) static"]


def _raw_candidates(conn: sqlite3.Connection, ev: dict, league: str,
                    kalshi: dict[tuple[str, str], KalshiQuote], stats: ScanStats,
                    rejections: list[dict], now: datetime,
                    ctx: GameContext) -> list[dict[str, Any]]:
    """Priced +EV candidates for one event, before event-level sizing.

    Fair probability = Bayesian blend of the de-vigged sharp anchor, the
    Kalshi mid (moneyline only) and the fundamental model (h2h/spreads).
    """
    event_name = f"{ev['away_team']} @ {ev['home_team']}"
    now_iso = db.iso(now)
    sharp, retail = split_books(ev)
    bench_key = next((k for k in SHARP_PRIORITY if k in sharp), None)
    out: list[dict[str, Any]] = []

    def reject(label: str, offered: int, reason: str) -> None:
        rejections.append({"event": event_name, "selection": label,
                           "offered_american": offered, "reason": reason})

    for mkey in BOOK_MARKETS:
        bench = _market(sharp[bench_key], mkey) if bench_key else None
        if not bench or len(bench) != 2:
            stats.no_benchmark += 1
            continue
        for side, other in ((bench[0], bench[1]), (bench[1], bench[0])):
            name, point = side["name"], side.get("point") if mkey != "h2h" else None
            label = display_selection(mkey, name, point)
            offers: list[tuple[str, dict, list[dict]]] = []
            for bkey, bm in retail.items():
                outcomes = _market(bm, mkey) or []
                same_name = [o for o in outcomes if o["name"] == name]
                exact = [o for o in same_name if mkey == "h2h" or o.get("point") == point]
                stats.point_mismatch += len(same_name) - len(exact)
                if exact:
                    offers.append((bkey, exact[0], outcomes))
            stats.retail_quotes += len(offers)
            eligible = [o for o in offers if validate_odds_floor(int(o[1]["price"]))]
            stats.floor_rejected += len(offers) - len(eligible)
            if offers and not eligible:
                best_bad = max(offers, key=lambda o: american_to_decimal(o[1]["price"]))
                reject(label, int(best_bad[1]["price"]), "All retail prices worse than -130")

            bench_dec = (american_to_decimal(side["price"]), american_to_decimal(other["price"]))
            # Power is the primary sharp de-vig; odds-ratio is reported as a comparator.
            p_pow = devig_power(*bench_dec)[0]
            p_or = devig_odds_ratio(*bench_dec)[0]
            pin_fair = p_pow
            team = normalize_team(name, league) if mkey == "h2h" else None
            quote = kalshi.get((ev["id"], team)) if team else None
            opp_team = normalize_team(other["name"], league) if mkey == "h2h" else None
            opp_quote = kalshi.get((ev["id"], opp_team)) if opp_team else None
            kalshi_mid, liq_method = exchange_probability(quote)
            model_prob, side_margin = ctx.model_prob_for(mkey, name, point)
            fair = compute_bayesian_true_prob(pin_fair, kalshi_mid, model_prob)[0]
            gates = model_gate_reasons(pin_fair, model_prob, ctx.key_questionable)
            if kalshi_mid is not None and abs(pin_fair - kalshi_mid) > DIVERGENCE_LIMIT:
                gates.append(DIVERGENCE_REASON)
            if liq_method == LIQ_ONE_SIDED and one_sided_depth(quote) < ONE_SIDED_HITL_LIQ_USD:
                gates.append(f"One-sided synthetic exchange line on "
                             f"${one_sided_depth(quote):,.0f} depth "
                             f"(< ${ONE_SIDED_HITL_LIQ_USD:,.0f})")
            canary = detect_exchange_canary(conn, ev, mkey, name, point, team, quote,
                                            opp_quote, now_iso)
            move = _line_move(conn, ev, mkey, name, point, now) if point is not None else None
            if move is not None and abs(move) > LINE_MOVE_LIMIT:
                gates.append(f"Line moved {move:+g} pts within 2h of tipoff")

            def thesis(offered_dec: float, _pin: float = pin_fair, _k: float | None = kalshi_mid,
                       _m: float | None = model_prob, _sm: float | None = side_margin,
                       _name: str = name, _label: str = label,
                       _canary: list[str] = canary, _lm: str | None = liq_method) -> str:
                text = compute_bayesian_true_prob(_pin, _k, _m, context={
                    "offered_prob": 1.0 / offered_dec,
                    "anchor": SHARP_BOOKS[bench_key],              # type: ignore[index]
                    "side": _label, "model_side_margin": _sm,
                    "model_drivers": ctx.drivers_for(_name) if _m is not None else []})[2]
                if _lm == LIQ_ILLIQUID:
                    text += " | Kalshi book ignored (ILLIQUID_NOISE: both sides < $250)"
                elif _lm == LIQ_ONE_SIDED:
                    text += " | Kalshi priced as ONE_SIDED_SYNTHETIC (4% synthetic hold)"
                if _canary:
                    text += f" | {CANARY_FLAG}: " + ", ".join(_canary)
                return text

            base = {
                "event_id": ev["id"], "event_name": event_name, "league": league,
                "sport": ev.get("sport_key"), "commence_time": ev["commence_time"],
                "benchmark_source": SHARP_BOOKS[bench_key],   # type: ignore[index]
                "benchmark_american": int(side["price"]),
                "benchmark_opp_american": int(other["price"]),
                "pin_fair": pin_fair, "p_power": p_pow, "p_odds_ratio": p_or, "fair": fair,
                "quote": quote, "gates": gates, "line_move": move,
                "model_prob": model_prob, "model_side_margin": side_margin, "ctx": ctx,
                "exchange_prob": kalshi_mid, "liq_method": liq_method, "canary": canary,
            }
            if eligible:
                bkey, best, book_outcomes = max(
                    eligible, key=lambda o: american_to_decimal(o[1]["price"]))
                offered = int(best["price"])
                dec = american_to_decimal(offered)
                ev_pct = calculate_ev(fair, dec) * 100
                if ev_pct < MIN_EV_PERCENT:
                    stats.below_min_ev += 1
                else:
                    other_book = next((o for o in book_outcomes if o is not best
                                       and (mkey != "spreads" or o.get("point") == -point)
                                       and (mkey != "totals" or o.get("point") == point)), None)
                    out.append({**base, "market_type": mkey, "selection": name, "point": point,
                                "bookmaker": RETAIL_BOOKS[bkey], "offered_american": offered,
                                "offered_decimal": dec, "ev_percent": ev_pct,
                                "thesis": thesis(dec), "extra_gates": _sharp_only_gate(pin_fair, dec),
                                "retail_novig": devig_power(dec, american_to_decimal(
                                    other_book["price"]))[0] if other_book else None})

            # Kalshi as execution venue: buy YES at the ask, net of taker fee.
            if quote and quote.yes_ask and 0.0 < quote.yes_ask < 1.0:
                net_dec = kalshi_net_odds(quote.yes_ask)
                k_american = decimal_to_american(net_dec)
                k_label = display_selection("kalshi_binary", name, None)
                if not validate_odds_floor(k_american):
                    stats.floor_rejected += 1
                    reject(k_label, k_american, "Kalshi net price (ask + fee) worse than -130")
                    continue
                ev_pct = calculate_ev(fair, net_dec) * 100
                if ev_pct < MIN_EV_PERCENT:
                    stats.below_min_ev += 1
                    continue
                out.append({**base, "market_type": "kalshi_binary", "selection": team or name,
                            "point": None, "bookmaker": "Kalshi", "offered_american": k_american,
                            "offered_decimal": net_dec, "ev_percent": ev_pct,
                            "thesis": thesis(net_dec), "retail_novig": None,
                            "extra_gates": _sharp_only_gate(pin_fair, net_dec)})
    return out


def _sharp_only_gate(sharp_prob: float, offered_dec: float) -> list[str]:
    """HITL when the price has no edge against the sharp anchor by itself.

    The blend can manufacture EV from the model alone; that must be a human
    call, never an automatic approval.
    """
    sharp_ev = calculate_ev(sharp_prob, offered_dec) * 100
    if sharp_ev > 0:
        return []
    return [f"Model-driven edge: sharp-only EV {sharp_ev:+.2f}% (price worse than anchor)"]


def game_analysis(ctx: GameContext, selection: str, model_prob: float | None,
                  side_margin: float | None, thesis: str) -> dict[str, Any]:
    """Serializable fundamental-model breakdown for one pick (JSON + ledger + cards)."""
    p = ctx.projection
    inj = [f"{cid.split('_', 1)[1]} {r['player']} ({r.get('position') or '?'}) {r['status']}"
           + (" [KEY]" if r.get("is_key") else "")
           for cid, rows in ctx.injuries.items() for r in rows
           if str(r.get("status", "")).lower() not in ("active",)][:8]
    out: dict[str, Any] = {
        "available": p is not None,
        "home_id": ctx.home_id, "away_id": ctx.away_id,
        "model_probability": None if model_prob is None else round(model_prob, 6),
        "selection_margin": None if side_margin is None else round(side_margin, 2),
        "rest": {"home": ctx.rest_home.label(), "away": ctx.rest_away.label()},
        "injuries": inj, "key_questionable": ctx.key_questionable,
        "notes": ctx.notes, "value_thesis": thesis,
    }
    if p:
        out.update({
            "projected_home_spread": p["projected_home_spread"],
            "projected_margin_home": p["projected_margin"],
            "home_win_prob": round(p["home_win_prob"], 6), "sigma": p["sigma"],
            "rating_label": p["rating_label"], "home_rating": p["home_rating"],
            "away_rating": p["away_rating"], "rating_delta": p["rating_delta"],
            "base_margin": p["base_margin"], "expected_pace": p.get("expected_pace"),
            "adjustments": [f"{label} {pts:+.1f}" for label, pts in p["adjustments"]],
            "games": list(p["games"]), "ratings_updated_at": p.get("updated_at"),
        })
    return out


def _finalize(conn: sqlite3.Connection, raw: list[dict[str, Any]], stats: ScanStats,
              rejections: list[dict], bankroll: float, origin: str, now_iso: str
              ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Event-capped sizing (highest EV first) -> (json opportunities, ledger rows)."""
    opps, rows = [], []
    by_event: dict[str, list[dict[str, Any]]] = {}
    for c in raw:
        c["pick_id"] = make_pick_id(c["event_id"], c["market_type"], c["selection"], c["point"])
        by_event.setdefault(c["event_id"], []).append(c)

    for event_id, cands in by_event.items():
        cands.sort(key=lambda c: -c["ev_percent"])
        scan_ids = [c["pick_id"] for c in cands]
        allocated = 0.0
        for c in cands:
            existing = db.open_event_exposure(conn, event_id, exclude=scan_ids) + allocated
            units, status, note = size_position(c["fair"], c["offered_decimal"], existing,
                                                bankroll)
            label = display_selection(c["market_type"], c["selection"], c["point"])
            if status == HITL_REJECTED:
                stats.event_cap_rejected += 1
                rejections.append({"event": c["event_name"], "selection": label,
                                   "offered_american": c["offered_american"], "reason": note})
                continue
            allocated += units
            gates = [*c["gates"], *c.get("extra_gates", [])]
            reasons = [r for r in [note, *gates] if r]
            if gates:
                status = HITL_PENDING
            if status == HITL_PENDING:
                stats.pending_hitl += 1
            stats.candidates += 1
            reason = "; ".join(reasons) or None
            q: KalshiQuote | None = c["quote"]
            raw_units = raw_quarter_kelly_units(c["fair"], c["offered_decimal"], bankroll)
            analysis = game_analysis(c["ctx"], c["selection"], c["model_prob"],
                                     c["model_side_margin"], c["thesis"])

            rows.append({
                "pick_id": c["pick_id"], "created_at": now_iso, "commence_time": c["commence_time"],
                "sport": c["sport"], "event_id": event_id, "event_name": c["event_name"],
                "market_type": c["market_type"], "selection": c["selection"],
                "line_point": c["point"], "bookmaker": c["bookmaker"],
                "offered_american": c["offered_american"],
                "offered_decimal": round(c["offered_decimal"], 6),
                "benchmark_source": c["benchmark_source"],
                "benchmark_american": c["benchmark_american"],
                "fair_prob": round(c["fair"], 6),
                "kalshi_mid_prob": q.mid_probability if q else None,
                "ev_percent": round(c["ev_percent"], 3), "kelly_units": units,
                "hitl_status": status, "hitl_reason": reason, "result": "UNSETTLED",
                "origin": origin,
                "model_prob": None if c["model_prob"] is None else round(c["model_prob"], 6),
                "analysis": json.dumps(analysis, separators=(",", ":")),
            })
            opp: dict[str, Any] = {
                "event": c["event_name"], "selection": label,
                "market_type": c["market_type"], "offered_american": c["offered_american"],
                "benchmark_source": c["benchmark_source"],
                "benchmark_american": c["benchmark_american"],
                "fair_probability": round(c["fair"], 6),
                "ev_percent": round(c["ev_percent"], 2), "quarter_kelly_units": units,
                "hitl_status": status, "hitl_reason": reason,
                "pick_id": c["pick_id"], "event_id": event_id, "sport": c["sport"],
                "league": c["league"], "bet_type": "single", "retail_book": c["bookmaker"],
                "benchmark_opp_american": c["benchmark_opp_american"],
                "fair_american": decimal_to_american(1.0 / c["fair"]),
                "pinnacle_fair_probability": round(c["pin_fair"], 6),
                "fair_probability_power": round(c["p_power"], 6),
                "fair_probability_odds_ratio": round(c["p_odds_ratio"], 6),
                "raw_quarter_kelly_units": round(raw_units, 4),
                "commence_time": c["commence_time"],
                "sharp_fair_american": decimal_to_american(1.0 / c["pin_fair"]),
                "offered_implied_probability": round(1.0 / c["offered_decimal"], 6),
                "value_thesis": c["thesis"],
                "confidence": "BOOSTED" if c["canary"] else "STANDARD",
                "model": analysis,
            }
            if c["canary"]:
                opp["signals"] = [CANARY_FLAG]
                opp["canary_evidence"] = c["canary"]
            if c["point"] is not None:
                opp["line_point"] = c["point"]
            if c["line_move"] is not None:
                opp["line_move_points"] = c["line_move"]
            if c["retail_novig"] is not None:
                opp["retail_novig_probability"] = round(c["retail_novig"], 6)
            if q:
                opp.update({"exchange_source": "Kalshi", "exchange_ticker": q.ticker,
                            "exchange_probability": q.mid_probability,
                            "exchange_yes_bid": q.yes_bid, "exchange_yes_ask": q.yes_ask,
                            "exchange_volume": q.volume,
                            "exchange_divergence": round(abs(c["pin_fair"] - q.mid_probability), 6),
                            "exchange_yes_bid_size_usd": q.yes_bid_size_usd,
                            "exchange_yes_ask_size_usd": q.yes_ask_size_usd,
                            "exchange_pricing_method": c["liq_method"],
                            "exchange_fair_probability": None if c["exchange_prob"] is None
                            else round(c["exchange_prob"], 6)})
            opps.append(opp)
    return opps, rows


# ==========================================================================
# Mock feeds (exact wire formats)
# ==========================================================================
def _bm(key: str, h2h: list[tuple[str, int]] | None = None,
        spreads: list[tuple[str, int, float]] | None = None,
        totals: list[tuple[str, int, float]] | None = None) -> dict:
    markets = []
    if h2h:
        markets.append({"key": "h2h", "outcomes": [{"name": n, "price": p} for n, p in h2h]})
    for mkey, rows in (("spreads", spreads), ("totals", totals)):
        if rows:
            markets.append({"key": mkey, "outcomes": [{"name": n, "price": p, "point": pt}
                                                      for n, p, pt in rows]})
    return {"key": key, "title": {**SHARP_BOOKS, **RETAIL_BOOKS}.get(key, key),
            "last_update": "2026-01-01T00:00:00Z", "markets": markets}


def build_mock_feeds(now: datetime) -> tuple[dict[str, list[dict]], dict[str, list[dict]],
                                             list[dict[str, Any]]]:
    """(odds events per sport, Kalshi markets per series, prior sharp snapshots).

    Scenarios: approvals; -130 floor rejects; spread/total point mismatch;
    Pinnacle/Kalshi divergence; 2-point spread move < 2h to tip; uncapped
    Kelly > 2u; event-cap rejection; Kalshi binary (net of fee) candidate;
    legacy cent-priced Kalshi market; unmatched Kalshi event; no-benchmark event.
    """
    soon = db.iso(now + timedelta(minutes=90))
    later = db.iso(now + timedelta(hours=5))
    sunday = (now + timedelta(days=(6 - now.weekday()) % 7 or 7)).replace(
        hour=18, minute=0, second=0, microsecond=0)
    nfl_t = db.iso(sunday)

    def ev(i: str, sport: str, home: str, away: str, t: str, books: list[dict]) -> dict:
        return {"id": i, "sport_key": sport, "sport_title": SPORT_TO_LEAGUE[sport],
                "commence_time": t, "home_team": home, "away_team": away, "bookmakers": books}

    nba = [
        ev("mock-nba-bos-mil", "basketball_nba", "Milwaukee Bucks", "Boston Celtics", soon, [
            _bm("pinnacle", [("Boston Celtics", -135), ("Milwaukee Bucks", 125)],
                [("Boston Celtics", -110, -4.5), ("Milwaukee Bucks", -110, 4.5)]),
            _bm("draftkings", [("Boston Celtics", -120), ("Milwaukee Bucks", 105)],
                [("Boston Celtics", -105, -4.5), ("Milwaukee Bucks", -115, 4.5)]),
            _bm("fanduel", [("Boston Celtics", -128), ("Milwaukee Bucks", 108)],
                [("Boston Celtics", -125, -4.5), ("Milwaukee Bucks", 105, 4.5)]),
        ]),
        ev("mock-nba-den-phx", "basketball_nba", "Phoenix Suns", "Denver Nuggets", later, [
            _bm("pinnacle", [("Denver Nuggets", -165), ("Phoenix Suns", 150)],
                totals=[("Over", -110, 229.5), ("Under", -110, 229.5)]),
            _bm("draftkings", [("Denver Nuggets", -150), ("Phoenix Suns", 130)],
                totals=[("Over", 104, 229.5), ("Under", -124, 229.5)]),
            _bm("fanduel", [("Denver Nuggets", -145), ("Phoenix Suns", 125)],
                totals=[("Over", 110, 230.5), ("Under", -132, 230.5)]),
            _bm("betmgm", [("Denver Nuggets", -155), ("Phoenix Suns", 160)]),
        ]),
        ev("mock-nba-lal-gsw", "basketball_nba", "Golden State Warriors", "Los Angeles Lakers",
           later, [
               _bm("pinnacle", [("Los Angeles Lakers", 105), ("Golden State Warriors", -115)]),
               _bm("draftkings", [("Los Angeles Lakers", 115), ("Golden State Warriors", -135)]),
               _bm("fanduel", [("Los Angeles Lakers", 110), ("Golden State Warriors", -130)]),
           ]),
    ]
    nfl = [
        ev("mock-nfl-buf-kc", "americanfootball_nfl", "Kansas City Chiefs", "Buffalo Bills",
           nfl_t, [
               _bm("pinnacle", [("Buffalo Bills", 120), ("Kansas City Chiefs", -130)]),
               _bm("williamhill_us", [("Buffalo Bills", 140), ("Kansas City Chiefs", -165)]),
               _bm("draftkings", [("Buffalo Bills", 125), ("Kansas City Chiefs", -145)]),
           ]),
        ev("mock-nfl-det-gb", "americanfootball_nfl", "Green Bay Packers", "Detroit Lions",
           nfl_t, [
               _bm("pinnacle", [("Detroit Lions", 110), ("Green Bay Packers", -120)],
                   [("Detroit Lions", -108, 1.5), ("Green Bay Packers", -112, -1.5)]),
               _bm("draftkings", [("Detroit Lions", 145), ("Green Bay Packers", -170)],
                   [("Detroit Lions", 105, 1.5), ("Green Bay Packers", -128, -1.5)]),
               _bm("fanduel", [("Detroit Lions", 120), ("Green Bay Packers", -140)],
                   [("Detroit Lions", 100, 2.5), ("Green Bay Packers", -135, -2.5)]),
           ]),
        ev("mock-nfl-pit-bal", "americanfootball_nfl", "Baltimore Ravens", "Pittsburgh Steelers",
           nfl_t, [_bm("draftkings", [("Pittsburgh Steelers", 150), ("Baltimore Ravens", -175)])]),
    ]

    def code(t: str, away: str, home: str, league: str) -> str:
        d = (db.parse_ts(t) - timedelta(hours=5)).date()       # US Eastern date
        return (f"{d.strftime('%y')}{d.strftime('%b').upper()}{d.strftime('%d')}"
                f"{abbreviation(away, league)}{abbreviation(home, league)}")

    def km(series: str, ev_code: str, abbr: str, sub: str, bid: float, ask: float, vol: float,
           legacy: bool = False, sizes: tuple[float, float] | None = None) -> dict:
        m = {"ticker": f"{series}-{ev_code}-{abbr}", "event_ticker": f"{series}-{ev_code}",
             "title": f"{sub} wins", "yes_sub_title": sub, "status": "active"}
        if legacy:
            m.update({"yes_bid": round(bid * 100), "yes_ask": round(ask * 100), "volume": int(vol)})
        else:
            m.update({"yes_bid_dollars": f"{bid:.4f}", "yes_ask_dollars": f"{ask:.4f}",
                      "volume_fp": f"{vol:.2f}"})
        if sizes:                                    # contracts at best YES bid / ask
            m.update({"yes_bid_size_fp": f"{sizes[0]:.2f}", "yes_ask_size_fp": f"{sizes[1]:.2f}"})
        return m

    c1 = code(soon, "Boston Celtics", "Milwaukee Bucks", "NBA")
    c2 = code(later, "Denver Nuggets", "Phoenix Suns", "NBA")
    c3 = code(later, "Los Angeles Lakers", "Golden State Warriors", "NBA")
    c4 = code(later, "Utah Jazz", "Houston Rockets", "NBA")
    n1 = code(nfl_t, "Buffalo Bills", "Kansas City Chiefs", "NFL")
    n2 = code(nfl_t, "Detroit Lions", "Green Bay Packers", "NFL")
    kalshi = {
        "KXNBAGAME": [
            # Deep two-sided, 2c spread -> TWO_SIDED_LIQUID
            km("KXNBAGAME", c1, "BOS", "Boston", 0.55, 0.57, 15234, sizes=(4000, 3500)),
            km("KXNBAGAME", c1, "MIL", "Milwaukee", 0.43, 0.45, 15234, sizes=(3500, 4000)),
            # PHX: $880 bid vs $92 ask -> ONE_SIDED_SYNTHETIC (>= $500, no gate) + canary
            km("KXNBAGAME", c2, "DEN", "Denver", 0.54, 0.56, 9120, sizes=(150, 1800)),
            km("KXNBAGAME", c2, "PHX", "Phoenix", 0.44, 0.46, 9120, sizes=(2000, 200)),
            # Legacy cent prices, no size fields -> MID_UNSIZED
            km("KXNBAGAME", c3, "LAL", "Los Angeles L", 0.41, 0.42, 8200, legacy=True),
            km("KXNBAGAME", c3, "GSW", "Golden State", 0.57, 0.59, 8200, legacy=True),
            km("KXNBAGAME", c4, "UTA", "Utah", 0.14, 0.15, 144, sizes=(100, 80)),
            km("KXNBAGAME", c4, "HOU", "Houston", 0.84, 0.86, 943, sizes=(20, 25)),
        ],
        "KXNFLGAME": [
            # BUF: $360 ask vs $13 bid -> ONE_SIDED_SYNTHETIC < $500 -> PENDING_CAPTAIN_HZ
            km("KXNFLGAME", n1, "BUF", "Buffalo", 0.43, 0.45, 210_400, sizes=(30, 800)),
            km("KXNFLGAME", n1, "KC", "Kansas City", 0.55, 0.57, 210_400, sizes=(800, 30)),
            # $20 stale orders both sides -> ILLIQUID_NOISE (Kalshi ignored)
            km("KXNFLGAME", n2, "DET", "Detroit", 0.46, 0.48, 96_300, sizes=(40, 30)),
            km("KXNFLGAME", n2, "GB", "Green Bay", 0.52, 0.54, 96_300, sizes=(30, 40)),
        ],
    }
    opened = db.iso(now - timedelta(hours=6))
    history = [
        {"event_id": "mock-nba-bos-mil", "timestamp": opened, "source": "pinnacle",
         "market_type": "spreads", "selection": sel, "point": pt, "american_odds": -110,
         "implied_prob": round(1 / american_to_decimal(-110), 6)}
        for sel, pt in (("Boston Celtics", -2.5), ("Milwaukee Bucks", 2.5))
    ]
    # Lead-lag canary: Pinnacle moved PHX +170 -> +150 while BetMGM sat at +160.
    history += [
        {"event_id": "mock-nba-den-phx", "timestamp": opened, "source": src,
         "market_type": "h2h", "selection": sel, "point": None, "american_odds": price,
         "implied_prob": round(1 / american_to_decimal(price), 6)}
        for src, sel, price in (("pinnacle", "Phoenix Suns", 170),
                                ("pinnacle", "Denver Nuggets", -190),
                                ("betmgm", "Phoenix Suns", 160),
                                ("betmgm", "Denver Nuggets", -155))
    ]
    return {"basketball_nba": nba, "americanfootball_nfl": nfl}, kalshi, history


def build_mock_model_fixture(now: datetime) -> dict[str, Any]:
    """In-memory ratings / rest / injuries for the mock slate (never persisted).

    Ratings are solved so that: BOS ML and BUF ML agree with the sharp line
    within 5% (APPROVED path); MIL +4.5 diverges from the sharp spread price
    (model gate); DEN is on a back-to-back; BUF is off a bye; a LAL starter is
    Questionable (injury gate).
    """
    ts = db.iso(now)
    nba = {"NBA_BOS": 5.0, "NBA_MIL": -3.7, "NBA_DEN": 7.0, "NBA_PHX": -6.5,
           "NBA_LAL": 1.5, "NBA_GSW": -1.4}
    nfl = {"NFL_BUF": 2.16, "NFL_KC": 3.0, "NFL_DET": 4.0, "NFL_GB": 2.83}
    ratings: dict[str, Any] = {"_league_avg_pace": 99.5}
    for cid, net in nba.items():
        ratings[cid] = {"sport": "NBA", "games": 10, "adj_net_rtg": net, "net_rtg": net,
                        "pace": 99.5, "updated_at": ts}
    for cid, srs in nfl.items():
        ratings[cid] = {"sport": "NFL", "games": 8, "srs": srs, "updated_at": ts}
    return {
        "ratings": ratings,
        "rest": {"NBA_BOS": 1, "NBA_MIL": 1, "NBA_DEN": 0, "NBA_PHX": 2, "NBA_LAL": 1,
                 "NBA_GSW": 1, "NFL_BUF": 13, "NFL_KC": 6, "NFL_DET": 6, "NFL_GB": 6},
        "injuries": {
            "NBA_LAL": [{"player": "Mock Starter (LAL)", "position": "F",
                         "status": "Questionable", "is_key": 1}],
            "NBA_GSW": [{"player": "Mock Reserve (GSW)", "position": "G", "status": "Out",
                         "is_key": 0}],
        },
    }


# ==========================================================================
# Orchestration
# ==========================================================================
def run_scan(conn: sqlite3.Connection, sports: tuple[str, ...] = ("basketball_nba",),
             mock: bool = False, output_path: Path = OUTPUT_PATH,
             now: datetime | None = None) -> dict[str, Any]:
    """Fetch (or mock) both sources, price, size, persist, and write the scan JSON."""
    now = now or db.utcnow()
    now_iso = db.iso(now)
    load_env()
    bankroll = bankroll_units()
    origin = db.ORIGIN_MOCK if mock else db.ORIGIN_LIVE
    stats = ScanStats()
    rejections: list[dict] = []
    raw: list[dict[str, Any]] = []

    fixture: dict[str, Any] | None = None
    if mock:
        odds_by_sport, kalshi_by_series, history = build_mock_feeds(now)
        fixture = build_mock_model_fixture(now)
        known = {r["event_id"] for r in conn.execute(
            "SELECT DISTINCT event_id FROM market_snapshots WHERE event_id LIKE 'mock-%'")}
        db.insert_snapshots(conn, [h for h in history if h["event_id"] not in known])

    for sport in sports:
        league = SPORT_TO_LEAGUE.get(sport)
        if not league:
            raise IngestionError(f"Unsupported sport {sport!r}; known: {list(SPORT_TO_LEAGUE)}")
        series = LEAGUE_TO_KALSHI_GAME_SERIES[league]
        if mock:
            events = odds_by_sport.get(sport, [])
            quotes = [q for q in map(parse_kalshi_market, kalshi_by_series.get(series, [])) if q]
            stats.quota[sport] = "mock"
        else:
            fetch = fetch_the_odds_api(conn, sport)
            events = fetch.events
            stats.quota[sport] = "cache" if fetch.from_cache else fetch.quota_remaining
            try:
                quotes = fetch_kalshi_markets(conn, series)
            except IngestionError as exc:
                log.warning("Kalshi unavailable for %s (%s); continuing books-only", series, exc)
                quotes = []
        aligned, unmatched = align_kalshi(events, quotes, league)
        stats.events += len(events)
        stats.kalshi_markets += len(quotes)
        stats.kalshi_matched += len(aligned)
        stats.kalshi_unmatched += unmatched
        if mock or not fetch.from_cache:
            stats.snapshots += db.insert_snapshots(conn, snapshot_rows(events, aligned, now_iso))
        for ev in events:
            ctx = build_game_context(None if mock else conn, league, ev["home_team"],
                                     ev["away_team"], ev["commence_time"], fixture)
            if ctx.projection is not None:
                stats.model_games += 1
            raw += _raw_candidates(conn, ev, league, aligned, stats, rejections, now, ctx)

    opps, rows = _finalize(conn, raw, stats, rejections, bankroll, origin, now_iso)
    stats.ledger_inserted, refused = Ledger(conn, origin).record_positions(rows)
    if refused:
        bad = dict(refused)
        for o in opps:
            if o["pick_id"] in bad:
                rejections.append({"event": o["event"], "selection": o["selection"],
                                   "offered_american": o["offered_american"],
                                   "reason": f"Ledger refused: {bad[o['pick_id']]}"})
        opps = [o for o in opps if o["pick_id"] not in bad]
    opps.sort(key=lambda o: (-o["ev_percent"], o["event"]))
    scan = {
        "timestamp": now_iso,
        "slate_date": now.date().isoformat(),
        "operating_authority": "Captain HZ",
        "mode": "MOCK" if mock else "LIVE",
        "bankroll_units": bankroll,
        "opportunities": opps,
        "rejections": rejections,
        "ingestion": {"sources": ["The-Odds-API", "Kalshi"], "sports": list(sports),
                      "stats": vars(stats)},
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(scan, indent=2) + "\n", encoding="utf-8")
    return scan


# ==========================================================================
# Polymarket (Polygon) exchange microstructure  -- Dubach (2026)
# ==========================================================================
# "The Anatomy of a Decentralized Prediction Market: Microstructure Evidence
# from the Polymarket Order Book".
#
# 1. Trade direction comes ONLY from on-chain CTF Exchange ``OrderFilled``
#    events. The public WebSocket ``change_side`` / quote-side heuristics agree
#    with on-chain ground truth only ~59% of the time and flip the sign of 60-67%
#    of effective spreads and Kyle's lambda, so they are rejected outright.
#    Asset id 0 is the USDC collateral leg:
#        makerAssetId == 0  -> maker pays USDC (maker buys)  -> taker SELLS, sign -1
#        takerAssetId == 0  -> taker pays USDC (taker buys)  -> taker BUYS,  sign +1
# 2. Depth is aggregated over all 10 L2 levels (L1/L10 cumulative-depth median
#    ~0.14: closer to a uniform geometric grid than a top-heavy book).
# 3. Self-matches (maker == taker) and reciprocal maker/taker round trips inside
#    the 128-block finality buffer are stripped as wash volume.

COLLATERAL_ASSET_ID = 0
POLYMARKET_DEPTH_LEVELS = 10
WASH_FINALITY_BLOCKS = 128
POLYMARKET_AMOUNT_SCALE = 1_000_000          # USDC and CTF outcome tokens: 6 decimals
_FEED_SIDE_KEYS = ("change_side", "side", "aggressor_side", "taker_side")


class FeedSideHeuristicError(ValueError):
    """Raised when a trade's direction would have to come from a feed-side heuristic."""


@dataclass(frozen=True)
class OnChainFill:
    """One CTF Exchange ``OrderFilled`` log, amounts in raw 6-decimal base units."""
    tx_hash: str
    log_index: int
    block_number: int
    maker: str
    taker: str
    maker_asset_id: int
    taker_asset_id: int
    maker_amount: int
    taker_amount: int
    fee: int = 0

    @property
    def sign(self) -> int:
        return trade_sign_from_fill(self)

    @property
    def token_id(self) -> int:
        """The outcome token traded (the non-collateral leg)."""
        return self.taker_asset_id if self.maker_asset_id == COLLATERAL_ASSET_ID \
            else self.maker_asset_id

    @property
    def usdc(self) -> float:
        raw = self.maker_amount if self.maker_asset_id == COLLATERAL_ASSET_ID \
            else self.taker_amount
        return raw / POLYMARKET_AMOUNT_SCALE

    @property
    def shares(self) -> float:
        raw = self.taker_amount if self.maker_asset_id == COLLATERAL_ASSET_ID \
            else self.maker_amount
        return raw / POLYMARKET_AMOUNT_SCALE

    @property
    def price(self) -> float:
        """USDC per outcome share."""
        return self.usdc / self.shares


def _asset_id(v: Any) -> int:
    return int(v, 16) if isinstance(v, str) and v.lower().startswith("0x") else int(v)


def parse_order_filled(event: dict[str, Any]) -> OnChainFill:
    """Decode an ``OrderFilled`` log (decoded-args dict, camelCase or snake_case).

    Anything without both asset ids (e.g. a WebSocket trade message carrying only
    ``change_side``) is refused: direction must come from the on-chain legs.
    """
    args = event.get("args", event)

    def pick(*names: str) -> Any:
        for n in names:
            if n in args:
                return args[n]
            if n in event:
                return event[n]
        return None

    maker_asset = pick("makerAssetId", "maker_asset_id")
    taker_asset = pick("takerAssetId", "taker_asset_id")
    if maker_asset is None or taker_asset is None:
        if any(k in event or k in args for k in _FEED_SIDE_KEYS):
            raise FeedSideHeuristicError(
                "feed-side trade direction rejected (Dubach 2026: ~59% sign agreement); "
                "supply on-chain OrderFilled makerAssetId/takerAssetId")
        raise ValueError("OrderFilled event missing makerAssetId/takerAssetId")
    fill = OnChainFill(
        tx_hash=str(pick("transactionHash", "tx_hash") or ""),
        log_index=int(pick("logIndex", "log_index") or 0),
        block_number=int(pick("blockNumber", "block_number")),
        maker=str(pick("maker")).lower(),
        taker=str(pick("taker")).lower(),
        maker_asset_id=_asset_id(maker_asset),
        taker_asset_id=_asset_id(taker_asset),
        maker_amount=int(pick("makerAmountFilled", "maker_amount")),
        taker_amount=int(pick("takerAmountFilled", "taker_amount")),
        fee=int(pick("fee") or 0),
    )
    trade_sign_from_fill(fill)                      # validates exactly one collateral leg
    if fill.maker_amount <= 0 or fill.taker_amount <= 0:
        raise ValueError("OrderFilled amounts must be positive")
    return fill


def trade_sign_from_fill(fill: OnChainFill) -> int:
    """Aggressor (taker) sign from the on-chain legs: +1 taker buy, -1 taker sell."""
    m0 = fill.maker_asset_id == COLLATERAL_ASSET_ID
    t0 = fill.taker_asset_id == COLLATERAL_ASSET_ID
    if m0 == t0:
        raise ValueError("exactly one of makerAssetId/takerAssetId must be the collateral (0)")
    return -1 if m0 else +1


def is_self_match(fill: OnChainFill) -> bool:
    return fill.maker.lower() == fill.taker.lower()


def filter_wash_trades(fills: list[OnChainFill], finality_blocks: int = WASH_FINALITY_BLOCKS
                       ) -> tuple[list[OnChainFill], list[OnChainFill]]:
    """Strip self-counterparty volume -> ``(clean, removed)``, input order kept.

    * Self-match: maker == taker.
    * Immediate round trip: two fills in the same outcome token whose maker and
      taker are swapped (A->B then B->A) within ``finality_blocks`` blocks. Both
      legs are removed; each fill pairs at most once (earliest partner first).
    """
    removed: set[int] = {i for i, f in enumerate(fills) if is_self_match(f)}
    order = sorted((i for i in range(len(fills)) if i not in removed),
                   key=lambda i: (fills[i].block_number, fills[i].log_index))
    for pos, i in enumerate(order):
        if i in removed:
            continue
        a = fills[i]
        for j in order[pos + 1:]:
            b = fills[j]
            if b.block_number - a.block_number > finality_blocks:
                break
            if j in removed:
                continue
            if b.token_id == a.token_id and b.maker == a.taker and b.taker == a.maker:
                removed.update((i, j))
                break
    clean = [f for k, f in enumerate(fills) if k not in removed]
    return clean, [f for k, f in enumerate(fills) if k in removed]


def onchain_vwap(fills: list[OnChainFill]) -> float | None:
    """Share-weighted average fill price (USDC notional / shares)."""
    shares = sum(f.shares for f in fills)
    return sum(f.usdc for f in fills) / shares if shares > 0 else None


def signed_volume(fills: list[OnChainFill]) -> float:
    """Net aggressor share volume (taker buys minus taker sells), on-chain signs only."""
    return sum(f.sign * f.shares for f in fills)


def effective_spread(fill: OnChainFill, midpoint: float) -> float:
    """Effective spread 2 * q * (p - m), q the on-chain aggressor sign (price units)."""
    if not 0.0 < midpoint < 1.0:
        raise ValueError(f"midpoint must be in (0, 1), got {midpoint}")
    return 2.0 * fill.sign * (fill.price - midpoint)


def volume_weighted_effective_spread(fills: list[OnChainFill],
                                     midpoints: list[float]) -> float | None:
    """Share-weighted mean effective spread; midpoints aligned 1:1 with fills."""
    if len(fills) != len(midpoints):
        raise ValueError("fills and midpoints must align")
    w = sum(f.shares for f in fills)
    if w <= 0:
        return None
    return sum(f.shares * effective_spread(f, m) for f, m in zip(fills, midpoints)) / w


# ---- L2 book --------------------------------------------------------------

@dataclass
class PolymarketBook:
    """L2 book: bids best (highest) first, asks best (lowest) first; (price, shares)."""
    bids: list[tuple[float, float]] = field(default_factory=list)
    asks: list[tuple[float, float]] = field(default_factory=list)

    @property
    def midpoint(self) -> float | None:
        if not self.bids or not self.asks:
            return None
        return 0.5 * (self.bids[0][0] + self.asks[0][0])


def parse_polymarket_book(payload: dict[str, Any]) -> PolymarketBook:
    """CLOB ``/book`` payload ({"bids": [{"price": "0.48", "size": "120"}], ...}).

    Levels are re-sorted best-first (the API lists bids ascending); empty or
    out-of-range levels are dropped.
    """
    def levels(side: str) -> list[tuple[float, float]]:
        out = []
        for lv in payload.get(side) or []:
            p, s = float(lv["price"]), float(lv["size"])
            if 0.0 < p < 1.0 and s > 0.0:
                out.append((p, s))
        return out

    return PolymarketBook(bids=sorted(levels("bids"), key=lambda x: -x[0]),
                          asks=sorted(levels("asks"), key=lambda x: x[0]))


def cumulative_depth(levels: list[tuple[float, float]],
                     n_levels: int = POLYMARKET_DEPTH_LEVELS, usd: bool = False) -> list[float]:
    """Cumulative depth L_1..L_n over the first ``n_levels`` (shares, or USD notional)."""
    out, run = [], 0.0
    for p, s in levels[:n_levels]:
        run += p * s if usd else s
        out.append(run)
    return out


def depth_ratio_l1_l10(levels: list[tuple[float, float]],
                       n_levels: int = POLYMARKET_DEPTH_LEVELS) -> float | None:
    """L_1 / L_n cumulative depth ratio (Dubach median ~0.14 for n = 10)."""
    cum = cumulative_depth(levels, n_levels)
    return cum[0] / cum[-1] if cum and cum[-1] > 0 else None


def available_liquidity_usd(book: PolymarketBook,
                            n_levels: int = POLYMARKET_DEPTH_LEVELS) -> tuple[float, float]:
    """(bid USD, ask USD) aggregated across all ``n_levels`` levels, not top-of-book."""
    bid = cumulative_depth(book.bids, n_levels, usd=True)
    ask = cumulative_depth(book.asks, n_levels, usd=True)
    return (bid[-1] if bid else 0.0, ask[-1] if ask else 0.0)


def book_vwap(book: PolymarketBook, side: str, shares: float,
              n_levels: int = POLYMARKET_DEPTH_LEVELS) -> tuple[float | None, float]:
    """Walk up to ``n_levels`` levels to fill ``shares`` -> ``(vwap, shares_filled)``.

    ``side="buy"`` consumes asks, ``side="sell"`` consumes bids. A partial fill
    returns the VWAP of what the 10-level book could absorb.
    """
    if shares <= 0:
        raise ValueError("shares must be positive")
    if side not in ("buy", "sell"):
        raise ValueError("side must be 'buy' or 'sell'")
    levels = book.asks if side == "buy" else book.bids
    left, cost = shares, 0.0
    for p, s in levels[:n_levels]:
        take = min(left, s)
        cost += take * p
        left -= take
        if left <= 0:
            break
    filled = shares - left
    return (cost / filled if filled > 0 else None), filled
