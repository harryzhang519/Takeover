"""Read models and Captain HZ actions behind the local web desk.

Everything here reads and writes ``ledger_picks`` through the same code paths
the CLI uses, so the browser can never do something ``run_pipeline.py`` would
refuse:

    approve      orchestrator.sign_off      pending -> APPROVED + Captain HZ sign-off
    decline      Ledger.decline_position    -> DECLINED (no stake, no exposure)
    undo         Ledger.undo_decision       APPROVED / DECLINED -> PENDING_CAPTAIN_HZ
    write-ups    writeup_generator          capper blurb, generated on approval only
    slate        open picks for today, grouped NBA / NFL (empty sports omitted)
    tier         free / premium card for a signed-off pick; locks at game time
    tracker      official picks only: APPROVED *and* signed off by Captain HZ
    discord      free and premium cards of signed-off open picks, written to OUTBOX

Tracker math (all in units):
    risked   = stakes of WIN + LOSS picks (a PUSH or VOID hands the stake back)
    net      = sum(net_units) over settled official picks
    ROI %    = net / risked * 100
    win rate = W / (W + L)
    record   = W-L-V, where V counts PUSH and VOID (CANCELLED: game off / DNP)
    CLV      = average closing line value % over settled picks that have one,
               and the share of them that beat the close (CLV > 0)
    splits   = by sport, by game lines vs player props, and by free vs premium
"""

from __future__ import annotations

import json
import logging
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any

from src.core import database as db
from src.core.engine import HITL_APPROVED, HITL_DECLINED, HITL_PENDING
from src.core.ledger import Ledger, undo_pick_decision
from src.core.orchestrator import OPERATOR, quantitative_filter_reasons, sign_off
from src.core.prop_model import PROP_MARKETS, is_prop_market
from src.core.tiers import (
    DEFAULT_TIER,
    TIER_LABELS,
    TIERS,
    effective_tier,
    normalize_tier,
    set_tier,
    tier_locked_reason,
)
from src.data.teams import SPORT_TO_LEAGUE
from templates import writeup_generator as wg
from templates.discord_templates import discord_post, render_tier_card

log = logging.getLogger("breeze.desk")

SPORT_ORDER: tuple[str, ...] = ("NBA", "NFL")
GAME_MARKET_LABELS = {"h2h": "Moneyline", "spreads": "Spread", "totals": "Total",
                      "kalshi_binary": "Moneyline (Kalshi)"}
SETTLED_RESULTS = ("WIN", "LOSS", "PUSH", "CANCELLED")
VOID_RESULTS = ("PUSH", "CANCELLED")
OFFICIAL_SQL = "hitl_status = 'APPROVED' AND signoff_at IS NOT NULL"
OPEN_SQL = "result = 'UNSETTLED' AND hitl_status IN ('APPROVED', 'PENDING_CAPTAIN_HZ')"
ORIGINS = (db.ORIGIN_LIVE, db.ORIGIN_MOCK)


class DeskError(ValueError):
    """An action the desk refuses (unknown pick, settled, failed mandate ...)."""

    def __init__(self, message: str, status: int = 409) -> None:
        super().__init__(message)
        self.status = status


# ==========================================================================
# Helpers
# ==========================================================================
def local_day_start(now: datetime) -> datetime:
    """Midnight of ``now``'s local calendar day (aware, local tz)."""
    local = now.astimezone()
    return local.replace(hour=0, minute=0, second=0, microsecond=0)


def league_of(p: dict[str, Any]) -> str:
    lg = SPORT_TO_LEAGUE.get(p.get("sport") or "")
    if lg:
        return lg
    m = PROP_MARKETS.get(p.get("market_type") or "")
    return m.league if m else "OTHER"


def _analysis(p: dict[str, Any]) -> dict[str, Any]:
    raw = p.get("analysis")
    if not raw:
        return {}
    try:
        out = json.loads(raw)
    except (TypeError, ValueError):
        return {}
    return out if isinstance(out, dict) else {}


def pick_label(p: dict[str, Any]) -> str:
    """Bet-slip wording: 'Celtics -3.5', 'Over 221.5', 'J. Tatum Over 27.5 PTS', 'Celtics ML'."""
    mt, sel, pt = p.get("market_type"), p.get("selection") or "", p.get("line_point")
    if is_prop_market(mt):
        line = "" if pt is None else f" {pt:g}"
        return f"{sel}{line} {PROP_MARKETS[mt].label}"
    if mt == "spreads" and pt is not None:
        return f"{sel} {pt:+g}"
    if mt == "totals" and pt is not None:
        return f"{sel} {pt:g}"
    return f"{sel} ML"


def pick_view(p: dict[str, Any], now: datetime | None = None) -> dict[str, Any]:
    """JSON-safe view of one ledger row for the browser."""
    now = now or db.utcnow()
    prop = is_prop_market(p.get("market_type"))
    a = _analysis(p)
    commence = p.get("commence_time")
    projection = a.get("projection")
    return {
        "pick_id": p["pick_id"], "origin": p.get("origin"), "league": league_of(p),
        "kind": "prop" if prop else "game", "market_type": p.get("market_type"),
        "market_label": (PROP_MARKETS[p["market_type"]].label if prop
                         else GAME_MARKET_LABELS.get(p.get("market_type") or "", "Line")),
        "event": p.get("event_name") or p.get("event_id"), "selection": p.get("selection"),
        "label": pick_label(p), "line_point": p.get("line_point"),
        "bookmaker": p.get("bookmaker"), "offered_american": p.get("offered_american"),
        "offered_decimal": p.get("offered_decimal"), "fair_prob": p.get("fair_prob"),
        "model_prob": p.get("model_prob"), "ev_percent": p.get("ev_percent"),
        "units": float(p.get("kelly_units") or 0.0), "hitl_status": p.get("hitl_status"),
        "hitl_reason": p.get("hitl_reason") or "", "signed": bool(p.get("signoff_at")),
        "signoff_at": p.get("signoff_at"), "commence_time": commence,
        "started": bool(commence) and db.parse_ts(commence) <= now,
        "projection": projection,
        "delta": (None if projection is None or p.get("line_point") is None
                  else round(float(projection) - float(p["line_point"]), 2)),
        "z": a.get("z"), "result": p.get("result"), "net_units": p.get("net_units"),
        "settled_at": p.get("settled_at"), "clv_percent": p.get("clv_percent"),
        "undoable": (p.get("result") == "UNSETTLED" and not p.get("dispatched_at")
                     and (p.get("hitl_status") == HITL_DECLINED
                          or (p.get("hitl_status") == HITL_APPROVED
                              and bool(p.get("signoff_at"))))),
        "writeup": p.get("discord_writeup"), "writeup_source": p.get("writeup_source"),
        "tier": effective_tier(p),
        "tier_locked": effective_tier(p) is not None and tier_locked_reason(p, now) is not None,
    }


def _sort_key(v: dict[str, Any]) -> tuple:
    return (v["signed"], v["commence_time"] or "", -(v["ev_percent"] or 0.0), v["pick_id"])


def _check_origin(origin: str) -> str:
    if origin not in ORIGINS:
        raise DeskError(f"unknown origin {origin!r}", 400)
    return origin


# ==========================================================================
# Slate
# ==========================================================================
def slate(conn: sqlite3.Connection, origin: str, now: datetime | None = None) -> dict[str, Any]:
    """Today's open picks grouped NBA / NFL; a sport with no picks is left out entirely."""
    now = now or db.utcnow()
    _check_origin(origin)
    since = db.iso(local_day_start(now))
    rows = db.fetch_picks(conn, f"origin = ? AND {OPEN_SQL} AND commence_time >= ?",
                          (origin, since))
    views = [with_post(pick_view(r, now)) for r in rows]
    sports = []
    for lg in SPORT_ORDER:
        mine = sorted((v for v in views if v["league"] == lg), key=_sort_key)
        if not mine:
            continue
        signed = [v for v in mine if v["signed"]]
        sports.append({
            "league": lg,
            "games": [v for v in mine if v["kind"] == "game"],
            "props": [v for v in mine if v["kind"] == "prop"],
            "queue": len(mine) - len(signed), "signed": len(signed),
            "signed_units": round(sum(v["units"] for v in signed), 2),
        })
    declined = db.fetch_picks(conn, "origin = ? AND hitl_status = ? AND result = 'UNSETTLED' "
                                    "AND commence_time >= ?", (origin, HITL_DECLINED, since))
    stale = conn.execute(f"SELECT COUNT(*) FROM ledger_picks WHERE origin = ? AND {OPEN_SQL} "
                         f"AND commence_time < ?", (origin, since)).fetchone()[0]
    signed_views = [v for v in views if v["signed"]]
    return {
        "origin": origin, "as_of": db.iso(now), "day_start": since, "sports": sports,
        "queue_count": sum(s["queue"] for s in sports),
        "signed_count": sum(s["signed"] for s in sports),
        "tier_counts": {t: sum(v["tier"] == t for v in signed_views) for t in TIERS},
        "stale_open": int(stale),
        "declined": [pick_view(r, now) for r in declined],
    }


# ==========================================================================
# Captain HZ actions
# ==========================================================================
def _require(conn: sqlite3.Connection, pick_id: str) -> dict[str, Any]:
    p = db.get_pick(conn, pick_id)
    if p is None:
        raise DeskError(f"unknown pick {pick_id}", 404)
    return p


def approve(conn: sqlite3.Connection, pick_id: str, now: datetime | None = None,
            tier: str | None = None) -> dict[str, Any]:
    """Captain HZ sign-off (same path as ``run_pipeline.py --approve``).

    ``tier`` picks the card: free (the default) or premium. Approving a pick
    that is already signed off only moves it when a different tier is named.
    """
    p = _require(conn, pick_id)
    try:
        wanted = normalize_tier(tier)
    except ValueError as exc:
        raise DeskError(str(exc), 400) from exc
    if p["signoff_at"] and p["hitl_status"] == HITL_APPROVED:
        if tier is not None and effective_tier(p) != wanted:
            return change_tier(conn, pick_id, wanted, now)
        if not p["discord_writeup"]:
            p = _ensure_writeup(conn, pick_id) or p
        return with_post(pick_view(p, now))
    if p["hitl_status"] not in (HITL_PENDING, HITL_APPROVED):
        raise DeskError(f"{pick_id} is {p['hitl_status']}; cannot sign off")
    try:
        p = sign_off(conn, pick_id, now=now, tier=wanted)
    except (KeyError, ValueError, PermissionError) as exc:
        raise DeskError(str(exc).strip("'\"")) from exc
    p = _ensure_writeup(conn, pick_id) or p
    return with_post(pick_view(p, now))


def change_tier(conn: sqlite3.Connection, pick_id: str, tier: str | None,
                now: datetime | None = None) -> dict[str, Any]:
    """Move a signed-off open pick between the free and premium cards (before game time)."""
    _require(conn, pick_id)
    try:
        p = set_tier(conn, pick_id, tier, now)
    except ValueError as exc:
        raise DeskError(str(exc), 400 if "unknown tier" in str(exc) else 409) from exc
    return with_post(pick_view(p, now))


def _ensure_writeup(conn: sqlite3.Connection, pick_id: str,
                    llm: wg.Llm | None = None) -> dict[str, Any] | None:
    """Write-up for a freshly approved pick. Never blocks or fails the approval itself."""
    try:
        return wg.attach_writeup(conn, pick_id, llm)
    except Exception as exc:                                  # noqa: BLE001
        log.warning("write-up for %s failed: %s", pick_id, exc)
        return None


def with_post(v: dict[str, Any]) -> dict[str, Any]:
    """Add the Discord post (header + write-up) to an approved, signed-off pick view."""
    if v["hitl_status"] == HITL_APPROVED and v["signed"]:
        v["post"] = discord_post(v, v.get("writeup"))
    return v


def regenerate_writeup(conn: sqlite3.Connection, pick_id: str, now: datetime | None = None,
                       llm: wg.Llm | None = None) -> dict[str, Any]:
    """Throw away the current write-up and generate a fresh one (Ollama, else template)."""
    _require(conn, pick_id)
    try:
        p = wg.attach_writeup(conn, pick_id, llm)
    except ValueError as exc:
        raise DeskError(str(exc)) from exc
    return with_post(pick_view(p, now))


def save_writeup(conn: sqlite3.Connection, pick_id: str, text: str,
                 now: datetime | None = None) -> dict[str, Any]:
    """Store Captain HZ's own edit of an approved pick's write-up."""
    _require(conn, pick_id)
    try:
        p = wg.save_custom_writeup(conn, pick_id, text)
    except ValueError as exc:
        bad_input = "empty" in str(exc) or "longer" in str(exc)
        raise DeskError(str(exc), 400 if bad_input else 409) from exc
    return with_post(pick_view(p, now))


def decline(conn: sqlite3.Connection, pick_id: str, now: datetime | None = None
            ) -> dict[str, Any]:
    """Captain HZ passes: DECLINED, out of the queue, the tracker and the drop."""
    p = _require(conn, pick_id)
    ts = db.iso(now or db.utcnow())
    try:
        p = Ledger(conn, p["origin"]).decline_position(pick_id, f"Declined by {OPERATOR} {ts}")
    except ValueError as exc:
        raise DeskError(str(exc)) from exc
    return pick_view(p, now)


def undo(conn: sqlite3.Connection, pick_id: str, now: datetime | None = None
         ) -> dict[str, Any]:
    """Pull an approved or declined open pick back into the Captain HZ queue."""
    _require(conn, pick_id)
    ts = db.iso(now or db.utcnow())
    try:
        p = undo_pick_decision(conn, pick_id, f"Undone by {OPERATOR} {ts}")
    except ValueError as exc:
        raise DeskError(str(exc)) from exc
    return pick_view(p, now)


def approve_all_qualified(conn: sqlite3.Connection, origin: str, now: datetime | None = None,
                          tier: str | None = None) -> dict[str, Any]:
    """Sign off every unsigned pick on today's slate that passes the quantitative filters.

    They all go to one card: ``tier`` (default free)."""
    now = now or db.utcnow()
    approved, skipped = [], []
    for sport in slate(conn, origin, now)["sports"]:
        for v in sport["games"] + sport["props"]:
            if v["signed"]:
                continue
            reasons = quantitative_filter_reasons(_require(conn, v["pick_id"]))
            if reasons:
                skipped.append({"pick_id": v["pick_id"], "reason": "; ".join(reasons)})
                continue
            try:
                approved.append(approve(conn, v["pick_id"], now, tier=tier))
            except DeskError as exc:
                skipped.append({"pick_id": v["pick_id"], "reason": str(exc)})
    return {"approved": approved, "skipped": skipped}


# ==========================================================================
# Tracker
# ==========================================================================
def _agg(rows: list[dict[str, Any]]) -> dict[str, Any]:
    w = sum(r["result"] == "WIN" for r in rows)
    lo = sum(r["result"] == "LOSS" for r in rows)
    v = sum(r["result"] in VOID_RESULTS for r in rows)
    risked = sum(float(r["kelly_units"] or 0.0) for r in rows if r["result"] in ("WIN", "LOSS"))
    net = sum(float(r["net_units"] or 0.0) for r in rows)
    clv = [float(r["clv_percent"]) for r in rows if r.get("clv_percent") is not None]
    return {
        "wins": w, "losses": lo, "voids": v, "record": f"{w}-{lo}-{v}",
        "risked_units": round(risked, 2), "net_units": round(net, 2),
        "roi_percent": None if risked <= 0 else round(net / risked * 100.0, 2),
        "win_rate": None if w + lo == 0 else round(w / (w + lo) * 100.0, 1),
        "picks": len(rows), "clv_n": len(clv),
        "clv_avg": round(sum(clv) / len(clv), 2) if clv else None,
        "clv_beat_rate": round(sum(c > 0 for c in clv) / len(clv) * 100.0, 1) if clv else None,
    }


def tracker(conn: sqlite3.Connection, origin: str, now: datetime | None = None,
            history_limit: int = 200) -> dict[str, Any]:
    """Official performance: signed-off picks only, in units and ROI %."""
    _check_origin(origin)
    rows = db.fetch_picks(conn, f"origin = ? AND {OFFICIAL_SQL}", (origin,))
    settled = sorted((r for r in rows if r["result"] in SETTLED_RESULTS),
                     key=lambda r: (r["settled_at"] or "", r["pick_id"]))
    open_rows = [r for r in rows if r["result"] == "UNSETTLED"]
    curve, cum = [], 0.0
    for r in settled:
        cum += float(r["net_units"] or 0.0)
        curve.append({"t": r["settled_at"], "net": round(cum, 2), "pick_id": r["pick_id"]})
    summary = _agg(settled)
    summary.update(open_count=len(open_rows),
                   open_units=round(sum(float(r["kelly_units"] or 0.0) for r in open_rows), 2))
    return {
        "origin": origin, "summary": summary,
        "by_league": {lg: _agg([r for r in settled if league_of(r) == lg])
                      for lg in SPORT_ORDER},
        "by_kind": {"game": _agg([r for r in settled if not is_prop_market(r["market_type"])]),
                    "prop": _agg([r for r in settled if is_prop_market(r["market_type"])])},
        "by_tier": {t: _agg([r for r in settled if effective_tier(r) == t]) for t in TIERS},
        "curve": curve,
        "open": [with_post(pick_view(r, now)) for r in open_rows],
        "settled": [pick_view(r, now) for r in reversed(settled[-history_limit:])],
    }


# ==========================================================================
# Discord cards (signed-off picks only): one free, one premium
# ==========================================================================
def discord_drop(conn: sqlite3.Connection, origin: str, now: datetime | None = None,
                 tier: str = DEFAULT_TIER) -> str:
    """Postable markdown for one card: today's signed-off, still-open picks of ``tier``.

    Each pick is a Discord post: ET start, emoji + bold selection + units,
    'Book: American / Decimal', then its write-up. A pick approved before
    write-ups existed gets a template blurb on the fly.
    """
    now = now or db.utcnow()
    _check_origin(origin)
    tier = normalize_tier(tier)
    since = db.iso(local_day_start(now))
    rows = db.fetch_picks(conn, f"origin = ? AND {OFFICIAL_SQL} AND result = 'UNSETTLED' "
                                f"AND commence_time >= ?", (origin, since))
    views = []
    for r in rows:
        if effective_tier(r) != tier:
            continue
        v = pick_view(r, now)
        if not v["writeup"]:
            v["writeup"] = wg.generate_writeup(conn, r, llm=lambda _prompt: None)[0]
        views.append(v)
    day = local_day_start(now).strftime("%A, %B %d").replace(" 0", " ")
    return render_tier_card(views, day, tier, mock=origin == db.ORIGIN_MOCK)


def drop_path(out_dir: Path, origin: str, now: datetime | None = None,
              tier: str = DEFAULT_TIER) -> Path:
    """<date>_free_card.md / <date>_premium_card.md (with a ``_mock`` suffix on the mock desk)."""
    day = local_day_start(now or db.utcnow()).strftime("%Y-%m-%d")
    suffix = "" if origin == db.ORIGIN_LIVE else f"_{origin.lower()}"
    return out_dir / f"{day}_{normalize_tier(tier).lower()}_card{suffix}.md"


def discord_cards(conn: sqlite3.Connection, origin: str, out_dir: Path,
                  now: datetime | None = None) -> list[dict[str, Any]]:
    """Both cards for the desk, free first: tier, label, markdown and OUTBOX path."""
    return [{"tier": t, "label": TIER_LABELS[t], "markdown": discord_drop(conn, origin, now, t),
             "path": str(drop_path(out_dir, origin, now, t))} for t in TIERS]


def write_discord_drop(conn: sqlite3.Connection, origin: str, out_dir: Path,
                       now: datetime | None = None) -> dict[str, Path]:
    """Write the free and premium cards to OUTBOX (a MOCK card never overwrites LIVE)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for t in TIERS:
        path = drop_path(out_dir, origin, now, t)
        path.write_text(discord_drop(conn, origin, now, t), encoding="utf-8")
        paths[t] = path
    return paths
