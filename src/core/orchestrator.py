"""Trading-floor orchestration with the mandatory Captain HZ sign-off gate.

One cycle (``run_trading_floor_cycle``):

    1. prices     market_fetcher.run_scan        live (or mock) books + Kalshi;
                                                  candidates recorded via Ledger
    2. edge       engine.calculate_ev             fair_prob already blends the
                                                  sharp de-vig and matchup_model
                                                  projection; re-verified here
    3. sizing     risk.size_bet_under_uncertainty quarter-Kelly (Baker-McHale at
                                                  the quarter-Kelly sigma), capped;
                                                  may only SHRINK the ledger stake
    4. ledger     ledger.Ledger                   mandates, exposure, bankroll
    5. HITL       captain_hz_approval_gate        no order payload leaves without
                                                  explicit Captain HZ sign-off
    6. dispatch   caller-supplied dispatcher      only for gate-passing picks
    7. settle     settler.reconcile_pending_bets  optional

Sign-off is explicit and bound to the exact terms: ``sign_off`` stores the
operator, time and a SHA-256 of (pick_id, event, market, selection, line,
book, price, stake). If any of those change afterwards the hash no longer
matches and the gate closes again. The scanner's automatic ``APPROVED``
status (no HITL triggers) is NOT a sign-off.
"""

from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from src.core import database as db
from src.core.engine import (
    HITL_APPROVED,
    HITL_PENDING,
    american_to_decimal,
    calculate_ev,
    validate_odds_floor,
)
from src.core.ledger import (
    MAX_STAKE_UNITS,
    Ledger,
    MandateViolation,
    mandate_violations,
)
from src.core.risk import implied_estimation_sigma, size_bet_under_uncertainty

log = logging.getLogger("breeze.floor")

OPERATOR = "Captain HZ"
MIN_EV_PERCENT = 1.5
_EV_TOL = 0.05                         # pct-points of rounding slack vs the recorded ev_percent

Approver = Callable[[dict[str, Any]], bool]
Dispatcher = Callable[[dict[str, Any]], str]


class HITLGateError(PermissionError):
    """An order was about to be dispatched without a valid Captain HZ sign-off."""


# ==========================================================================
# Quantitative filters and the sign-off gate
# ==========================================================================
def quantitative_filter_reasons(pick: dict[str, Any]) -> list[str]:
    """Why a pick is NOT a qualified +EV bet (empty list = qualified)."""
    reasons: list[str] = []
    if pick.get("result", "UNSETTLED") != "UNSETTLED":
        reasons.append(f"Position already settled ({pick.get('result')})")
    if pick.get("hitl_status") not in (HITL_APPROVED, HITL_PENDING):
        reasons.append(f"HITL status {pick.get('hitl_status')}")
    odds = pick.get("offered_american")
    if odds is None or not validate_odds_floor(int(odds)):
        reasons.append(f"Odds {odds} fail the -130 floor")
        return reasons
    dec = pick.get("offered_decimal") or american_to_decimal(int(odds))
    fair = pick.get("fair_prob")
    if fair is None or not 0.0 < float(fair) < 1.0:
        reasons.append("No fair probability")
    else:
        ev_pct = calculate_ev(float(fair), float(dec)) * 100.0
        if ev_pct < MIN_EV_PERCENT - _EV_TOL:
            reasons.append(f"EV {ev_pct:.2f}% below {MIN_EV_PERCENT:.1f}% threshold")
    reasons += [r for r in mandate_violations(pick) if "Event exposure" not in r]
    return reasons


def signoff_hash(pick: dict[str, Any]) -> str:
    """SHA-256 over the economic terms a sign-off authorises."""
    terms = [pick.get("pick_id"), pick.get("event_id"), pick.get("market_type"),
             pick.get("selection"), pick.get("line_point"), pick.get("bookmaker"),
             int(pick["offered_american"]), round(float(pick.get("kelly_units") or 0.0), 2)]
    return hashlib.sha256(json.dumps(terms, separators=(",", ":")).encode()).hexdigest()


def captain_hz_approval_gate(pick: dict[str, Any]) -> bool:
    """Mandatory checkpoint before any order payload is dispatched.

    True only when the pick (a) is a qualified +EV bet passing every
    quantitative filter and mandate, (b) is APPROVED, (c) carries an explicit
    Captain HZ sign-off whose hash still matches its current terms, and
    (d) has not already been dispatched.
    """
    try:
        if quantitative_filter_reasons(pick):
            return False
        return (pick.get("hitl_status") == HITL_APPROVED
                and pick.get("signoff_by") == OPERATOR
                and bool(pick.get("signoff_at"))
                and pick.get("signoff_hash") == signoff_hash(pick)
                and not pick.get("dispatched_at"))
    except (KeyError, TypeError, ValueError):
        return False


def sign_off(conn: sqlite3.Connection, pick_id: str, operator: str = OPERATOR,
             now: datetime | None = None) -> dict[str, Any]:
    """Record Captain HZ's explicit sign-off on a qualified pick.

    Pending picks are approved through the ledger first (committing the stake
    against bankroll and event caps). Raises ValueError if the pick fails the
    quantitative filters, PermissionError for any operator but Captain HZ.
    """
    if operator != OPERATOR:
        raise PermissionError(f"only {OPERATOR} may sign off, not {operator!r}")
    p = db.get_pick(conn, pick_id)
    if p is None:
        raise KeyError(f"unknown pick {pick_id}")
    reasons = quantitative_filter_reasons(p)
    if reasons:
        raise ValueError(f"{pick_id} is not a qualified bet: {'; '.join(reasons)}")
    ts = db.iso(now or db.utcnow())
    if p["hitl_status"] == HITL_PENDING:
        note = f"{p['hitl_reason'] or ''} | Approved by {OPERATOR} {ts}".strip(" |")
        p = Ledger(conn, p["origin"]).approve_position(pick_id, note)
    db.update_pick(conn, pick_id, signoff_by=operator, signoff_at=ts,
                   signoff_hash=signoff_hash(p))
    return db.get_pick(conn, pick_id)  # type: ignore[return-value]


def build_order_payload(pick: dict[str, Any]) -> dict[str, Any]:
    return {
        "pick_id": pick["pick_id"], "event_id": pick["event_id"],
        "event_name": pick.get("event_name"), "bookmaker": pick.get("bookmaker"),
        "market_type": pick["market_type"], "selection": pick["selection"],
        "line_point": pick.get("line_point"), "bet_type": "single",
        "american_odds": int(pick["offered_american"]),
        "decimal_odds": float(pick["offered_decimal"]),
        "stake_units": float(pick["kelly_units"]),
        "signoff": {"by": pick["signoff_by"], "at": pick["signoff_at"],
                    "hash": pick["signoff_hash"]},
    }


def dispatch_order(conn: sqlite3.Connection, pick_id: str, dispatcher: Dispatcher,
                   now: datetime | None = None) -> dict[str, Any]:
    """Send one order through ``dispatcher`` -- only past the Captain HZ gate.

    The pick is re-read from the ledger immediately before dispatch so a stale
    in-memory copy can never bypass the gate. Returns the payload sent.
    """
    p = db.get_pick(conn, pick_id)
    if p is None or not captain_hz_approval_gate(p):
        raise HITLGateError(f"{pick_id}: no valid {OPERATOR} sign-off; order NOT dispatched")
    payload = build_order_payload(p)
    ref = dispatcher(payload)
    db.update_pick(conn, pick_id, dispatched_at=db.iso(now or db.utcnow()),
                   dispatch_ref=None if ref is None else str(ref))
    return payload


# ==========================================================================
# Pipeline loop
# ==========================================================================
@dataclass
class CycleReport:
    mode: str
    scanned: int = 0
    qualified: list[str] = field(default_factory=list)
    resized: list[tuple[str, float, float]] = field(default_factory=list)
    blocked: list[tuple[str, str]] = field(default_factory=list)
    awaiting_signoff: list[str] = field(default_factory=list)
    signed_off: list[str] = field(default_factory=list)
    ready_to_dispatch: list[str] = field(default_factory=list)
    dispatched: list[str] = field(default_factory=list)
    dispatch_errors: list[tuple[str, str]] = field(default_factory=list)
    settlement: dict[str, Any] | None = None
    bankroll: dict[str, Any] = field(default_factory=dict)


def refine_stake(ledger: Ledger, pick: dict[str, Any]) -> float:
    """Quarter-Kelly via risk.py at the event's remaining capacity (never above the ledger stake)."""
    p, dec = float(pick["fair_prob"]), float(pick["offered_decimal"])
    sigma = implied_estimation_sigma(p, dec - 1.0)
    existing = ledger.event_exposure(pick["event_id"], exclude=[pick["pick_id"]])
    sized = size_bet_under_uncertainty(p, dec, sigma, existing_event_units=existing)
    return min(float(sized["units"]), float(pick["kelly_units"] or 0.0), MAX_STAKE_UNITS)


def run_trading_floor_cycle(
        conn: sqlite3.Connection, sports: tuple[str, ...] = ("basketball_nba",),
        mock: bool = False, approver: Approver | None = None,
        dispatcher: Dispatcher | None = None, settle: bool = False,
        now: datetime | None = None, output_path: Path | None = None,
        fetch: Callable[..., dict[str, Any]] | None = None) -> CycleReport:
    """One end-to-end pass: prices -> edge -> sizing -> ledger -> HITL -> dispatch.

    ``approver`` is Captain HZ's interactive confirmation (called once per
    qualified, unsigned pick); without it nothing is signed. ``dispatcher``
    places a signed order and returns a reference; without it gate-passing
    picks are only listed as ready. Neither has a default that auto-approves.
    """
    origin = db.ORIGIN_MOCK if mock else db.ORIGIN_LIVE
    ledger = Ledger(conn, origin)
    report = CycleReport(mode="MOCK" if mock else "LIVE")

    if fetch is None:
        from src.data.market_fetcher import run_scan
        fetch = run_scan
    kwargs: dict[str, Any] = {"mock": mock, "now": now}
    if output_path is not None:
        kwargs["output_path"] = output_path
    scan = fetch(conn, sports, **kwargs)
    opps = scan.get("opportunities", [])
    report.scanned = len(opps)

    for opp in opps:
        pick = db.get_pick(conn, opp["pick_id"])
        if pick is None or pick["origin"] != origin:
            continue
        reasons = quantitative_filter_reasons(pick)
        if reasons:
            report.blocked.append((pick["pick_id"], "; ".join(reasons)))
            continue
        if not pick["signoff_at"] and not pick["dispatched_at"]:
            units = refine_stake(ledger, pick)
            if units <= 0.0:
                ledger.cancel_position(pick["pick_id"], "risk sizing returned zero stake", now)
                report.blocked.append((pick["pick_id"], "risk sizing returned zero stake"))
                continue
            if units < float(pick["kelly_units"]) - 1e-9:
                report.resized.append((pick["pick_id"], float(pick["kelly_units"]), units))
                pick = ledger.resize_position(pick["pick_id"], units)
        report.qualified.append(pick["pick_id"])

        if pick["signoff_at"] or pick["dispatched_at"]:
            continue
        if approver is not None and approver(dict(pick)):
            try:
                sign_off(conn, pick["pick_id"], now=now)
                report.signed_off.append(pick["pick_id"])
            except (ValueError, MandateViolation) as exc:
                report.blocked.append((pick["pick_id"], str(exc)))
        else:
            report.awaiting_signoff.append(pick["pick_id"])

    # Every signed, undispatched pick of this origin -- including CLI sign-offs.
    for p in db.fetch_picks(conn, "origin = ? AND result = 'UNSETTLED' AND signoff_at IS NOT "
                                  "NULL AND dispatched_at IS NULL", (origin,)):
        if not captain_hz_approval_gate(p):
            report.blocked.append((p["pick_id"], "sign-off no longer matches pick terms"))
            continue
        if dispatcher is None:
            report.ready_to_dispatch.append(p["pick_id"])
            continue
        try:
            dispatch_order(conn, p["pick_id"], dispatcher, now)
            report.dispatched.append(p["pick_id"])
        except Exception as exc:                      # broker failure: keep it undispatched
            log.error("dispatch failed for %s: %s", p["pick_id"], exc)
            report.dispatch_errors.append((p["pick_id"], str(exc)))

    if settle:
        from src.data.settler import reconcile_pending_bets, run_settlement
        report.settlement = run_settlement(conn, mock=True) if mock \
            else reconcile_pending_bets(conn, now=now)
    report.bankroll = ledger.state().as_dict()
    return report
