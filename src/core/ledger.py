"""Ledger and bankroll state manager over ``ledger_picks`` (data/takeover.db, WAL).

Position lifecycle (``status`` is the desk-facing view of the ``result`` column):

    PENDING    result UNSETTLED     open: stake committed (APPROVED) or reserved
                                     (PENDING_CAPTAIN_HZ)
    WON        result WIN           net = stake * (dec - 1),  payout = stake * dec
    LOST       result LOSS          net = -stake,             payout = 0
    PUSH       result PUSH          net = 0,                  payout = stake
    CANCELLED  result CANCELLED     void (e.g. game cancelled): net = 0, payout = stake
               result RULE_VIOLATION (quarantined by the backtester) also reads CANCELLED

Bankroll (per origin, so MOCK and SYNTHETIC_SEED never touch LIVE cash):
    capital    = sum(bankroll_adjustments)            initial deposit + deposits/withdrawals
    realized   = sum(net_units) of settled APPROVED picks
    exposure   = sum(stake) of open APPROVED picks    (cash already committed)
    reserved   = sum(stake) of open PENDING_CAPTAIN_HZ picks
    cash       = capital + realized - exposure
    available  = cash - reserved                      what a new position may use

Everything is derived from the ledger rows themselves, so picks written by any
module stay consistent; ``bankroll_journal`` is an append-only audit trail.

Hard-coded desk mandates (independent of BANKROLL_UNITS):
    * odds floor: -130 American or longer
    * straight singles only
    * max 2.0u per bet
    * max 2.0u total open exposure per event_id
"""

from __future__ import annotations

import sqlite3
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any, Iterable

from src.core import database as db
from src.core.config import bankroll_units
from src.core.engine import (
    HITL_APPROVED,
    HITL_PENDING,
    HITL_REJECTED,
    MIN_AMERICAN_ODDS,
    american_to_decimal,
    validate_odds_floor,
)

ODDS_FLOOR_AMERICAN: int = MIN_AMERICAN_ODDS             # -130
MAX_STAKE_UNITS: float = 2.0
MAX_EVENT_UNITS: float = 2.0
SINGLE_MARKETS: frozenset[str] = frozenset({"h2h", "spreads", "totals", "kalshi_binary"})
_EPS = 1e-9

STATUS_PENDING = "PENDING"
STATUS_WON = "WON"
STATUS_LOST = "LOST"
STATUS_PUSH = "PUSH"
STATUS_CANCELLED = "CANCELLED"
STATUSES = (STATUS_PENDING, STATUS_WON, STATUS_LOST, STATUS_PUSH, STATUS_CANCELLED)

RESULT_TO_STATUS: dict[str, str] = {
    "UNSETTLED": STATUS_PENDING, "WIN": STATUS_WON, "LOSS": STATUS_LOST, "PUSH": STATUS_PUSH,
    "CANCELLED": STATUS_CANCELLED, "RULE_VIOLATION": STATUS_CANCELLED,
}
STATUS_TO_RESULT: dict[str, str] = {
    STATUS_PENDING: "UNSETTLED", STATUS_WON: "WIN", STATUS_LOST: "LOSS",
    STATUS_PUSH: "PUSH", STATUS_CANCELLED: "CANCELLED",
}
_FINAL_RESULTS = ("WIN", "LOSS", "PUSH", "CANCELLED")

LEDGER_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS bankroll_adjustments (
    id            INTEGER PRIMARY KEY,
    ts            TIMESTAMP NOT NULL,
    origin        TEXT NOT NULL,
    kind          TEXT NOT NULL CHECK (kind IN ('INITIAL', 'DEPOSIT', 'WITHDRAWAL')),
    amount_units  REAL NOT NULL,
    note          TEXT
);
CREATE INDEX IF NOT EXISTS ix_adj_origin ON bankroll_adjustments (origin);

CREATE TABLE IF NOT EXISTS bankroll_journal (
    id            INTEGER PRIMARY KEY,
    ts            TIMESTAMP NOT NULL,
    origin        TEXT NOT NULL,
    pick_id       TEXT,
    kind          TEXT NOT NULL,
    amount_units  REAL NOT NULL,
    cash_after    REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_journal_pick ON bankroll_journal (pick_id);
"""


class MandateViolation(ValueError):
    """A position breaches a hard desk mandate (floor, singles, stake or event cap)."""


class InsufficientBankroll(ValueError):
    """A position needs more available cash than the bankroll holds."""


def to_status(result: str) -> str:
    try:
        return RESULT_TO_STATUS[result]
    except KeyError:
        raise ValueError(f"unknown ledger result {result!r}") from None


def to_result(status: str) -> str:
    """Accept desk statuses (WON/LOST/...) or raw results (WIN/LOSS/...)."""
    s = str(status).upper()
    if s in STATUS_TO_RESULT:
        return STATUS_TO_RESULT[s]
    if s in RESULT_TO_STATUS and s != "RULE_VIOLATION":
        return s
    raise ValueError(f"unknown status {status!r}; use one of {STATUSES}")


def settlement_amounts(result: str, stake: float, offered_decimal: float) -> tuple[float, float]:
    """(net_units, payout_units) for a final result."""
    r = to_result(result)
    if r == "WIN":
        return round(stake * (offered_decimal - 1.0), 4), round(stake * offered_decimal, 4)
    if r == "LOSS":
        return round(-stake, 4), 0.0
    if r in ("PUSH", "CANCELLED"):
        return 0.0, round(stake, 4)
    raise ValueError("a PENDING position has no settlement amounts")


def mandate_violations(pick: dict[str, Any], existing_event_units: float = 0.0) -> list[str]:
    """Every hard-mandate breach for a candidate position (empty list = compliant)."""
    out: list[str] = []
    odds = pick.get("offered_american")
    if odds is None:
        out.append("offered_american missing")
    elif not validate_odds_floor(int(odds)):
        out.append(f"Odds {int(odds):+d} worse than {ODDS_FLOOR_AMERICAN} floor")
    legs = pick.get("legs", 1)
    n_legs = len(legs) if isinstance(legs, (list, tuple)) else int(legs or 1)
    bet_type = str(pick.get("bet_type", "single")).lower()
    if n_legs != 1 or bet_type != "single" or pick.get("market_type") not in SINGLE_MARKETS:
        out.append(f"Straight singles only (bet_type={bet_type}, legs={n_legs}, "
                   f"market={pick.get('market_type')})")
    stake = float(pick.get("kelly_units") or 0.0)
    if stake <= 0.0:
        out.append("Stake must be positive")
    elif stake > MAX_STAKE_UNITS + _EPS:
        out.append(f"Stake {stake:.2f}u exceeds {MAX_STAKE_UNITS:.1f}u per-bet cap")
    if existing_event_units + stake > MAX_EVENT_UNITS + _EPS:
        out.append(f"Event exposure {existing_event_units + stake:.2f}u exceeds "
                   f"{MAX_EVENT_UNITS:.1f}u cap for {pick.get('event_id')}")
    return out


@dataclass(frozen=True)
class BankrollState:
    origin: str
    capital: float
    realized_pnl: float
    exposure: float
    reserved: float
    cash: float
    available: float
    open_positions: int
    settled_positions: int

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class Ledger:
    """Bankroll-aware position manager for one origin (LIVE by default)."""

    def __init__(self, conn: sqlite3.Connection, origin: str = db.ORIGIN_LIVE,
                 starting_bankroll: float | None = None) -> None:
        self.conn = conn
        self.origin = origin
        conn.executescript(LEDGER_SCHEMA_SQL)
        has = conn.execute("SELECT 1 FROM bankroll_adjustments WHERE origin = ? LIMIT 1",
                           (origin,)).fetchone()
        if has is None:
            amount = bankroll_units() if starting_bankroll is None else float(starting_bankroll)
            if amount <= 0:
                raise ValueError("starting bankroll must be positive")
            with conn:
                conn.execute("INSERT INTO bankroll_adjustments (ts, origin, kind, amount_units,"
                             " note) VALUES (?, ?, 'INITIAL', ?, ?)",
                             (db.iso(db.utcnow()), origin, amount, "baseline bankroll"))

    # ---- state ---------------------------------------------------------
    def _sum(self, sql: str, params: Iterable[Any] = ()) -> float:
        return float(self.conn.execute(sql, (self.origin, *params)).fetchone()[0] or 0.0)

    def capital(self) -> float:
        return self._sum("SELECT SUM(amount_units) FROM bankroll_adjustments WHERE origin = ?")

    def realized_pnl(self) -> float:
        return self._sum("SELECT SUM(net_units) FROM ledger_picks WHERE origin = ? AND "
                         "hitl_status = 'APPROVED' AND result IN ('WIN','LOSS','PUSH','CANCELLED')")

    def open_exposure(self, hitl_status: str = HITL_APPROVED) -> float:
        return self._sum("SELECT SUM(kelly_units) FROM ledger_picks WHERE origin = ? AND "
                         "result = 'UNSETTLED' AND hitl_status = ?", (hitl_status,))

    def event_exposure(self, event_id: str, exclude: Iterable[str] = ()) -> float:
        """Open units (approved + pending) on an event, across this origin."""
        excl = tuple(exclude)
        marks = ",".join("?" * len(excl)) or "''"
        return self._sum(
            f"SELECT SUM(kelly_units) FROM ledger_picks WHERE origin = ? AND event_id = ? AND "
            f"result = 'UNSETTLED' AND hitl_status IN ('APPROVED', 'PENDING_CAPTAIN_HZ') AND "
            f"pick_id NOT IN ({marks})", (event_id, *excl))

    def cash(self) -> float:
        return self.capital() + self.realized_pnl() - self.open_exposure()

    def available_cash(self) -> float:
        return self.cash() - self.open_exposure(HITL_PENDING)

    def state(self) -> BankrollState:
        cap, real = self.capital(), self.realized_pnl()
        exp, res = self.open_exposure(), self.open_exposure(HITL_PENDING)
        n_open = int(self._sum("SELECT COUNT(*) FROM ledger_picks WHERE origin = ? AND "
                               "result = 'UNSETTLED' AND hitl_status != 'REJECTED'"))
        n_set = int(self._sum("SELECT COUNT(*) FROM ledger_picks WHERE origin = ? AND "
                              "result IN ('WIN','LOSS','PUSH','CANCELLED')"))
        cash = cap + real - exp
        return BankrollState(self.origin, round(cap, 4), round(real, 4), round(exp, 4),
                             round(res, 4), round(cash, 4), round(cash - res, 4), n_open, n_set)

    def adjust(self, amount_units: float, note: str = "") -> BankrollState:
        """Deposit (> 0) or withdraw (< 0) capital; withdrawals cannot exceed available cash."""
        if amount_units == 0:
            raise ValueError("adjustment must be non-zero")
        if amount_units < 0 and -amount_units > self.available_cash() + _EPS:
            raise InsufficientBankroll(f"cannot withdraw {-amount_units:.2f}u; available "
                                       f"{self.available_cash():.2f}u")
        kind = "DEPOSIT" if amount_units > 0 else "WITHDRAWAL"
        with self.conn:
            self.conn.execute("INSERT INTO bankroll_adjustments (ts, origin, kind, amount_units,"
                              " note) VALUES (?, ?, ?, ?, ?)",
                              (db.iso(db.utcnow()), self.origin, kind, amount_units, note))
        return self.state()

    def _journal(self, pick_id: str | None, kind: str, amount: float) -> None:
        with self.conn:
            self.conn.execute("INSERT INTO bankroll_journal (ts, origin, pick_id, kind, "
                              "amount_units, cash_after) VALUES (?, ?, ?, ?, ?, ?)",
                              (db.iso(db.utcnow()), self.origin, pick_id, kind,
                               round(amount, 4), round(self.cash(), 4)))

    def journal(self, pick_id: str | None = None) -> list[dict[str, Any]]:
        sql = "SELECT * FROM bankroll_journal WHERE origin = ?"
        params: tuple[Any, ...] = (self.origin,)
        if pick_id is not None:
            sql += " AND pick_id = ?"
            params += (pick_id,)
        return [dict(r) for r in self.conn.execute(sql + " ORDER BY id", params)]

    # ---- positions -----------------------------------------------------
    def check_position(self, pick: dict[str, Any]) -> list[str]:
        """Mandate + bankroll problems for a would-be position (empty = acceptable)."""
        existing = self.event_exposure(pick["event_id"], exclude=[pick["pick_id"]])
        problems = mandate_violations(pick, existing)
        stake = float(pick.get("kelly_units") or 0.0)
        if pick.get("hitl_status", HITL_APPROVED) != HITL_REJECTED and \
                stake > self.available_cash() + _EPS:
            problems.append(f"Stake {stake:.2f}u exceeds available cash "
                            f"{self.available_cash():.2f}u")
        return problems

    def record_position(self, pick: dict[str, Any]) -> bool:
        """Validate hard mandates and bankroll, then insert into ledger_picks.

        Returns False if the pick_id already exists (first record stands).
        Raises MandateViolation / InsufficientBankroll instead of recording.
        """
        if db.get_pick(self.conn, pick["pick_id"]) is not None:
            return False
        row = dict(pick)
        row.setdefault("created_at", db.iso(db.utcnow()))
        row.setdefault("origin", self.origin)
        row.setdefault("hitl_status", HITL_APPROVED)
        row.setdefault("result", "UNSETTLED")
        if row["origin"] != self.origin:
            raise ValueError(f"pick origin {row['origin']} != ledger origin {self.origin}")
        if row.get("offered_decimal") is None and row.get("offered_american") is not None:
            row["offered_decimal"] = round(american_to_decimal(int(row["offered_american"])), 6)
        if row["hitl_status"] != HITL_REJECTED:
            problems = self.check_position(row)
            cash_only = [p for p in problems if "available cash" in p]
            if len(cash_only) != len(problems):
                raise MandateViolation("; ".join(problems))
            if cash_only:
                raise InsufficientBankroll(cash_only[0])
        inserted = db.insert_pick(self.conn, row)
        if inserted and row["hitl_status"] == HITL_APPROVED:
            self._journal(row["pick_id"], "STAKE_COMMITTED", -float(row["kelly_units"]))
        elif inserted and row["hitl_status"] == HITL_PENDING:
            self._journal(row["pick_id"], "STAKE_RESERVED", 0.0)
        return inserted

    def record_positions(self, picks: Iterable[dict[str, Any]]
                         ) -> tuple[int, list[tuple[str, str]]]:
        """Record many; returns (inserted, [(pick_id, reason)] refused by mandate/bankroll)."""
        inserted, refused = 0, []
        for p in picks:
            try:
                inserted += self.record_position(p)
            except (MandateViolation, InsufficientBankroll) as exc:
                refused.append((p["pick_id"], str(exc)))
        return inserted, refused

    def _open_pick(self, pick_id: str) -> dict[str, Any]:
        p = db.get_pick(self.conn, pick_id)
        if p is None:
            raise KeyError(f"unknown pick {pick_id}")
        if p["origin"] != self.origin:
            raise ValueError(f"{pick_id} belongs to origin {p['origin']}, not {self.origin}")
        return p

    def approve_position(self, pick_id: str, note: str | None = None) -> dict[str, Any]:
        """Move a PENDING_CAPTAIN_HZ position to APPROVED, committing its stake."""
        p = self._open_pick(pick_id)
        if p["result"] != "UNSETTLED":
            raise ValueError(f"{pick_id} is already settled ({p['result']})")
        if p["hitl_status"] == HITL_APPROVED:
            return p
        if p["hitl_status"] != HITL_PENDING:
            raise ValueError(f"{pick_id} is {p['hitl_status']}; only pending picks can be approved")
        problems = mandate_violations(p, self.event_exposure(p["event_id"], exclude=[pick_id]))
        if problems:
            raise MandateViolation("; ".join(problems))
        # its own reservation is released on approval, so compare against cash net of others
        others_reserved = self.open_exposure(HITL_PENDING) - float(p["kelly_units"] or 0.0)
        if float(p["kelly_units"]) > self.cash() - others_reserved + _EPS:
            raise InsufficientBankroll(f"stake {p['kelly_units']:.2f}u exceeds available cash")
        fields: dict[str, Any] = {"hitl_status": HITL_APPROVED}
        if note:
            fields["hitl_reason"] = note
        db.update_pick(self.conn, pick_id, **fields)
        self._journal(pick_id, "STAKE_COMMITTED", -float(p["kelly_units"]))
        return db.get_pick(self.conn, pick_id)  # type: ignore[return-value]

    def resize_position(self, pick_id: str, units: float) -> dict[str, Any]:
        """Reduce an open, unsigned, undispatched stake (sizing refinement only shrinks)."""
        p = self._open_pick(pick_id)
        if p["result"] != "UNSETTLED" or p["signoff_at"] or p["dispatched_at"]:
            raise ValueError(f"{pick_id} can no longer be resized")
        if not 0.0 < units <= float(p["kelly_units"]) + _EPS:
            raise ValueError(f"resize must be in (0, {p['kelly_units']}]")
        db.update_pick(self.conn, pick_id, kelly_units=round(units, 2))
        return db.get_pick(self.conn, pick_id)  # type: ignore[return-value]

    def settle_position(self, pick_id: str, status: str, now: datetime | None = None) -> bool:
        """Settle an open position as WON / LOST / PUSH / CANCELLED (or WIN / LOSS ...).

        Writes result, net_units, payout_units, settled_at and journals the
        bankroll credit/debit. Returns False if the position was already settled.
        """
        result = to_result(status)
        if result == "UNSETTLED":
            raise ValueError("cannot settle to PENDING")
        p = self._open_pick(pick_id)
        if p["result"] != "UNSETTLED":
            return False
        stake = float(p["kelly_units"] or 0.0)
        net, payout = settlement_amounts(result, stake, float(p["offered_decimal"]))
        db.update_pick(self.conn, pick_id, result=result, net_units=net, payout_units=payout,
                       settled_at=db.iso(now or db.utcnow()))
        if p["hitl_status"] == HITL_APPROVED:
            self._journal(pick_id, f"SETTLED_{to_status(result)}", payout)
        return True

    def cancel_position(self, pick_id: str, reason: str,
                        now: datetime | None = None) -> bool:
        """Void an open position (stake returned)."""
        p = self._open_pick(pick_id)
        if p["result"] != "UNSETTLED":
            return False
        note = f"{p['hitl_reason'] or ''} | CANCELLED: {reason}".strip(" |")
        db.update_pick(self.conn, pick_id, hitl_reason=note)
        return self.settle_position(pick_id, STATUS_CANCELLED, now)

    def positions(self, status: str | None = None) -> list[dict[str, Any]]:
        """Ledger rows for this origin with a desk ``status`` field added."""
        rows = db.fetch_picks(self.conn, "origin = ?", (self.origin,))
        for r in rows:
            r["status"] = to_status(r["result"])
        if status is not None:
            want = to_status(to_result(status))
            rows = [r for r in rows if r["status"] == want]
        return rows
