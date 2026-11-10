"""Batch 3 tests: ledger & bankroll (Phase 8), Captain HZ HITL gate and
trading-floor orchestration (Phase 9), automated settlement (Phase 10).

Every test uses a throwaway SQLite file under tmp_path; nothing touches the
network or data/takeover.db.

Run: python -m pytest tests/test_batch3.py -v
"""

from __future__ import annotations

import sqlite3
import sys
import types
from datetime import date, datetime, timezone
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.core import database as db  # noqa: E402
from src.core import orchestrator as orch  # noqa: E402
from src.core.engine import HITL_APPROVED, HITL_PENDING, HITL_REJECTED  # noqa: E402
from src.core.ledger import (  # noqa: E402
    MAX_EVENT_UNITS,
    MAX_STAKE_UNITS,
    InsufficientBankroll,
    Ledger,
    MandateViolation,
    mandate_violations,
    settlement_amounts,
    to_result,
    to_status,
)
from src.core.orchestrator import (  # noqa: E402
    HITLGateError,
    captain_hz_approval_gate,
    dispatch_order,
    quantitative_filter_reasons,
    run_trading_floor_cycle,
    sign_off,
)
from src.data import settler  # noqa: E402
from src.data.http_client import HttpError  # noqa: E402
from src.data.settler import GameScore  # noqa: E402

NOW = datetime(2026, 11, 10, 12, 0, tzinfo=timezone.utc)
HOME, AWAY = "Boston Celtics", "Milwaukee Bucks"


@pytest.fixture
def conn(tmp_path):
    c = db.connect(tmp_path / "takeover.db")
    yield c
    c.close()


@pytest.fixture
def ledger(conn):
    return Ledger(conn, starting_bankroll=100.0)


def make_pick(pick_id="P1", event_id="E1", units=1.0, american=120, fair=0.50,
              status=HITL_APPROVED, origin=db.ORIGIN_LIVE, **kw):
    dec = 1 + american / 100 if american > 0 else 1 + 100 / -american
    p = {"pick_id": pick_id, "created_at": "2026-11-08T12:00:00Z",
         "commence_time": "2026-11-09T00:30:00Z", "sport": "basketball_nba",
         "event_id": event_id, "event_name": f"{AWAY} @ {HOME}", "market_type": "h2h",
         "selection": HOME, "line_point": None, "bookmaker": "DraftKings",
         "offered_american": american, "offered_decimal": round(dec, 6), "fair_prob": fair,
         "ev_percent": round((fair * dec - 1) * 100, 3), "kelly_units": units,
         "hitl_status": status, "result": "UNSETTLED", "origin": origin}
    p.update(kw)
    return p


# ==========================================================================
# PART 1: Ledger & bankroll
# ==========================================================================

class TestSchema:
    def test_wal_and_tables(self, conn, ledger):
        assert conn.execute("PRAGMA journal_mode").fetchone()[0].lower() == "wal"
        tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        assert {"ledger_picks", "bankroll_adjustments", "bankroll_journal"} <= tables
        cols = {r["name"] for r in conn.execute("PRAGMA table_info(ledger_picks)")}
        assert {"payout_units", "signoff_by", "signoff_at", "signoff_hash",
                "dispatched_at", "dispatch_ref", "closing_american"} <= cols

    def test_legacy_ledger_rebuilt_for_cancelled_without_data_loss(self, tmp_path):
        path = tmp_path / "legacy.db"
        raw = sqlite3.connect(path)
        legacy = db.SCHEMA_SQL.replace("'PUSH', 'CANCELLED', ", "'PUSH', ")
        assert "'CANCELLED'" not in legacy
        raw.executescript(legacy)
        raw.execute("ALTER TABLE ledger_picks ADD COLUMN model_prob REAL")
        raw.execute("ALTER TABLE ledger_picks ADD COLUMN analysis TEXT")
        raw.execute("INSERT INTO ledger_picks (pick_id, created_at, event_id, market_type, "
                    "selection, offered_american, offered_decimal, kelly_units, model_prob, "
                    "result, net_units) VALUES ('OLD', '2026-01-01T00:00:00Z', 'E', 'h2h', "
                    "'X', 110, 2.1, 1.0, 0.55, 'WIN', 1.1)")
        raw.execute("INSERT INTO ledger_picks (pick_id, created_at, event_id, market_type, "
                    "selection, offered_american, offered_decimal) VALUES "
                    "('BAD', '2026-01-01T00:00:00Z', 'E', 'h2h', 'Y', -145, 1.69)")
        raw.execute("INSERT INTO rule_violations VALUES ('BAD', '2026-01-02T00:00:00Z', -145, "
                    "'floor', 'WIN', 0.69, '{}')")
        raw.commit()
        with pytest.raises(sqlite3.IntegrityError):
            raw.execute("UPDATE ledger_picks SET result = 'CANCELLED' WHERE pick_id = 'OLD'")
        raw.close()

        c = db.connect(path)
        try:
            sql = c.execute("SELECT sql FROM sqlite_master WHERE name = 'ledger_picks'").fetchone()[0]
            assert "'CANCELLED'" in sql
            old = db.get_pick(c, "OLD")
            assert old["model_prob"] == 0.55 and old["net_units"] == 1.1 and old["result"] == "WIN"
            assert db.count_picks(c) == 2
            idx = {r[0] for r in c.execute("SELECT name FROM sqlite_master WHERE type='index'")}
            assert {"ix_picks_event", "ix_picks_result"} <= idx
            assert c.execute("PRAGMA foreign_key_check").fetchall() == []
            assert len(db.fetch_violations(c)) == 1
            db.update_pick(c, "OLD", result="CANCELLED")
            assert db.get_pick(c, "OLD")["result"] == "CANCELLED"
        finally:
            c.close()
        c2 = db.connect(path)                          # second connect is a no-op
        assert db.count_picks(c2) == 2
        c2.close()


class TestStatusAndAmounts:
    def test_status_mapping(self):
        assert [to_status(r) for r in ("UNSETTLED", "WIN", "LOSS", "PUSH", "CANCELLED",
                                       "RULE_VIOLATION")] == \
            ["PENDING", "WON", "LOST", "PUSH", "CANCELLED", "CANCELLED"]
        assert to_result("won") == "WIN" and to_result("LOSS") == "LOSS"
        with pytest.raises(ValueError):
            to_result("VOIDED")
        with pytest.raises(ValueError):
            to_status("???")

    def test_settlement_amounts(self):
        assert settlement_amounts("WON", 1.5, 2.2) == pytest.approx((1.8, 3.3))
        assert settlement_amounts("LOST", 1.5, 2.2) == (-1.5, 0.0)
        assert settlement_amounts("PUSH", 1.5, 2.2) == (0.0, 1.5)
        assert settlement_amounts("CANCELLED", 1.5, 2.2) == (0.0, 1.5)
        with pytest.raises(ValueError):
            settlement_amounts("PENDING", 1.0, 2.0)
        assert settler.net_units_for("WIN", 1.0, 1.91) == pytest.approx(0.91)


class TestMandates:
    def test_hard_coded_caps(self):
        assert MAX_STAKE_UNITS == 2.0 and MAX_EVENT_UNITS == 2.0

    @pytest.mark.parametrize("odds,ok", [(-130, True), (-125, True), (150, True),
                                         (-131, False), (-200, False), (50, False)])
    def test_odds_floor(self, odds, ok):
        assert (not mandate_violations(make_pick(american=odds))) is ok

    @pytest.mark.parametrize("extra", [{"legs": 2}, {"legs": ["a", "b"]},
                                       {"bet_type": "parlay"}, {"market_type": "teaser"}])
    def test_singles_only(self, extra):
        assert any("singles" in r for r in mandate_violations(make_pick(**extra)))

    def test_stake_cap(self):
        assert not mandate_violations(make_pick(units=2.0))
        assert any("per-bet" in r for r in mandate_violations(make_pick(units=2.01)))
        assert mandate_violations(make_pick(units=0.0))

    def test_record_rejects_floor_breach(self, ledger):
        with pytest.raises(MandateViolation, match="-130"):
            ledger.record_position(make_pick(american=-135))
        assert ledger.positions() == []

    def test_event_exposure_cap(self, ledger):
        assert ledger.record_position(make_pick("A", units=1.5))
        with pytest.raises(MandateViolation, match="Event exposure"):
            ledger.record_position(make_pick("B", units=0.6, status=HITL_PENDING))
        assert ledger.record_position(make_pick("C", units=0.5))
        assert ledger.event_exposure("E1") == pytest.approx(2.0)
        assert ledger.record_position(make_pick("D", event_id="E2", units=2.0))
        ledger.settle_position("A", "LOST")                 # settled stake frees capacity
        assert ledger.record_position(make_pick("F", units=1.5))

    def test_rejected_picks_recorded_without_checks(self, ledger):
        assert ledger.record_position(make_pick("R", units=0.0, status=HITL_REJECTED))
        assert ledger.state().exposure == 0.0

    def test_duplicate_returns_false(self, ledger):
        assert ledger.record_position(make_pick())
        assert ledger.record_position(make_pick(units=1.9)) is False
        assert db.get_pick(ledger.conn, "P1")["kelly_units"] == 1.0

    def test_insufficient_bankroll(self, conn):
        small = Ledger(conn, starting_bankroll=1.5)
        with pytest.raises(InsufficientBankroll):
            small.record_position(make_pick(units=2.0))
        assert small.record_position(make_pick(units=1.5))

    def test_record_many_collects_refusals(self, ledger):
        n, refused = ledger.record_positions([make_pick("A"), make_pick("B", american=-140)])
        assert n == 1 and refused[0][0] == "B"


class TestBankroll:
    def test_commit_and_reserve(self, ledger):
        ledger.record_position(make_pick("A", units=1.0))
        ledger.record_position(make_pick("B", event_id="E2", units=0.5, status=HITL_PENDING))
        s = ledger.state()
        assert (s.capital, s.exposure, s.reserved, s.cash, s.available) == \
            pytest.approx((100.0, 1.0, 0.5, 99.0, 98.5))
        assert s.open_positions == 2
        assert [j["kind"] for j in ledger.journal()] == ["STAKE_COMMITTED", "STAKE_RESERVED"]

    @pytest.mark.parametrize("status,cash,net,payout", [
        ("WON", 101.2, 1.2, 2.2), ("LOST", 99.0, -1.0, 0.0),
        ("PUSH", 100.0, 0.0, 1.0), ("CANCELLED", 100.0, 0.0, 1.0)])
    def test_settlement_credits_bankroll(self, ledger, status, cash, net, payout):
        ledger.record_position(make_pick(units=1.0, american=120))
        assert ledger.settle_position("P1", status, NOW)
        p = db.get_pick(ledger.conn, "P1")
        assert (p["net_units"], p["payout_units"]) == pytest.approx((net, payout))
        assert p["settled_at"] == db.iso(NOW)
        s = ledger.state()
        assert s.cash == pytest.approx(cash) and s.exposure == 0.0
        assert s.realized_pnl == pytest.approx(net) and s.settled_positions == 1
        assert ledger.positions(status)[0]["pick_id"] == "P1"
        assert ledger.journal("P1")[-1]["amount_units"] == pytest.approx(payout)
        assert ledger.settle_position("P1", "WON") is False          # idempotent

    def test_pending_settles_without_cash_effect(self, ledger):
        ledger.record_position(make_pick(status=HITL_PENDING))
        ledger.settle_position("P1", "WON")
        assert ledger.state().cash == pytest.approx(100.0)

    def test_approve_pending_commits_stake(self, ledger):
        ledger.record_position(make_pick(status=HITL_PENDING, units=1.5))
        ledger.approve_position("P1", "ok")
        s = ledger.state()
        assert (s.exposure, s.reserved, s.cash) == pytest.approx((1.5, 0.0, 98.5))
        assert db.get_pick(ledger.conn, "P1")["hitl_status"] == HITL_APPROVED

    def test_cancel_position(self, ledger):
        ledger.record_position(make_pick())
        assert ledger.cancel_position("P1", "game cancelled")
        p = db.get_pick(ledger.conn, "P1")
        assert p["result"] == "CANCELLED" and "game cancelled" in p["hitl_reason"]
        assert ledger.state().cash == pytest.approx(100.0)

    def test_cannot_settle_to_pending_or_unknown(self, ledger):
        ledger.record_position(make_pick())
        with pytest.raises(ValueError):
            ledger.settle_position("P1", "PENDING")
        with pytest.raises(KeyError):
            ledger.settle_position("NOPE", "WON")

    def test_origins_isolated(self, conn, ledger):
        mock = Ledger(conn, db.ORIGIN_MOCK, starting_bankroll=50.0)
        mock.record_position(make_pick("M", origin=db.ORIGIN_MOCK, units=2.0))
        mock.settle_position("M", "LOST")
        assert mock.state().cash == pytest.approx(48.0)
        assert ledger.state().cash == pytest.approx(100.0)
        with pytest.raises(ValueError):
            ledger.settle_position("M", "WON")

    def test_deposit_and_withdraw(self, ledger):
        ledger.record_position(make_pick(units=2.0))
        assert ledger.adjust(25.0, "top-up").capital == pytest.approx(125.0)
        assert ledger.adjust(-23.0).cash == pytest.approx(100.0)
        with pytest.raises(InsufficientBankroll):
            ledger.adjust(-100.5)
        with pytest.raises(ValueError):
            ledger.adjust(0.0)

    def test_initial_deposit_written_once(self, conn):
        Ledger(conn, starting_bankroll=100.0)
        Ledger(conn, starting_bankroll=999.0)
        assert Ledger(conn).capital() == pytest.approx(100.0)


# ==========================================================================
# PART 2: Captain HZ gate & orchestration
# ==========================================================================

class TestApprovalGate:
    def test_auto_approved_pick_is_not_signed_off(self, ledger):
        ledger.record_position(make_pick())
        p = db.get_pick(ledger.conn, "P1")
        assert not quantitative_filter_reasons(p)
        assert captain_hz_approval_gate(p) is False

    def test_signoff_opens_gate(self, conn, ledger):
        ledger.record_position(make_pick())
        p = sign_off(conn, "P1", now=NOW)
        assert p["signoff_by"] == "Captain HZ" and p["signoff_at"] == db.iso(NOW)
        assert captain_hz_approval_gate(p) is True

    @pytest.mark.parametrize("field,value", [("offered_american", 115), ("kelly_units", 1.9),
                                             ("selection", AWAY), ("bookmaker", "FanDuel")])
    def test_changed_terms_close_gate(self, conn, ledger, field, value):
        ledger.record_position(make_pick())
        p = sign_off(conn, "P1")
        p[field] = value
        assert captain_hz_approval_gate(p) is False

    def test_forged_signoff_rejected(self, ledger):
        ledger.record_position(make_pick())
        p = db.get_pick(ledger.conn, "P1")
        p.update(signoff_by="Captain HZ", signoff_at="2026-11-10T00:00:00Z", signoff_hash="x")
        assert captain_hz_approval_gate(p) is False
        p.update(signoff_by="Someone Else", signoff_hash=orch.signoff_hash(p))
        assert captain_hz_approval_gate(p) is False

    def test_unqualified_picks_never_pass(self, conn, ledger):
        ledger.record_position(make_pick("LOW", fair=0.46))          # EV +1.2% < 1.5%
        assert any("EV" in r for r in quantitative_filter_reasons(db.get_pick(conn, "LOW")))
        with pytest.raises(ValueError, match="not a qualified"):
            sign_off(conn, "LOW")
        bad = make_pick(american=-140)
        bad.update(signoff_by="Captain HZ", signoff_at="t", signoff_hash=orch.signoff_hash(bad))
        assert captain_hz_approval_gate(bad) is False
        assert captain_hz_approval_gate({}) is False

    def test_only_captain_hz(self, conn, ledger):
        ledger.record_position(make_pick())
        with pytest.raises(PermissionError):
            sign_off(conn, "P1", operator="Intern")

    def test_signoff_of_pending_commits_through_ledger(self, conn, ledger):
        ledger.record_position(make_pick(status=HITL_PENDING, units=1.5))
        p = sign_off(conn, "P1")
        assert p["hitl_status"] == HITL_APPROVED and "Captain HZ" in p["hitl_reason"]
        assert ledger.state().exposure == pytest.approx(1.5)


class TestDispatch:
    def test_unsigned_order_never_reaches_dispatcher(self, conn, ledger):
        ledger.record_position(make_pick())
        sent = []
        with pytest.raises(HITLGateError):
            dispatch_order(conn, "P1", sent.append)
        assert sent == []

    def test_signed_order_dispatched_once(self, conn, ledger):
        ledger.record_position(make_pick())
        sign_off(conn, "P1")
        sent = []
        payload = dispatch_order(conn, "P1", lambda o: sent.append(o) or "BR-1", NOW)
        assert sent == [payload]
        assert payload["bet_type"] == "single" and payload["stake_units"] == 1.0
        assert payload["american_odds"] == 120 and payload["signoff"]["by"] == "Captain HZ"
        p = db.get_pick(conn, "P1")
        assert p["dispatched_at"] == db.iso(NOW) and p["dispatch_ref"] == "BR-1"
        with pytest.raises(HITLGateError):
            dispatch_order(conn, "P1", sent.append)
        assert len(sent) == 1

    def test_db_tamper_after_signoff_blocks(self, conn, ledger):
        ledger.record_position(make_pick())
        sign_off(conn, "P1")
        db.update_pick(conn, "P1", offered_american=105, offered_decimal=2.05)
        with pytest.raises(HITLGateError):
            dispatch_order(conn, "P1", lambda o: "x")


def _fake_fetch(picks):
    def fetch(conn, sports, mock=False, now=None, output_path=None):
        led = Ledger(conn, db.ORIGIN_MOCK if mock else db.ORIGIN_LIVE)
        led.record_positions(picks)
        return {"opportunities": [{"pick_id": p["pick_id"]} for p in picks]}
    return fetch


class TestTradingFloorCycle:
    PICKS = [make_pick("GOOD", "E1", units=1.0),
             make_pick("BIG", "E2", units=2.0, fair=0.48),              # QK = 1.17u
             make_pick("THIN", "E3", units=0.5, fair=0.46)]             # EV 1.2%

    def test_without_approver_nothing_is_signed_or_sent(self, conn, ledger):
        sent = []
        r = run_trading_floor_cycle(conn, fetch=_fake_fetch(self.PICKS), dispatcher=sent.append)
        assert r.scanned == 3
        assert set(r.awaiting_signoff) == {"GOOD", "BIG"}
        assert r.blocked[0][0] == "THIN" and "EV" in r.blocked[0][1]
        assert r.signed_off == [] and r.dispatched == [] and sent == []

    def test_risk_sizing_only_shrinks(self, conn, ledger):
        r = run_trading_floor_cycle(conn, fetch=_fake_fetch(self.PICKS))
        assert r.resized == [("BIG", 2.0, pytest.approx(1.17))]
        assert db.get_pick(conn, "BIG")["kelly_units"] == pytest.approx(1.17)
        assert db.get_pick(conn, "GOOD")["kelly_units"] == 1.0

    def test_approved_cycle_dispatches_signed_orders(self, conn, ledger):
        sent, seen = [], []
        approve = lambda p: seen.append(p["pick_id"]) or p["pick_id"] == "GOOD"  # noqa: E731
        r = run_trading_floor_cycle(conn, fetch=_fake_fetch(self.PICKS), approver=approve,
                                    dispatcher=lambda o: sent.append(o) or "ok", now=NOW)
        assert set(seen) == {"GOOD", "BIG"}                         # THIN never offered
        assert r.signed_off == ["GOOD"] and r.awaiting_signoff == ["BIG"]
        assert r.dispatched == ["GOOD"] and [o["pick_id"] for o in sent] == ["GOOD"]
        assert r.bankroll["exposure"] == pytest.approx(1.0 + 1.17 + 0.5)
        # next cycle: nothing re-sent
        r2 = run_trading_floor_cycle(conn, fetch=_fake_fetch([]), approver=lambda p: True,
                                     dispatcher=lambda o: sent.append(o) or "ok")
        assert len(sent) == 1 and r2.dispatched == []

    def test_cli_signoff_picked_up_and_listed_ready(self, conn, ledger):
        ledger.record_position(make_pick("CLI", "E9"))
        sign_off(conn, "CLI")
        r = run_trading_floor_cycle(conn, fetch=_fake_fetch([]))
        assert r.ready_to_dispatch == ["CLI"]

    def test_dispatcher_failure_keeps_order_open(self, conn, ledger):
        def boom(order):
            raise ConnectionError("broker down")
        r = run_trading_floor_cycle(conn, fetch=_fake_fetch([self.PICKS[0]]),
                                    approver=lambda p: True, dispatcher=boom)
        assert r.dispatch_errors[0][0] == "GOOD"
        assert db.get_pick(conn, "GOOD")["dispatched_at"] is None

    def test_mock_end_to_end(self, conn, tmp_path):
        sent = []
        r = run_trading_floor_cycle(conn, sports=("basketball_nba", "americanfootball_nfl"),
                                    mock=True, approver=lambda p: True,
                                    dispatcher=lambda o: sent.append(o) or "sim",
                                    settle=True, output_path=tmp_path / "scan.json")
        assert r.mode == "MOCK" and r.scanned > 0
        assert r.dispatched and len(sent) == len(r.dispatched)
        assert all(o["signoff"]["by"] == "Captain HZ" for o in sent)
        assert all(o["american_odds"] >= -130 for o in sent)
        assert all(o["stake_units"] <= MAX_STAKE_UNITS for o in sent)
        assert r.settlement["grading"]["graded"] > 0
        led = Ledger(conn, db.ORIGIN_MOCK)
        s = led.state()
        assert s.cash == pytest.approx(s.capital + s.realized_pnl - s.exposure)
        assert db.count_picks(conn, (db.ORIGIN_LIVE,)) == 0
        for e, units in conn.execute("SELECT event_id, SUM(kelly_units) FROM ledger_picks "
                                     "GROUP BY event_id"):
            assert units <= MAX_EVENT_UNITS + 1e-9


# ==========================================================================
# PART 3: Settlement & reconciliation
# ==========================================================================

def _espn_event(home_score, away_score, completed=True, name="STATUS_FINAL"):
    return {"status": {"type": {"completed": completed, "name": name}},
            "competitions": [{"competitors": [
                {"homeAway": "home", "score": str(home_score), "team": {"displayName": HOME}},
                {"homeAway": "away", "score": str(away_score), "team": {"displayName": AWAY}}]}]}


class TestScoreSources:
    def test_parse_espn_statuses(self):
        games = settler.parse_espn({"events": [
            _espn_event(110, 100), _espn_event(0, 0, True, "STATUS_CANCELED"),
            _espn_event(0, 0, False, "STATUS_POSTPONED"), _espn_event(50, 48, False, "STATUS_IN_PROGRESS")]})
        assert [(g.completed, g.cancelled, g.postponed) for g in games] == [
            (True, False, False), (False, True, False), (False, False, True),
            (False, False, False)]

    def test_nba_api_fallback(self, monkeypatch):
        class Board:
            def __init__(self, game_date, timeout):
                assert game_date == "2026-11-09"

            def get_normalized_dict(self):
                return {"GameHeader": [
                    {"GAME_ID": "1", "HOME_TEAM_ID": 10, "VISITOR_TEAM_ID": 20, "GAME_STATUS_ID": 3},
                    {"GAME_ID": "2", "HOME_TEAM_ID": 30, "VISITOR_TEAM_ID": 40, "GAME_STATUS_ID": 2}],
                    "LineScore": [
                        {"GAME_ID": "1", "TEAM_ID": 10, "TEAM_ABBREVIATION": "BOS", "PTS": 112},
                        {"GAME_ID": "1", "TEAM_ID": 20, "TEAM_ABBREVIATION": "MIL", "PTS": 108},
                        {"GAME_ID": "2", "TEAM_ID": 30, "TEAM_ABBREVIATION": "LAL", "PTS": 60},
                        {"GAME_ID": "2", "TEAM_ID": 40, "TEAM_ABBREVIATION": "GSW", "PTS": 58}]}

        endpoints = types.ModuleType("nba_api.stats.endpoints")
        endpoints.scoreboardv2 = types.SimpleNamespace(ScoreboardV2=Board)
        monkeypatch.setitem(sys.modules, "nba_api", types.ModuleType("nba_api"))
        monkeypatch.setitem(sys.modules, "nba_api.stats", types.ModuleType("nba_api.stats"))
        monkeypatch.setitem(sys.modules, "nba_api.stats.endpoints", endpoints)
        games = settler.fetch_nba_api_scores(date(2026, 11, 9))
        assert games == [GameScore("BOS", "MIL", 112, 108, True),
                         GameScore("LAL", "GSW", 60, 58, False)]

    def test_nfl_data_py_fallback(self, monkeypatch):
        rows = [{"gameday": "2026-11-08", "home_team": "KC", "away_team": "BUF",
                 "home_score": 27.0, "away_score": 24.0},
                {"gameday": "2026-11-08", "home_team": "LA", "away_team": "SF",
                 "home_score": float("nan"), "away_score": float("nan")},
                {"gameday": "2026-11-09", "home_team": "GB", "away_team": "DET",
                 "home_score": 20.0, "away_score": 23.0}]
        fake = types.ModuleType("nfl_data_py")
        seasons = []
        fake.import_schedules = lambda years: seasons.append(years) or types.SimpleNamespace(
            to_dict=lambda orient: rows)
        monkeypatch.setitem(sys.modules, "nfl_data_py", fake)
        games = settler.fetch_nfl_data_py_scores(date(2026, 11, 8))
        assert seasons == [[2026]]
        assert games == [GameScore("KC", "BUF", 27, 24, True), GameScore("LA", "SF", 0, 0, False)]
        assert settler._nfl_season(date(2027, 1, 15)) == 2026

    def test_missing_package_returns_empty(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "nfl_data_py", None)
        assert settler.fetch_nfl_data_py_scores(date(2026, 11, 8)) == []
        assert settler.fetch_package_scores("hockey", date(2026, 11, 8)) == []

    def test_fallback_chain(self, conn, monkeypatch):
        def espn_down(*a):
            raise HttpError("503")
        pkg = [GameScore("BOS", "MIL", 1, 0, True)]
        monkeypatch.setattr(settler, "fetch_espn_scoreboard", espn_down)
        monkeypatch.setitem(settler.PACKAGE_FETCHERS, "basketball_nba", lambda d: pkg)
        assert settler.fetch_scores(conn, "basketball_nba", date(2026, 11, 9)) == pkg
        odds = [GameScore(HOME, AWAY, 2, 1, True)]
        monkeypatch.setitem(settler.PACKAGE_FETCHERS, "basketball_nba", lambda d: [])
        monkeypatch.setattr(settler, "fetch_odds_api_scores", lambda c, s: odds)
        assert settler.fetch_scores(conn, "basketball_nba", date(2026, 11, 9)) == odds


class TestReconciliation:
    def _with_board(self, monkeypatch, games):
        monkeypatch.setattr(settler, "fetch_scores", lambda c, s, d: games)

    def test_reconcile_grades_and_credits_bankroll(self, conn, ledger, monkeypatch):
        ledger.record_position(make_pick("ML", "E1", units=1.0, american=120))
        ledger.record_position(make_pick("SPR", "E2", units=1.0, american=-110,
                                         market_type="spreads", line_point=-3.5, fair=0.55))
        ledger.record_position(make_pick("TOT", "E3", units=0.5, american=-105,
                                         market_type="totals", selection="Over",
                                         line_point=220.5, fair=0.55))
        self._with_board(monkeypatch, [GameScore(HOME, AWAY, 112, 108, True)])
        out = settler.reconcile_pending_bets(conn, now=NOW)
        assert out["grading"]["graded"] == 3
        res = {p["pick_id"]: (p["status"], p["net_units"]) for p in ledger.positions()}
        assert res["ML"] == ("WON", pytest.approx(1.2))
        assert res["SPR"] == ("WON", pytest.approx(0.9091, abs=1e-4))
        assert res["TOT"] == ("LOST", pytest.approx(-0.5))
        b = out["bankroll"]
        assert b["exposure"] == 0.0
        assert b["cash"] == pytest.approx(100.0 + 1.2 + 0.9091 - 0.5, abs=1e-4)
        assert out["closing"]["no_snapshot"] == 3

    def test_cancelled_postponed_and_live_games(self, conn, ledger, monkeypatch):
        ledger.record_position(make_pick("C", units=1.0))
        self._with_board(monkeypatch, [GameScore(HOME, AWAY, 0, 0, False, cancelled=True)])
        assert settler.grade_settled_games(conn, now=NOW)["cancelled"] == 1
        assert db.get_pick(conn, "C")["result"] == "CANCELLED"
        assert ledger.state().cash == pytest.approx(100.0)

        ledger.record_position(make_pick("P", event_id="E2"))
        self._with_board(monkeypatch, [GameScore(HOME, AWAY, 0, 0, False, postponed=True)])
        assert settler.grade_settled_games(conn, now=NOW)["postponed"] == 1
        self._with_board(monkeypatch, [GameScore(HOME, AWAY, 50, 40, False)])
        assert settler.grade_settled_games(conn, now=NOW)["not_final"] == 1
        self._with_board(monkeypatch, [])
        assert settler.grade_settled_games(conn, now=NOW)["unmatched"] == 1
        assert ledger.positions("PENDING")[0]["pick_id"] == "P"

    def test_future_games_untouched(self, conn, ledger, monkeypatch):
        ledger.record_position(make_pick(commence_time="2026-11-20T00:00:00Z"))
        self._with_board(monkeypatch, [GameScore(HOME, AWAY, 1, 0, True)])
        assert settler.grade_settled_games(conn, now=NOW)["graded"] == 0

    def test_mock_settlement_routes_through_ledger(self, conn, tmp_path):
        from src.data.market_fetcher import run_scan
        run_scan(conn, ("basketball_nba", "americanfootball_nfl"), mock=True,
                 output_path=tmp_path / "scan.json")
        out = settler.run_settlement(conn, mock=True)
        assert out["grading"]["graded"] > 0
        settled = db.fetch_picks(conn, "origin = 'MOCK' AND result != 'UNSETTLED'")
        assert settled and all(p["payout_units"] is not None for p in settled)
        assert out["bankroll"]["origin"] == "MOCK"
