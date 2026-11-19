"""Free / premium tiers on signed-off picks, the split Discord cards, and CLV in the tracker.

Every test uses tmp_path for SQLite and OUTBOX output; nothing touches the
network, data/takeover.db or the real OUTBOX/.
Run: python -m pytest tests/test_tiers.py -v
"""

from __future__ import annotations

import json
import sqlite3
import sys
import threading
import urllib.error
import urllib.request
from datetime import datetime, timedelta
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import run_pipeline  # noqa: E402
import web_server  # noqa: E402
from src.core import database as db  # noqa: E402
from src.core.engine import HITL_APPROVED, HITL_PENDING  # noqa: E402
from src.core.ledger import Ledger  # noqa: E402
from src.core.tiers import TIER_FREE, TIER_PREMIUM, normalize_tier, set_tier  # noqa: E402
from src.web import desk  # noqa: E402
from templates.discord_templates import HONEST_TAGLINE  # noqa: E402

NOW = datetime(2026, 11, 14, 12, 0).astimezone()
MOCK = db.ORIGIN_MOCK


@pytest.fixture
def conn(tmp_path):
    c = db.connect(tmp_path / "takeover.db")
    yield c
    c.close()


def _pick(pick_id: str, *, event: str = "ev-1", hours: float = 3.0,
          selection: str = "Boston Celtics", start: datetime = NOW, **kw) -> dict:
    p = {"pick_id": pick_id, "commence_time": db.iso(start + timedelta(hours=hours)),
         "sport": "basketball_nba", "event_id": event,
         "event_name": "Boston Celtics @ Milwaukee Bucks", "market_type": "spreads",
         "selection": selection, "line_point": -3.5, "bookmaker": "DraftKings",
         "offered_american": 105, "fair_prob": 0.55, "ev_percent": 12.75, "kelly_units": 1.0,
         "hitl_status": HITL_PENDING, "hitl_reason": "", "origin": MOCK}
    p.update(kw)
    return p


def _record(conn, *picks) -> Ledger:
    led = Ledger(conn, MOCK)
    for p in picks:
        assert led.record_position(p)
    return led


# ==========================================================================
# Sign-off sets the tier
# ==========================================================================
def test_normalize_tier() -> None:
    assert normalize_tier(None) == normalize_tier("") == TIER_FREE
    assert normalize_tier("premium") == normalize_tier(" Premium ") == TIER_PREMIUM
    with pytest.raises(ValueError, match="unknown tier"):
        normalize_tier("vip")


def test_approve_defaults_to_free_and_takes_premium(conn) -> None:
    _record(conn, _pick("PK-a00000000001"), _pick("PK-a00000000002", event="ev-2"))
    free = desk.approve(conn, "PK-a00000000001", now=NOW)
    premium = desk.approve(conn, "PK-a00000000002", now=NOW, tier="premium")
    assert (free["tier"], premium["tier"]) == (TIER_FREE, TIER_PREMIUM)
    row = db.get_pick(conn, "PK-a00000000002")
    assert row["tier"] == TIER_PREMIUM and row["tier_set_at"] == row["signoff_at"]
    with pytest.raises(desk.DeskError) as bad:
        desk.approve(conn, "PK-a00000000001", now=NOW, tier="gold")
    assert bad.value.status == 400


def test_unsigned_picks_have_no_tier(conn) -> None:
    _record(conn, _pick("PK-a00000000003"),
            _pick("PK-a00000000004", event="ev-2", hitl_status=HITL_APPROVED))   # auto-approved
    views = {v["pick_id"]: v for s in desk.slate(conn, MOCK, NOW)["sports"] for v in s["games"]}
    assert views["PK-a00000000003"]["tier"] is None
    assert views["PK-a00000000004"]["tier"] is None                      # no sign-off yet


def test_reapprove_with_another_tier_moves_it(conn) -> None:
    _record(conn, _pick("PK-a00000000005"))
    desk.approve(conn, "PK-a00000000005", now=NOW)
    assert desk.approve(conn, "PK-a00000000005", now=NOW)["tier"] == TIER_FREE   # no tier: kept
    assert desk.approve(conn, "PK-a00000000005", now=NOW, tier="premium")["tier"] == TIER_PREMIUM


def test_approve_all_uses_one_tier(conn) -> None:
    _record(conn, _pick("PK-a00000000006"), _pick("PK-a00000000007", event="ev-2"))
    out = desk.approve_all_qualified(conn, MOCK, NOW, tier="premium")
    assert {v["tier"] for v in out["approved"]} == {TIER_PREMIUM}
    assert desk.slate(conn, MOCK, NOW)["tier_counts"] == {TIER_FREE: 0, TIER_PREMIUM: 2}


# ==========================================================================
# Moving between cards, and the game-time lock
# ==========================================================================
def test_change_tier_moves_between_cards(conn) -> None:
    _record(conn, _pick("PK-b00000000001"))
    desk.approve(conn, "PK-b00000000001", now=NOW)
    assert "**Celtics -3.5**" in desk.discord_drop(conn, MOCK, NOW, TIER_FREE)
    v = desk.change_tier(conn, "PK-b00000000001", "premium", now=NOW + timedelta(minutes=5))
    assert v["tier"] == TIER_PREMIUM and not v["tier_locked"]
    assert "**Celtics" not in desk.discord_drop(conn, MOCK, NOW, TIER_FREE)
    assert "**Celtics -3.5**" in desk.discord_drop(conn, MOCK, NOW, TIER_PREMIUM)
    assert db.get_pick(conn, "PK-b00000000001")["tier_set_at"] == db.iso(NOW + timedelta(minutes=5))


def test_tier_locks_at_game_time(conn) -> None:
    _record(conn, _pick("PK-b00000000002", hours=1))
    desk.approve(conn, "PK-b00000000002", now=NOW)
    after_tip = NOW + timedelta(hours=2)
    with pytest.raises(desk.DeskError, match="locks at game time") as locked:
        desk.change_tier(conn, "PK-b00000000002", "premium", now=after_tip)
    assert locked.value.status == 409
    assert desk.pick_view(db.get_pick(conn, "PK-b00000000002"), after_tip)["tier_locked"]
    assert db.get_pick(conn, "PK-b00000000002")["tier"] == TIER_FREE


def test_tier_refused_for_unsigned_settled_and_unknown(conn) -> None:
    led = _record(conn, _pick("PK-b00000000003"), _pick("PK-b00000000004", event="ev-2"))
    with pytest.raises(desk.DeskError, match="not signed off"):
        desk.change_tier(conn, "PK-b00000000003", "premium", now=NOW)
    desk.approve(conn, "PK-b00000000004", now=NOW)
    led.settle_position("PK-b00000000004", "WIN")
    with pytest.raises(desk.DeskError, match="part of the record"):
        desk.change_tier(conn, "PK-b00000000004", "premium", now=NOW)
    with pytest.raises(desk.DeskError) as unknown:
        desk.change_tier(conn, "PK-fffff0000000", "premium", now=NOW)
    assert unknown.value.status == 404
    assert set_tier(conn, "PK-b00000000004", "free")["tier"] == TIER_FREE   # same tier: no-op


def test_undo_clears_the_tier(conn) -> None:
    _record(conn, _pick("PK-b00000000005"))
    desk.approve(conn, "PK-b00000000005", now=NOW, tier="premium")
    desk.undo(conn, "PK-b00000000005", now=NOW)
    row = db.get_pick(conn, "PK-b00000000005")
    assert (row["hitl_status"], row["tier"], row["tier_set_at"]) == (HITL_PENDING, None, None)
    assert desk.approve(conn, "PK-b00000000005", now=NOW)["tier"] == TIER_FREE


# ==========================================================================
# Split Discord cards
# ==========================================================================
def test_cards_split_free_and_premium(conn, tmp_path) -> None:
    _record(conn, _pick("PK-c00000000001"),
            _pick("PK-c00000000002", event="ev-2", selection="Milwaukee Bucks", line_point=3.5))
    desk.approve(conn, "PK-c00000000001", now=NOW)
    desk.approve(conn, "PK-c00000000002", now=NOW, tier="premium")
    paths = desk.write_discord_drop(conn, MOCK, tmp_path / "drops", NOW)
    free = paths[TIER_FREE].read_text(encoding="utf-8")
    premium = paths[TIER_PREMIUM].read_text(encoding="utf-8")
    assert free.startswith("# Breeze Bets - Free Plays") and "**Celtics -3.5**" in free
    assert "**Bucks" not in free
    assert premium.startswith("# Breeze Bets - Premium Card") and "**Bucks +3.5**" in premium
    assert "**Celtics" not in premium
    for md in (free, premium):
        assert md.count("@everyone") == 1 and HONEST_TAGLINE in md and "MOCK DESK" in md
    cards = desk.discord_cards(conn, MOCK, tmp_path / "drops", NOW)
    assert [(c["tier"], c["label"]) for c in cards] == [(TIER_FREE, "Free play"),
                                                        (TIER_PREMIUM, "Premium")]
    assert cards[1]["markdown"] == premium


def test_live_cards_have_no_mock_suffix(tmp_path) -> None:
    assert desk.drop_path(tmp_path, db.ORIGIN_LIVE, NOW, "premium").name == "2026-11-14_premium_card.md"
    assert desk.drop_path(tmp_path, MOCK, NOW).name == "2026-11-14_free_card_mock.md"


# ==========================================================================
# Tracker: free / premium split and CLV
# ==========================================================================
def test_tracker_splits_by_tier_and_reports_clv(conn) -> None:
    led = _record(conn, _pick("PK-d00000000001", units=2.0), _pick("PK-d00000000002", event="ev-2"),
                  _pick("PK-d00000000003", event="ev-3"), _pick("PK-d00000000004", event="ev-4"))
    desk.approve(conn, "PK-d00000000001", now=NOW)
    desk.approve(conn, "PK-d00000000002", now=NOW)
    desk.approve(conn, "PK-d00000000003", now=NOW, tier="premium")
    desk.approve(conn, "PK-d00000000004", now=NOW, tier="premium")   # open: not in the record
    for pid, clv in (("PK-d00000000001", 2.5), ("PK-d00000000002", -1.0),
                     ("PK-d00000000003", 4.0)):
        db.update_pick(conn, pid, clv_percent=clv)
    led.settle_position("PK-d00000000001", "WIN")
    led.settle_position("PK-d00000000002", "LOSS")
    led.settle_position("PK-d00000000003", "LOSS")
    t = desk.tracker(conn, MOCK, NOW)
    s = t["summary"]
    assert (s["picks"], s["clv_n"]) == (3, 3)
    assert s["clv_avg"] == pytest.approx(round((2.5 - 1.0 + 4.0) / 3, 2))
    assert s["clv_beat_rate"] == pytest.approx(66.7)
    free, premium = t["by_tier"][TIER_FREE], t["by_tier"][TIER_PREMIUM]
    assert (free["record"], premium["record"]) == ("1-1-0", "0-1-0")
    assert free["clv_avg"] == pytest.approx(0.75) and free["clv_beat_rate"] == 50.0
    assert premium["net_units"] == pytest.approx(-1.0)
    assert {v["pick_id"]: v["tier"] for v in t["open"]} == {"PK-d00000000004": TIER_PREMIUM}


def test_tracker_without_closing_lines_has_no_clv(conn) -> None:
    led = _record(conn, _pick("PK-d00000000005"))
    desk.approve(conn, "PK-d00000000005", now=NOW)
    led.settle_position("PK-d00000000005", "WIN")
    s = desk.tracker(conn, MOCK, NOW)["summary"]
    assert (s["clv_n"], s["clv_avg"], s["clv_beat_rate"]) == (0, None, None)


# ==========================================================================
# Migration: an existing ledger files its signed-off picks under FREE
# ==========================================================================
def test_legacy_ledger_backfills_free_for_signed_off_picks(tmp_path) -> None:
    path = tmp_path / "legacy.db"
    c = db.connect(path)
    c.close()
    raw = sqlite3.connect(path)
    raw.execute("ALTER TABLE ledger_picks DROP COLUMN tier")
    raw.execute("ALTER TABLE ledger_picks DROP COLUMN tier_set_at")
    cols = ("pick_id, created_at, event_id, market_type, selection, offered_american, "
            "offered_decimal, hitl_status, signoff_at")
    raw.execute(f"INSERT INTO ledger_picks ({cols}) VALUES ('PK-signed', '2026-11-01T00:00:00Z', "
                "'e1', 'h2h', 'X', 110, 2.1, 'APPROVED', '2026-11-01T09:00:00Z')")
    raw.execute(f"INSERT INTO ledger_picks ({cols}) VALUES ('PK-pending', '2026-11-01T00:00:00Z', "
                "'e2', 'h2h', 'Y', 110, 2.1, 'PENDING_CAPTAIN_HZ', NULL)")
    raw.commit()
    raw.close()
    c = db.connect(path)
    try:
        signed, pending = db.get_pick(c, "PK-signed"), db.get_pick(c, "PK-pending")
        assert (signed["tier"], signed["tier_set_at"]) == (TIER_FREE, "2026-11-01T09:00:00Z")
        assert pending["tier"] is None
    finally:
        c.close()


# ==========================================================================
# CLI
# ==========================================================================
def test_cli_approve_with_tier_and_set_tier(conn, tmp_path, capsys) -> None:
    hours = (db.utcnow() - NOW).total_seconds() / 3600 + 3          # starts in about 3 hours
    _record(conn, _pick("PK-e00000000001", hours=hours))
    path = str(tmp_path / "takeover.db")
    assert run_pipeline.main(["--approve", "PK-e00000000001", "--tier", "premium",
                              "--db", path]) == 0
    assert "premium card" in capsys.readouterr().out
    assert db.get_pick(conn, "PK-e00000000001")["tier"] == TIER_PREMIUM
    assert run_pipeline.main(["--set-tier", "PK-e00000000001", "--tier", "FREE",
                              "--db", path]) == 0
    assert db.get_pick(conn, "PK-e00000000001")["tier"] == TIER_FREE
    assert run_pipeline.main(["--set-tier", "PK-ffffffffffff", "--tier", "free",
                              "--db", path]) == 1
    for bad in (["--set-tier", "PK-e00000000001"], ["--tier", "premium", "--settle"]):
        with pytest.raises(SystemExit):
            run_pipeline.main([*bad, "--db", path])


# ==========================================================================
# HTTP
# ==========================================================================
@pytest.fixture
def server(tmp_path):
    app = web_server.DeskApp(db_path=tmp_path / "takeover.db",
                             settings_path=tmp_path / "web_desk.json",
                             lock_path=tmp_path / ".lock", discord_dir=tmp_path / "drops",
                             runner=lambda a: (0, "ok"))
    web_server.save_settings(app.settings_path, {"mode": "mock"})
    srv = web_server.make_server(app, "127.0.0.1", 0)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield app, f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()
    srv.server_close()


def _post(url: str, body: dict | None = None, csrf: bool = True) -> tuple[int, dict]:
    req = urllib.request.Request(url, data=json.dumps(body or {}).encode(), method="POST",
                                 headers={"X-Breeze-Desk": "1"} if csrf else {})
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            return r.status, json.load(r)
    except urllib.error.HTTPError as e:
        return e.code, json.load(e)


def test_http_approve_premium_and_move_tier(server) -> None:
    app, base = server
    conn = app.connect()
    hours = (db.utcnow() - NOW).total_seconds() / 3600 + 2
    _record(conn, _pick("PK-f00000000001", hours=hours))
    conn.close()
    pick = f"{base}/api/picks/PK-f00000000001"
    assert _post(f"{pick}/approve", {"tier": "vip"})[0] == 400
    status, out = _post(f"{pick}/approve", {"tier": "premium"})
    assert status == 200 and out["pick"]["tier"] == TIER_PREMIUM
    premium = Path(out["drop_paths"][TIER_PREMIUM]).read_text(encoding="utf-8")
    assert "**Celtics -3.5**" in premium
    assert "**Celtics" not in Path(out["drop_paths"][TIER_FREE]).read_text(encoding="utf-8")

    assert _post(f"{pick}/tier", {"tier": "free"}, csrf=False)[0] == 403
    assert _post(f"{pick}/tier", {})[0] == 400
    status, out = _post(f"{pick}/tier", {"tier": "free"})
    assert status == 200 and out["pick"]["tier"] == TIER_FREE
    assert "**Celtics -3.5**" in Path(out["drop_paths"][TIER_FREE]).read_text(encoding="utf-8")
    with urllib.request.urlopen(f"{base}/api/tracker", timeout=10) as r:
        tracker = json.load(r)
    assert set(tracker["by_tier"]) == {TIER_FREE, TIER_PREMIUM}
    assert "clv_avg" in tracker["summary"]
