"""Web desk: DECLINED status, slate / tracker / Discord read models, jobs, scheduler, HTTP API.

Every test uses tmp_path for SQLite, settings, locks and OUTBOX output; nothing
touches the network, data/takeover.db or the real OUTBOX/.
Run: python -m pytest tests/test_web_desk.py -v
"""

from __future__ import annotations

import json
import sqlite3
import sys
import threading
import time
import urllib.error
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import run_pipeline  # noqa: E402
import web_server  # noqa: E402
from src.core import database as db  # noqa: E402
from src.core.engine import HITL_APPROVED, HITL_DECLINED, HITL_PENDING  # noqa: E402
from src.core.ledger import Ledger  # noqa: E402
from src.data.market_fetcher import run_scan  # noqa: E402
from src.data.prop_market_fetcher import run_prop_scan  # noqa: E402
from src.data.settler import run_mock_settlement  # noqa: E402
from src.web import desk  # noqa: E402
from src.web.jobs import (  # noqa: E402
    DESK_SPORTS,
    DailyScheduler,
    JobBusy,
    JobRunner,
    RunLock,
    commands_for,
    next_run_at,
)

# Local noon: picks a few hours later fall on the same local day in any timezone.
NOW = datetime(2026, 11, 14, 12, 0).astimezone()
MOCK = db.ORIGIN_MOCK


@pytest.fixture
def conn(tmp_path):
    c = db.connect(tmp_path / "takeover.db")
    yield c
    c.close()


def _pick(pick_id: str, *, event: str = "ev-nba-1", sport: str = "basketball_nba",
          market: str = "spreads", status: str = HITL_PENDING, units: float = 1.0,
          hours: float = 3.0, **kw) -> dict:
    prop = market.startswith("player_")
    p = {"pick_id": pick_id, "commence_time": db.iso(NOW + timedelta(hours=hours)),
         "sport": sport, "event_id": event, "event_name": "Boston Celtics @ Milwaukee Bucks",
         "market_type": market,
         "selection": "Jayson Tatum Over" if prop else "Boston Celtics",
         "line_point": 27.5 if prop else -3.5, "bookmaker": "DraftKings",
         "offered_american": 105, "fair_prob": 0.55, "ev_percent": 12.75,
         "kelly_units": units, "hitl_status": status, "hitl_reason": "", "origin": MOCK}
    if prop:
        p["model_prob"] = 0.6
        p["analysis"] = json.dumps({"projection": 30.1, "z": 0.8})
    p.update(kw)
    return p


def _record(conn, *picks) -> Ledger:
    led = Ledger(conn, MOCK)
    for p in picks:
        assert led.record_position(p)
    return led


# ==========================================================================
# DECLINED status
# ==========================================================================
def test_decline_pending_pick_leaves_queue_and_exposure(conn) -> None:
    led = _record(conn, _pick("PK-aaaaaa000001"))
    assert db.open_event_exposure(conn, "ev-nba-1") == 1.0
    p = desk.decline(conn, "PK-aaaaaa000001", now=NOW)
    assert p["hitl_status"] == HITL_DECLINED
    assert "Declined by Captain HZ" in p["hitl_reason"]
    assert db.open_event_exposure(conn, "ev-nba-1") == 0.0
    assert led.state().open_positions == 0
    assert led.journal("PK-aaaaaa000001")[-1]["kind"] == "RESERVATION_RELEASED"
    assert desk.slate(conn, MOCK, NOW)["sports"] == []


def test_decline_auto_approved_pick_releases_stake(conn) -> None:
    led = _record(conn, _pick("PK-aaaaaa000002", status=HITL_APPROVED, units=1.5))
    cash_before = led.cash()
    desk.decline(conn, "PK-aaaaaa000002", now=NOW)
    assert led.cash() == pytest.approx(cash_before + 1.5)
    assert led.journal("PK-aaaaaa000002")[-1]["kind"] == "STAKE_RELEASED"


def test_decline_refuses_signed_off_and_settled_picks(conn) -> None:
    led = _record(conn, _pick("PK-aaaaaa000003"), _pick("PK-aaaaaa000004", event="ev-2",
                                                        status=HITL_APPROVED))
    desk.approve(conn, "PK-aaaaaa000003", now=NOW)
    with pytest.raises(desk.DeskError, match="signed off"):
        desk.decline(conn, "PK-aaaaaa000003", now=NOW)
    led.settle_position("PK-aaaaaa000004", "WIN")
    with pytest.raises(desk.DeskError, match="settled"):
        desk.decline(conn, "PK-aaaaaa000004", now=NOW)
    with pytest.raises(desk.DeskError) as unknown:
        desk.decline(conn, "PK-ffffff000000", now=NOW)
    assert unknown.value.status == 404


def test_declined_pick_cannot_be_approved(conn) -> None:
    _record(conn, _pick("PK-aaaaaa000005"))
    desk.decline(conn, "PK-aaaaaa000005", now=NOW)
    with pytest.raises(desk.DeskError, match="DECLINED"):
        desk.approve(conn, "PK-aaaaaa000005", now=NOW)


def test_legacy_ledger_is_rebuilt_to_accept_declined(tmp_path) -> None:
    path = tmp_path / "legacy.db"
    legacy = db.SCHEMA_SQL.replace(", 'DECLINED')", ")")
    assert legacy != db.SCHEMA_SQL
    raw = sqlite3.connect(path)
    raw.executescript(legacy)
    raw.execute("INSERT INTO ledger_picks (pick_id, created_at, event_id, market_type, selection,"
                " offered_american, offered_decimal, hitl_status) VALUES ('PK-old', "
                "'2026-11-01T00:00:00Z', 'e', 'h2h', 'X', 110, 2.1, 'PENDING_CAPTAIN_HZ')")
    raw.commit()
    with pytest.raises(sqlite3.IntegrityError):
        raw.execute("UPDATE ledger_picks SET hitl_status = 'DECLINED'")
    raw.close()
    c = db.connect(path)
    try:
        db.update_pick(c, "PK-old", hitl_status=HITL_DECLINED)
        assert db.get_pick(c, "PK-old")["hitl_status"] == HITL_DECLINED
    finally:
        c.close()


def test_cli_decline(conn, tmp_path, capsys) -> None:
    _record(conn, _pick("PK-aaaaaa000006"))
    assert run_pipeline.main(["--decline", "PK-aaaaaa000006", "--db",
                              str(tmp_path / "takeover.db")]) == 0
    assert "DECLINED" in capsys.readouterr().out
    assert db.get_pick(conn, "PK-aaaaaa000006")["hitl_status"] == HITL_DECLINED


# ==========================================================================
# Slate
# ==========================================================================
def test_slate_groups_by_sport_and_hides_empty_sports(conn) -> None:
    _record(conn,
            _pick("PK-bbbbbb000001"),
            _pick("PK-bbbbbb000002", market="player_points"),
            _pick("PK-bbbbbb000003", event="ev-nba-2"),
            _pick("PK-bbbbbb000004", event="ev-old", hours=-48))     # an earlier slate
    desk.decline(conn, "PK-bbbbbb000003", now=NOW)
    desk.approve(conn, "PK-bbbbbb000001", now=NOW)
    s = desk.slate(conn, MOCK, NOW)
    assert [sp["league"] for sp in s["sports"]] == ["NBA"]            # no NFL picks -> no NFL
    nba = s["sports"][0]
    assert [v["pick_id"] for v in nba["games"]] == ["PK-bbbbbb000001"]
    assert [v["pick_id"] for v in nba["props"]] == ["PK-bbbbbb000002"]
    assert nba["games"][0]["signed"] and not nba["props"][0]["signed"]
    assert (nba["queue"], nba["signed"], s["stale_open"]) == (1, 1, 1)
    prop = nba["props"][0]
    assert prop["label"] == "Jayson Tatum Over 27.5 PTS"
    assert prop["delta"] == pytest.approx(2.6)
    assert nba["games"][0]["label"] == "Boston Celtics -3.5"


def test_slate_shows_nfl_only_when_it_has_picks(conn) -> None:
    _record(conn, _pick("PK-cccccc000001", sport="americanfootball_nfl", event="ev-nfl",
                        market="h2h"))
    assert [sp["league"] for sp in desk.slate(conn, MOCK, NOW)["sports"]] == ["NFL"]
    assert desk.slate(conn, db.ORIGIN_LIVE, NOW)["sports"] == []
    with pytest.raises(desk.DeskError):
        desk.slate(conn, "SYNTHETIC_SEED", NOW)


def test_approve_all_qualified_skips_picks_failing_filters(conn) -> None:
    _record(conn, _pick("PK-dddddd000001"),
            _pick("PK-dddddd000002", event="ev-2", fair_prob=0.40, ev_percent=-18.0))
    out = desk.approve_all_qualified(conn, MOCK, NOW)
    assert [v["pick_id"] for v in out["approved"]] == ["PK-dddddd000001"]
    assert out["skipped"][0]["pick_id"] == "PK-dddddd000002"
    assert "EV" in out["skipped"][0]["reason"]


# ==========================================================================
# Tracker + Discord drop
# ==========================================================================
def test_tracker_counts_only_signed_off_picks_in_units(conn) -> None:
    led = _record(conn,
                  _pick("PK-eeeeee000001", units=2.0),                         # game WIN
                  _pick("PK-eeeeee000002", event="ev-2", market="player_points"),  # prop LOSS
                  _pick("PK-eeeeee000003", event="ev-3", sport="americanfootball_nfl",
                        market="h2h"),                                        # NFL VOID
                  _pick("PK-eeeeee000004", event="ev-4", units=1.5),          # still open
                  _pick("PK-eeeeee000005", event="ev-5", status=HITL_APPROVED))  # never signed
    for pid in ("PK-eeeeee000001", "PK-eeeeee000002", "PK-eeeeee000003", "PK-eeeeee000004"):
        desk.approve(conn, pid, now=NOW)
    led.settle_position("PK-eeeeee000001", "WIN")
    led.settle_position("PK-eeeeee000002", "LOSS")
    led.cancel_position("PK-eeeeee000003", "0 snaps (did not play)")
    led.settle_position("PK-eeeeee000005", "WIN")                           # unsigned: excluded
    t = desk.tracker(conn, MOCK, NOW)
    s = t["summary"]
    assert (s["wins"], s["losses"], s["voids"], s["record"]) == (1, 1, 1, "1-1-1")
    assert s["net_units"] == pytest.approx(2.0 * 1.05 - 1.0)
    assert s["risked_units"] == pytest.approx(3.0)                  # voids hand the stake back
    assert s["roi_percent"] == pytest.approx(round(1.1 / 3.0 * 100, 2))
    assert s["win_rate"] == 50.0
    assert (s["open_count"], s["open_units"]) == (1, 1.5)
    assert t["by_kind"]["game"]["record"] == "1-0-1"
    assert t["by_kind"]["prop"]["record"] == "0-1-0"
    assert t["by_league"]["NFL"]["record"] == "0-0-1"
    assert t["curve"][-1]["net"] == pytest.approx(1.1)
    assert {v["pick_id"] for v in t["settled"]} == {"PK-eeeeee000001", "PK-eeeeee000002",
                                                   "PK-eeeeee000003"}


def test_tracker_empty_has_no_roi(conn) -> None:
    s = desk.tracker(conn, MOCK, NOW)["summary"]
    assert (s["record"], s["roi_percent"], s["win_rate"]) == ("0-0-0", None, None)


def test_discord_drop_lists_only_signed_off_open_picks(conn, tmp_path) -> None:
    _record(conn, _pick("PK-ffffff000001"),
            _pick("PK-ffffff000002", event="ev-2", market="player_points"),
            _pick("PK-ffffff000003", event="ev-3", selection="Milwaukee Bucks"))
    desk.approve(conn, "PK-ffffff000001", now=NOW)
    desk.approve(conn, "PK-ffffff000002", now=NOW)
    desk.decline(conn, "PK-ffffff000003", now=NOW)
    md = desk.discord_drop(conn, MOCK, NOW)
    assert "Boston Celtics -3.5" in md and "Jayson Tatum Over 27.5 PTS" in md
    assert "Milwaukee Bucks" not in md.split("## NBA", 1)[1].replace(
        "Boston Celtics @ Milwaukee Bucks", "")
    assert "MOCK DESK" in md and "2 plays, 2.00u" in md and "-130" in md
    path = desk.write_discord_drop(conn, MOCK, tmp_path / "drops", NOW)
    assert path.name.endswith("_approved_drop_mock.md")
    assert path.read_text(encoding="utf-8") == md


def test_discord_drop_empty_card(conn) -> None:
    assert "No official plays signed off yet" in desk.discord_drop(conn, MOCK, NOW)


def test_mock_scan_approve_settle_flows_into_tracker(conn, tmp_path) -> None:
    sports = ("basketball_nba", "americanfootball_nfl")
    utc_now = NOW.astimezone(timezone.utc)
    run_scan(conn, sports, mock=True, output_path=tmp_path / "games.json", now=utc_now)
    run_prop_scan(conn, sports, mock=True, output_path=tmp_path / "props.json", now=utc_now)
    s = desk.slate(conn, MOCK, NOW)
    assert {sp["league"] for sp in s["sports"]} == {"NBA", "NFL"}
    assert any(sp["props"] for sp in s["sports"]) and any(sp["games"] for sp in s["sports"])
    approved = desk.approve_all_qualified(conn, MOCK, NOW)["approved"]
    assert approved and all(v["signed"] for v in approved)
    assert all(v["units"] == 1.0 for v in approved if v["kind"] == "prop")
    run_mock_settlement(conn)
    t = desk.tracker(conn, MOCK, NOW)
    assert len(t["settled"]) == len(approved)
    assert t["summary"]["net_units"] == pytest.approx(
        sum(v["net_units"] for v in t["settled"]), abs=0.02)


# ==========================================================================
# Jobs + scheduler
# ==========================================================================
def test_commands_settle_before_scan_and_mock_skips_network() -> None:
    live = commands_for("daily", "live", Path("x.db"))
    assert [c[0] for c in live] == ["--settle", "--update-stats"]
    assert "--scan" in live[1] and DESK_SPORTS in live[1] and live[1][-2:] == ["--db", "x.db"]
    mock = commands_for("daily", "mock")
    assert mock == [["--settle", "--mock"], ["--mock", "--sport", DESK_SPORTS]]
    with pytest.raises(ValueError):
        commands_for("backtest", "live")


def test_next_run_at_nine_local() -> None:
    tz = timezone(timedelta(hours=-5))
    assert next_run_at(datetime(2026, 11, 14, 8, 59, tzinfo=tz)) == \
        datetime(2026, 11, 14, 9, 0, tzinfo=tz)
    assert next_run_at(datetime(2026, 11, 14, 9, 0, tzinfo=tz)) == \
        datetime(2026, 11, 15, 9, 0, tzinfo=tz)


def test_scheduler_fires_once_when_due() -> None:
    fired = threading.Event()
    clock = {"now": datetime(2026, 11, 14, 8, 59, 59, 500000).astimezone()}

    def tick() -> datetime:
        clock["now"] += timedelta(milliseconds=200)
        return clock["now"]
    sched = DailyScheduler(fired.set, clock=tick)
    sched.start()
    assert fired.wait(5)
    sched.stop()


def test_job_runner_is_exclusive_and_records_steps(tmp_path) -> None:
    gate, seen = threading.Event(), []

    def runner(args):
        seen.append(args)
        gate.wait(5)
        return 0, "line 1\nall good"
    jr = JobRunner(tmp_path / "lock", db_path=tmp_path / "t.db", runner=runner)
    jr.submit("daily", "mock")
    with pytest.raises(JobBusy):
        jr.submit("scan", "mock")
    assert not RunLock(tmp_path / "lock").acquire()            # other processes are locked out
    gate.set()
    for _ in range(100):
        if jr.status()["current"] is None:
            break
        time.sleep(0.05)
    last = jr.status()["last"]
    assert last["status"] == "ok" and len(last["steps"]) == 2
    assert last["steps"][1]["tail"].endswith("all good")
    assert seen[0][:2] == ["--settle", "--mock"]
    assert RunLock(tmp_path / "lock").acquire()                # released


def test_failed_step_marks_job_failed(tmp_path) -> None:
    jr = JobRunner(tmp_path / "lock", runner=lambda a: (1, "ERROR: boom"))
    assert jr.run("settle", "live").status == "failed"


# ==========================================================================
# HTTP API
# ==========================================================================
@pytest.fixture
def server(tmp_path):
    calls: list[list[str]] = []
    app = web_server.DeskApp(db_path=tmp_path / "takeover.db",
                             settings_path=tmp_path / "web_desk.json",
                             lock_path=tmp_path / ".lock", discord_dir=tmp_path / "drops",
                             runner=lambda a: (calls.append(a), (0, "ok"))[1])
    web_server.save_settings(app.settings_path, {"mode": "mock"})
    srv = web_server.make_server(app, "127.0.0.1", 0)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{srv.server_address[1]}"
    yield app, base, calls
    srv.shutdown()
    srv.server_close()


def _call(url: str, method: str = "GET", body: dict | None = None,
          headers: dict | None = None) -> tuple[int, dict | str]:
    h = {"X-Breeze-Desk": "1"} if method == "POST" else {}
    h.update(headers or {})
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(url, data=data, method=method, headers=h)
    if method == "POST" and data is None:
        req.data = b""
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            raw, status, ctype = r.read(), r.status, r.headers.get("Content-Type", "")
    except urllib.error.HTTPError as e:
        raw, status, ctype = e.read(), e.code, e.headers.get("Content-Type", "")
    return status, json.loads(raw) if "json" in ctype else raw.decode()


def test_http_serves_frontend_and_blocks_traversal(server) -> None:
    _, base, _ = server
    status, html = _call(base + "/")
    assert status == 200 and "Breeze Bets" in html and "/static/app.js" in html
    assert _call(base + "/static/styles.css")[0] == 200
    assert _call(base + "/static/../web_server.py")[0] == 404
    assert _call(base + "/static/%2e%2e/web_server.py")[0] == 404


def test_http_guards_host_and_csrf(server) -> None:
    app, base, _ = server
    conn = app.connect()
    _record(conn, _pick("PK-111111000001", commence_time=db.iso(db.utcnow() + timedelta(hours=2))))
    conn.close()
    url = base + "/api/picks/PK-111111000001/approve"
    assert _call(url, "POST", headers={"X-Breeze-Desk": ""})[0] == 403
    assert _call(base + "/api/slate", headers={"Host": "evil.example:8000"})[0] == 403
    assert db.get_pick(app.connect(), "PK-111111000001")["signoff_at"] is None


def test_http_approve_decline_slate_tracker_and_drop(server) -> None:
    app, base, _ = server
    soon = db.iso(db.utcnow() + timedelta(hours=2))
    conn = app.connect()
    _record(conn, _pick("PK-222222000001", commence_time=soon),
            _pick("PK-222222000002", event="ev-2", market="player_points", commence_time=soon),
            _pick("PK-222222000003", event="ev-3", commence_time=soon,
                  sport="americanfootball_nfl", market="h2h"))
    conn.close()
    status, slate = _call(base + "/api/slate")
    assert status == 200 and slate["origin"] == MOCK
    assert [s["league"] for s in slate["sports"]] == ["NBA", "NFL"]

    status, out = _call(base + "/api/picks/PK-222222000001/approve", "POST")
    assert status == 200 and out["pick"]["signed"]
    assert Path(out["drop_path"]).read_text(encoding="utf-8").count("Boston Celtics -3.5") == 1
    status, out = _call(base + "/api/picks/PK-222222000003/decline", "POST")
    assert status == 200 and out["pick"]["hitl_status"] == HITL_DECLINED
    assert _call(base + "/api/picks/PK-222222000003/approve", "POST")[0] == 409
    assert _call(base + "/api/picks/PK-999999000000/approve", "POST")[0] == 404

    _, slate = _call(base + "/api/slate")
    assert [s["league"] for s in slate["sports"]] == ["NBA"]          # NFL emptied -> hidden
    _, tr = _call(base + "/api/tracker")
    assert tr["summary"]["open_count"] == 1
    _, drop = _call(base + "/api/discord-drop")
    assert "Boston Celtics -3.5" in drop["markdown"] and "PTS" not in drop["markdown"]

    status, out = _call(base + "/api/picks/approve-all", "POST")
    assert status == 200 and [v["pick_id"] for v in out["approved"]] == ["PK-222222000002"]


def test_http_jobs_and_settings(server) -> None:
    app, base, calls = server
    status, out = _call(base + "/api/scan", "POST")
    assert status == 202 and out["job"]["kind"] == "scan"
    for _ in range(100):
        if _call(base + "/api/status")[1]["jobs"]["current"] is None:
            break
        time.sleep(0.05)
    assert calls[-1][:3] == ["--mock", "--sport", DESK_SPORTS]
    assert _call(base + "/api/settle", "POST", {"mode": "live"})[0] == 202
    for _ in range(100):
        if _call(base + "/api/status")[1]["jobs"]["current"] is None:
            break
        time.sleep(0.05)
    assert calls[-1][0] == "--settle" and "--mock" not in calls[-1]
    assert _call(base + "/api/scan", "POST", {"mode": "paper"})[0] == 400

    status, st = _call(base + "/api/settings", "POST", {"mode": "live"})
    assert status == 200 and st["mode"] == "live" and st["origin"] == db.ORIGIN_LIVE
    assert 0 < st["seconds_to_next_run"] <= 86400
    assert _call(base + "/api/settings", "POST", {"mode": "yolo"})[0] == 400
    assert web_server.load_settings(app.settings_path)["mode"] == "live"
