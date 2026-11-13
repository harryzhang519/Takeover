"""Player-prop pipeline: mock scan end-to-end, gates, quota/slate guards, ledger,
settlement, stats ingestion, DB migration, run_pipeline and monitor integration.

Every test uses tmp_path for SQLite and JSON output; nothing touches the
network, data/takeover.db or OUTBOX/.
Run: python -m pytest tests/test_prop_pipeline.py -v
"""

from __future__ import annotations

import json
import re
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import infinite_monitor as mon  # noqa: E402
from src.core import database as db  # noqa: E402
from src.core.ledger import Ledger, MandateViolation, mandate_violations  # noqa: E402
from src.core.schema_validator import load_schema, validate  # noqa: E402
from src.data import prop_market_fetcher as pmf  # noqa: E402
from src.data import prop_stats_fetcher as psf  # noqa: E402
from src.data import settler  # noqa: E402
from src.data.prop_market_fetcher import (  # noqa: E402
    MOCK_PLAYER_FINALS,
    PropConfig,
    eastern_date,
    eastern_offset_hours,
    market_fair,
    parse_prop_quotes,
    run_prop_scan,
    slate_events,
)

NOW = datetime(2026, 11, 10, 12, 0, tzinfo=timezone.utc)
SPORTS = ("basketball_nba", "americanfootball_nfl")
SCHEMA = load_schema(ROOT / "schemas" / "prop_scan_schema.json")


@pytest.fixture
def conn(tmp_path):
    c = db.connect(tmp_path / "takeover.db")
    yield c
    c.close()


@pytest.fixture
def mock_scan(conn, tmp_path):
    return run_prop_scan(conn, SPORTS, mock=True, output_path=tmp_path / "props.json", now=NOW)


def _by_reason(scan, text):
    return [r for r in scan["rejections"] if text in r["reason"]]


# ==========================================================================
# Mock scan end-to-end
# ==========================================================================
def test_mock_scan_validates_against_schema(mock_scan, tmp_path) -> None:
    errors, _ = validate(mock_scan, SCHEMA)
    assert errors == []
    on_disk = json.loads((tmp_path / "props.json").read_text(encoding="utf-8"))
    assert validate(on_disk, SCHEMA)[0] == []
    try:
        import jsonschema
    except ImportError:
        return
    jsonschema.Draft7Validator(SCHEMA).validate(on_disk)


def test_mock_scan_flags_the_designed_plays(mock_scan) -> None:
    flagged = {(p["player"], p["market_type"], p["side"]) for p in mock_scan["props"]}
    assert flagged == {("BOS Mock Wing", "player_points", "Under"),     # hot streak regressed
                       ("GSW Mock Wing", "player_points", "Over"),      # key teammate newly Out
                       ("BUF Mock WR1", "player_receptions", "Over")}   # count (NB)
    for p in mock_scan["props"]:
        assert p["hitl_status"] == "PENDING_CAPTAIN_HZ"
        assert p["market_ev_percent"] > 0 and p["ev_percent"] >= 4.0
        assert p["z_score"] >= 0.65 and len(p["triggers"]) == 2       # AND gate default
        assert p["offered_american"] >= -130
        assert p["market_fair_probability"] < p["model_fair_probability"] \
            <= p["model_raw_probability"]
        assert (p["delta"] > 0) == (p["side"] == "Over")
    assert mock_scan["config"]["divergence_mode"] == "all"


def test_mock_scan_rejects_each_failure_class(mock_scan) -> None:
    assert _by_reason(mock_scan, "worse than -130 floor")
    assert _by_reason(mock_scan, "No market benchmark")
    assert _by_reason(mock_scan, "GSW Mock Star listed Out")
    assert _by_reason(mock_scan, "Market EV")
    shrunk_only = [r for r in _by_reason(mock_scan, "Divergence below threshold")
                   if r["selection"].startswith("DET Mock RB Under")]
    assert shrunk_only, "a shrunk-EV-only play must fail the default AND gate"
    s = mock_scan["ingestion"]["stats"]
    assert s["injury_out"] == 1 and s["no_benchmark"] == 1 and s["candidates"] == \
        len(mock_scan["props"])
    assert s["slate_gate"] == "bypassed (mock)" and s["quota"]["basketball_nba"] == "mock"


def test_injury_flags_and_usage_boost(mock_scan) -> None:
    gsw = next(p for p in mock_scan["props"] if p["player"] == "GSW Mock Wing")
    assert gsw["injury_flag"] and "key teammate GSW Mock Star (Out)" in gsw["hitl_reason"]
    assert gsw["projection"]["usage_boost"] == pytest.approx(0.04)
    assert "Usage +4% (key teammate GSW Mock Star newly Out)" in gsw["hitl_reason"]
    bos = next(p for p in mock_scan["props"] if p["player"] == "BOS Mock Wing")
    assert not bos["injury_flag"] and "usage_boost" not in bos["projection"]
    assert mon.is_injury_flag(gsw)                      # monitor highlights it


def test_questionable_player_gets_no_boost(conn) -> None:
    """LAL Mock Starter is himself Questionable: flagged elsewhere, never boosted."""
    src = pmf.MockModelSource()
    hist = src.players[psf.player_key("LAL Mock Starter")]
    proj, _ = pmf.project_player(src, "NBA", hist, "player_points", "NBA_GSW", "NBA_LAL",
                                 PropConfig())
    assert "usage_boost" not in proj.details


def test_long_absence_not_boosted() -> None:
    """A teammate out for weeks is already in the player's logs: no bump."""
    class Stale(pmf.MockModelSource):
        def recently_active(self, sport, canonical_id, name):
            return False
    src = Stale()
    hist = src.players[psf.player_key("GSW Mock Wing")]
    proj, _ = pmf.project_player(src, "NBA", hist, "player_points", "NBA_GSW", "NBA_LAL",
                                 PropConfig())
    assert "usage_boost" not in proj.details


def test_every_prop_is_a_flat_official_unit(conn, mock_scan) -> None:
    assert mock_scan["props"]
    per_event: dict[str, float] = {}
    for p in mock_scan["props"]:
        per_event[p["event_id"]] = per_event.get(p["event_id"], 0.0) + p["official_units"]
        assert p["official_units"] == 1.0
        assert p["sizing"]["method"] == "flat_official"
        assert p["raw_kelly_units"] == p["sizing"]["raw_kelly_units"] > 0     # diagnostic kept
    assert all(v <= 2.0 + 1e-9 for v in per_event.values())
    assert {r["kelly_units"] for r in db.fetch_picks(conn, "market_type LIKE 'player_%'")} \
        == {1.0}
    cfg = mock_scan["config"]
    assert cfg["official_prop_units"] == 1.0 and cfg["prop_event_cap_units"] == 2.0
    assert cfg["event_total_cap_units"] == 4.0


def test_official_prop_units_clamps_raw_kelly() -> None:
    from src.core.prop_model import MIN_PROP_OFFICIAL_UNITS, official_prop_units
    assert MIN_PROP_OFFICIAL_UNITS == 1.0
    assert official_prop_units(0.12) == official_prop_units(3.4) == 1.0


def _game_pick(units: float, pick_id: str = "GAME1", **kw) -> dict:
    p = {"pick_id": pick_id, "commence_time": "2026-11-10T17:00:00Z",
         "sport": "basketball_nba", "event_id": "mock-nba-lal-gsw",
         "event_name": "Los Angeles Lakers @ Golden State Warriors", "market_type": "spreads",
         "selection": "Los Angeles Lakers", "line_point": 4.5, "bookmaker": "DraftKings",
         "offered_american": 105, "fair_prob": 0.52, "ev_percent": 6.6, "kelly_units": units,
         "hitl_status": "APPROVED", "origin": db.ORIGIN_MOCK}
    p.update(kw)
    return p


def test_full_game_line_bet_does_not_starve_props(conn, tmp_path) -> None:
    base_conn = db.connect(tmp_path / "base.db")
    try:
        baseline = run_prop_scan(base_conn, SPORTS, mock=True, output_path=tmp_path / "b.json",
                                 now=NOW)
    finally:
        base_conn.close()
    want = [p["pick_id"] for p in baseline["props"] if p["event_id"] == "mock-nba-lal-gsw"]
    assert want, "mock slate should flag at least one LAL @ GSW prop"
    Ledger(conn, db.ORIGIN_MOCK, starting_bankroll=100.0).record_position(_game_pick(2.0))
    scan = run_prop_scan(conn, SPORTS, mock=True, output_path=tmp_path / "p.json", now=NOW)
    lal = [p for p in scan["props"] if p["event_id"] == "mock-nba-lal-gsw"]
    assert [p["pick_id"] for p in lal] == want                  # 2.0u spread, props untouched
    assert all(p["official_units"] == 1.0 for p in lal)
    assert not [r for r in scan["rejections"] if "Event exposure" in r["reason"]]


def test_prop_bucket_full_refuses_further_props(conn, tmp_path) -> None:
    led = Ledger(conn, db.ORIGIN_MOCK, starting_bankroll=100.0)
    for i in (1, 2):
        led.record_position(_prop_pick(pick_id=f"OPEN{i}", event_id="mock-nba-lal-gsw",
                                       selection=f"Someone{i} Over", origin=db.ORIGIN_MOCK))
    scan = run_prop_scan(conn, SPORTS, mock=True, output_path=tmp_path / "p.json", now=NOW)
    assert not [p for p in scan["props"] if p["event_id"] == "mock-nba-lal-gsw"]
    refused = [r for r in scan["rejections"] if "prop bucket 2.00u of 2.00u" in r["reason"]]
    assert refused and scan["ingestion"]["stats"]["event_cap_rejected"] >= len(refused)


def test_ledger_rows_and_dedupe(conn, tmp_path, mock_scan) -> None:
    rows = db.fetch_picks(conn, "market_type LIKE 'player_%'")
    assert len(rows) == len(mock_scan["props"]) == mock_scan["ingestion"]["stats"][
        "ledger_inserted"]
    for r in rows:
        assert r["hitl_status"] == "PENDING_CAPTAIN_HZ" and r["origin"] == db.ORIGIN_MOCK
        assert re.fullmatch(r".+ (Over|Under)", r["selection"]) and r["line_point"] is not None
        assert json.loads(r["analysis"])["triggers"]
    again = run_prop_scan(conn, SPORTS, mock=True, output_path=tmp_path / "p2.json", now=NOW)
    assert again["ingestion"]["stats"]["ledger_inserted"] == 0
    assert len(db.fetch_picks(conn, "market_type LIKE 'player_%'")) == len(rows)


def test_prop_snapshots_recorded(mock_scan, conn) -> None:
    n = conn.execute("SELECT COUNT(*) FROM market_snapshots WHERE market_type LIKE 'player_%'"
                     ).fetchone()[0]
    assert n == mock_scan["ingestion"]["stats"]["prop_quotes"] > 0


def test_config_defaults_and_toggles(conn, tmp_path) -> None:
    d = PropConfig()
    assert (d.divergence_mode, d.quota_floor, d.lookahead_hours) == ("all", 40, 6.0)
    loose = run_prop_scan(conn, SPORTS, mock=True, output_path=tmp_path / "b.json", now=NOW,
                          cfg=PropConfig(divergence_mode="any", pace_method="multiplicative"))
    assert loose["config"]["pace_method"] == "multiplicative"
    flagged = {(p["player"], p["market_type"]) for p in loose["props"]}
    assert ("DET Mock RB", "player_rush_yds") in flagged       # shrunk-EV only, OR mode


def test_sport_filter_limits_prop_scan(conn, tmp_path) -> None:
    nba = run_prop_scan(conn, ("basketball_nba",), mock=True, output_path=tmp_path / "n.json",
                        now=NOW)
    assert nba["props"] and {p["sport"] for p in nba["props"]} == {"basketball_nba"}
    assert all("Mock" in r["selection"] and not r["selection"].startswith(("BUF", "KC", "DET",
                                                                           "GB"))
               for r in nba["rejections"])
    assert set(nba["ingestion"]["stats"]["quota"]) == {"basketball_nba"}


# ==========================================================================
# Market benchmark and wire format
# ==========================================================================
def test_market_fair_prefers_pinnacle_then_consensus_excluding_offer() -> None:
    pairs = {"pinnacle": {"Over": -110, "Under": -110},
             "draftkings": {"Over": 105, "Under": -125}, "fanduel": {"Over": -115, "Under": -105}}
    p, src, pin = market_fair(pairs, "Over", "draftkings")
    assert src == "Pinnacle" and p == pytest.approx(0.5) and pin == -110
    no_pin = {k: v for k, v in pairs.items() if k != "pinnacle"}
    p2, src2, pin2 = market_fair(no_pin, "Over", "draftkings")
    assert src2 == "Consensus (1 book)" and pin2 is None and p2 > 0.5     # FanDuel only
    assert market_fair({"draftkings": pairs["draftkings"]}, "Over", "draftkings") is None


def test_parse_prop_quotes_wire_format() -> None:
    ev = {"id": "e1", "bookmakers": [{"key": "draftkings", "markets": [
        {"key": "player_points", "outcomes": [
            {"name": "Over", "description": "A Player", "price": -115, "point": 24.5},
            {"name": "Under", "description": "A Player", "price": -105, "point": 24.5}]},
        {"key": "h2h", "outcomes": [{"name": "Team", "price": 100}]}]}]}
    qs = parse_prop_quotes(ev)
    assert [(q.side, q.point, q.price) for q in qs] == [("Over", 24.5, -115),
                                                         ("Under", 24.5, -105)]


# ==========================================================================
# Slate gate and quota guard (live path, network mocked)
# ==========================================================================
def test_eastern_offset_dst_boundaries() -> None:
    assert eastern_offset_hours(datetime(2026, 7, 1, tzinfo=timezone.utc)) == -4
    assert eastern_offset_hours(datetime(2026, 12, 1, tzinfo=timezone.utc)) == -5
    assert eastern_offset_hours(datetime(2026, 3, 8, 6, 59, tzinfo=timezone.utc)) == -5
    assert eastern_offset_hours(datetime(2026, 3, 8, 7, 0, tzinfo=timezone.utc)) == -4
    assert eastern_offset_hours(datetime(2026, 11, 1, 5, 59, tzinfo=timezone.utc)) == -4
    assert eastern_offset_hours(datetime(2026, 11, 1, 6, 0, tzinfo=timezone.utc)) == -5
    assert eastern_date(datetime(2026, 11, 11, 3, 0, tzinfo=timezone.utc)).day == 10


def _ev(i, t):
    return {"id": f"ev{i}", "sport_key": "basketball_nba", "commence_time": t,
            "home_team": "Boston Celtics", "away_team": "Milwaukee Bucks"}


SLATE = [_ev(1, "2026-11-11T00:30:00Z"),    # 19:30 ET today, 12.5h out
         _ev(2, "2026-11-11T05:00:00Z"),    # 00:00 ET tomorrow
         _ev(3, "2026-11-10T11:00:00Z"),    # already started
         _ev(4, "2026-11-10T23:00:00Z"),    # 18:00 ET today, 11h out
         _ev(5, "2026-11-10T16:30:00Z")]    # 11:30 ET today, 4.5h out


def test_slate_gate_today_et_not_started() -> None:
    no_limit = dict(lookahead_hours=None)
    assert [e["id"] for e in slate_events(SLATE, NOW, 12, **no_limit)] == ["ev5", "ev4", "ev1"]
    assert [e["id"] for e in slate_events(SLATE, NOW, 1, **no_limit)] == ["ev5"]
    assert {e["id"] for e in slate_events(SLATE, NOW, 12, gate=False)} == \
        {"ev1", "ev2", "ev4", "ev5"}


def test_lookahead_window_default_six_hours() -> None:
    assert [e["id"] for e in slate_events(SLATE, NOW, 12)] == ["ev5"]
    assert [e["id"] for e in slate_events(SLATE, NOW, 12, lookahead_hours=11.0)] == \
        ["ev5", "ev4"]


def _fake_api(monkeypatch, remaining: str):
    calls: list[str] = []

    def fake_get_json(url, params=None, **kw):
        calls.append(url)
        if url.endswith("/events/"):
            return SLATE, {"x-requests-remaining": remaining, "x-requests-last": "0"}
        eid = url.rstrip("/").split("/")[-2]
        return ({**next(e for e in SLATE if e["id"] == eid), "bookmakers": []},
                {"x-requests-remaining": remaining, "x-requests-last": "4"})

    monkeypatch.setattr(pmf, "get_json", fake_get_json)
    monkeypatch.setenv("ODDS_API_KEY", "test-key")
    return calls


def test_quota_floor_stops_prop_requests(conn, tmp_path, monkeypatch) -> None:
    calls = _fake_api(monkeypatch, "450")
    scan = run_prop_scan(conn, ("basketball_nba",), mock=False, now=NOW, refresh_stats=False,
                         output_path=tmp_path / "p.json", cfg=PropConfig(quota_floor=500))
    s = scan["ingestion"]["stats"]
    assert s["quota_guard_tripped"] and s["events_queried"] == 0
    assert [c for c in calls if "/odds" in c] == []
    assert s["quota"]["basketball_nba"] == "quota floor reached"


def test_default_quota_floor_lets_free_tier_through(conn, tmp_path, monkeypatch) -> None:
    calls = _fake_api(monkeypatch, "450")              # a fresh free-tier month
    scan = run_prop_scan(conn, ("basketball_nba",), mock=False, now=NOW, refresh_stats=False,
                         output_path=tmp_path / "p.json")
    assert not scan["ingestion"]["stats"]["quota_guard_tripped"]
    assert [c.split("/")[-3] for c in calls if "/odds" in c] == ["ev5"]   # 6h lookahead
    calls = _fake_api(monkeypatch, "39")
    conn.execute("DELETE FROM api_cache")
    conn.commit()
    low = run_prop_scan(conn, ("basketball_nba",), mock=False, now=NOW, refresh_stats=False,
                        output_path=tmp_path / "p.json")
    assert low["ingestion"]["stats"]["quota_guard_tripped"]
    assert [c for c in calls if "/odds" in c] == []


def test_only_slate_games_queried_and_cached(conn, tmp_path, monkeypatch) -> None:
    calls = _fake_api(monkeypatch, "9000")
    cfg = PropConfig(lookahead_hours=24.0)
    scan = run_prop_scan(conn, ("basketball_nba",), mock=False, now=NOW, refresh_stats=False,
                         output_path=tmp_path / "p.json", cfg=cfg)
    odds_calls = [c for c in calls if "/odds" in c]
    assert sorted(c.split("/")[-3] for c in odds_calls) == ["ev1", "ev4", "ev5"]
    assert scan["ingestion"]["stats"]["events_on_slate"] == 3
    assert validate(scan, SCHEMA)[0] == []
    calls.clear()
    again = run_prop_scan(conn, ("basketball_nba",), mock=False, now=NOW, refresh_stats=False,
                          output_path=tmp_path / "p.json", cfg=cfg)
    assert calls == [] and again["ingestion"]["stats"]["events_from_cache"] == 3


# ==========================================================================
# Ledger and DB migration
# ==========================================================================
def _prop_pick(**kw):
    p = {"pick_id": "PP1", "commence_time": "2026-11-11T00:30:00Z", "sport": "basketball_nba",
         "event_id": "E1", "event_name": "Milwaukee Bucks @ Boston Celtics",
         "market_type": "player_points", "selection": "A Player Over", "line_point": 24.5,
         "bookmaker": "DraftKings", "offered_american": 105, "fair_prob": 0.53,
         "ev_percent": 8.65, "kelly_units": 1.0, "hitl_status": "PENDING_CAPTAIN_HZ"}
    p.update(kw)
    return p


def test_ledger_accepts_props_and_enforces_mandates(conn) -> None:
    led = Ledger(conn, starting_bankroll=100.0)
    assert led.record_position(_prop_pick())
    with pytest.raises(MandateViolation):
        led.record_position(_prop_pick(pick_id="PP2", offered_american=-135))
    with pytest.raises(MandateViolation, match="per-prop cap"):
        led.record_position(_prop_pick(pick_id="PP3", event_id="E2", kelly_units=1.2))
    assert led.record_position(_prop_pick(pick_id="PP4", kelly_units=1.0))   # E1 props 2.0u
    with pytest.raises(MandateViolation, match="player-prop cap"):
        led.record_position(_prop_pick(pick_id="PP5", kelly_units=0.5))
    # Game picks keep their 2.0u single cap.
    assert mandate_violations({**_prop_pick(), "market_type": "h2h", "kelly_units": 1.8}) == []


def test_game_and_prop_exposure_buckets_are_independent(conn) -> None:
    led = Ledger(conn, starting_bankroll=100.0)
    game = {**_prop_pick(pick_id="G1"), "market_type": "spreads", "selection": "Bucks",
            "kelly_units": 2.0, "hitl_status": "APPROVED"}
    assert led.record_position(game)                                # game bucket full
    assert led.record_position(_prop_pick(pick_id="P1"))            # props still fit
    assert led.record_position(_prop_pick(pick_id="P2", selection="B Player Over"))
    assert led.event_exposure("E1") == pytest.approx(4.0)            # combined ceiling
    assert led.event_exposure("E1", bucket="game") == pytest.approx(2.0)
    assert led.event_exposure("E1", bucket="prop") == pytest.approx(2.0)
    with pytest.raises(MandateViolation, match="game-line cap"):
        led.record_position({**game, "pick_id": "G2", "selection": "Celtics",
                             "kelly_units": 0.5})
    with pytest.raises(MandateViolation, match="player-prop cap"):
        led.record_position(_prop_pick(pick_id="P3", selection="C Player Over"))
    # The combined 4.0u ceiling still binds when a caller only knows the total.
    assert any("combined single-game cap" in v for v in mandate_violations(
        _prop_pick(), existing_event_units=3.5, existing_bucket_units=0.5))
    assert db.open_event_exposure(conn, "E1", bucket="prop") == pytest.approx(2.0)
    assert db.open_event_exposure(conn, "E1", bucket="game") == pytest.approx(2.0)


def test_legacy_ledger_without_prop_markets_is_rebuilt(tmp_path) -> None:
    path = tmp_path / "legacy.db"
    legacy_ddl = re.sub(r",\s*'player_points'.*?'player_receptions'", "", db._ledger_ddl(),
                        flags=re.S)
    assert "player_points" not in legacy_ddl and "'CANCELLED'" in legacy_ddl
    raw = sqlite3.connect(path)
    raw.executescript(legacy_ddl)
    raw.execute("INSERT INTO ledger_picks (pick_id, created_at, event_id, market_type, "
                "selection, offered_american, offered_decimal) VALUES ('OLD', "
                "'2026-01-01T00:00:00Z', 'E', 'h2h', 'X', 110, 2.1)")
    raw.commit()
    raw.close()
    c = db.connect(path)
    try:
        sql = c.execute("SELECT sql FROM sqlite_master WHERE name = 'ledger_picks'").fetchone()[0]
        assert "'player_receptions'" in sql
        assert db.get_pick(c, "OLD")["offered_american"] == 110
        assert Ledger(c, starting_bankroll=100.0).record_position(_prop_pick())
    finally:
        c.close()


# ==========================================================================
# Settlement
# ==========================================================================
def test_mock_settlement_grades_props_from_box_scores(conn, mock_scan) -> None:
    out = settler.run_mock_settlement(conn)
    g = out["grading"]
    assert g["props_graded"] + g["props_void"] == len(mock_scan["props"])
    assert g["props_unresolved"] == 0
    for p in db.fetch_picks(conn, "market_type LIKE 'player_%'"):
        player = p["selection"].rpartition(" ")[0]
        final = MOCK_PLAYER_FINALS[(p["event_id"], player)]
        stat = settler.PROP_MARKETS[p["market_type"]].stat
        assert p["result"] == settler.grade_prop(p, final[stat])
        assert p["closing_fair_prob"] is not None and p["clv_percent"] is not None
        expect = {"WIN": p["kelly_units"] * (p["offered_decimal"] - 1),
                  "LOSS": -p["kelly_units"], "PUSH": 0.0}[p["result"]]
        assert p["net_units"] == pytest.approx(expect, abs=1e-3)


NBA_BOX = {"boxscore": {"players": [{"team": {"id": "2"}, "statistics": [{
    "labels": ["MIN", "FG", "3PT", "FT", "OREB", "DREB", "REB", "AST", "STL", "BLK", "TO", "PF",
               "+/-", "PTS"],
    "athletes": [
        {"athlete": {"displayName": "P.J. Shooter Jr."},
         "stats": ["36", "10-21", "4-9", "5-6", "1", "8", "9", "6", "1", "0", "3", "2", "+8", "29"]},
        {"athlete": {"displayName": "Bench Guy"}, "stats": [], "didNotPlay": True}]}]}]}}
NFL_BOX = {"boxscore": {"players": [{"team": {"id": "2"}, "statistics": [
    {"name": "passing", "labels": ["C/ATT", "YDS", "AVG", "TD", "INT", "SACKS", "QBR", "RTG"],
     "athletes": [{"athlete": {"displayName": "Q Back"},
                   "stats": ["22/33", "251", "7.6", "2", "0", "1-6", "60", "101"]}]},
    {"name": "receiving", "labels": ["REC", "YDS", "AVG", "TD", "LONG", "TGTS"],
     "athletes": [{"athlete": {"displayName": "Wide Out"},
                   "stats": ["7", "88", "12.6", "1", "31", "10"]}]},
    {"name": "rushing", "labels": ["CAR", "YDS", "AVG", "TD", "LONG"],
     "athletes": [{"athlete": {"displayName": "Wide Out"},
                   "stats": ["1", "-2", "-2.0", "0", "0"]}]}]}]}}


def test_parse_espn_box_nba_and_nfl() -> None:
    nba = settler.parse_espn_box(NBA_BOX, "basketball_nba")
    row = nba[psf.player_key("PJ Shooter")]
    assert row["played"] and (row["pts"], row["reb"], row["ast"], row["fg3m"]) == (29, 9, 6, 4)
    assert nba["bench guy"]["played"] is False
    nfl = settler.parse_espn_box(NFL_BOX, "americanfootball_nfl")
    assert nfl["q back"]["pass_yds"] == 251
    assert (nfl["wide out"]["rec_yds"], nfl["wide out"]["receptions"]) == (88, 7)
    assert nfl["wide out"]["rush_yds"] == -2


def test_prop_outcomes_and_voids() -> None:
    nba = {"sport": "basketball_nba", "market_type": "player_points",
           "selection": "X Over", "line_point": 24.5}
    assert settler._prop_outcome(nba, {"played": True, "pts": 29})[0] == "WIN"
    assert settler._prop_outcome({**nba, "selection": "X Under"},
                                 {"played": True, "pts": 29})[0] == "LOSS"
    assert settler._prop_outcome({**nba, "line_point": 29.0}, {"played": True, "pts": 29})[0] \
        == "PUSH"
    assert settler._prop_outcome(nba, {"played": False})[0] == "CANCELLED"
    assert settler._prop_outcome(nba, None)[0] == "CANCELLED"         # NBA lists every DNP
    nfl = {"sport": "americanfootball_nfl", "market_type": "player_receptions",
           "selection": "Y Over", "line_point": 4.5}
    assert settler._prop_outcome(nfl, {"played": True, "rush_yds": 12})[0] == "LOSS"   # 0 rec


def test_nfl_absent_player_resolved_by_participation() -> None:
    over = {"sport": "americanfootball_nfl", "market_type": "player_receptions",
            "selection": "Y Over", "line_point": 0.5}
    under = {**over, "selection": "Y Under"}
    assert settler._prop_outcome(over, None)[0] is None                       # no data
    assert settler._prop_outcome(over, None, ("unknown", None))[0] is None
    assert settler._prop_outcome(over, None, ("played", 31.0)) == \
        ("LOSS", "no stat line but 31 snaps: graded on 0")
    assert settler._prop_outcome(under, None, ("played", 31.0))[0] == "WIN"
    assert settler._prop_outcome(over, None, ("inactive", 0.0))[0] == "CANCELLED"


SCHED = [{"game_id": "2026_10_BUF_KC", "gameday": "2026-11-15", "home_team": "KC",
          "away_team": "BUF"},
         {"game_id": "2026_15_KC_BUF", "gameday": "2026-12-20", "home_team": "BUF",
          "away_team": "KC"}]
SNAPS = [{"game_id": "2026_10_BUF_KC", "player": "Wide Out", "offense_snaps": 0.0,
          "st_snaps": 6.0},
         {"game_id": "2026_10_BUF_KC", "player": "Benchwarmer", "offense_snaps": 0.0,
          "st_snaps": 0.0}]


def test_nfl_participation_lookup() -> None:
    from datetime import date
    day = {date(2026, 11, 15)}
    look = lambda name, d=day: settler.nfl_participation(  # noqa: E731
        SNAPS, SCHED, "NFL_KC", "NFL_BUF", d, name)
    assert look("Wide Out") == ("played", 6.0)               # special-teams snaps count
    assert look("Benchwarmer") == ("inactive", 0.0)
    assert look("Not Listed") == ("inactive", 0.0)           # game published, no snaps
    assert look("Wide Out", {date(2026, 12, 20)}) == ("unknown", None)   # not published yet
    assert look("Wide Out", {date(2026, 11, 16)}) == ("unknown", None)   # no such game


def test_live_nfl_settlement_uses_snap_counts(conn, monkeypatch) -> None:
    led = Ledger(conn, starting_bankroll=100.0)
    pick = {**_prop_pick(pick_id="NFL1", sport="americanfootball_nfl",
                         event_name="Buffalo Bills @ Kansas City Chiefs",
                         market_type="player_receptions", selection="Wide Out Over",
                         line_point=0.5, kelly_units=0.5, hitl_status="APPROVED"),
            "commence_time": "2026-11-15T18:00:00Z"}
    led.record_position(pick)
    monkeypatch.setattr(settler, "fetch_scores", lambda c, s, d: [
        GameScoreFinal("Kansas City Chiefs", "Buffalo Bills")])
    monkeypatch.setattr(settler, "_espn_event_id", lambda c, p, lg: "401")
    monkeypatch.setattr(settler, "fetch_espn_box", lambda c, s, i: {})     # no stat line
    monkeypatch.setattr(settler, "_load_nfl_participation", lambda season: (SNAPS, SCHED))
    out = settler.grade_settled_games(conn, now=datetime(2026, 11, 16, tzinfo=timezone.utc))
    assert out["props_graded"] == 1
    assert db.get_pick(conn, "NFL1")["result"] == "LOSS"                 # played, 0 rec


def GameScoreFinal(home: str, away: str):
    return settler.GameScore(home, away, 24, 20, True)


# ==========================================================================
# Stats ingestion
# ==========================================================================
def test_player_key_normalization() -> None:
    assert psf.player_key("P.J. Washington Jr.") == "pj washington"
    assert psf.player_key("Nikola Jokić") == "nikola jokic"
    assert psf.player_key("Amon-Ra St. Brown") == "amon ra st brown"


def test_season_labels() -> None:
    from datetime import date
    assert psf.nba_season_label(date(2026, 11, 10)) == "2026-27"
    assert psf.nba_season_label(date(2027, 2, 1)) == "2026-27"
    assert psf.nfl_season(date(2027, 1, 20)) == 2026


def _team_row(gid, abbr, matchup, pts, fga, oreb, tov, fta, reb=44, ast=25, fg3m=12):
    return {"GAME_ID": gid, "TEAM_ABBREVIATION": abbr, "MATCHUP": matchup, "PTS": pts,
            "FGA": fga, "OREB": oreb, "TOV": tov, "FTA": fta, "REB": reb, "AST": ast,
            "FG3M": fg3m}


def test_build_nba_tables() -> None:
    teams = [_team_row("G1", "BOS", "BOS vs. MIL", 118, 88, 10, 12, 20),
             _team_row("G1", "MIL", "MIL @ BOS", 104, 90, 12, 14, 18, reb=40)]
    players = [{"GAME_ID": "G1", "TEAM_ABBREVIATION": "BOS", "PLAYER_NAME": "A Wing",
                "GAME_DATE": "2026-11-08", "MIN": 35, "PTS": 30, "REB": 7, "AST": 5,
                "FG3M": 4, "FG3A": 9, "FGM": 11, "FGA": 22, "FTM": 4, "FTA": 5, "TOV": 2}]
    logs, ctx = psf.build_nba_tables(players, teams, "2026-27", "2026-11-09T00:00:00Z")
    poss = ((88 - 10 + 12 + 0.44 * 20) + (90 - 12 + 14 + 0.44 * 18)) / 2
    assert logs[0]["game_poss"] == pytest.approx(poss)
    assert logs[0]["team_margin"] == 14 and logs[0]["canonical_id"] == "NBA_BOS"
    assert logs[0]["opponent_id"] == "NBA_MIL"
    bos = next(t for t in ctx if t["canonical_id"] == "NBA_BOS")
    assert bos["pace"] == pytest.approx(poss)
    assert bos["drtg"] == pytest.approx(100 * 104 / poss)
    assert bos["allowed_reb_100"] == pytest.approx(100 * 40 / poss)


def test_build_nfl_tables_neutral_script() -> None:
    pbp = [{"game_id": "g1", "posteam": "BUF", "defteam": "KC", "season_type": "REG",
            "pass_attempt": 1, "rush_attempt": 0, "wp": 0.5, "qtr": 1, "yards_gained": 8,
            "epa": 0.4},
           {"game_id": "g1", "posteam": "BUF", "defteam": "KC", "season_type": "REG",
            "pass_attempt": 0, "rush_attempt": 1, "wp": 0.5, "qtr": 2, "yards_gained": 4,
            "epa": 0.0},
           {"game_id": "g1", "posteam": "BUF", "defteam": "KC", "season_type": "REG",
            "pass_attempt": 0, "rush_attempt": 1, "wp": 0.95, "qtr": 4, "yards_gained": 2,
            "epa": -0.2},
           {"game_id": "g1", "posteam": "KC", "defteam": "BUF", "season_type": "REG",
            "pass_attempt": 1, "rush_attempt": 0, "wp": 0.5, "qtr": 1, "yards_gained": 6,
            "epa": 0.1}]
    weekly = [{"player_display_name": "Wide Out", "position": "WR", "team": "BUF",
               "opponent_team": "KC", "week": 3, "season_type": "REG", "targets": 9,
               "receptions": 6, "receiving_yards": 77, "target_share": 0.26},
              {"player_display_name": "Kicker", "position": "K", "team": "BUF", "week": 3}]
    logs, ctx = psf.build_nfl_tables(weekly, pbp, 2026, "2026-11-09T00:00:00Z")
    buf = next(t for t in ctx if t["canonical_id"] == "NFL_BUF")
    assert buf["plays_pg"] == 3 and buf["neutral_pass_rate"] == pytest.approx(0.5)
    kc = next(t for t in ctx if t["canonical_id"] == "NFL_KC")
    assert kc["def_rush_ypc"] == pytest.approx(3.0) and kc["def_pass_ypa"] == pytest.approx(8.0)
    assert len(logs) == 1 and logs[0]["rec_yds"] == 77 and logs[0]["game_date"] == "2026-W03"


def test_load_player_history_exact_and_fuzzy(conn) -> None:
    rows = [{"sport": "NBA", "player_key": "nicolas claxton", "game_id": f"G{i}",
             "player_name": "Nicolas Claxton", "canonical_id": "NBA_BKN",
             "game_date": f"2026-11-0{i}", "season": "2026-27", "minutes": 30, "reb": 9,
             "fetched_at": "2026-11-09T00:00:00Z"} for i in range(1, 6)]
    psf._store(conn, "NBA", rows, [])
    h = psf.load_player_history(conn, "NBA", "Nicolas Claxton", ("NBA_BKN", "NBA_BOS"))
    assert h and h.team_id == "NBA_BKN" and len(h.logs) == 5
    assert h.logs[0]["game_date"] == "2026-11-05"                  # most recent first
    assert psf.load_player_history(conn, "NBA", "Nic Claxton", ("NBA_BKN",)) is not None
    assert psf.load_player_history(conn, "NBA", "Nicolas Claxtn", ("NBA_BKN",)) is not None
    assert psf.load_player_history(conn, "NBA", "Nicolas Claxton", ("NBA_BOS",)) is None
    assert psf.load_player_history(conn, "NBA", "Someone Else", ("NBA_BKN",)) is None


def test_nickname_match_requires_uniqueness() -> None:
    assert psf._nickname_match("nic claxton", ["nicolas claxton", "a b"]) == "nicolas claxton"
    assert psf._nickname_match("j williams", ["jalen williams", "jaylin williams"]) is None
    assert psf._nickname_match("claxton", ["nicolas claxton"]) is None


def test_prior_season_fetched_once_and_attached(conn, monkeypatch) -> None:
    assert psf.previous_nba_season("2026-27") == "2025-26"
    assert psf.previous_nba_season("2000-01") == "1999-00"
    prior_calls: list[str] = []
    team_rows = [_team_row("G1", "BKN", "BKN vs. BOS", 110, 88, 10, 12, 20),
                 _team_row("G1", "BOS", "BOS @ BKN", 104, 90, 12, 14, 18)]
    player_rows = [{"GAME_ID": "G1", "TEAM_ABBREVIATION": "BKN", "PLAYER_NAME": "New Guy",
                    "GAME_DATE": "2026-11-08", "MIN": 30, "PTS": 12, "REB": 4, "AST": 2,
                    "FG3M": 1, "FG3A": 4, "FGM": 5, "FGA": 11, "FTM": 1, "FTA": 2, "TOV": 1}]
    monkeypatch.setattr(psf, "_nba_rows", lambda mode, season: player_rows if mode == "P"
                        else team_rows)

    def prior(season):
        prior_calls.append(season)
        return [{"PLAYER_NAME": "New Guy", "GP": 70, "MIN": 2100, "PTS": 900, "REB": 350,
                 "AST": 150, "FGM": 330, "FGA": 700, "FG3M": 90, "FG3A": 250, "FTM": 150,
                 "FTA": 190}]
    monkeypatch.setattr(psf, "_nba_prior_rows", prior)
    first = psf.update_nba_prop_stats(conn, "2026-27")
    second = psf.update_nba_prop_stats(conn, "2026-27")
    assert prior_calls == ["2025-26"]                       # one bulk call, ever
    assert first["prior_players"] == second["prior_players"] == 1
    h = psf.load_player_history(conn, "NBA", "New Guy", ("NBA_BKN", "NBA_BOS"))
    assert h.prior and h.prior["season"] == "2025-26" and h.prior["reb"] == 350


def test_prior_failure_is_soft(conn, monkeypatch) -> None:
    monkeypatch.setattr(psf, "_nba_rows", lambda mode, season: [])

    def boom(season):
        raise psf.PropStatsUnavailable("stats.nba.com timeout")
    monkeypatch.setattr(psf, "_nba_prior_rows", boom)
    out = psf.update_nba_prop_stats(conn, "2026-27")
    assert out["prior_players"] == 0 and "timeout" in out["prior_error"]


def test_teammate_recently_active(conn) -> None:
    def row(player, gid, date_, minutes):
        return {"sport": "NBA", "player_key": psf.player_key(player), "game_id": gid,
                "player_name": player, "canonical_id": "NBA_GSW", "game_date": date_,
                "season": "2026-27", "minutes": minutes, "fetched_at": "x"}
    games = [("G1", "2026-11-01"), ("G2", "2026-11-03"), ("G3", "2026-11-05"),
             ("G4", "2026-11-07")]
    rows = [row("Wing", g, d, 30) for g, d in games]
    rows += [row("Fresh Injury", g, d, 33) for g, d in games]              # played last game
    rows += [row("Long Out", g, d, 32) for g, d in games[:1]]              # last seen G1
    psf._store(conn, "NBA", rows, [])
    assert psf.teammate_recently_active(conn, "NBA", "NBA_GSW", "Fresh Injury") is True
    assert psf.teammate_recently_active(conn, "NBA", "NBA_GSW", "Long Out") is False
    assert psf.teammate_recently_active(conn, "NBA", "NBA_GSW", "Never Logged") is None


def test_missing_package_is_reported_not_raised(conn, monkeypatch) -> None:
    def boom(*a, **k):
        raise psf.PropStatsUnavailable("nba_api not installed")
    monkeypatch.setattr(psf, "_nba_rows", boom)
    out = psf.update_prop_stats(conn, ("NBA",))
    assert out[0]["error"] == "nba_api not installed" and out[0]["player_rows"] == 0


# ==========================================================================
# run_pipeline and monitor integration
# ==========================================================================
def test_run_pipeline_do_props_scan(conn, tmp_path, monkeypatch, capsys) -> None:
    import run_pipeline
    real = run_pipeline.run_prop_scan
    monkeypatch.setattr(run_pipeline, "run_prop_scan",
                        lambda c, s, mock: real(c, s, mock=mock, now=NOW,
                                                output_path=tmp_path / "p.json"))
    code, scan = run_pipeline.do_props_scan(conn, SPORTS, True)
    assert code == 0 and scan["props"]
    out = capsys.readouterr().out
    assert "[schema] prop_candidates.json: PASS" in out and "PENDING_CAPTAIN_HZ" in out


def _stub_pipeline(monkeypatch, seen):
    import run_pipeline
    monkeypatch.setattr(run_pipeline, "do_scan",
                        lambda conn, sports, mock: seen.append(("scan", sports)) or (0, None))
    monkeypatch.setattr(run_pipeline, "do_props_scan",
                        lambda conn, sports, mock: seen.append(("props", sports)) or (0, None))
    monkeypatch.setattr(run_pipeline, "do_update_stats",
                        lambda conn: seen.append(("stats", ())) or 0)
    monkeypatch.setattr(run_pipeline, "do_update_prop_stats",
                        lambda conn: seen.append(("prop-stats", ())) or 0)
    return run_pipeline


def test_run_pipeline_props_run_by_default(tmp_path, monkeypatch) -> None:
    seen: list[tuple[str, tuple[str, ...]]] = []
    rp = _stub_pipeline(monkeypatch, seen)
    dbp = str(tmp_path / "t.db")
    assert rp.main(["--mock", "--sport", "basketball_nba", "--db", dbp]) == 0
    assert seen == [("scan", ("basketball_nba",)), ("props", ("basketball_nba",))]
    seen.clear()
    assert rp.main(["--mock", "--db", dbp]) == 0                     # mock default: all
    assert seen == [("scan", SPORTS), ("props", SPORTS)]
    seen.clear()
    for flag in ("--scan", "--live"):
        assert rp.main([flag, "--db", dbp]) == 0
        assert seen == [("scan", ("basketball_nba",)), ("props", ("basketball_nba",))]
        seen.clear()
    assert rp.main(["--update-stats", "--db", dbp]) == 0
    assert seen == [("stats", ()), ("prop-stats", ())]
    seen.clear()
    assert rp.main(["--props", "--sport", "americanfootball_nfl", "--db", dbp]) == 0
    assert seen == [("props", ("americanfootball_nfl",))]          # props-only scan


def test_run_pipeline_no_props_opt_out(tmp_path, monkeypatch) -> None:
    seen: list[tuple[str, tuple[str, ...]]] = []
    rp = _stub_pipeline(monkeypatch, seen)
    dbp = str(tmp_path / "t.db")
    assert rp.main(["--mock", "--no-props", "--db", dbp]) == 0
    assert seen == [("scan", SPORTS)]
    seen.clear()
    assert rp.main(["--update-stats", "--scan", "--no-props", "--db", dbp]) == 0
    assert seen == [("stats", ()), ("scan", ("basketball_nba",))]
    with pytest.raises(SystemExit):
        rp.main(["--mock", "--props", "--no-props", "--db", dbp])


def test_unified_outbox_drop_and_content(conn, tmp_path, mock_scan) -> None:
    from src.data.market_fetcher import run_scan
    from templates.discord_templates import select_free_pick, write_discord_briefs
    from templates.social_templates import write_social_content
    games = run_scan(conn, SPORTS, mock=True, now=NOW, output_path=tmp_path / "g.json")
    opps, props = games["opportunities"], mock_scan["props"]
    assert opps and props
    drop = write_discord_briefs(opps, "2026-11-10", tmp_path, props=props).read_text("utf-8")
    assert "## Today's Card" in drop and "| Player props | " in drop
    assert "## VIP Syndicate Card - Game Lines" in drop and "## Official Player Props" in drop
    for o in opps:
        assert o["selection"] in drop
    for p in props:
        assert p["selection"] in drop
        assert f"{p['model_projection']:.2f}" in drop and f"+{p['ev_percent']:.2f}" in drop
    assert drop.count("| 1.00 | PENDING_CAPTAIN_HZ |") == len(props)
    content = write_social_content(select_free_pick(opps), "2026-11-10", tmp_path, opps=opps,
                                   props=props).read_text("utf-8")
    assert "### Game Picks" in content and "### Player Props" in content
    assert all(p["selection"] in content for p in props)
    assert content.count("| 1.00u | PENDING_CAPTAIN_HZ |") == len(props)
    off = write_discord_briefs(opps, "2026-11-11", tmp_path, props=None).read_text("utf-8")
    assert "## Official Player Props" not in off and "| Player props | off |" in off


def test_monitor_scan_args_and_schedule() -> None:
    assert mon.scan_args(False) == ["--mock", "--sport", "all"]            # props by default
    assert mon.scan_args(True, False) == ["--scan", "--sport", "all", "--no-props"]
    assert mon.scan_args(True, True, "basketball_nba") == ["--scan", "--sport",
                                                           "basketball_nba"]
    assert mon.stats_args() == ["--update-stats"]
    assert mon.stats_args(False) == ["--update-stats", "--no-props"]
    assert [c for c in range(1, 12) if mon.should_scan_props(c, True, 5)] == [1, 6, 11]
    assert not mon.should_scan_props(1, False, 1)
    assert mon.parse_args([]).props and not mon.parse_args(["--no-props"]).props


def test_monitor_renders_prop_table(mock_scan, tmp_path, monkeypatch) -> None:
    path = tmp_path / "prop_candidates.json"
    path.write_text(json.dumps(mock_scan), encoding="utf-8")
    monkeypatch.setattr(mon, "PROPS_PATH", path)
    monkeypatch.setattr(mon, "ROOT", tmp_path)
    pv = mon.collect_props(True, True, "MOCK", 0.0, tmp_path / "none.db")
    assert pv.source == "json" and len(pv.props) == len(mock_scan["props"])
    text = mon.render_props(pv, mon.Style(False), 5)
    assert "== MISPRICED PLAYER PROPS (MODEL vs MARKET) ==" in text
    for col in ("Player", "Prop Market", "Line", "Model Proj", "Delta", "Book & Odds",
                "Model Fair %", "EV %", "Units"):
        assert col in text
    assert "GSW Mock Wing" in text and "[INJURY]" in text
    evs = [float(m) for m in re.findall(r"^  \S.*?\s(\d+\.\d\d)\s+\d\.\d\d  PENDING", text, re.M)]
    assert evs == sorted(evs, reverse=True)
    assert mon.collect_props(True, True, "LIVE", 0.0, tmp_path / "none.db").source == "none"
