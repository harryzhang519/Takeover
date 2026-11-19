"""Discord posts and capper write-ups for APPROVED picks.

Covers odds / time formatting, the post layout, grounded context extraction
(schedule spots, injuries, matchup, pace, form, market), the Ollama path with
its validation and fallback, approval-only triggering, the desk API and the
free / premium card files. No network: Ollama is patched, everything else
lives in tmp_path.
Run: python -m pytest tests/test_writeups.py -v
"""

from __future__ import annotations

import json
import re
import sqlite3
import sys
import threading
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
from src.core.engine import HITL_APPROVED, HITL_PENDING  # noqa: E402
from src.core.ledger import Ledger  # noqa: E402
from src.data.team_registry import REGISTRY, sync_registry  # noqa: E402
from src.web import desk  # noqa: E402
from templates import writeup_generator as wg  # noqa: E402
from templates.discord_templates import (  # noqa: E402
    decimal_odds_str,
    discord_post,
    eastern_start,
    post_selection,
    HONEST_TAGLINE,
    render_tier_card,
)

MOCK = db.ORIGIN_MOCK
NBA_TIP = "2026-11-16T00:30:00Z"            # Sun Nov 15, 7:30 PM ET
NFL_KICK = "2026-11-16T18:00:00Z"           # Mon Nov 16, 1:00 PM ET
NOW_LOCAL = datetime(2026, 11, 15, 12, 0).astimezone()
NO_SCAN = (Path("does-not-exist.json"),)


# ==========================================================================
# Seeded world
# ==========================================================================
def _teams(league: str) -> list[str]:
    return sorted(cid for cid, t in REGISTRY.items() if t.sport == league)


def _ranked(league: str, top: list[str], bottom: list[str] = ()) -> dict[str, float]:
    """Values so ``top`` rank 1, 2, ... and ``bottom`` sit at the very end."""
    rest = [c for c in _teams(league) if c not in top and c not in bottom]
    order = list(top) + rest + list(bottom)
    return {cid: float(len(order) - i) for i, cid in enumerate(order)}


def _seed_ratings(conn, league: str, rating: dict[str, float], pace: dict[str, float] | None = None):
    col = "adj_net_rtg" if league == "NBA" else "srs"
    with conn:
        for cid, v in rating.items():
            conn.execute(f"INSERT OR REPLACE INTO team_ratings (canonical_id, sport, updated_at, "
                         f"source, {col}, pace) VALUES (?, ?, '2026-11-15T00:00:00Z', 'test', ?, ?)",
                         (cid, league, v, (pace or {}).get(cid)))


def _game(conn, cid, opp, when, pf=None, pa=None):
    with conn:
        conn.execute("INSERT INTO team_schedule VALUES (?, ?, ?, ?, 0, ?, ?, ?)",
                     (cid, f"{cid}-{when}", when, opp, int(pf is not None), pf, pa))


def _injury(conn, cid, league, player, pos, status="Out"):
    with conn:
        conn.execute("INSERT INTO injury_reports (canonical_id, sport, player, position, status, "
                     "is_key, fetched_at) VALUES (?, ?, ?, ?, ?, 1, '2026-11-15T00:00:00Z')",
                     (cid, league, player, pos, status))


@pytest.fixture
def conn(tmp_path):
    c = db.connect(tmp_path / "takeover.db")
    sync_registry(c)
    yield c
    c.close()


def _nba_game_pick(**kw) -> dict:
    p = {"pick_id": "PK-0a0a0a000001", "commence_time": NBA_TIP, "sport": "basketball_nba",
         "event_id": "ev-bos-mil", "event_name": "Boston Celtics @ Milwaukee Bucks",
         "market_type": "spreads", "selection": "Boston Celtics", "line_point": -3.5,
         "bookmaker": "BetMGM", "offered_american": -108, "fair_prob": 0.56, "ev_percent": 6.9,
         "kelly_units": 1.5, "hitl_status": HITL_PENDING, "hitl_reason": "", "origin": MOCK,
         "analysis": json.dumps({"selection_margin": 7.2})}
    p.update(kw)
    return p


@pytest.fixture
def nba_world(conn):
    _seed_ratings(conn, "NBA", _ranked("NBA", ["NBA_OKC", "NBA_BOS", "NBA_DEN", "NBA_CLE",
                                               "NBA_NYK"], ["NBA_MIL"]))
    _game(conn, "NBA_MIL", "NBA_IND", "2026-11-15T00:30:00Z", 110, 112)   # B2B, lost by 2
    _game(conn, "NBA_MIL", "NBA_OKC", "2026-11-18T00:30:00Z")             # top-5 up next
    _game(conn, "NBA_BOS", "NBA_PHI", "2026-11-13T00:30:00Z", 90, 120)    # blown out, rested
    _injury(conn, "NBA_MIL", "NBA", "Giannis Antetokounmpo", "F")
    _injury(conn, "NBA_MIL", "NBA", "Bench Guy", "G", status="Questionable")
    return conn


# ==========================================================================
# Formatting
# ==========================================================================
@pytest.mark.parametrize("american, decimal", [(-108, "1.93"), (105, "2.05"), (-110, "1.91"),
                                               (150, "2.50"), (-200, "1.50"), (100, "2.00")])
def test_american_to_decimal_formatting(american, decimal) -> None:
    assert decimal_odds_str(american) == decimal


def test_eastern_start_handles_standard_and_daylight_time() -> None:
    assert eastern_start("2026-11-16T01:15:00Z") == "8:15 PM ET"            # EST, UTC-5
    assert eastern_start("2026-07-01T23:00:00Z") == "7:00 PM ET"            # EDT, UTC-4
    assert eastern_start(None) == "Time TBD"


def test_discord_post_layout() -> None:
    pick = {"league": "NFL", "market_type": "spreads", "selection": "Jacksonville Jaguars",
            "line_point": -9.5, "units": 1.5, "offered_american": -108, "bookmaker": "BetMGM",
            "commence_time": "2026-11-16T01:15:00Z"}
    assert discord_post(pick, "Short week for Baltimore. @everyone") == (
        "8:15 PM ET\n\U0001F3C8 **Jaguars -9.5** (1.50u)\nBetMGM: -108 / 1.93\n\n"
        "Short week for Baltimore. @everyone")


@pytest.mark.parametrize("pick, label", [
    ({"league": "NBA", "market_type": "player_threes", "selection": "Matas Buzelis Over",
      "line_point": 2.5}, "Matas Buzelis Over 2.5 Threes"),
    ({"league": "NBA", "market_type": "player_points", "selection": "Matas Buzelis Over",
      "line_point": 14.5}, "Matas Buzelis Over 14.5 Pts"),
    ({"league": "NBA", "market_type": "totals", "selection": "Over", "line_point": 229.5,
      "event": "Denver Nuggets @ Phoenix Suns"}, "Nuggets/Suns Over 229.5"),
    ({"league": "NBA", "market_type": "h2h", "selection": "Portland Trail Blazers"},
     "Trail Blazers ML"),
    ({"league": "NFL", "market_type": "spreads", "selection": "Baltimore Ravens",
      "line_point": 3.0}, "Ravens +3"),
])
def test_post_selection_wording(pick, label) -> None:
    assert post_selection(pick) == label


# ==========================================================================
# Context extraction
# ==========================================================================
def test_nba_situational_injury_and_rating_context(nba_world) -> None:
    ctx = wg.gather_context(nba_world, _nba_game_pick(), NO_SCAN)
    assert (ctx.team, ctx.opp) == ("NBA_BOS", "NBA_MIL")
    text = {f.tag: f.text for f in ctx.facts}
    assert {"opp_b2b", "hangover", "bounce_back", "lookahead", "opp_injuries", "ratings",
            "projection"} <= set(text)
    assert "second night of a back-to-back" in text["opp_b2b"]
    assert "110-112 heartbreaker against the Pacers" in text["hangover"]
    assert "90-120 by the 76ers" in text["bounce_back"]
    assert "Thunder up next, 1st in net rating" in text["lookahead"]
    assert text["opp_injuries"] == "The Bucks are down Giannis Antetokounmpo."  # not the GTD
    assert text["ratings"] == "The Celtics are 2nd in net rating and the Bucks are 30th."
    assert "winning this by 7.2 and we're only laying 3.5" in text["projection"]
    assert [f.tag for f in ctx.facts[:2]] == ["opp_b2b", "opp_injuries"]   # strongest first


def test_sandwich_spot_when_both_neighbours_are_top_five(nba_world) -> None:
    with nba_world:
        nba_world.execute("UPDATE team_schedule SET opponent_id = 'NBA_DEN' "
                          "WHERE canonical_id = 'NBA_MIL' AND completed = 1")
    tags = wg.gather_context(nba_world, _nba_game_pick(), NO_SCAN).tags()
    assert "sandwich" in tags and "lookahead" not in tags


def test_no_spot_claims_without_schedule_data(conn) -> None:
    ctx = wg.gather_context(conn, _nba_game_pick(), NO_SCAN)
    assert not ctx.tags() & {"opp_b2b", "hangover", "bounce_back", "lookahead", "sandwich",
                             "opp_injuries", "ratings"}
    assert ctx.tags() == {"projection"}                                      # pick's own data


def test_nfl_short_week_and_backup_qb(conn) -> None:
    _seed_ratings(conn, "NFL", _ranked("NFL", ["NFL_JAX"], ["NFL_BAL"]))
    _game(conn, "NFL_BAL", "NFL_PIT", "2026-11-13T01:15:00Z", 17, 20)        # Thursday night
    _game(conn, "NFL_JAX", "NFL_HOU", "2026-11-09T18:00:00Z", 24, 10)
    _injury(conn, "NFL_BAL", "NFL", "Lamar Jackson", "QB")
    _injury(conn, "NFL_BAL", "NFL", "Some Tackle", "OT")
    pick = _nba_game_pick(sport="americanfootball_nfl", event_id="ev-jax-bal",
                          event_name="Jacksonville Jaguars @ Baltimore Ravens",
                          selection="Jacksonville Jaguars", line_point=-9.5,
                          commence_time=NFL_KICK, analysis=None)
    text = {f.tag: f.text for f in wg.gather_context(conn, pick, NO_SCAN).facts}
    assert text["opp_qb_out"] == "The Ravens will be without their QB Lamar Jackson."
    assert "short week, just 4 days after playing the Steelers" in text["short_week"]
    assert "17-20 heartbreaker" in text["hangover"]
    assert "1st in SRS" in text["ratings"]


@pytest.fixture
def prop_world(conn):
    pace = _ranked("NBA", ["NBA_CHI", "NBA_ATL"])
    _seed_ratings(conn, "NBA", _ranked("NBA", ["NBA_BOS"]), pace)
    threes = _ranked("NBA", ["NBA_WAS", "NBA_ATL"])
    with conn:
        for cid, v in threes.items():
            conn.execute("INSERT INTO team_prop_context (canonical_id, sport, updated_at, "
                         "allowed_fg3m_100) VALUES (?, 'NBA', '2026-11-15T00:00:00Z', ?)", (cid, v))
        for i, made in enumerate([3, 4, 2, 3, 5, 3, 1, 4]):
            conn.execute("INSERT INTO player_game_logs (sport, player_key, game_id, player_name, "
                         "game_date, minutes, fg3m, fetched_at) VALUES ('NBA', 'matas buzelis', ?, "
                         "'Matas Buzelis', ?, 33, ?, '2026-11-15T00:00:00Z')",
                         (f"g{i}", f"2026-11-{14 - i:02d}", made))
    db.insert_snapshots(conn, [
        {"event_id": "ev-chi-atl", "timestamp": "2026-11-15T20:00:00Z", "source": "pinnacle",
         "market_type": "totals", "selection": sel, "point": 245.5, "american_odds": odds}
        for sel, odds in (("Over", -115), ("Under", -105))])
    return conn


def _prop_pick(**kw) -> dict:
    p = _nba_game_pick(pick_id="PK-0b0b0b000001", event_id="ev-chi-atl",
                       event_name="Chicago Bulls @ Atlanta Hawks", market_type="player_threes",
                       selection="Matas Buzelis Over", line_point=2.5, offered_american=110,
                       bookmaker="DraftKings", kelly_units=1.0,
                       analysis=json.dumps({"projection": 3.4, "team_id": "NBA_CHI",
                                            "opponent_id": "NBA_ATL",
                                            "details": {"minutes": 33.2}}))
    p.update(kw)
    return p


def test_prop_matchup_pace_total_and_form_context(prop_world) -> None:
    ctx = wg.gather_context(prop_world, _prop_pick(), NO_SCAN)
    text = {f.tag: f.text for f in ctx.facts}
    assert text["defense"] == ("The Hawks give up the 2nd most threes per 100 possessions in "
                               "the league.")
    assert text["pace"].startswith("The Bulls and Hawks are #1 and #2 in pace")
    assert text["total"] == "The game total is 245.5 and it's juiced to the over."
    assert "averaging 3.1 threes over the last 8 games" in text["form"]
    assert "over 2.5 in 6 of them" in text["form"]
    assert "projected for 3.4" in text["projection"]
    assert "33 minutes a night" in text["minutes"]


def test_under_prop_only_uses_facts_that_support_the_under(prop_world) -> None:
    ctx = wg.gather_context(prop_world, _prop_pick(selection="Matas Buzelis Under",
                                                   analysis=json.dumps({"projection": 3.4})),
                            NO_SCAN)
    assert not ctx.tags() & {"defense", "pace", "total", "form", "projection", "minutes"}


# ==========================================================================
# Template voice + rules
# ==========================================================================
def _rules_hold(text: str, ctx) -> None:
    assert text.endswith("@everyone") and text.count("@everyone") == 1
    assert "—" not in text and "–" not in text
    assert not wg.BANNED.search(text)
    assert 2 <= wg.sentence_count(text) <= 5
    assert wg.validate(text, ctx)[0] == text                      # every number is grounded


def test_fallback_game_writeup_follows_the_rules(nba_world) -> None:
    ctx = wg.gather_context(nba_world, _nba_game_pick(), NO_SCAN)
    text = wg.fallback_writeup(ctx)
    _rules_hold(text, ctx)
    assert text.startswith("The Bucks are on the second night of a back-to-back")
    assert "The Bucks are down Giannis Antetokounmpo." in text
    assert wg.fallback_writeup(ctx) == text                       # deterministic


def test_fallback_prop_writeup_names_player_without_pronouns(prop_world) -> None:
    ctx = wg.gather_context(prop_world, _prop_pick(), NO_SCAN)
    text = wg.fallback_writeup(ctx)
    _rules_hold(text, ctx)
    assert "Matas Buzelis is averaging" in text
    assert "Buzelis projected" in text or "Buzelis" in text.split(".")[-2]
    assert not re.search(r"\b(he|him|his)\b", text, re.I)


def test_fallback_with_no_facts_still_posts(conn) -> None:
    ctx = wg.gather_context(conn, _nba_game_pick(analysis=None), NO_SCAN)
    assert ctx.facts == []
    text = wg.fallback_writeup(ctx)
    _rules_hold(text, ctx)
    assert "BetMGM is hanging -108" in text


# ==========================================================================
# Ollama
# ==========================================================================
@pytest.fixture
def ollama(monkeypatch):
    """Turn Ollama on and capture calls; tests set ``reply`` (str or exception)."""
    monkeypatch.setenv("BREEZE_WRITEUP_LLM", "on")
    calls: list[dict] = []
    box: dict = {"reply": None}

    def fake_post(url, payload, timeout):
        calls.append({"url": url, "payload": payload, "timeout": timeout})
        if isinstance(box["reply"], Exception):
            raise box["reply"]
        return {"response": box["reply"]}
    monkeypatch.setattr(wg, "_post_json", fake_post)
    return box, calls


def test_ollama_writeup_is_cleaned_and_used(nba_world, ollama) -> None:
    box, calls = ollama
    box["reply"] = ("@everyone The Bucks are on the second night of a back-to-back — and "
                    "Giannis Antetokounmpo is out. The Celtics are 2nd in net rating. Lay it.")
    text, source = wg.generate_writeup(nba_world, _nba_game_pick(), scan_paths=NO_SCAN)
    assert source == wg.SOURCE_OLLAMA
    assert text == ("The Bucks are on the second night of a back-to-back, and Giannis "
                    "Antetokounmpo is out. The Celtics are 2nd in net rating. Lay it. @everyone")
    call = calls[0]
    assert call["url"] == "http://localhost:11434/api/generate" and call["timeout"] == 4.0
    assert call["payload"]["model"] == "qwen2.5-coder:7b" and call["payload"]["stream"] is False
    prompt = call["payload"]["prompt"]
    assert all(ex[:40] in prompt for ex in wg.STYLE_EXAMPLES)
    assert "Giannis Antetokounmpo" in prompt and "Celtics -3.5 at -108 (BetMGM)" in prompt


@pytest.mark.parametrize("reply, why", [
    ("The Bucks have lost 7 straight. Lay it with the Celtics. @everyone", "invented number"),
    ("Our model gives the Celtics a 61% chance. Lay it. @everyone", "math talk"),
    ("Lay it. @everyone", "one sentence"),
    ("", "empty"),
])
def test_bad_ollama_writeups_fall_back_to_template(nba_world, ollama, reply, why) -> None:
    box, _ = ollama
    box["reply"] = reply
    text, source = wg.generate_writeup(nba_world, _nba_game_pick(), scan_paths=NO_SCAN)
    assert source == wg.SOURCE_TEMPLATE, why
    assert text == wg.fallback_writeup(wg.gather_context(nba_world, _nba_game_pick(), NO_SCAN))


def test_ollama_offline_falls_back_and_cools_down(nba_world, ollama) -> None:
    box, calls = ollama
    box["reply"] = urllib.error.URLError("connection refused")
    assert wg.generate_writeup(nba_world, _nba_game_pick(), scan_paths=NO_SCAN)[1] == \
        wg.SOURCE_TEMPLATE
    assert wg.generate_writeup(nba_world, _nba_game_pick(), scan_paths=NO_SCAN)[1] == \
        wg.SOURCE_TEMPLATE
    assert len(calls) == 1                                        # second approval skipped Ollama
    box["reply"] = TimeoutError("timed out")
    wg.reset_ollama_cooldown()
    assert wg.generate_writeup(nba_world, _nba_game_pick(), scan_paths=NO_SCAN)[1] == \
        wg.SOURCE_TEMPLATE


def test_ollama_disabled_by_environment(nba_world, ollama, monkeypatch) -> None:
    box, calls = ollama
    monkeypatch.setenv("BREEZE_WRITEUP_LLM", "off")
    box["reply"] = "never used"
    assert wg.generate_writeup(nba_world, _nba_game_pick(), scan_paths=NO_SCAN)[1] == \
        wg.SOURCE_TEMPLATE
    assert calls == []


def test_scan_json_exchange_price_is_used(conn, tmp_path) -> None:
    scan = tmp_path / "live.json"
    scan.write_text(json.dumps({"opportunities": [
        {"pick_id": "PK-0a0a0a000001", "exchange_probability": 0.58}]}), encoding="utf-8")
    pick = _nba_game_pick(market_type="h2h", line_point=None, analysis=None)
    text = {f.tag: f.text for f in wg.gather_context(conn, pick, (scan,)).facts}
    assert text["exchange"] == "Kalshi has the Celtics trading at 58 cents."


# ==========================================================================
# Approval-only triggering
# ==========================================================================
def _record(conn, *picks) -> Ledger:
    led = Ledger(conn, MOCK)
    for p in picks:
        assert led.record_position(p)
    return led


def _commence_soon() -> str:
    return db.iso(db.utcnow() + timedelta(hours=3))


def test_writeups_only_on_approval_and_cleared_on_undo(nba_world) -> None:
    _record(nba_world, _nba_game_pick(),
            _nba_game_pick(pick_id="PK-0a0a0a000002", event_id="ev-2"),
            _nba_game_pick(pick_id="PK-0a0a0a000003", event_id="ev-3"))
    get = lambda pid: db.get_pick(nba_world, pid)   # noqa: E731
    assert get("PK-0a0a0a000001")["discord_writeup"] is None             # pending: nothing
    with pytest.raises(ValueError, match="approved picks only"):
        wg.attach_writeup(nba_world, "PK-0a0a0a000001")

    v = desk.approve(nba_world, "PK-0a0a0a000001", now=NOW_LOCAL)
    p = get("PK-0a0a0a000001")
    assert p["writeup_source"] == wg.SOURCE_TEMPLATE and p["discord_writeup"].endswith("@everyone")
    assert v["post"].startswith("7:30 PM ET\n\U0001F3C0 **Celtics -3.5** (1.50u)\nBetMGM: -108 / 1.93")
    assert v["post"].endswith(p["discord_writeup"])

    desk.decline(nba_world, "PK-0a0a0a000002", now=NOW_LOCAL)
    assert get("PK-0a0a0a000002")["discord_writeup"] is None             # declined: nothing

    desk.undo(nba_world, "PK-0a0a0a000001", now=NOW_LOCAL)
    p = get("PK-0a0a0a000001")
    assert (p["hitl_status"], p["discord_writeup"], p["writeup_source"]) == (HITL_PENDING,
                                                                            None, None)


def test_approve_all_writes_up_every_approved_pick(nba_world) -> None:
    soon = _commence_soon()
    _record(nba_world, _nba_game_pick(commence_time=soon),
            _nba_game_pick(pick_id="PK-0a0a0a000004", event_id="ev-4", commence_time=soon),
            _nba_game_pick(pick_id="PK-0a0a0a000005", event_id="ev-5", commence_time=soon,
                           fair_prob=0.40, ev_percent=-15.0))                 # fails filters
    out = desk.approve_all_qualified(nba_world, MOCK)
    assert {v["pick_id"] for v in out["approved"]} == {"PK-0a0a0a000001", "PK-0a0a0a000004"}
    assert all(v["writeup"] and v["post"] for v in out["approved"])
    assert db.get_pick(nba_world, "PK-0a0a0a000005")["discord_writeup"] is None


def test_cli_approve_writes_up(nba_world, tmp_path, capsys) -> None:
    _record(nba_world, _nba_game_pick())
    assert run_pipeline.main(["--approve", "PK-0a0a0a000001", "--db",
                              str(tmp_path / "takeover.db")]) == 0
    assert "[writeup] (template)" in capsys.readouterr().out
    assert db.get_pick(nba_world, "PK-0a0a0a000001")["discord_writeup"].endswith("@everyone")


def test_failed_writeup_never_blocks_approval(nba_world, monkeypatch) -> None:
    _record(nba_world, _nba_game_pick())
    monkeypatch.setattr(wg, "gather_context", lambda *a, **k: (_ for _ in ()).throw(
        RuntimeError("boom")))
    v = desk.approve(nba_world, "PK-0a0a0a000001", now=NOW_LOCAL)
    assert v["signed"] and v["hitl_status"] == HITL_APPROVED


def test_legacy_ledger_gains_writeup_columns(tmp_path) -> None:
    path = tmp_path / "legacy.db"
    raw = sqlite3.connect(path)
    raw.executescript(db.SCHEMA_SQL)                      # base table, no migration columns
    raw.execute("INSERT INTO ledger_picks (pick_id, created_at, event_id, market_type, selection,"
                " offered_american, offered_decimal) VALUES ('PK-old', '2026-11-01T00:00:00Z', "
                "'e', 'h2h', 'X', 110, 2.1)")
    raw.commit()
    raw.close()
    c = db.connect(path)
    try:
        cols = {r["name"] for r in c.execute("PRAGMA table_info(ledger_picks)")}
        assert {"discord_writeup", "writeup_source"} <= cols
        assert db.get_pick(c, "PK-old")["selection"] == "X"
    finally:
        c.close()


# ==========================================================================
# Free / premium card files
# ==========================================================================
def test_free_card_renders_posts_in_start_order(nba_world, tmp_path) -> None:
    soon = _commence_soon()
    later = db.iso(db.utcnow() + timedelta(hours=5))
    _record(nba_world, _nba_game_pick(commence_time=later),
            _nba_game_pick(pick_id="PK-0a0a0a000006", event_id="ev-6", commence_time=soon,
                           selection="Milwaukee Bucks", line_point=3.5),
            _nba_game_pick(pick_id="PK-0a0a0a000007", event_id="ev-7", commence_time=soon))
    desk.approve(nba_world, "PK-0a0a0a000001")
    desk.approve(nba_world, "PK-0a0a0a000006")
    paths = desk.write_discord_drop(nba_world, MOCK, tmp_path / "drops")
    md = paths["FREE"].read_text(encoding="utf-8")
    assert md.index("**Bucks +3.5**") < md.index("**Celtics -3.5**")       # earlier tip first
    assert md.count("@everyone") == 2 and "PK-0a0a0a000007" not in md
    block = md.split("\n\n")[2]                                            # first post header
    lines = block.split("\n")
    assert re.fullmatch(r"\d{1,2}:\d{2} [AP]M ET", lines[0])
    assert lines[1] == "\U0001F3C0 **Bucks +3.5** (1.50u)" and lines[2] == "BetMGM: -108 / 1.93"
    assert "|" not in md                                                   # no tables


def test_render_tier_cards_empty_and_mock_banner() -> None:
    free = render_tier_card([], "Sunday, November 15", "FREE", mock=True)
    assert free.startswith("# Breeze Bets - Free Plays - Sunday, November 15")
    assert "MOCK DESK" in free and "No free play signed off yet" in free
    premium = render_tier_card([], "Sunday, November 15", "PREMIUM")
    assert premium.startswith("# Breeze Bets - Premium Card") and "MOCK" not in premium
    for md in (free, premium):
        assert md.rstrip().endswith(f"_{HONEST_TAGLINE}_")
    with pytest.raises(ValueError):
        render_tier_card([], "Sunday", "VIP")


# ==========================================================================
# HTTP: edit / regenerate
# ==========================================================================
@pytest.fixture
def server(tmp_path, nba_world):
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


def _post(url, body=None, csrf=True):
    req = urllib.request.Request(url, data=json.dumps(body or {}).encode(), method="POST",
                                 headers={"X-Breeze-Desk": "1"} if csrf else {})
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            return r.status, json.load(r)
    except urllib.error.HTTPError as e:
        return e.code, json.load(e)


def test_http_edit_and_regenerate_writeup(server, nba_world) -> None:
    app, base = server
    conn = app.connect()
    _record(conn, _nba_game_pick(commence_time=_commence_soon()),
            _nba_game_pick(pick_id="PK-0a0a0a000008", event_id="ev-8",
                           commence_time=_commence_soon()))
    conn.close()
    url = f"{base}/api/picks/PK-0a0a0a000001/writeup"
    assert _post(url, {"text": "x"})[0] == 409                          # still pending
    assert _post(f"{base}/api/picks/PK-0a0a0a000008/writeup", {"regenerate": True})[0] == 409

    status, out = _post(f"{base}/api/picks/PK-0a0a0a000001/approve")
    assert status == 200 and out["pick"]["writeup_source"] == "template"
    assert _post(url, {"text": "Mine."}, csrf=False)[0] == 403
    assert _post(url, {"text": "   "})[0] == 400
    assert _post(url, {"nope": 1})[0] == 400

    status, out = _post(url, {"text": "Celtics in a bounce-back spot. @everyone"})
    assert status == 200 and out["pick"]["writeup_source"] == "edited"
    assert out["pick"]["post"].endswith("\n\nCeltics in a bounce-back spot. @everyone")
    drop = Path(out["drop_paths"]["FREE"]).read_text(encoding="utf-8")
    assert "Celtics in a bounce-back spot. @everyone" in drop

    status, out = _post(url, {"regenerate": True})
    assert status == 200 and out["pick"]["writeup_source"] == "template"
    assert "bounce-back spot. @everyone" not in out["pick"]["writeup"]
