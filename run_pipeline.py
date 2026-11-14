"""Breeze quant backend - unified CLI. All workflows report to Captain HZ.

    python run_pipeline.py --setup-env [--odds-key K --kalshi-key-id ID --kalshi-key-path P]
    python run_pipeline.py --update-stats                     # ESPN + prop stats
    python run_pipeline.py --scan --sport basketball_nba      # live games + props
    python run_pipeline.py --live --sport all                 # same as --scan
    python run_pipeline.py --mock                             # offline games + props
    python run_pipeline.py --settle [--mock]                  # closing lines, CLV, grading
    python run_pipeline.py --approve PICK_ID                  # Captain HZ sign-off
    python run_pipeline.py --decline PICK_ID                  # Captain HZ passes
    python run_pipeline.py --backtest [--include-mock]        # seeds 250 trades if empty
    python run_pipeline.py --mock --no-props                  # game lines only
    python run_pipeline.py --props                            # live props only (no games)

Player props are an equal pillar to game lines: --mock, --scan/--live and
--update-stats all include props unless --no-props is given. The day's
OUTBOX drop (<date>_drop.md) and content file (<date>_content.md) carry game
picks and official 1.00u player props side by side.

Actions combine and run in order: setup -> update-stats (+ prop stats) -> scan
-> props -> OUTBOX drop -> settle -> approve -> decline -> backtest.
The local web desk (python web_server.py) drives the same actions from a browser.
Everything persists to data/takeover.db (SQLite, WAL).
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.core import database as db  # noqa: E402
from src.core.backtester import run_backtest, seed_if_empty  # noqa: E402
from src.core.config import bankroll_units, load_env, setup_env  # noqa: E402
from src.core.engine import HITL_APPROVED, HITL_PENDING  # noqa: E402
from src.core.ledger import Ledger  # noqa: E402
from src.core.orchestrator import sign_off  # noqa: E402
from src.core.schema_validator import load_schema, validate  # noqa: E402
from src.data.market_fetcher import OUTPUT_PATH, IngestionError, run_scan  # noqa: E402
from src.data.prop_market_fetcher import PROP_OUTPUT_PATH, run_prop_scan  # noqa: E402
from src.data.prop_stats_fetcher import update_prop_stats  # noqa: E402
from src.data.settler import run_settlement  # noqa: E402
from src.data.stats_fetcher import update_stats  # noqa: E402
from src.data.team_registry import sync_registry  # noqa: E402
from src.data.teams import SPORT_TO_LEAGUE  # noqa: E402
from templates.analysis_cards import render_analysis_card  # noqa: E402
from templates.discord_templates import select_free_pick, write_discord_briefs  # noqa: E402
from templates.social_templates import write_social_content  # noqa: E402

SCHEMA_PATH = ROOT / "schemas" / "market_scan_schema.json"
PROP_SCHEMA_PATH = ROOT / "schemas" / "prop_scan_schema.json"
REQUIRED_DIRS = ("OUTBOX/market_scans", "OUTBOX/discord_briefs", "OUTBOX/social_distribution",
                 "data")
BANNERS = {
    "MOCK": "> **MOCK SCAN - SYNTHETIC DATA.** Simulated Odds-API/Kalshi payloads. "
            "Do not distribute or bet.",
    "LIVE": "> **LIVE DATA.** Requires Captain HZ review before distribution. "
            "PENDING_CAPTAIN_HZ positions are not to be placed.",
}


def ensure_dirs() -> dict[str, Path]:
    out = {}
    for rel in REQUIRED_DIRS:
        p = ROOT / rel
        existed = p.is_dir()
        p.mkdir(parents=True, exist_ok=True)
        print(f"[dir] {rel + '/':<30} {'exists' if existed else 'created'}")
        out[rel.rsplit("/", 1)[-1]] = p
    return out


def do_scan(conn, sports: tuple[str, ...], mock: bool) -> tuple[int, dict | None]:
    """Game-line scan -> (exit code, schema-valid scan or None)."""
    print(f"\n== GAME LINES SCAN | {'MOCK' if mock else 'LIVE'} | {', '.join(sports)}")
    try:
        scan = run_scan(conn, sports, mock=mock)
    except IngestionError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1, None
    errors, backend = validate(scan, load_schema(SCHEMA_PATH))
    if errors:
        print(f"SCHEMA VALIDATION FAILED ({backend}):", file=sys.stderr)
        for e in errors:
            print(f"  - {e}", file=sys.stderr)
        return 1, None
    s = scan["ingestion"]["stats"]
    print(f"[schema] live_candidates.json: PASS ({backend})")
    print(f"events {s['events']} | retail quotes {s['retail_quotes']} | point mismatch "
          f"{s['point_mismatch']} | floor rejected {s['floor_rejected']} | below 1.5% EV "
          f"{s['below_min_ev']} | event-cap rejected {s['event_cap_rejected']} | Kalshi matched "
          f"{s['kalshi_matched']}/{s['kalshi_markets']} | snapshots +{s['snapshots']} | "
          f"quota {s['quota']}")
    for o in scan["opportunities"]:
        print(f"  [{o['hitl_status']:<18}] {o['selection']:<32} {o['offered_american']:+5d} "
              f"{o['retail_book']:<10} fair {o['fair_probability'] * 100:5.2f}%  "
              f"EV {o['ev_percent']:5.2f}%  {o['quarter_kelly_units']:.2f}u  {o['pick_id']}"
              + (f"\n      -> {o['hitl_reason']}" if o["hitl_reason"] else ""))
    for r in scan["rejections"]:
        print(f"  [REJECTED          ] {r['selection']:<32} {r['offered_american']:+5d}  "
              f"{r['reason']}")
    if scan["opportunities"]:
        print(f"\n== GAME ANALYSIS CARDS ({len(scan['opportunities'])} +EV plays; model on "
              f"{s['model_games']}/{s['events']} games)")
        for o in scan["opportunities"]:
            print(render_analysis_card(o))
    print(f"[ledger] {s['ledger_inserted']} new picks -> ledger_picks")
    print(f"[write] {OUTPUT_PATH.relative_to(ROOT)}")
    return 0, scan


def do_props_scan(conn, sports: tuple[str, ...], mock: bool) -> tuple[int, dict | None]:
    """Player-prop scan -> (exit code, schema-valid scan or None)."""
    print(f"\n== PLAYER PROPS SCAN | {'MOCK' if mock else 'LIVE'} | {', '.join(sports)}")
    try:
        scan = run_prop_scan(conn, sports, mock=mock)
    except (IngestionError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1, None
    errors, backend = validate(scan, load_schema(PROP_SCHEMA_PATH))
    if errors:
        print(f"PROP SCHEMA VALIDATION FAILED ({backend}):", file=sys.stderr)
        for e in errors:
            print(f"  - {e}", file=sys.stderr)
        return 1, None
    s = scan["ingestion"]["stats"]
    print(f"[schema] prop_candidates.json: PASS ({backend})")
    print(f"slate {s['events_on_slate']}/{s['events_listed']} events ({s['slate_gate']}) | "
          f"queried {s['events_queried']} (cache {s['events_from_cache']}) | quotes "
          f"{s['prop_quotes']} | lines {s['player_lines']} | modeled {s['players_modeled']} | "
          f"no model {s['no_model']} | floor {s['floor_rejected']} | no benchmark "
          f"{s['no_benchmark']} | market EV<=0 {s['negative_market_ev']} | no divergence "
          f"{s['no_divergence']} | injury out {s['injury_out']} | event-cap "
          f"{s['event_cap_rejected']} | quota {s['quota']}"
          + (" | QUOTA FLOOR HIT" if s["quota_guard_tripped"] else ""))
    for r in scan["ingestion"].get("stats_refresh") or []:
        print(f"[prop-stats] {r['sport']}: " + (f"ERROR {r['error']}" if r.get("error") else
              f"{r['players']} players, {r['player_rows']} rows, {r['teams']} teams"))
    for p in scan["props"]:
        print(f"  [{p['hitl_status']:<18}] {p['selection']:<36} proj {p['model_projection']:7.2f} "
              f"({p['delta']:+.2f}) {p['offered_american']:+5d} {p['retail_book']:<10} model "
              f"{p['model_raw_probability'] * 100:5.1f}% EV {p['ev_percent']:5.2f}% "
              f"{p['official_units']:.2f}u (raw Kelly {p['raw_kelly_units']:.2f}u)  "
              f"{p['pick_id']}\n      -> {p['hitl_reason']}")
    for r in scan["rejections"]:
        odds = r.get("offered_american")
        print(f"  [REJECTED          ] {r['selection']:<36} "
              f"{'' if odds is None else f'{odds:+5d}'}  {r['reason']}")
    print(f"[ledger] {s['ledger_inserted']} new prop picks -> ledger_picks")
    print(f"[write] {PROP_OUTPUT_PATH.relative_to(ROOT)}")
    return 0, scan


def write_drops(game_scan: dict, prop_scan: dict | None, props_enabled: bool,
                dirs: dict[str, Path], focus: str) -> int:
    """Unified OUTBOX drop + content files: game picks and player props together.

    ``props_enabled`` with no ``prop_scan`` means the prop scan failed; the
    files then say so instead of silently dropping the pillar.
    """
    slate, opps = game_scan["slate_date"], game_scan["opportunities"]
    banner = BANNERS[game_scan["mode"]]
    props = None if not props_enabled else (prop_scan or {}).get("props", [])
    props_note = ("Prop scan failed this run; player props are missing from this drop."
                  if props_enabled and prop_scan is None else "")
    print(f"\n== OUTBOX DROP | {slate} | game picks {len(opps)} | player props "
          + ("off (--no-props)" if props is None else f"{len(props)}"))
    for p in (write_discord_briefs(opps, slate, dirs["discord_briefs"], banner, props=props,
                                   props_note=props_note),
              write_social_content(select_free_pick(opps), slate, dirs["social_distribution"],
                                   focus, banner, opps=opps, props=props,
                                   props_note=props_note)):
        print(f"[write] {p.relative_to(ROOT)}")
    return 0


def do_update_prop_stats(conn) -> int:
    print("\n== UPDATE PROP STATS (nba_api, nflverse)")
    results = update_prop_stats(conn)
    for r in results:
        print(f"{r['sport']}: " + (f"ERROR {r['error']}" if r.get("error") else
              f"{r['players']} players | {r['player_rows']} game rows | {r['teams']} teams | "
              f"season {r.get('season')}" + (f" | source {r['source']}" if r.get("source")
                                             else "")
              + (f" | prior-season players {r['prior_players']}" if "prior_players" in r else "")
              + (f" (prior unavailable: {r['prior_error']})" if r.get("prior_error") else "")))
    return 0 if all(not r.get("error") for r in results) else 1


def do_update_stats(conn) -> int:
    print("\n== UPDATE STATS (ESPN, no key)")
    results = update_stats(conn)
    for r in results:
        print(f"{r['sport']}: {r['teams_with_games']}/{r['teams']} teams rated | season "
              f"{r['season']} | source {r['source']} | schedule rows {r['schedule_rows']} | "
              f"injuries {r['injuries']} (key Questionable {r['key_questionable']}) | "
              f"failed requests {r['failed_requests']}")
    for sport, col, label in (("NBA", "adj_net_rtg", "Adj NetRtg"), ("NFL", "srs", "SRS")):
        rows = conn.execute(
            f"SELECT canonical_id, wins, losses, {col} AS r FROM team_ratings WHERE sport = ? "
            f"AND {col} IS NOT NULL ORDER BY {col} DESC", (sport,)).fetchall()
        if rows:
            fmt = lambda x: f"{x['canonical_id']} {x['wins']}-{x['losses']} {x['r']:+.1f}"  # noqa
            print(f"  {sport} {label} top: {', '.join(map(fmt, rows[:3]))} | bottom: "
                  f"{', '.join(map(fmt, rows[-3:]))}")
    key = conn.execute("SELECT canonical_id, player, position, status FROM injury_reports "
                       "WHERE is_key = 1 AND lower(status) IN ('questionable', 'day-to-day', "
                       "'game-time decision', 'gtd') ORDER BY canonical_id").fetchall()
    print("  Key players Questionable (HITL trigger): "
          + (", ".join(f"{r['canonical_id']} {r['player']} ({r['position']}, {r['status']})"
                       for r in key) or "none"))
    return 0 if all(r["failed_requests"] == 0 for r in results) else 1


def do_settle(conn, mock: bool) -> int:
    print(f"\n== SETTLE | {'MOCK' if mock else 'LIVE'}")
    out = run_settlement(conn, mock=mock)
    print(f"closing lines: {out['closing']}")
    print(f"grading:       {out['grading']}")
    if out.get("bankroll"):
        print(f"bankroll:      {out['bankroll']}")
    origin = db.ORIGIN_MOCK if mock else db.ORIGIN_LIVE
    for p in db.fetch_picks(conn, "origin = ? AND settled_at IS NOT NULL", (origin,))[-10:]:
        print(f"  {p['pick_id']} {p['selection']:<22} {p['market_type']:<13} "
              f"{p['offered_american']:+5d} close {p['closing_american'] or 'n/a':>5} "
              f"CLV {'n/a' if p['clv_percent'] is None else f'{p['clv_percent']:+.2f}%':>7}  "
              f"{p['result']:<5} {p['net_units'] if p['net_units'] is not None else 0:+.2f}u "
              f"[{p['hitl_status']}]")
    return 0


def do_approve(conn, pick_id: str) -> int:
    p = db.get_pick(conn, pick_id)
    if p is None:
        print(f"ERROR: unknown pick {pick_id}", file=sys.stderr)
        return 1
    if p["hitl_status"] not in (HITL_PENDING, HITL_APPROVED):
        print(f"ERROR: {pick_id} is {p['hitl_status']}; cannot sign off", file=sys.stderr)
        return 1
    if p["signoff_at"]:
        print(f"[approve] {pick_id} already signed off by {p['signoff_by']} at {p['signoff_at']}")
        return 0
    try:
        p = sign_off(conn, pick_id)
    except (ValueError, PermissionError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(f"[approve] {pick_id} {p['selection']} -> APPROVED + signed off by "
          f"{p['signoff_by']} ({p['kelly_units']:.2f}u)")
    return 0


def do_decline(conn, pick_id: str) -> int:
    p = db.get_pick(conn, pick_id)
    if p is None:
        print(f"ERROR: unknown pick {pick_id}", file=sys.stderr)
        return 1
    try:
        p = Ledger(conn, p["origin"]).decline_position(
            pick_id, f"Declined by Captain HZ {db.iso(db.utcnow())}")
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(f"[decline] {pick_id} {p['selection']} -> {p['hitl_status']}")
    return 0


def do_backtest(conn, include_mock: bool, dirs: dict[str, Path]) -> int:
    print("\n== BACKTEST")
    bankroll = bankroll_units()
    seeded = seed_if_empty(conn, bankroll)
    if seeded:
        print(f"[seed] ledger had no history: inserted {seeded} SYNTHETIC_SEED trades")
    run_backtest(conn, bankroll=bankroll, include_mock=include_mock,
                 out_dir=dirs["market_scans"])
    print(f"\n[db] {db.table_counts(conn)}")
    for name in ("backtest_summary.md", "equity_curve.csv"):
        print(f"[write] {(dirs['market_scans'] / name).relative_to(ROOT)}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--setup-env", action="store_true", help="Write API keys to .env + ping")
    ap.add_argument("--odds-key", help="with --setup-env: The-Odds-API key")
    ap.add_argument("--kalshi-key-id", help="with --setup-env: Kalshi key ID")
    ap.add_argument("--kalshi-key-path", help="with --setup-env: Kalshi private key PEM path")
    ap.add_argument("--update-stats", action="store_true",
                    help="Refresh NBA/NFL ratings, schedules and injuries from ESPN, plus "
                         "player-prop stats (nba_api, nflverse) unless --no-props")
    ap.add_argument("--scan", "--live", dest="scan", action="store_true",
                    help="Live scan: game lines (The-Odds-API + Kalshi) and player props")
    ap.add_argument("--sport", default=None,
                    help=f"comma list or 'all' ({', '.join(SPORT_TO_LEAGUE)}); applies to the "
                         f"game scan and the prop scan. Default: all with --mock, else "
                         f"basketball_nba")
    ap.add_argument("--mock", action="store_true",
                    help="Offline synthetic games + props (also applies to --settle)")
    ap.add_argument("--no-props", action="store_true",
                    help="Skip player props (prop scan, prop stats refresh, props in OUTBOX)")
    ap.add_argument("--props", action="store_true",
                    help="Run the player-prop scan on its own (live, or synthetic with "
                         "--mock); props already run by default with --scan/--live/--mock")
    ap.add_argument("--settle", action="store_true", help="Closing lines, CLV, grading")
    ap.add_argument("--approve", metavar="PICK_ID", help="Captain HZ approves a pending pick")
    ap.add_argument("--decline", metavar="PICK_ID",
                    help="Captain HZ passes on an open pick (hitl_status DECLINED)")
    ap.add_argument("--backtest", action="store_true", help="Run the backtester")
    ap.add_argument("--include-mock", action="store_true", help="Backtest MOCK picks too")
    ap.add_argument("--focus", default="both", choices=["both", "parlay_tax", "house_money"])
    ap.add_argument("--db", type=Path, default=db.DB_PATH, help="SQLite path")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="[%(name)s] %(message)s", stream=sys.stdout)

    if not (args.setup_env or args.update_stats or args.scan or args.mock or args.settle
            or args.approve or args.decline or args.backtest or args.props):
        ap.print_help()
        return 2
    if args.props and args.no_props:
        ap.error("--props and --no-props are mutually exclusive")
    props_on = not args.no_props
    scanning = args.scan or args.mock
    # No --sport: mock covers the full synthetic slate, live defaults to NBA. An
    # explicit --sport is always honoured (game scan and prop scan alike).
    sport_arg = args.sport or ("all" if args.mock and not args.scan else "basketball_nba")
    sports = tuple(SPORT_TO_LEAGUE) if sport_arg == "all" else tuple(
        s.strip() for s in sport_arg.split(",") if s.strip())

    dirs = ensure_dirs()
    load_env()
    rc = 0
    if args.setup_env:
        rc |= 0 if setup_env(args.odds_key, args.kalshi_key_id, args.kalshi_key_path,
                             interactive=None) else 1
    conn = db.connect(args.db)
    try:
        print(f"[db] {args.db.relative_to(ROOT) if args.db.is_relative_to(ROOT) else args.db} "
              f"journal_mode={conn.execute('PRAGMA journal_mode').fetchone()[0]}")
        sync_registry(conn)
        if args.update_stats:
            rc |= do_update_stats(conn)
            if props_on:
                rc |= do_update_prop_stats(conn)
        game_scan = prop_scan = None
        if scanning:
            code, game_scan = do_scan(conn, sports, args.mock)
            rc |= code
        if props_on and (scanning or args.props):
            code, prop_scan = do_props_scan(conn, sports, args.mock)
            rc |= code
        if game_scan is not None:
            rc |= write_drops(game_scan, prop_scan, props_on, dirs, args.focus)
        if args.settle:
            rc |= do_settle(conn, args.mock)
        if args.approve:
            rc |= do_approve(conn, args.approve)
        if args.decline:
            rc |= do_decline(conn, args.decline)
        if args.backtest:
            rc |= do_backtest(conn, args.include_mock, dirs)
    finally:
        conn.close()
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
