"""Breeze quant backend - unified CLI. All workflows report to Captain HZ.

    python run_pipeline.py --setup-env [--odds-key K --kalshi-key-id ID --kalshi-key-path P]
    python run_pipeline.py --update-stats                     # ESPN ratings/rest/injuries
    python run_pipeline.py --scan --sport basketball_nba      # live (cached 300s)
    python run_pipeline.py --mock                             # offline synthetic scan
    python run_pipeline.py --settle [--mock]                  # closing lines, CLV, grading
    python run_pipeline.py --approve PICK_ID                  # Captain HZ sign-off
    python run_pipeline.py --backtest [--include-mock]        # seeds 250 trades if empty

Actions combine and run in order: setup -> update-stats -> scan -> settle -> approve
-> backtest.
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
from src.core.orchestrator import sign_off  # noqa: E402
from src.core.schema_validator import load_schema, validate  # noqa: E402
from src.data.market_fetcher import OUTPUT_PATH, IngestionError, run_scan  # noqa: E402
from src.data.settler import run_settlement  # noqa: E402
from src.data.stats_fetcher import update_stats  # noqa: E402
from src.data.team_registry import sync_registry  # noqa: E402
from src.data.teams import SPORT_TO_LEAGUE  # noqa: E402
from templates.analysis_cards import render_analysis_card  # noqa: E402
from templates.discord_templates import select_free_pick, write_discord_briefs  # noqa: E402
from templates.social_templates import write_social_content  # noqa: E402

SCHEMA_PATH = ROOT / "schemas" / "market_scan_schema.json"
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


def do_scan(conn, sports: tuple[str, ...], mock: bool, dirs: dict[str, Path],
            focus: str) -> int:
    print(f"\n== SCAN | {'MOCK' if mock else 'LIVE'} | {', '.join(sports)}")
    try:
        scan = run_scan(conn, sports, mock=mock)
    except IngestionError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    errors, backend = validate(scan, load_schema(SCHEMA_PATH))
    if errors:
        print(f"SCHEMA VALIDATION FAILED ({backend}):", file=sys.stderr)
        for e in errors:
            print(f"  - {e}", file=sys.stderr)
        return 1
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
    slate = scan["slate_date"]
    opps = scan["opportunities"]
    for p in (write_discord_briefs(opps, slate, dirs["discord_briefs"], BANNERS[scan["mode"]]),
              write_social_content(select_free_pick(opps), slate, dirs["social_distribution"],
                                   focus, BANNERS[scan["mode"]])):
        print(f"[write] {p.relative_to(ROOT)}")
    return 0


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
                    help="Refresh NBA/NFL ratings, schedules and injuries from ESPN")
    ap.add_argument("--scan", action="store_true", help="Live scan (The-Odds-API + Kalshi)")
    ap.add_argument("--sport", default="basketball_nba",
                    help=f"comma list or 'all' ({', '.join(SPORT_TO_LEAGUE)})")
    ap.add_argument("--mock", action="store_true",
                    help="Offline synthetic data (scan; also applies to --settle)")
    ap.add_argument("--settle", action="store_true", help="Closing lines, CLV, grading")
    ap.add_argument("--approve", metavar="PICK_ID", help="Captain HZ approves a pending pick")
    ap.add_argument("--backtest", action="store_true", help="Run the backtester")
    ap.add_argument("--include-mock", action="store_true", help="Backtest MOCK picks too")
    ap.add_argument("--focus", default="both", choices=["both", "parlay_tax", "house_money"])
    ap.add_argument("--db", type=Path, default=db.DB_PATH, help="SQLite path")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="[%(name)s] %(message)s", stream=sys.stdout)

    if not (args.setup_env or args.update_stats or args.scan or args.mock or args.settle
            or args.approve or args.backtest):
        ap.print_help()
        return 2
    sports = tuple(SPORT_TO_LEAGUE) if args.sport == "all" else tuple(
        s.strip() for s in args.sport.split(",") if s.strip())
    if args.mock and args.sport == "basketball_nba" and not args.scan:
        sports = tuple(SPORT_TO_LEAGUE)                 # mock covers the full synthetic slate

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
        if args.scan or args.mock:
            rc |= do_scan(conn, sports, args.mock, dirs, args.focus)
        if args.settle:
            rc |= do_settle(conn, args.mock)
        if args.approve:
            rc |= do_approve(conn, args.approve)
        if args.backtest:
            rc |= do_backtest(conn, args.include_mock, dirs)
    finally:
        conn.close()
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
