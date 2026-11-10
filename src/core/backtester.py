"""Institutional backtester over the SQLite ledger (or an in-memory pick list).

Scope: graded (WIN/LOSS/PUSH), APPROVED, mandate-compliant picks. MOCK-origin
picks are excluded unless requested. Any pick priced worse than -130 is first
quarantined into ``rule_violations`` and marked RULE_VIOLATION, so it never
reaches a KPI.

Staking simulations (both off the fixed BANKROLL_UNITS baseline):
    Flat 1.0u        every bet risks 1.0u
    Quarter-Kelly    the recorded ``kelly_units`` (already single- and
                     event-capped at 2.0u); recomputed if missing

Metrics
    Core:  N, W-L-P, win %, units staked, net units, ROI %, yield %, profit factor
    Risk:  max drawdown (u, % of peak), drawdown duration (bets), streaks,
           per-bet Sharpe and Sortino
    Edge:  expected EV % vs realised ROI %, CLV avg and beat rate, Brier score
           and log-loss (model vs offered-price baseline), odds-weighted
           t-statistic and one-sided p-value for H0: no skill
Outputs: multi-panel ASCII report, backtest_summary.md, equity_curve.csv.
"""

from __future__ import annotations

import csv
import math
import random
import sqlite3
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Literal

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.core import database as db  # noqa: E402
from src.core.engine import (  # noqa: E402
    HITL_APPROVED,
    HITL_PENDING,
    MIN_AMERICAN_ODDS,
    american_to_decimal,
    decimal_to_american,
    devig_odds_ratio,
    devig_power,
    devig_shin_n,
    kalshi_net_odds,
    size_position,
    validate_odds_floor,
)
from src.data.teams import teams  # noqa: E402

REPORT_DIR = ROOT / "OUTBOX" / "market_scans"
Strategy = Literal["flat", "quarter_kelly"]
LABEL = {"flat": "Flat 1.0u", "quarter_kelly": "Quarter-Kelly"}
GRADED = ("WIN", "LOSS", "PUSH")

# Offered-odds buckets covering every valid price at the -130 floor or longer.
ODDS_BUCKETS: tuple[tuple[str, int, int], ...] = (
    ("-130 to -101", -130, -100),       # -100 is even money; grouped with favourites' edge
    ("+100 to +140", 100, 140),
    ("+141 and up", 141, 100_000),
)


# ==========================================================================
# Quarantine
# ==========================================================================
def violation_reason(pick: dict[str, Any]) -> str | None:
    """Why a pick breaches the mandate, or None if compliant."""
    odds = pick.get("offered_american")
    if not isinstance(odds, int) or not validate_odds_floor(odds):
        return f"Offered {odds:+d} worse than {MIN_AMERICAN_ODDS} floor" \
            if isinstance(odds, int) and odds <= -100 else f"Invalid offered odds {odds!r}"
    if (pick.get("kelly_units") or 0) > 2.0 + 1e-9:
        return f"Stake {pick['kelly_units']:.2f}u exceeds 2.0u cap"
    return None


def quarantine_violations(conn: sqlite3.Connection) -> int:
    """Move every non-compliant ledger pick into rule_violations (idempotent)."""
    n = 0
    for p in db.fetch_picks(conn, "result != 'RULE_VIOLATION'"):
        reason = violation_reason(p)
        if reason and db.quarantine_pick(conn, p, reason):
            n += 1
    return n


# ==========================================================================
# Simulation
# ==========================================================================
@dataclass
class Bet:
    pick: dict[str, Any]
    stake: float
    pnl: float
    equity: float


def _pnl(result: str, stake: float, dec: float) -> float:
    return stake * (dec - 1.0) if result == "WIN" else -stake if result == "LOSS" else 0.0


def _kelly_stake(p: dict[str, Any], bankroll: float) -> float:
    if p.get("kelly_units") is not None:
        return float(p["kelly_units"])
    return size_position(p["fair_prob"], p["offered_decimal"], 0.0, bankroll)[0]


def simulate(picks: list[dict[str, Any]], strategy: Strategy, bankroll: float) -> list[Bet]:
    """Chronological equity path. Stakes never depend on running equity."""
    equity, out = bankroll, []
    for p in picks:
        stake = 1.0 if strategy == "flat" else _kelly_stake(p, bankroll)
        if stake <= 0:
            continue
        pnl = _pnl(p["result"], stake, p["offered_decimal"])
        equity += pnl
        out.append(Bet(p, stake, pnl, equity))
    return out


# ==========================================================================
# Metrics
# ==========================================================================
def _streaks(results: list[str]) -> tuple[int, int]:
    """Longest win / loss streaks; pushes neither extend nor break a streak."""
    bw = bl = cw = cl = 0
    for r in results:
        if r == "WIN":
            cw, cl = cw + 1, 0
        elif r == "LOSS":
            cl, cw = cl + 1, 0
        bw, bl = max(bw, cw), max(bl, cl)
    return bw, bl


def _drawdown(bets: list[Bet], start: float) -> tuple[float, float, int]:
    """(max DD units, max DD % of running peak, longest spell below peak in bets)."""
    peak, mdd, mdd_pct, run, longest = start, 0.0, 0.0, 0, 0
    for b in bets:
        if b.equity >= peak:
            peak, run = b.equity, 0
            continue
        run += 1
        longest = max(longest, run)
        mdd = max(mdd, peak - b.equity)
        mdd_pct = max(mdd_pct, (peak - b.equity) / peak * 100)
    return mdd, mdd_pct, longest


def skill_test(bets: list[Bet]) -> tuple[float | None, float | None]:
    """Odds-weighted test of H0: no skill (win prob = break-even 1/decimal).

    Under H0 each non-push bet's return per unit R = d*X - 1 has mean 0 and
    variance d - 1 (heteroscedastic in the odds). With stakes s_i:
        t = sum(s_i * R_i) / sqrt(sum(s_i^2 * (d_i - 1)))
    One-sided p-value from the normal tail (large-N CLT).
    """
    num = var = 0.0
    for b in bets:
        if b.pick["result"] == "PUSH":
            continue
        d = b.pick["offered_decimal"]
        num += b.pnl
        var += b.stake ** 2 * (d - 1.0)
    if var <= 0:
        return None, None
    t = num / math.sqrt(var)
    return t, 0.5 * math.erfc(t / math.sqrt(2.0))


def strategy_metrics(bets: list[Bet], start: float, strategy: Strategy) -> dict[str, Any]:
    results = [b.pick["result"] for b in bets]
    w, l, p = (results.count(r) for r in GRADED)
    staked = sum(b.stake for b in bets)
    net = sum(b.pnl for b in bets)
    won = sum(b.pnl for b in bets if b.pnl > 0)
    lost = -sum(b.pnl for b in bets if b.pnl < 0)
    pnls = [b.pnl for b in bets]
    n = len(bets)
    mean = net / n if n else 0.0
    std = math.sqrt(sum((x - mean) ** 2 for x in pnls) / (n - 1)) if n > 1 else 0.0
    downside = math.sqrt(sum(min(x, 0.0) ** 2 for x in pnls) / n) if n else 0.0
    mdd, mdd_pct, dd_bets = _drawdown(bets, start)
    lw, ll = _streaks(results)
    t, pval = skill_test(bets)
    return {
        "strategy": LABEL[strategy], "n": n, "wins": w, "losses": l, "pushes": p,
        "record": f"{w}-{l}-{p}",
        "win_pct": w / (w + l) * 100 if w + l else 0.0,
        "staked": staked, "net": net,
        "roi_pct": net / staked * 100 if staked else 0.0,
        "yield_pct": sum(b.pnl / b.stake for b in bets) / n * 100 if n else 0.0,
        "profit_factor": won / lost if lost else None,
        "final_equity": start + net,
        "mdd_units": mdd, "mdd_pct": mdd_pct, "dd_duration_bets": dd_bets,
        "longest_win": lw, "longest_loss": ll,
        "sharpe": mean / std if std else None,
        "sortino": mean / downside if downside else None,
        "expected_ev_pct": (sum(b.stake * (b.pick.get("ev_percent") or 0.0) for b in bets)
                            / staked if staked else 0.0),
        "t_stat": t, "p_value": pval,
    }


def edge_metrics(picks: list[dict[str, Any]]) -> dict[str, Any]:
    """CLV, Brier and log-loss on graded non-push picks (strategy-independent)."""
    clv = [p["clv_percent"] for p in picks if p.get("clv_percent") is not None]
    scored = [p for p in picks if p["result"] in ("WIN", "LOSS") and p.get("fair_prob")]
    out: dict[str, Any] = {
        "clv_n": len(clv),
        "clv_avg_pct": sum(clv) / len(clv) if clv else None,
        "clv_beat_rate_pct": sum(c > 0 for c in clv) / len(clv) * 100 if clv else None,
        "brier_n": len(scored),
    }
    if scored:
        eps = 1e-12

        def brier(probs: list[float], ys: list[int]) -> float:
            return sum((q - y) ** 2 for q, y in zip(probs, ys)) / len(ys)

        def logloss(probs: list[float], ys: list[int]) -> float:
            return -sum(y * math.log(max(q, eps)) + (1 - y) * math.log(max(1 - q, eps))
                        for q, y in zip(probs, ys)) / len(ys)

        ys = [1 if p["result"] == "WIN" else 0 for p in scored]
        model = [p["fair_prob"] for p in scored]
        market = [1.0 / p["offered_decimal"] for p in scored]
        out.update({"brier_model": brier(model, ys), "brier_market": brier(market, ys),
                    "logloss_model": logloss(model, ys), "logloss_market": logloss(market, ys),
                    "mean_fair_prob": sum(model) / len(model),
                    "hit_rate": sum(ys) / len(ys)})
    return out


def bucket_metrics(picks: list[dict[str, Any]], bankroll: float) -> list[dict[str, Any]]:
    rows = []
    for label, lo, hi in ODDS_BUCKETS:
        sub = [p for p in picks if lo <= p["offered_american"] <= hi]
        f = strategy_metrics(simulate(sub, "flat", bankroll), bankroll, "flat")
        k = strategy_metrics(simulate(sub, "quarter_kelly", bankroll), bankroll, "quarter_kelly")
        clv = [p["clv_percent"] for p in sub if p.get("clv_percent") is not None]
        rows.append({"bucket": label, "n": f["n"], "record": f["record"], "win_pct": f["win_pct"],
                     "flat_net": f["net"], "flat_roi": f["roi_pct"], "qk_net": k["net"],
                     "qk_roi": k["roi_pct"],
                     "avg_ev": sum(p["ev_percent"] or 0 for p in sub) / len(sub) if sub else None,
                     "avg_clv": sum(clv) / len(clv) if clv else None})
    return rows


# ==========================================================================
# Rendering
# ==========================================================================
def _f(v: Any, kind: str = "") -> str:
    if v is None:
        return "n/a"
    spec = {"pct": ".2f", "u": ".2f", "s": "+.2f", "3": ".3f", "4": ".4f", "p": ".4f"}.get(kind)
    if spec is None or not isinstance(v, (int, float)):
        return str(v)
    return format(v, spec) + ("%" if kind == "pct" else "")


def ascii_table(title: str, header: list[str], rows: list[list[str]]) -> str:
    widths = [max(len(str(c)) for c in col) for col in zip(header, *rows)]
    sep = "+" + "+".join("-" * (w + 2) for w in widths) + "+"
    fmt = lambda cells: "| " + " | ".join(  # noqa: E731
        (str(c).ljust(w) if i == 0 else str(c).rjust(w))
        for i, (c, w) in enumerate(zip(cells, widths))) + " |"
    return "\n".join([f"== {title} ==", sep, fmt(header), sep, *map(fmt, rows), sep])


def md_table(header: list[str], rows: list[list[str]]) -> str:
    out = ["| " + " | ".join(header) + " |",
           "|" + "|".join(["---"] + ["---:"] * (len(header) - 1)) + "|"]
    return "\n".join(out + ["| " + " | ".join(map(str, r)) + " |" for r in rows])


CORE_ROWS = [("Sample size (N)", "n", ""), ("Record (W-L-P)", "record", ""),
             ("Win % (ex. pushes)", "win_pct", "pct"), ("Units staked", "staked", "u"),
             ("Net units", "net", "s"), ("ROI", "roi_pct", "pct"), ("Yield (mean per-bet)",
             "yield_pct", "pct"), ("Profit factor", "profit_factor", "3"),
             ("Final equity", "final_equity", "u")]
RISK_ROWS = [("Max drawdown (u)", "mdd_units", "u"), ("Max drawdown (% peak)", "mdd_pct", "pct"),
             ("Max DD duration (bets)", "dd_duration_bets", ""),
             ("Longest win streak", "longest_win", ""), ("Longest loss streak", "longest_loss", ""),
             ("Sharpe (per bet)", "sharpe", "4"), ("Sortino (per bet)", "sortino", "4")]
EDGE_ROWS = [("Expected EV % (stake-wtd)", "expected_ev_pct", "pct"),
             ("Realised ROI %", "roi_pct", "pct"), ("t-stat (H0: no skill)", "t_stat", "3"),
             ("p-value (one-sided)", "p_value", "p")]


@dataclass
class BacktestReport:
    generated_at: str
    bankroll: float
    scope: str
    origins: dict[str, int]
    flat: list[Bet]
    kelly: list[Bet]
    m: dict[str, dict[str, Any]]
    edge: dict[str, Any]
    buckets: list[dict[str, Any]]
    violations: list[dict[str, Any]]
    excluded: dict[str, int]


def _panels(r: BacktestReport) -> list[tuple[str, list[str], list[list[str]]]]:
    f, k = r.m["flat"], r.m["quarter_kelly"]
    two = lambda spec: [[lbl, _f(f[key], kd), _f(k[key], kd)] for lbl, key, kd in spec]  # noqa
    e = r.edge
    edge_single = [
        ["CLV average (n=%d)" % e["clv_n"], _f(e["clv_avg_pct"], "pct")],
        ["CLV beat rate", _f(e["clv_beat_rate_pct"], "pct")],
        ["Brier: model / market", f"{_f(e.get('brier_model'), '4')} / "
                                  f"{_f(e.get('brier_market'), '4')}"],
        ["Log-loss: model / market", f"{_f(e.get('logloss_model'), '4')} / "
                                     f"{_f(e.get('logloss_market'), '4')}"],
        ["Mean fair prob / hit rate", f"{_f(e.get('mean_fair_prob'), '4')} / "
                                      f"{_f(e.get('hit_rate'), '4')}"],
    ]
    buckets = [[b["bucket"], str(b["n"]), b["record"], _f(b["win_pct"], "pct"),
                _f(b["flat_net"], "s"), _f(b["flat_roi"], "pct"), _f(b["qk_net"], "s"),
                _f(b["qk_roi"], "pct"), _f(b["avg_ev"], "pct"), _f(b["avg_clv"], "pct")]
               for b in r.buckets]
    viol = [[v["pick_id"], str(v["offered_american"]), v["original_result"] or "-",
             v["reason"]] for v in r.violations] or [["-", "-", "-", "none"]]
    hdr = ["Metric", LABEL["flat"], LABEL["quarter_kelly"]]
    return [
        ("1. CORE PERFORMANCE", hdr, two(CORE_ROWS)),
        ("2. RISK", hdr, two(RISK_ROWS)),
        ("3a. EDGE VALIDATION (by staking)", hdr, two(EDGE_ROWS)),
        ("3b. EDGE VALIDATION (pick-level)", ["Metric", "Value"], edge_single),
        ("4. ODDS BUCKETS", ["Offered odds", "N", "W-L-P", "Win %", "Flat net", "Flat ROI",
                             "QK net", "QK ROI", "Avg EV", "Avg CLV"], buckets),
        ("5. RULE QUARANTINE (excluded from all KPIs)",
         ["Pick", "Offered", "Orig. result", "Reason"], viol),
    ]


def _notes(r: BacktestReport) -> list[str]:
    out = [f"Scope: {r.scope} | origins {r.origins} | excluded: {r.excluded}"]
    if r.origins.get(db.ORIGIN_SEED):
        out.append("NOTE: includes SYNTHETIC_SEED trades (seeded verification data, not a real "
                   "track record).")
    return out


def render_ascii(r: BacktestReport) -> str:
    head = [f"BACKTEST | Captain HZ | singles only, >= {MIN_AMERICAN_ODDS} | "
            f"baseline {r.bankroll:.0f}u fixed | {r.generated_at}"]
    return "\n\n".join(["\n".join(head + _notes(r))]
                       + [ascii_table(*p) for p in _panels(r)])


def render_markdown(r: BacktestReport) -> str:
    out = ["# Backtest Summary", "",
           f"Generated {r.generated_at} | Operating authority: Captain HZ | Baseline bankroll "
           f"{r.bankroll:.0f}u (fixed; stakes never compound P&L)", ""]
    if r.origins.get(db.ORIGIN_SEED):
        out += ["> **Synthetic data.** This run includes seeded verification trades. It is not "
                "a real track record and must not be presented as one.", ""]
    out += [f"Scope: {r.scope}. Origins: {r.origins}. Excluded: {r.excluded}.", ""]
    for title, header, rows in _panels(r):
        out += [f"## {title.split('. ', 1)[1].title()}", "", md_table(header, rows), ""]
    out += [
        "## Methodology", "",
        "- Universe: graded (WIN/LOSS/PUSH), APPROVED, compliant singles. PENDING picks never "
        "approved by Captain HZ are excluded.",
        "- Flat: 1.0u per bet. Quarter-Kelly: the recorded stake (0.25x Kelly on consensus fair "
        "probability, capped at 2.0u per single and 2.0u per event).",
        "- ROI = net / staked. Yield = mean of per-bet return on stake. Profit factor = gross won "
        "/ gross lost.",
        "- Sharpe = mean / stdev of per-bet P&L; Sortino uses downside deviation. Per bet, not "
        "annualised.",
        "- CLV = offered decimal x closing de-vigged sharp probability - 1. Beat rate = share "
        "of bets with CLV > 0.",
        "- Brier / log-loss compare the model's fair probability with the offered price's raw "
        "implied probability on the same bets (lower is better).",
        "- Skill test: under H0 (win prob = 1/decimal) each bet's unit return has mean 0 and "
        "variance (decimal - 1); t = sum(stake x R) / sqrt(sum(stake^2 x (decimal - 1))). "
        "One-sided normal p-value.",
        f"- Picks priced worse than {MIN_AMERICAN_ODDS} are quarantined to `rule_violations`.",
    ]
    return "\n".join(out) + "\n"


def write_equity_curve(r: BacktestReport, path: Path) -> None:
    by_id = {b.pick["pick_id"]: b for b in r.kelly}
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["step", "pick_id", "commence_time", "selection", "market_type",
                    "offered_american", "result", "clv_percent", "flat_stake", "flat_pnl",
                    "flat_equity", "qk_stake", "qk_pnl", "qk_equity"])
        w.writerow([0, "", "", "BASELINE", "", "", "", "", "", "", f"{r.bankroll:.2f}", "", "",
                    f"{r.bankroll:.2f}"])
        qk_eq = r.bankroll
        for i, b in enumerate(r.flat, 1):
            p = b.pick
            k = by_id.get(p["pick_id"])
            if k:
                qk_eq = k.equity
            sel = p["selection"] + (f" {p['line_point']:+g}" if p["market_type"] == "spreads"
                                    else f" {p['line_point']:g}" if p["market_type"] == "totals"
                                    else "")
            w.writerow([i, p["pick_id"], p["commence_time"], sel, p["market_type"],
                        p["offered_american"], p["result"],
                        "" if p.get("clv_percent") is None else f"{p['clv_percent']:.3f}",
                        f"{b.stake:.2f}", f"{b.pnl:.4f}", f"{b.equity:.4f}",
                        f"{k.stake:.2f}" if k else "0.00", f"{k.pnl:.4f}" if k else "0.0000",
                        f"{qk_eq:.4f}"])


# ==========================================================================
# Entry point
# ==========================================================================
def run_backtest(conn: sqlite3.Connection | None = None,
                 picks: list[dict[str, Any]] | None = None,
                 bankroll: float = 100.0, include_mock: bool = False,
                 out_dir: Path = REPORT_DIR, echo: bool = True) -> BacktestReport:
    """Backtest the SQLite ledger, or an in-memory ``picks`` simulation list."""
    excluded: dict[str, int] = {}
    if picks is None:
        if conn is None:
            raise ValueError("pass a connection or a pick list")
        quarantine_violations(conn)
        origins = (db.ORIGIN_LIVE, db.ORIGIN_SEED) + ((db.ORIGIN_MOCK,) if include_mock else ())
        marks = ",".join("?" * len(origins))
        universe = db.fetch_picks(conn, f"origin IN ({marks})", origins)
        violations = db.fetch_violations(conn)
        scope = "SQLite ledger_picks"
    else:
        universe, violations = [], []
        for p in picks:
            reason = violation_reason(p)
            if reason:
                violations.append({"pick_id": p.get("pick_id"), "offered_american":
                                   p.get("offered_american"), "original_result": p.get("result"),
                                   "reason": reason})
            else:
                universe.append(p)
        scope = "in-memory simulation list"

    compliant = []
    for p in universe:
        if p["result"] == "RULE_VIOLATION":
            continue
        if p["result"] not in GRADED:
            excluded["unsettled"] = excluded.get("unsettled", 0) + 1
        elif p["hitl_status"] != HITL_APPROVED:
            excluded[p["hitl_status"].lower()] = excluded.get(p["hitl_status"].lower(), 0) + 1
        else:
            compliant.append(p)
    compliant.sort(key=lambda p: (p["commence_time"] or "", p["pick_id"]))
    origins_count: dict[str, int] = {}
    for p in compliant:
        origins_count[p.get("origin", "LIST")] = origins_count.get(p.get("origin", "LIST"), 0) + 1

    flat = simulate(compliant, "flat", bankroll)
    kelly = simulate(compliant, "quarter_kelly", bankroll)
    report = BacktestReport(
        generated_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        bankroll=bankroll, scope=scope, origins=origins_count, flat=flat, kelly=kelly,
        m={"flat": strategy_metrics(flat, bankroll, "flat"),
           "quarter_kelly": strategy_metrics(kelly, bankroll, "quarter_kelly")},
        edge=edge_metrics(compliant), buckets=bucket_metrics(compliant, bankroll),
        violations=violations, excluded=excluded)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "backtest_summary.md").write_text(render_markdown(report), encoding="utf-8")
    write_equity_curve(report, out_dir / "equity_curve.csv")
    if echo:
        print(render_ascii(report))
    return report


# ==========================================================================
# Seed dataset
# ==========================================================================
# Seed chosen so realised flat ROI (+3.2%) sits near average CLV (+2.3%), the
# generator's true expectation; seeds 4240-59 span -9% to +12% flat ROI.
SEED = 4258
SEED_N = 250
SEED_VIOLATION_SLOTS = (23, 71, 118, 166, 214)          # five intentional -145 trades
_PRICES_2WAY = [-125, -120, -115, -112, -110, -108, -105, 100, 102, 105, 110]
_PRICES_ML = [-130, -125, -120, -115, -110, 105, 110, 115, 120, 130, 140, 150, 165, 180]
_BOOKS = ["DraftKings", "FanDuel", "BetMGM", "Caesars"]


def _pin_price(prob: float, vig: float = 1.022) -> int:
    """Pinnacle-style vigged American price for a fair probability."""
    return decimal_to_american(1.0 / min(prob * vig, 0.99))


def generate_seed_trades(n: int = SEED_N, seed: int = SEED,
                         bankroll: float = 100.0) -> list[dict[str, Any]]:
    """Deterministic, realistic graded history for verification.

    * Entry fair probability = offered implied + edge (EV >= 1.5% by construction).
    * Closing fair probability = entry fair + N(-1.0pt, 2.0pt): the model is
      slightly overconfident and the close is sharper than the entry estimate.
    * Outcomes are drawn from the CLOSING fair probability, so realised ROI
      tracks average CLV, as it does in efficient markets.
    * Integer spreads push ~6% of the time. Five trades at -145 breach the floor.
    """
    rng = random.Random(seed)
    start = datetime(2026, 8, 25, tzinfo=timezone.utc)
    nba_open = datetime(2026, 10, 21, tzinfo=timezone.utc)
    rows = []
    for i in range(n):
        tip = start + timedelta(days=75 * i / n, hours=rng.choice([17, 20, 23, 24, 25]))
        league = "NFL" if tip < nba_open or rng.random() < 0.35 else "NBA"
        sport = "americanfootball_nfl" if league == "NFL" else "basketball_nba"
        home, away = rng.sample(teams(league), 2)
        side = rng.choice([home, away])
        violation = i in SEED_VIOLATION_SLOTS
        mkt = "h2h" if violation else rng.choices(
            ["spreads", "totals", "h2h", "kalshi_binary"], [42, 22, 28, 8])[0]
        point: float | None = None
        selection, book = side, rng.choice(_BOOKS)
        if mkt == "spreads":
            point = rng.choice([1.5, 2.5, 3.0, 3.5, 6.5, 7.0] if league == "NFL"
                               else [1.5, 2.5, 4.0, 4.5, 5.5, 7.5]) * rng.choice([-1, 1])
            offered = rng.choice(_PRICES_2WAY)
        elif mkt == "totals":
            point = rng.choice([41.5, 44.5, 47.5] if league == "NFL" else [216.5, 224.5, 231.5])
            selection, offered = rng.choice(["Over", "Under"]), rng.choice(_PRICES_2WAY)
        elif mkt == "kalshi_binary":
            book = "Kalshi"
            offered = decimal_to_american(kalshi_net_odds(round(rng.uniform(0.30, 0.53), 2)))
        else:
            offered = -145 if violation else rng.choice(_PRICES_ML)
        dec = american_to_decimal(offered)
        implied = 1.0 / dec
        edge = rng.uniform(max(0.0155 / dec, 0.008), 0.035)
        fair = min(implied + edge, 0.95)
        units, status, note = size_position(fair, dec, 0.0, bankroll)
        if status == HITL_PENDING:
            status, note = HITL_APPROVED, f"Captain HZ sign-off (seed): {note}"
        close_fair = min(max(fair + rng.gauss(-0.010, 0.020), 0.03), 0.97)
        if point is not None and float(point).is_integer() and rng.random() < 0.06:
            result = "PUSH"
        else:
            result = "WIN" if rng.random() < close_fair else "LOSS"
        net = units * (dec - 1) if result == "WIN" else -units if result == "LOSS" else 0.0
        kalshi_mid = (round(min(max(fair + rng.gauss(0, 0.015), 0.02), 0.98), 4)
                      if mkt in ("h2h", "kalshi_binary") else None)
        rows.append({
            "pick_id": f"SEED-{i + 1:04d}",
            "created_at": db.iso(tip - timedelta(hours=rng.uniform(2, 20))),
            "commence_time": db.iso(tip), "sport": sport, "event_id": f"seed-evt-{i + 1:04d}",
            "event_name": f"{away} @ {home}", "market_type": mkt, "selection": selection,
            "line_point": point, "bookmaker": book, "offered_american": offered,
            "offered_decimal": round(dec, 6), "benchmark_source": rng.choice(
                ["Pinnacle", "Pinnacle", "Pinnacle", "Circa"]),
            "benchmark_american": _pin_price(fair), "fair_prob": round(fair, 6),
            "kalshi_mid_prob": kalshi_mid, "ev_percent": round((fair * dec - 1) * 100, 3),
            "kelly_units": units, "closing_american": _pin_price(close_fair),
            "closing_fair_prob": round(close_fair, 6),
            "clv_percent": round((dec * close_fair - 1) * 100, 3),
            "hitl_status": status, "hitl_reason": note or None, "result": result,
            "net_units": round(net, 4), "origin": db.ORIGIN_SEED,
            "settled_at": db.iso(tip + timedelta(hours=3)),
        })
    return rows


def seed_if_empty(conn: sqlite3.Connection, bankroll: float = 100.0) -> int:
    """Seed 250 synthetic trades when the ledger has no LIVE or SEED history."""
    if db.count_picks(conn, (db.ORIGIN_LIVE, db.ORIGIN_SEED)):
        return 0
    return db.insert_picks(conn, generate_seed_trades(bankroll=bankroll))


# ==========================================================================
# Closing Line Value & skill testing -- Buchdahl
# ==========================================================================
# "What is the Closing Line?", "What is the Closing Line Value?", "Making the
# closing odds fair". The fair (no-vig) closing price is the best available
# estimate of the true probability; CLV = offered decimal x closing NVP prob
# (= offered dec / fair closing dec). CLV's per-bet sigma (~0.1) is ~10x
# smaller than P/L's (~1.0), so skill can be detected in ~50 bets.

NVP_METHODS = ("shin", "odds_ratio", "logit", "power")


def calculate_no_vig_price(closing_odds_a: float, closing_odds_b: float,
                           method: str = "shin") -> tuple[float, float]:
    """Fair closing probabilities (sum to 1) from two-way DECIMAL closing odds.

    Methods: ``shin`` (Clarke et al. 2017; == additive for n = 2),
    ``odds_ratio``/``logit`` (Strumbelj 2014), ``power``. Crude equal-margin
    division is deliberately not offered.
    """
    for d in (closing_odds_a, closing_odds_b):
        if not d > 1.0:
            raise ValueError(f"decimal odds must be > 1.0, got {d}")
    m = method.lower()
    if m not in NVP_METHODS:
        raise ValueError(f"unknown method {method!r}; use one of {NVP_METHODS}")
    pis = [1.0 / closing_odds_a, 1.0 / closing_odds_b]
    if m == "shin":
        if sum(pis) <= 1.0:
            raise ValueError("closing book has no overround; nothing to de-vig")
        pa, pb = devig_shin_n(pis)[0]
    elif m in ("odds_ratio", "logit"):
        pa, pb = devig_odds_ratio(closing_odds_a, closing_odds_b)
    else:
        pa, pb = devig_power(closing_odds_a, closing_odds_b)
    s = pa + pb
    return pa / s, pb / s


def compute_clv(offered_dec_odds: float, closing_nvp_prob: float) -> float:
    """CLV ratio = offered decimal x fair closing probability (> 1.0 beats the close)."""
    if not offered_dec_odds > 1.0:
        raise ValueError(f"offered decimal odds must be > 1.0, got {offered_dec_odds}")
    if not 0.0 < closing_nvp_prob < 1.0:
        raise ValueError(f"closing NVP probability must be in (0, 1), got {closing_nvp_prob}")
    return offered_dec_odds * closing_nvp_prob


def clv_skill_test(clv_ratios: list[float]) -> tuple[float, float]:
    """One-sample t-test of H0: mean CLV ratio == 1 -> (t, one-sided p via normal tail)."""
    n = len(clv_ratios)
    if n < 2:
        raise ValueError("need at least 2 CLV observations")
    mean = sum(clv_ratios) / n
    sd = math.sqrt(sum((c - mean) ** 2 for c in clv_ratios) / (n - 1))
    if sd == 0.0:
        t = math.copysign(math.inf, mean - 1.0) if mean != 1.0 else 0.0
    else:
        t = (mean - 1.0) / (sd / math.sqrt(n))
    return t, 0.5 * math.erfc(t / math.sqrt(2.0))


def evaluate_forecast_skill(predictions: list[float], outcomes: list[int]) -> float:
    """Brier score mean((p - y)^2): 0 perfect, 0.25 for a coin-flip forecaster."""
    if len(predictions) != len(outcomes):
        raise ValueError("predictions and outcomes must be the same length")
    if not predictions:
        raise ValueError("no forecasts to score")
    for p in predictions:
        if not 0.0 <= p <= 1.0:
            raise ValueError(f"prediction must be in [0, 1], got {p}")
    for o in outcomes:
        if o not in (0, 1):
            raise ValueError(f"outcome must be 0 or 1, got {o}")
    return sum((p - o) ** 2 for p, o in zip(predictions, outcomes)) / len(predictions)
