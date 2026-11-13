"""Breeze infinite monitor: run the scan pipeline on a loop and show +EV picks.

    python infinite_monitor.py                      # mock scan every 60s (default)
    python infinite_monitor.py --live               # live scan (--scan --sport all)
    python infinite_monitor.py --live --interval 300 --refresh-stats-every 3
    python infinite_monitor.py --once               # one cycle, then exit
    python infinite_monitor.py --mock --no-props    # game lines only, skip player props
    python infinite_monitor.py --live --props-every 5   # props every 5th cycle

Player props are a first-class pillar: every cycle scans game lines AND props
(and --update-stats refreshes both) unless --no-props is given.

Each cycle calls ``run_pipeline.py`` as a subprocess (same interpreter), then
reads the structured result. Sources are tried in order:

    1. OUTBOX/market_scans/live_candidates.json, if this cycle rewrote it and
       its ``mode`` matches the requested mode (exact fields).
    2. The pipeline's stdout (opportunity rows, stats line, analysis cards).
    3. data/takeover.db ledger_picks: open picks for the origin (last resort).

The monitor only reads. It never signs off or dispatches anything: approving a
PENDING_CAPTAIN_HZ pick still goes through ``run_pipeline.py --approve PICK_ID``.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sqlite3
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
PIPELINE = ROOT / "run_pipeline.py"
CANDIDATES_PATH = ROOT / "OUTBOX" / "market_scans" / "live_candidates.json"
PROPS_PATH = ROOT / "OUTBOX" / "market_scans" / "prop_candidates.json"
DB_PATH = ROOT / "data" / "takeover.db"

PENDING = "PENDING_CAPTAIN_HZ"
INJURY_WORDS = ("injur", "questionable", "day-to-day", "game-time", "gtd", "out ")


# ==========================================================================
# Terminal styling
# ==========================================================================
class Style:
    """ANSI colours, disabled for non-TTY output, NO_COLOR, or --no-color."""

    def __init__(self, enabled: bool) -> None:
        self.on = enabled
        if enabled and os.name == "nt":
            os.system("")                       # turns on VT escape handling in Windows consoles

    def _wrap(self, code: str, text: str) -> str:
        return f"\033[{code}m{text}\033[0m" if self.on else text

    def bold(self, t: str) -> str: return self._wrap("1", t)
    def dim(self, t: str) -> str: return self._wrap("2", t)
    def red(self, t: str) -> str: return self._wrap("1;31", t)
    def green(self, t: str) -> str: return self._wrap("32", t)
    def yellow(self, t: str) -> str: return self._wrap("1;33", t)
    def magenta(self, t: str) -> str: return self._wrap("1;35", t)
    def cyan(self, t: str) -> str: return self._wrap("36", t)


def _configure_stdout() -> None:
    """Never crash on a character the Windows console code page can't encode."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
        except (AttributeError, ValueError):
            pass


# ==========================================================================
# Subprocess execution
# ==========================================================================
@dataclass
class RunResult:
    label: str
    returncode: int | None              # None = timed out or failed to start
    stdout: str = ""
    stderr: str = ""
    seconds: float = 0.0
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.returncode == 0


def run_pipeline(args: list[str], timeout: float, label: str) -> RunResult:
    """Call run_pipeline.py with the active interpreter. Never raises (except Ctrl+C)."""
    env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1")
    cmd = [sys.executable, str(PIPELINE), *args]
    t0 = time.monotonic()
    try:
        cp = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True,
                            encoding="utf-8", errors="replace", timeout=timeout)
        return RunResult(label, cp.returncode, cp.stdout or "", cp.stderr or "",
                         time.monotonic() - t0)
    except subprocess.TimeoutExpired as exc:   # run() has already killed the child
        out = exc.stdout.decode("utf-8", "replace") if isinstance(exc.stdout, bytes) \
            else (exc.stdout or "")
        return RunResult(label, None, out, "", time.monotonic() - t0,
                         f"timed out after {timeout:.0f}s")
    except OSError as exc:
        return RunResult(label, None, "", "", time.monotonic() - t0,
                         f"could not start pipeline: {exc}")


# ==========================================================================
# Sourcing the scan
# ==========================================================================
@dataclass
class ScanView:
    source: str                                          # json | stdout | sqlite | none
    mode: str
    timestamp: str | None = None
    stats: dict[str, Any] = field(default_factory=dict)
    opportunities: list[dict[str, Any]] = field(default_factory=list)
    rejections: list[dict[str, Any]] = field(default_factory=list)
    cards: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


def load_json_scan(path: Path, since: float, mode: str) -> tuple[dict[str, Any] | None, str]:
    """The candidates file, only if this cycle wrote it and the mode matches."""
    try:
        if path.stat().st_mtime < since:
            return None, "candidates file not rewritten this cycle"
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None, f"{path.name} not found"
    except (OSError, json.JSONDecodeError) as exc:
        return None, f"{path.name} unreadable: {exc}"
    if not isinstance(data, dict) or "opportunities" not in data:
        return None, f"{path.name} has no opportunities list"
    if str(data.get("mode", "")).upper() != mode:
        return None, f"{path.name} mode {data.get('mode')!r} != {mode}"
    return data, ""


def render_cards(opps: list[dict[str, Any]]) -> list[str] | None:
    """Render cards with the repo's own renderer (identical to the scan output)."""
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    try:
        from templates.analysis_cards import render_analysis_card
        return [render_analysis_card(o) for o in opps]
    except Exception:                                    # renderer missing / changed
        return None


_STATS_RE = re.compile(
    r"events (?P<events>\d+) \| retail quotes (?P<retail_quotes>\d+) \| point mismatch "
    r"(?P<point_mismatch>\d+) \| floor rejected (?P<floor_rejected>\d+) \| below 1\.5% EV "
    r"(?P<below_min_ev>\d+) \| event-cap rejected (?P<event_cap_rejected>\d+) \| Kalshi matched "
    r"(?P<kalshi_matched>\d+)/(?P<kalshi_markets>\d+) \| snapshots \+(?P<snapshots>\d+) \| "
    r"quota (?P<quota>.*)$")
_OPP_RE = re.compile(
    r"^\s+\[(?P<hitl_status>[A-Z_]+)\s*\] (?P<selection>.+?)\s+(?P<offered_american>[+-]\d+) "
    r"(?P<retail_book>\S+)\s+fair\s+(?P<fair>[\d.]+)%\s+EV\s+(?P<ev>-?[\d.]+)%\s+"
    r"(?P<units>[\d.]+)u\s+(?P<pick_id>\S+)\s*$")
_REJ_RE = re.compile(r"^\s+\[REJECTED\s*\] (?P<selection>.+?)\s+(?P<offered_american>[+-]\d+)"
                     r"\s+(?P<reason>.+)$")


def parse_stdout(text: str, mode: str) -> ScanView:
    """Rebuild the scan from run_pipeline.py's printed rows."""
    view = ScanView("stdout", mode)
    lines = text.splitlines()
    for i, line in enumerate(lines):
        m = _STATS_RE.search(line)
        if m:
            view.stats = {k: (v if k == "quota" else int(v)) for k, v in m.groupdict().items()}
            continue
        m = _OPP_RE.match(line)
        if m and m["hitl_status"] != "REJECTED":
            nxt = lines[i + 1] if i + 1 < len(lines) else ""
            reason = nxt.strip()[3:] if nxt.strip().startswith("-> ") else ""
            view.opportunities.append({
                "pick_id": m["pick_id"], "selection": m["selection"].strip(),
                "offered_american": int(m["offered_american"]), "retail_book": m["retail_book"],
                "fair_probability": float(m["fair"]) / 100.0, "ev_percent": float(m["ev"]),
                "quarter_kelly_units": float(m["units"]), "hitl_status": m["hitl_status"],
                "hitl_reason": reason})
            continue
        m = _REJ_RE.match(line)
        if m:
            view.rejections.append({"selection": m["selection"].strip(),
                                    "offered_american": int(m["offered_american"]),
                                    "reason": m["reason"].strip()})
    # Cards: everything between the cards header and the "[ledger]" line.
    start = next((i for i, l in enumerate(lines) if l.startswith("== GAME ANALYSIS CARDS")), None)
    if start is not None:
        end = next((i for i in range(start + 1, len(lines)) if lines[i].startswith("[ledger]")),
                   len(lines))
        block = "\n".join(lines[start + 1:end]).strip("\n")
        if block:
            view.cards = [block]
    return view


def load_sqlite(db_path: Path, mode: str) -> ScanView:
    """Last resort: open, unrejected picks for this origin from ledger_picks.

    The ledger keeps the FIRST record of each pick, so prices here may be older
    than the latest scan.
    """
    view = ScanView("sqlite", mode)
    if not db_path.exists():
        view.notes.append(f"{db_path.name} not found")
        return view
    try:
        conn = sqlite3.connect(f"file:{db_path.as_posix()}?mode=ro", uri=True, timeout=5)
        conn.row_factory = sqlite3.Row
        try:
            rows = conn.execute(
                "SELECT pick_id, selection, offered_american, bookmaker, fair_prob, ev_percent, "
                "kelly_units, hitl_status, hitl_reason, created_at FROM ledger_picks "
                "WHERE origin = ? AND result = 'UNSETTLED' AND hitl_status != 'REJECTED' "
                "ORDER BY created_at DESC LIMIT 200", (mode,)).fetchall()
        finally:
            conn.close()
    except sqlite3.Error as exc:
        view.notes.append(f"SQLite read failed: {exc}")
        return view
    for r in rows:
        view.opportunities.append({
            "pick_id": r["pick_id"], "selection": r["selection"],
            "offered_american": r["offered_american"], "retail_book": r["bookmaker"] or "?",
            "fair_probability": r["fair_prob"], "ev_percent": r["ev_percent"],
            "quarter_kelly_units": r["kelly_units"], "hitl_status": r["hitl_status"],
            "hitl_reason": r["hitl_reason"] or ""})
    if rows:
        view.notes.append("open ledger positions (first-recorded prices), not this cycle's scan")
    return view


@dataclass
class PropView:
    source: str                                  # json | json-earlier | sqlite | none
    props: list[dict[str, Any]] = field(default_factory=list)
    rejections: list[dict[str, Any]] = field(default_factory=list)
    stats: dict[str, Any] = field(default_factory=dict)
    config: dict[str, Any] = field(default_factory=dict)
    timestamp: str | None = None
    note: str = ""


def collect_props(ran_this_cycle: bool, ok: bool, mode: str, since: float,
                  db_path: Path) -> PropView:
    """Prop scan output: this cycle's file, else the last same-mode file, else the ledger."""
    try:
        mtime = PROPS_PATH.stat().st_mtime
        data = json.loads(PROPS_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        mtime, data = 0.0, None
    if isinstance(data, dict) and str(data.get("mode", "")).upper() == mode \
            and isinstance(data.get("props"), list):
        fresh = mtime >= since
        view = PropView("json" if fresh else "json-earlier", list(data["props"]),
                        list(data.get("rejections") or []),
                        dict((data.get("ingestion") or {}).get("stats") or {}),
                        dict(data.get("config") or {}), data.get("timestamp"))
        if ran_this_cycle and not fresh:
            view.note = "prop scan did not rewrite its file this cycle; showing the previous scan"
        elif not ran_this_cycle:
            view.note = "props not scanned this cycle (--props-every); showing the previous scan"
        if ran_this_cycle and fresh and not ok:
            view.note = "pipeline exited non-zero; prop file may not have passed its schema"
        return view
    view = PropView("sqlite")
    try:
        conn = sqlite3.connect(f"file:{db_path.as_posix()}?mode=ro", uri=True, timeout=5)
        conn.row_factory = sqlite3.Row
        try:
            rows = conn.execute(
                "SELECT pick_id, selection, market_type, line_point, bookmaker, "
                "offered_american, model_prob, fair_prob, ev_percent, kelly_units, hitl_status, "
                "hitl_reason, analysis FROM ledger_picks WHERE origin = ? AND result = "
                "'UNSETTLED' AND market_type LIKE 'player_%' ORDER BY ev_percent DESC",
                (mode,)).fetchall()
        finally:
            conn.close()
    except sqlite3.Error:
        rows = []
    for r in rows:
        try:
            a = json.loads(r["analysis"] or "{}")
        except json.JSONDecodeError:
            a = {}
        player, _, side = str(r["selection"]).rpartition(" ")
        proj = a.get("projection")
        view.props.append({
            "pick_id": r["pick_id"], "player": player, "side": side, "line": r["line_point"],
            "stat_label": r["market_type"].replace("player_", ""),
            "model_projection": proj,
            "delta": None if proj is None or r["line_point"] is None else proj - r["line_point"],
            "retail_book": r["bookmaker"], "offered_american": r["offered_american"],
            "model_raw_probability": r["model_prob"], "ev_percent": r["ev_percent"],
            "official_units": r["kelly_units"], "hitl_status": r["hitl_status"],
            "hitl_reason": r["hitl_reason"] or "",
            "injury_flag": "injury" in str(r["hitl_reason"] or "").lower()})
    if view.props:
        view.note = "no prop scan file for this mode; showing open prop picks from the ledger"
    else:
        view.source = "none"
    return view


def collect(scan: RunResult, mode: str, since: float, db_path: Path) -> ScanView:
    # A scan that times out never finishes writing; one that exits non-zero may
    # still have written this cycle's file (run_scan writes before the CLI
    # validates the schema), so that file is shown, flagged as unvalidated.
    timed_out = scan.returncode is None
    data, why = (None, "") if timed_out else load_json_scan(CANDIDATES_PATH, since, mode)
    if data is not None:
        opps = list(data.get("opportunities") or [])
        view = ScanView("json", mode, data.get("timestamp"),
                        dict((data.get("ingestion") or {}).get("stats") or {}), opps,
                        list(data.get("rejections") or []))
        if not scan.ok:
            view.source = "json-unvalidated"
            view.notes.append(f"pipeline exited {scan.returncode}: showing this cycle's "
                              f"{CANDIDATES_PATH.name} WITHOUT schema validation")
        cards = render_cards(opps)
        view.cards = cards if cards is not None else parse_stdout(scan.stdout, mode).cards
        return view
    view = parse_stdout(scan.stdout, mode)
    if why:
        view.notes.append(why)
    if view.opportunities or view.stats:
        return view
    fallback = load_sqlite(db_path, mode)
    fallback.notes = view.notes + fallback.notes
    if not fallback.opportunities:
        fallback.source = "none"
    return fallback


# ==========================================================================
# Display
# ==========================================================================
def _num(v: Any, fmt: str, missing: str = "n/a") -> str:
    try:
        return format(float(v), fmt)
    except (TypeError, ValueError):
        return missing


def _fit(text: str, width: int) -> str:
    text = str(text)
    return text if len(text) <= width else text[: width - 1] + "~"


def is_injury_flag(o: dict[str, Any]) -> bool:
    reason = str(o.get("hitl_reason") or "").lower()
    model = o.get("model") or {}
    return bool(model.get("key_questionable")) or any(w in reason for w in INJURY_WORDS)


def render_stats(view: ScanView, st: Style) -> str:
    s = view.stats
    if not s:
        return st.dim("  ingestion stats unavailable")
    quota = s.get("quota")
    if isinstance(quota, dict):
        quota = ", ".join(f"{k.split('_')[-1].upper()}={v}" for k, v in quota.items())
    km = f"{s.get('kalshi_matched', '?')}/{s.get('kalshi_markets', '?')}"
    line1 = (f"  events {s.get('events', '?')} | retail quotes {s.get('retail_quotes', '?')} | "
             f"Kalshi matched {km} | quota remaining {quota if quota not in (None, '') else '?'}")
    line2 = (f"  floor rejected {s.get('floor_rejected', '?')} | below 1.5% EV "
             f"{s.get('below_min_ev', '?')} | event-cap rejected "
             f"{s.get('event_cap_rejected', '?')} | point mismatch "
             f"{s.get('point_mismatch', '?')} | new ledger picks "
             f"{s.get('ledger_inserted', '?')}")
    return line1 + "\n" + st.dim(line2)


COLS = (("Pick ID", 16), ("Selection", 30), ("Book @ Odds", 18), ("Fair %", 7),
        ("EV %", 7), ("QK (u)", 6))


def render_table(opps: list[dict[str, Any]], st: Style) -> str:
    plus_ev = [o for o in opps if (o.get("ev_percent") or 0) > 0]
    plus_ev.sort(key=lambda o: float(o.get("ev_percent") or 0.0), reverse=True)
    if not plus_ev:
        return st.dim("  No +EV opportunities this cycle.")
    head = "  " + " ".join(f"{n:<{w}}" if i < 3 else f"{n:>{w}}"
                           for i, (n, w) in enumerate(COLS)) + "  HITL status / reason"
    out = [st.bold(head), "  " + "-" * 106]
    for o in plus_ev:
        odds = o.get("offered_american")
        odds_s = f"{int(odds):+d}" if isinstance(odds, (int, float)) else "?"
        book = _fit(f"{o.get('retail_book') or '?'} {odds_s}", COLS[2][1])
        fair = o.get("fair_probability")
        cells = [_fit(o.get("pick_id", "?"), COLS[0][1]), _fit(o.get("selection", "?"), COLS[1][1]),
                 book, _num(None if fair is None else float(fair) * 100, ".2f"),
                 _num(o.get("ev_percent"), ".2f"), _num(o.get("quarter_kelly_units"), ".2f")]
        row = "  " + " ".join(f"{c:<{w}}" if i < 3 else f"{c:>{w}}"
                              for i, (c, (_, w)) in enumerate(zip(cells, COLS)))
        status, reason = str(o.get("hitl_status") or "?"), str(o.get("hitl_reason") or "")
        injury = is_injury_flag(o)
        if status == PENDING:
            tag = st.yellow(f"{status}")
        elif status == "APPROVED":
            tag = st.green(status)
        else:
            tag = st.red(status)
        if injury:
            tag += " " + st.magenta("[INJURY]")
        out.append(f"{row}  {tag}")
        if reason:
            colour = st.magenta if injury else (st.yellow if status == PENDING else st.dim)
            out.append(" " * 6 + colour(f"-> {reason}"))
    return "\n".join(out)


def render_rejections(rej: list[dict[str, Any]], st: Style, limit: int) -> str:
    if not rej:
        return st.dim("  none")
    out = []
    for r in rej[:limit]:
        odds = r.get("offered_american")
        odds_s = f"{int(odds):+d}" if isinstance(odds, (int, float)) else "?"
        out.append(st.dim(f"  {_fit(r.get('selection', '?'), 34):<34} {odds_s:>5}  "
                          f"{r.get('reason', '')}"))
    if len(rej) > limit:
        out.append(st.dim(f"  ... {len(rej) - limit} more"))
    return "\n".join(out)


PROP_COLS = (("Player", 22), ("Prop Market", 11), ("Line", 7), ("Model Proj", 10),
             ("Delta", 7), ("Book & Odds", 16), ("Model Fair %", 12), ("EV %", 6),
             ("Units", 6))


def prop_units(p: dict[str, Any]) -> float | None:
    """Official stake of a prop row (``quarter_kelly_units`` for pre-flat-sizing files)."""
    v = p.get("official_units", p.get("quarter_kelly_units"))
    return None if v is None else float(v)


def render_slate_summary(view: ScanView, props: PropView | None, st: Style) -> str:
    """One line putting both pillars side by side: game picks and player props."""
    games = [o for o in view.opportunities if (o.get("ev_percent") or 0) > 0]
    g_units = sum(float(o.get("quarter_kelly_units") or 0.0) for o in games)
    line = f" OFFICIAL SLATE | game picks {len(games)} ({g_units:.2f}u)"
    if props is None:
        return st.bold(line + " | player props off (--no-props)")
    p_units = sum(prop_units(p) or 0.0 for p in props.props)
    return st.bold(line + f" | player props {len(props.props)} ({p_units:.2f}u, 1.00u each)"
                   f" | total {g_units + p_units:.2f}u")


def render_props(pv: PropView, st: Style, max_rejections: int) -> str:
    """The MISPRICED PLAYER PROPS table (sorted by shrunk EV %)."""
    out = ["", st.bold(" == MISPRICED PLAYER PROPS (MODEL vs MARKET) ==")]
    src = {"json": f"{PROPS_PATH.relative_to(ROOT)} (scan @ {pv.timestamp or '?'})",
           "json-earlier": f"{PROPS_PATH.relative_to(ROOT)} (previous scan @ "
                           f"{pv.timestamp or '?'})",
           "sqlite": "data/takeover.db ledger_picks (fallback)", "none": "no data"}[pv.source]
    out.append(f"  source: {src}")
    if pv.note:
        out.append(st.yellow(f"  note: {pv.note}"))
    s, c = pv.stats, pv.config
    if s:
        out.append(f"  slate {s.get('events_on_slate', '?')}/{s.get('events_listed', '?')} "
                   f"events | lines {s.get('player_lines', '?')} | modeled "
                   f"{s.get('players_modeled', '?')} | no model {s.get('no_model', '?')} | "
                   f"quota {s.get('quota', '?')}"
                   + (st.red("  QUOTA FLOOR HIT") if s.get("quota_guard_tripped") else ""))
    if c:
        out.append(st.dim(f"  gate: market EV > 0 AND (z >= {c.get('sigma_threshold')} "
                          f"{'OR' if c.get('divergence_mode') == 'any' else 'AND'} shrunk EV >= "
                          f"{c.get('min_shrunk_ev_percent')}%) | model weight "
                          f"{c.get('model_weight')} | pace {c.get('pace_method')} | "
                          f"lookahead {c.get('lookahead_hours', '?')}h | quota floor "
                          f"{c.get('quota_floor', '?')}"))
    props = sorted(pv.props, key=lambda p: float(p.get("ev_percent") or 0.0), reverse=True)
    if not props:
        out.append(st.dim("  No mispriced props."))
    else:
        head = "  " + " ".join(f"{n:<{w}}" if i in (0, 1, 5) else f"{n:>{w}}"
                               for i, (n, w) in enumerate(PROP_COLS)) + "  Status"
        out += [st.bold(head), "  " + "-" * 106]
        for p in props:
            odds = p.get("offered_american")
            odds_s = f"{int(odds):+d}" if isinstance(odds, (int, float)) else "?"
            side = str(p.get("side") or "?")[:1]
            line = p.get("line")
            fair = p.get("model_raw_probability")
            cells = [_fit(p.get("player", "?"), 22), _fit(p.get("stat_label", "?"), 11),
                     f"{side} {line:g}" if isinstance(line, (int, float)) else "?",
                     _num(p.get("model_projection"), ".2f"), _num(p.get("delta"), "+.2f"),
                     _fit(f"{p.get('retail_book') or '?'} {odds_s}", 16),
                     _num(None if fair is None else float(fair) * 100, ".1f"),
                     _num(p.get("ev_percent"), ".2f"), _num(prop_units(p), ".2f")]
            row = "  " + " ".join(f"{v:<{w}}" if i in (0, 1, 5) else f"{v:>{w}}"
                                  for i, (v, (_, w)) in enumerate(zip(cells, PROP_COLS)))
            status = str(p.get("hitl_status") or "?")
            tag = st.yellow("PENDING") if status == PENDING else st.green(status)
            if p.get("injury_flag") or is_injury_flag(p):
                tag += " " + st.magenta("[INJURY]")
            out.append(f"{row}  {tag}")
            extra = [t.split(">=")[0].strip() for t in p.get("triggers") or []]
            reason = str(p.get("hitl_reason") or "")
            injuries = [r for r in reason.split("; ") if r.lower().startswith("injury")]
            raw_k = p.get("raw_kelly_units")
            detail = (f"triggers {', '.join(extra)}" if extra else "") + \
                (f" | raw Kelly {float(raw_k):.2f}u" if raw_k is not None else "") + \
                (f" | {'; '.join(injuries)}" if injuries else "")
            if detail:
                colour = st.magenta if injuries else st.dim
                out.append(" " * 6 + colour(f"-> {detail.strip(' |')}"))
    rej = pv.rejections
    if pv.source.startswith("json"):
        out += ["", st.bold(f"  PROP REJECTIONS ({len(rej)})"),
                render_rejections(rej, st, max_rejections)]
    return "\n".join(out)


def render_cycle(cycle: int, mode: str, runs: list[RunResult], view: ScanView, st: Style,
                 show_cards: bool, max_rejections: int, clear: bool,
                 props: PropView | None = None) -> str:
    now = datetime.now().astimezone()
    bar = "=" * 108
    parts = []
    if clear:
        parts.append("\033[2J\033[H" if st.on else "")
    banner = st.red("LIVE DATA - Captain HZ sign-off required before any bet") if mode == "LIVE" \
        else st.cyan("MOCK - synthetic data, do not bet")
    parts += [st.bold(bar),
              st.bold(f" BREEZE MONITOR | cycle {cycle} | {mode} | {now:%Y-%m-%d %H:%M:%S %Z}")
              + "  " + banner,
              st.bold(bar)]
    for r in runs:
        if r.ok:
            parts.append(st.green(f"  [{r.label}] ok in {r.seconds:.1f}s"))
        else:
            err = r.error or f"exit code {r.returncode}"
            tail = [l for l in (r.stderr or r.stdout).strip().splitlines()[-3:] if l.strip()]
            parts.append(st.red(f"  [{r.label}] FAILED ({err}) after {r.seconds:.1f}s"))
            parts += [st.dim(f"      {l}") for l in tail]
    src = {"json": f"{CANDIDATES_PATH.relative_to(ROOT)} (scan @ {view.timestamp or '?'})",
           "json-unvalidated": f"{CANDIDATES_PATH.relative_to(ROOT)} (scan @ "
                               f"{view.timestamp or '?'}) - SCHEMA VALIDATION FAILED",
           "stdout": "pipeline stdout (JSON unavailable)",
           "sqlite": "data/takeover.db ledger_picks (fallback)",
           "none": "no data"}[view.source]
    parts.append(st.red(f"  source: {src}") if view.source == "json-unvalidated"
                 else f"  source: {src}")
    parts += [st.yellow(f"  note: {n}") for n in view.notes]

    n_pos = sum(1 for o in view.opportunities if (o.get("ev_percent") or 0) > 0)
    n_pend = sum(1 for o in view.opportunities if o.get("hitl_status") == PENDING)
    n_inj = sum(1 for o in view.opportunities if is_injury_flag(o))
    parts += ["", render_slate_summary(view, props, st),
              "", st.bold(" INGESTION"), render_stats(view, st), "",
              st.bold(f" GAME PICKS - +EV OPPORTUNITIES ({n_pos})  ")
              + (st.yellow(f"{n_pend} pending Captain HZ") if n_pend else "")
              + (("  " + st.magenta(f"{n_inj} injury-flagged")) if n_inj else ""),
              render_table(view.opportunities, st)]
    if view.source in ("json", "json-unvalidated", "stdout"):
        parts += ["", st.bold(f" REJECTIONS ({len(view.rejections)})"),
                  render_rejections(view.rejections, st, max_rejections)]
    if props is not None:
        parts.append(render_props(props, st, max_rejections))
        n_pend += sum(1 for p in props.props if p.get("hitl_status") == PENDING)
    if show_cards and view.cards:
        count = len(view.cards) if view.source.startswith("json") else "from stdout"
        parts += ["", st.bold(f" GAME ANALYSIS CARDS ({count})")]
        parts += view.cards
    if n_pend:
        parts += ["", st.dim("  Approve a pending pick: python run_pipeline.py --approve PICK_ID")]
    return "\n".join(p for p in parts if p is not None)


# ==========================================================================
# Loop
# ==========================================================================
def scan_args(live: bool, props: bool = True, sport: str = "all") -> list[str]:
    """run_pipeline.py scans props by default; only the skip needs a flag."""
    base = ["--scan" if live else "--mock", "--sport", sport]
    return base + ([] if props else ["--no-props"])


def stats_args(props: bool = True) -> list[str]:
    return ["--update-stats"] + ([] if props else ["--no-props"])


def should_scan_props(cycle: int, enabled: bool, every: int) -> bool:
    """Cycle 1 and every ``every`` cycles after it, unless --no-props."""
    return enabled and every > 0 and (cycle - 1) % every == 0


def should_refresh_stats(cycle: int, every: int) -> bool:
    """Cycle 1 and every ``every`` cycles after it; 0 disables."""
    return every > 0 and (cycle - 1) % every == 0


def run_cycle(cycle: int, a: argparse.Namespace, st: Style) -> None:
    mode = "LIVE" if a.live else "MOCK"
    runs: list[RunResult] = []
    if should_refresh_stats(cycle, a.refresh_stats_every):
        sargs = stats_args(a.props)
        print(st.dim(f"[cycle {cycle}] refreshing stats ({' '.join(sargs)}) ..."), flush=True)
        runs.append(run_pipeline(sargs, a.timeout, "update-stats"))
    props_now = should_scan_props(cycle, a.props, a.props_every)
    print(st.dim(f"[cycle {cycle}] running {mode} scan{' + props' if props_now else ''} ..."),
          flush=True)
    since = time.time() - 1.0                       # 1s slack for filesystem mtime resolution
    scan = run_pipeline(scan_args(a.live, props_now, a.sport), a.timeout,
                        "scan + props" if props_now else "scan")
    runs.append(scan)
    view = collect(scan, mode, since, DB_PATH)
    pview = collect_props(props_now, scan.ok, mode, since, DB_PATH) if a.props else None
    print(render_cycle(cycle, mode, runs, view, st, not a.no_cards, a.max_rejections, a.clear,
                       pview), flush=True)


def sleep_until(deadline: float, st: Style) -> None:
    remaining = deadline - time.monotonic()
    if remaining > 0:
        print(st.dim(f"\nNext cycle in {remaining:.0f}s (Ctrl+C to stop)"), flush=True)
        time.sleep(remaining)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--live", action="store_true", help="live scan: --scan --sport all")
    mode.add_argument("--mock", action="store_true", help="synthetic scan (default)")
    ap.add_argument("--interval", type=float, default=60.0,
                    help="seconds between cycle starts (default 60)")
    ap.add_argument("--refresh-stats-every", type=int, default=5, metavar="N",
                    help="run --update-stats before the scan every N cycles, starting with "
                         "the first (default 5; 0 = never)")
    ap.add_argument("--sport", default="all",
                    help="passed to run_pipeline.py: all (default), basketball_nba, "
                         "americanfootball_nfl, or a comma list; filters games and props")
    ap.add_argument("--no-props", action="store_true",
                    help="skip the player-prop scan, prop stats refresh and the MISPRICED "
                         "PLAYER PROPS table (props run by default)")
    ap.add_argument("--props", action="store_true", help=argparse.SUPPRESS)  # legacy no-op
    ap.add_argument("--props-every", type=int, default=1, metavar="N",
                    help="scan props every N cycles, starting with the first (default 1). "
                         "Live prop odds are also cached 15 min per game.")
    ap.add_argument("--timeout", type=float, default=300.0,
                    help="per-subprocess timeout in seconds (default 300)")
    ap.add_argument("--once", action="store_true", help="run one cycle and exit")
    ap.add_argument("--no-cards", action="store_true", help="hide game analysis cards")
    ap.add_argument("--max-rejections", type=int, default=15,
                    help="rejection rows to show (default 15)")
    ap.add_argument("--clear", action="store_true", help="clear the screen each cycle")
    ap.add_argument("--no-color", action="store_true", help="disable ANSI colours")
    a = ap.parse_args(argv)
    a.props = not a.no_props
    if a.interval < 0 or a.timeout <= 0 or a.refresh_stats_every < 0 or a.props_every < 1:
        ap.error("--interval must be >= 0, --timeout > 0, --refresh-stats-every >= 0, "
                 "--props-every >= 1")
    return a


def main(argv: list[str] | None = None) -> int:
    _configure_stdout()
    a = parse_args(argv)
    colour = not a.no_color and "NO_COLOR" not in os.environ and sys.stdout.isatty()
    st = Style(colour)
    if not PIPELINE.exists():
        print(f"ERROR: {PIPELINE} not found", file=sys.stderr)
        return 1
    print(st.bold(f"Breeze monitor | {'LIVE' if a.live else 'MOCK'} | interval {a.interval:.0f}s"
                  f" | props {'every ' + str(a.props_every) + ' cycle(s)' if a.props else 'off'}"
                  f" | stats every {a.refresh_stats_every or 'never'} cycles | "
                  f"python {sys.executable}"))
    cycle = 0
    try:
        while True:
            cycle += 1
            started = time.monotonic()
            try:
                run_cycle(cycle, a, st)
            except KeyboardInterrupt:
                raise
            except Exception as exc:                 # never let one bad cycle kill the loop
                print(st.red(f"[cycle {cycle}] monitor error: {type(exc).__name__}: {exc}"),
                      flush=True)
            if a.once:
                return 0
            sleep_until(started + a.interval, st)
    except KeyboardInterrupt:
        print(st.dim(f"\nStopped after {cycle} cycle(s)."), flush=True)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
