"""SQLite (WAL) persistence: API cache, market snapshots, pick ledger.

One file, ``data/takeover.db``, opened per command and closed on exit; no
daemon, no ORM. WAL mode lets a reader (e.g. the backtester) run while a scan
writes. Timestamps are stored as ISO-8601 UTC strings, which sort correctly.

Tables:
    api_cache         response cache keyed by request; protects API quota
    market_snapshots  every sharp/retail/exchange price seen, for line movement
                      and closing-line capture
    ledger_picks      every recommended single, graded in place
    rule_violations   picks quarantined by the backtester (mandate breaches)
"""

from __future__ import annotations

import json
import re
import sqlite3
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[2]
DB_PATH = ROOT / "data" / "takeover.db"

ORIGIN_LIVE = "LIVE"
ORIGIN_MOCK = "MOCK"
ORIGIN_SEED = "SYNTHETIC_SEED"

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS api_cache (
    cache_key   TEXT PRIMARY KEY,
    payload     TEXT NOT NULL,
    expires_at  TIMESTAMP NOT NULL
);

CREATE TABLE IF NOT EXISTS market_snapshots (
    id            INTEGER PRIMARY KEY,
    event_id      TEXT NOT NULL,
    timestamp     TIMESTAMP NOT NULL,
    source        TEXT NOT NULL,
    market_type   TEXT NOT NULL,
    selection     TEXT NOT NULL,
    point         REAL,
    american_odds INTEGER,
    implied_prob  REAL
);
CREATE INDEX IF NOT EXISTS ix_snap_lookup
    ON market_snapshots (event_id, market_type, source, timestamp);
CREATE UNIQUE INDEX IF NOT EXISTS ux_snap_dedupe
    ON market_snapshots (event_id, timestamp, source, market_type, selection, IFNULL(point, -9999));

CREATE TABLE IF NOT EXISTS ledger_picks (
    pick_id            TEXT PRIMARY KEY,
    created_at         TIMESTAMP NOT NULL,
    commence_time      TIMESTAMP,
    sport              TEXT,
    event_id           TEXT NOT NULL,
    event_name         TEXT,
    market_type        TEXT NOT NULL
        CHECK (market_type IN ('h2h', 'spreads', 'totals', 'kalshi_binary')),
    selection          TEXT NOT NULL,
    line_point         REAL,
    bookmaker          TEXT,
    offered_american   INTEGER NOT NULL,
    offered_decimal    REAL NOT NULL,
    benchmark_source   TEXT,
    benchmark_american INTEGER,
    fair_prob          REAL,
    kalshi_mid_prob    REAL,
    ev_percent         REAL,
    kelly_units        REAL,
    closing_american   INTEGER,
    closing_fair_prob  REAL,
    clv_percent        REAL,
    hitl_status        TEXT NOT NULL DEFAULT 'APPROVED'
        CHECK (hitl_status IN ('APPROVED', 'PENDING_CAPTAIN_HZ', 'REJECTED')),
    hitl_reason        TEXT,
    result             TEXT NOT NULL DEFAULT 'UNSETTLED'
        CHECK (result IN ('UNSETTLED', 'WIN', 'LOSS', 'PUSH', 'CANCELLED', 'RULE_VIOLATION')),
    net_units          REAL,
    origin             TEXT NOT NULL DEFAULT 'LIVE',
    settled_at         TIMESTAMP
);
CREATE INDEX IF NOT EXISTS ix_picks_event ON ledger_picks (event_id, result);
CREATE INDEX IF NOT EXISTS ix_picks_result ON ledger_picks (result, commence_time);

CREATE TABLE IF NOT EXISTS teams_metadata (
    canonical_id      TEXT PRIMARY KEY,
    sport             TEXT NOT NULL,
    full_name         TEXT NOT NULL,
    kalshi_code       TEXT NOT NULL,
    espn_id           TEXT NOT NULL,
    espn_abbr         TEXT NOT NULL,
    arena_altitude_ft INTEGER,
    high_altitude     INTEGER NOT NULL DEFAULT 0,
    timezone          TEXT,
    utc_offset_std    INTEGER
);

CREATE TABLE IF NOT EXISTS team_ratings (
    canonical_id      TEXT PRIMARY KEY REFERENCES teams_metadata (canonical_id),
    sport             TEXT NOT NULL,
    season            TEXT,
    updated_at        TIMESTAMP NOT NULL,
    source            TEXT NOT NULL,
    games             INTEGER,
    wins              INTEGER,
    losses            INTEGER,
    home_record       TEXT,
    away_record       TEXT,
    points_for_pg     REAL,
    points_against_pg REAL,
    point_diff_pg     REAL,
    pythag_win_pct    REAL,
    srs               REAL,          -- opponent- and venue-adjusted margin, pts/game
    ortg              REAL,          -- NBA: points per 100 possessions
    drtg              REAL,
    net_rtg           REAL,          -- NBA: raw ORTG - DRTG
    adj_net_rtg       REAL,          -- NBA: SRS per 100 possessions (opponent-adjusted)
    pace              REAL,          -- NBA: possessions per game
    efg_pct           REAL,
    yards_per_play    REAL,          -- NFL
    points_per_drive  REAL,          -- NFL (needs drive data; NULL from ESPN feed)
    turnover_margin   REAL,          -- NFL, season total
    epa_per_play      REAL           -- NFL (optional nfl_data_py hook)
);

CREATE TABLE IF NOT EXISTS team_schedule (
    canonical_id    TEXT NOT NULL,
    espn_event_id   TEXT NOT NULL,
    game_time       TIMESTAMP NOT NULL,
    opponent_id     TEXT,
    is_home         INTEGER NOT NULL,
    completed       INTEGER NOT NULL,
    points_for      REAL,
    points_against  REAL,
    PRIMARY KEY (canonical_id, espn_event_id)
);
CREATE INDEX IF NOT EXISTS ix_sched_time ON team_schedule (canonical_id, game_time);

CREATE TABLE IF NOT EXISTS injury_reports (
    id            INTEGER PRIMARY KEY,
    canonical_id  TEXT NOT NULL,
    sport         TEXT NOT NULL,
    player        TEXT NOT NULL,
    position      TEXT,
    status        TEXT NOT NULL,     -- Out / Doubtful / Questionable / Day-To-Day ...
    is_key        INTEGER NOT NULL DEFAULT 0,
    key_reason    TEXT,
    detail        TEXT,
    reported_at   TIMESTAMP,
    fetched_at    TIMESTAMP NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_inj_team ON injury_reports (canonical_id, status);

CREATE TABLE IF NOT EXISTS rule_violations (
    pick_id          TEXT PRIMARY KEY REFERENCES ledger_picks (pick_id),
    quarantined_at   TIMESTAMP NOT NULL,
    offered_american INTEGER,
    reason           TEXT NOT NULL,
    original_result  TEXT,
    original_net_units REAL,
    snapshot         TEXT NOT NULL
);
"""

PICK_COLUMNS: tuple[str, ...] = (
    "pick_id", "created_at", "commence_time", "sport", "event_id", "event_name",
    "market_type", "selection", "line_point", "bookmaker", "offered_american",
    "offered_decimal", "benchmark_source", "benchmark_american", "fair_prob",
    "kalshi_mid_prob", "ev_percent", "kelly_units", "closing_american",
    "closing_fair_prob", "clv_percent", "hitl_status", "hitl_reason", "result",
    "net_units", "origin", "settled_at", "model_prob", "analysis",
    "payout_units", "signoff_by", "signoff_at", "signoff_hash", "dispatched_at", "dispatch_ref",
)

# Columns added after the first release: (table, column, DDL type). Applied by connect().
MIGRATIONS: tuple[tuple[str, str, str], ...] = (
    ("ledger_picks", "model_prob", "REAL"),
    ("ledger_picks", "analysis", "TEXT"),
    ("ledger_picks", "payout_units", "REAL"),          # gross return credited at settlement
    ("ledger_picks", "signoff_by", "TEXT"),            # explicit Captain HZ sign-off
    ("ledger_picks", "signoff_at", "TIMESTAMP"),
    ("ledger_picks", "signoff_hash", "TEXT"),          # binds sign-off to the exact terms
    ("ledger_picks", "dispatched_at", "TIMESTAMP"),
    ("ledger_picks", "dispatch_ref", "TEXT"),
)


def _ledger_ddl() -> str:
    m = re.search(r"CREATE TABLE IF NOT EXISTS ledger_picks \(.*?\n\);", SCHEMA_SQL, re.S)
    assert m is not None
    return m.group(0)


def _rebuild_ledger_if_stale(conn: sqlite3.Connection) -> bool:
    """Rebuild ledger_picks when its result CHECK predates 'CANCELLED'.

    SQLite cannot alter a CHECK constraint, so this follows the documented
    create-copy-drop-rename procedure in one transaction (foreign keys off so
    rule_violations keeps pointing at ledger_picks). Every row and column is kept.
    """
    row = conn.execute("SELECT sql FROM sqlite_master WHERE type = 'table' "
                       "AND name = 'ledger_picks'").fetchone()
    if row is None or "'CANCELLED'" in row[0]:
        return False
    old_cols = [r["name"] for r in conn.execute("PRAGMA table_info(ledger_picks)")]
    ddl = _ledger_ddl().replace("CREATE TABLE IF NOT EXISTS ledger_picks",
                                "CREATE TABLE ledger_picks__new")
    extra = [f"ALTER TABLE ledger_picks__new ADD COLUMN {c} {d};"
             for t, c, d in MIGRATIONS if t == "ledger_picks" and c in old_cols]
    new_cols = re.findall(r"^\s{4}(\w+)\s", ddl, re.M) + \
        [c for t, c, _ in MIGRATIONS if t == "ledger_picks" and c in old_cols]
    common = ", ".join(c for c in old_cols if c in new_cols)
    conn.commit()
    conn.execute("PRAGMA foreign_keys=OFF;")
    try:
        conn.executescript(
            "BEGIN;\nDROP TABLE IF EXISTS ledger_picks__new;\n" + ddl + "\n" + "\n".join(extra)
            + f"\nINSERT INTO ledger_picks__new ({common}) SELECT {common} FROM ledger_picks;"
            + "\nDROP TABLE ledger_picks;\nALTER TABLE ledger_picks__new RENAME TO ledger_picks;"
            + "\nCOMMIT;")
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.execute("PRAGMA foreign_keys=ON;")
    return True


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def iso(dt: datetime) -> str:
    """UTC ISO-8601 with 'Z', second precision."""
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_ts(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def connect(path: Path = DB_PATH) -> sqlite3.Connection:
    """Open the database in WAL mode and ensure the schema exists."""
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, timeout=10.0)
    conn.row_factory = sqlite3.Row
    mode = conn.execute("PRAGMA journal_mode=WAL;").fetchone()[0]
    if str(mode).lower() != "wal":
        print(f"[db] warning: journal_mode is {mode}, expected wal", file=sys.stderr)
    conn.execute("PRAGMA synchronous=NORMAL;")
    conn.execute("PRAGMA foreign_keys=ON;")
    conn.executescript(SCHEMA_SQL)
    if _rebuild_ledger_if_stale(conn):
        conn.executescript(SCHEMA_SQL)                  # recreate the dropped indexes
    for table, column, ddl in MIGRATIONS:
        cols = {r["name"] for r in conn.execute(f"PRAGMA table_info({table})")}
        if column not in cols:
            with conn:
                conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {ddl}")
    return conn


# --------------------------------------------------------------------------
# api_cache
# --------------------------------------------------------------------------
def cache_get(conn: sqlite3.Connection, key: str) -> Any | None:
    """Return a cached JSON payload if present and unexpired."""
    row = conn.execute("SELECT payload, expires_at FROM api_cache WHERE cache_key = ?",
                       (key,)).fetchone()
    if row is None or parse_ts(row["expires_at"]) <= utcnow():
        return None
    return json.loads(row["payload"])


def cache_put(conn: sqlite3.Connection, key: str, payload: Any, ttl_s: int) -> None:
    expires = iso(utcnow() + timedelta(seconds=ttl_s))
    with conn:
        conn.execute("INSERT OR REPLACE INTO api_cache VALUES (?, ?, ?)",
                     (key, json.dumps(payload, separators=(",", ":")), expires))
        conn.execute("DELETE FROM api_cache WHERE expires_at < ?", (iso(utcnow()),))


# --------------------------------------------------------------------------
# market_snapshots
# --------------------------------------------------------------------------
def insert_snapshots(conn: sqlite3.Connection, rows: Iterable[dict[str, Any]]) -> int:
    """Insert snapshot rows; exact duplicates (same instant/source/selection) are ignored."""
    data = [(r["event_id"], r["timestamp"], r["source"], r["market_type"], r["selection"],
             r.get("point"), r.get("american_odds"), r.get("implied_prob")) for r in rows]
    with conn:
        before = conn.total_changes
        conn.executemany(
            "INSERT OR IGNORE INTO market_snapshots (event_id, timestamp, source, market_type,"
            " selection, point, american_odds, implied_prob) VALUES (?, ?, ?, ?, ?, ?, ?, ?)", data)
    return conn.total_changes - before


def earliest_point(conn: sqlite3.Connection, event_id: str, market_type: str,
                   selection: str, sources: tuple[str, ...]) -> float | None:
    """First recorded line point for a selection at any of ``sources``."""
    marks = ",".join("?" * len(sources))
    row = conn.execute(
        f"SELECT point FROM market_snapshots WHERE event_id = ? AND market_type = ? "
        f"AND selection = ? AND source IN ({marks}) AND point IS NOT NULL "
        f"ORDER BY timestamp ASC, id ASC LIMIT 1",
        (event_id, market_type, selection, *sources)).fetchone()
    return None if row is None else row["point"]


def latest_snapshot_pair(conn: sqlite3.Connection, event_id: str, market_type: str,
                         sources: tuple[str, ...], before: str
                         ) -> tuple[str, list[sqlite3.Row]] | None:
    """Rows from the most recent sharp snapshot taken at or before ``before``.

    Returns (source, rows) where rows all share that snapshot timestamp.
    """
    marks = ",".join("?" * len(sources))
    row = conn.execute(
        f"SELECT source, timestamp FROM market_snapshots WHERE event_id = ? AND market_type = ?"
        f" AND source IN ({marks}) AND timestamp <= ? ORDER BY timestamp DESC, id DESC LIMIT 1",
        (event_id, market_type, *sources, before)).fetchone()
    if row is None:
        return None
    rows = conn.execute(
        "SELECT * FROM market_snapshots WHERE event_id = ? AND market_type = ? AND source = ?"
        " AND timestamp = ?", (event_id, market_type, row["source"], row["timestamp"])).fetchall()
    return row["source"], rows


# --------------------------------------------------------------------------
# ledger_picks
# --------------------------------------------------------------------------
def insert_pick(conn: sqlite3.Connection, pick: dict[str, Any]) -> bool:
    """Insert a pick; returns False if ``pick_id`` already exists (first call stands)."""
    cols = [c for c in PICK_COLUMNS if c in pick]
    sql = (f"INSERT OR IGNORE INTO ledger_picks ({', '.join(cols)}) "
           f"VALUES ({', '.join('?' * len(cols))})")
    with conn:
        cur = conn.execute(sql, [pick[c] for c in cols])
    return cur.rowcount == 1


def insert_picks(conn: sqlite3.Connection, picks: Iterable[dict[str, Any]]) -> int:
    return sum(insert_pick(conn, p) for p in picks)


def update_pick(conn: sqlite3.Connection, pick_id: str, **fields: Any) -> None:
    bad = set(fields) - set(PICK_COLUMNS)
    if bad:
        raise KeyError(f"unknown ledger columns: {bad}")
    sets = ", ".join(f"{k} = ?" for k in fields)
    with conn:
        conn.execute(f"UPDATE ledger_picks SET {sets} WHERE pick_id = ?",
                     [*fields.values(), pick_id])


def get_pick(conn: sqlite3.Connection, pick_id: str) -> dict[str, Any] | None:
    row = conn.execute("SELECT * FROM ledger_picks WHERE pick_id = ?", (pick_id,)).fetchone()
    return dict(row) if row else None


def fetch_picks(conn: sqlite3.Connection, where: str = "1=1",
                params: Iterable[Any] = ()) -> list[dict[str, Any]]:
    rows = conn.execute(f"SELECT * FROM ledger_picks WHERE {where} "
                        f"ORDER BY commence_time, created_at, pick_id", tuple(params)).fetchall()
    return [dict(r) for r in rows]


def open_event_exposure(conn: sqlite3.Connection, event_id: str,
                        exclude: Iterable[str] = ()) -> float:
    """Units already committed (unsettled, approved or pending) on an event."""
    excl = tuple(exclude)
    marks = ",".join("?" * len(excl)) or "''"
    row = conn.execute(
        f"SELECT COALESCE(SUM(kelly_units), 0) FROM ledger_picks WHERE event_id = ? "
        f"AND result = 'UNSETTLED' AND hitl_status IN ('APPROVED', 'PENDING_CAPTAIN_HZ') "
        f"AND pick_id NOT IN ({marks})", (event_id, *excl)).fetchone()
    return float(row[0])


def count_picks(conn: sqlite3.Connection, origins: tuple[str, ...] | None = None) -> int:
    if not origins:
        return int(conn.execute("SELECT COUNT(*) FROM ledger_picks").fetchone()[0])
    marks = ",".join("?" * len(origins))
    return int(conn.execute(f"SELECT COUNT(*) FROM ledger_picks WHERE origin IN ({marks})",
                            origins).fetchone()[0])


# --------------------------------------------------------------------------
# rule_violations
# --------------------------------------------------------------------------
def quarantine_pick(conn: sqlite3.Connection, pick: dict[str, Any], reason: str) -> bool:
    """Copy a pick into rule_violations and mark it RULE_VIOLATION / REJECTED.

    Idempotent: returns False if the pick is already quarantined.
    """
    with conn:
        cur = conn.execute(
            "INSERT OR IGNORE INTO rule_violations VALUES (?, ?, ?, ?, ?, ?, ?)",
            (pick["pick_id"], iso(utcnow()), pick.get("offered_american"), reason,
             pick.get("result"), pick.get("net_units"), json.dumps(pick)))
        if cur.rowcount != 1:
            return False
        conn.execute("UPDATE ledger_picks SET result = 'RULE_VIOLATION', hitl_status = 'REJECTED',"
                     " hitl_reason = ?, net_units = NULL WHERE pick_id = ?",
                     (reason, pick["pick_id"]))
    return True


def fetch_violations(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    return [dict(r) for r in conn.execute(
        "SELECT * FROM rule_violations ORDER BY quarantined_at, pick_id").fetchall()]


def table_counts(conn: sqlite3.Connection) -> dict[str, int]:
    return {t: int(conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0])
            for t in ("api_cache", "market_snapshots", "ledger_picks", "rule_violations",
                      "teams_metadata", "team_ratings", "team_schedule", "injury_reports")}
