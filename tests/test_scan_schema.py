"""The scan JSON the pipeline writes must validate against schemas/market_scan_schema.json."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.core import database as db  # noqa: E402
from src.core.schema_validator import load_schema, validate  # noqa: E402
from src.data.market_fetcher import run_scan  # noqa: E402


def test_mock_scan_matches_schema(tmp_path):
    conn = db.connect(tmp_path / "takeover.db")
    try:
        scan = run_scan(conn, ("basketball_nba", "americanfootball_nfl"), mock=True,
                        output_path=tmp_path / "scan.json")
    finally:
        conn.close()
    assert scan["opportunities"], "mock slate should produce opportunities"
    errors, _ = validate(scan, load_schema(ROOT / "schemas" / "market_scan_schema.json"))
    assert errors == []
