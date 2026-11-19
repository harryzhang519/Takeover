"""Free / premium tiers for Captain HZ's signed-off picks.

Every signed-off pick goes to exactly one Discord audience:

    FREE     the public free plays (the default: today's Discord is free plays)
    PREMIUM  the paid membership card

The tier is chosen at sign-off and stored on the ledger row (``tier``, with
``tier_set_at``). It can be switched while the pick is open, but it locks once
the game starts. The public free and premium records are only honest if a pick
can't be moved between them after the result is in sight.

Undo clears the tier along with the sign-off. Pending, declined and rejected
picks have no tier.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime
from typing import Any

from src.core import database as db
from src.core.engine import HITL_APPROVED

TIER_FREE = "FREE"
TIER_PREMIUM = "PREMIUM"
TIERS: tuple[str, ...] = (TIER_FREE, TIER_PREMIUM)
DEFAULT_TIER = TIER_FREE
TIER_LABELS = {TIER_FREE: "Free play", TIER_PREMIUM: "Premium"}


def normalize_tier(value: Any) -> str:
    """'free' / 'Premium' / None -> 'FREE' / 'PREMIUM' / the default. Anything else is refused."""
    if value is None or value == "":
        return DEFAULT_TIER
    tier = str(value).strip().upper()
    if tier not in TIERS:
        raise ValueError(f"unknown tier {value!r}; use free or premium")
    return tier


def is_official(p: dict[str, Any]) -> bool:
    return p.get("hitl_status") == HITL_APPROVED and bool(p.get("signoff_at"))


def effective_tier(p: dict[str, Any]) -> str | None:
    """The audience of a signed-off pick (a pre-tier sign-off counts as FREE); None otherwise."""
    if not is_official(p):
        return None
    return p.get("tier") if p.get("tier") in TIERS else DEFAULT_TIER


def tier_locked_reason(p: dict[str, Any], now: datetime | None = None) -> str | None:
    """Why the tier of ``p`` can no longer change, or None while it still can."""
    now = now or db.utcnow()
    pid = p.get("pick_id")
    if not is_official(p):
        return f"{pid} is not signed off; approve it first"
    if p.get("result") != "UNSETTLED":
        return f"{pid} is already settled; its tier is part of the record"
    if p.get("dispatched_at"):
        return f"{pid} was already dispatched to a book"
    start = p.get("commence_time")
    if start and db.parse_ts(start) <= now:
        return f"{pid} has started; the tier locks at game time"
    return None


def set_tier(conn: sqlite3.Connection, pick_id: str, tier: Any,
             now: datetime | None = None) -> dict[str, Any]:
    """Move an open, signed-off pick between the free and premium cards."""
    p = db.get_pick(conn, pick_id)
    if p is None:
        raise KeyError(f"unknown pick {pick_id}")
    tier = normalize_tier(tier)
    if p.get("tier") == tier:
        return p
    reason = tier_locked_reason(p, now)
    if reason:
        raise ValueError(reason)
    db.update_pick(conn, pick_id, tier=tier, tier_set_at=db.iso(now or db.utcnow()))
    return db.get_pick(conn, pick_id)  # type: ignore[return-value]
