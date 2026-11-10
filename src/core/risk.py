"""Kelly sizing under parameter uncertainty and the desk-mandate gate.

After Baker, R. D. & McHale, I. G. (2013), "Optimal betting under parameter
uncertainty: improving the Kelly criterion", Decision Analysis.

Raw Kelly plugs an ESTIMATED win probability p into a formula derived for the
true one, which overstates out-of-sample growth. If p carries estimation
variance sigma^2, the growth-optimal stake is the Kelly stake shrunk by

    k* = s*(p)^2 / (s*(p)^2 + ((b+1)/b)^2 sigma^2)
       = ((b+1)p - 1)^2 / (((b+1)p - 1)^2 + (b+1)^2 sigma^2)

where b is the fractional odds (decimal - 1) and s*(p) = ((b+1)p - 1)/b is the
raw Kelly fraction. The shrunk stake is the closed form

    s = ((b+1)p - 1)^3 / (b [((b+1)p - 1)^2 + (b+1)^2 sigma^2]).

Desk mandate: quarter-Kelly (k = 0.25) is the CEILING; it equals the
Baker-McHale shrinkage when sigma = :func:`implied_estimation_sigma`. The
stake is the smaller of quarter-Kelly and Baker-McHale, then hard-gated:
odds floor -130 or longer, straight singles only, max 2.0u per bet and max
2.0u aggregate per unique event_id on a fixed baseline bankroll.
"""

from __future__ import annotations

import math

from src.core.engine import (
    DEFAULT_BANKROLL_UNITS,
    HITL_APPROVED,
    HITL_PENDING,
    HITL_REJECTED,
    KELLY_FRACTION,
    MAX_EVENT_FRACTION,
    MAX_STAKE_FRACTION,
    MIN_DECIMAL_ODDS,
)


def _validate(p: float, b: float, sigma: float) -> None:
    if not 0.0 < p < 1.0:
        raise ValueError(f"p must be in (0, 1), got {p}")
    if not b > 0.0:
        raise ValueError(f"fractional odds b must be > 0, got {b}")
    if sigma < 0.0:
        raise ValueError(f"sigma must be >= 0, got {sigma}")


def raw_kelly_fraction(p: float, b: float) -> float:
    """s*(p) = ((b+1)p - 1) / b; may be negative (no bet)."""
    _validate(p, b, 0.0)
    return ((b + 1.0) * p - 1.0) / b


def calculate_baker_mchale_shrinkage(p: float, b: float, sigma: float) -> float:
    """Optimal shrinkage k* in [0, 1] for estimated p with standard error ``sigma``.

    ``sigma`` is the standard deviation of the estimate of p (sigma^2 is its
    estimation variance). Returns 0 when the edge is non-positive and 1 when
    sigma = 0 (p known exactly -> full Kelly).
    """
    _validate(p, b, sigma)
    edge = (b + 1.0) * p - 1.0
    if edge <= 0.0:
        return 0.0
    e2 = edge * edge
    return e2 / (e2 + (b + 1.0) ** 2 * sigma * sigma)


def baker_mchale_stake_fraction(p: float, b: float, sigma: float) -> float:
    """Shrunk bankroll fraction ((b+1)p-1)^3 / (b[((b+1)p-1)^2 + (b+1)^2 sigma^2]).

    Identical to ``calculate_baker_mchale_shrinkage * raw_kelly_fraction``;
    0 when the edge is non-positive.
    """
    _validate(p, b, sigma)
    edge = (b + 1.0) * p - 1.0
    if edge <= 0.0:
        return 0.0
    return edge ** 3 / (b * (edge * edge + (b + 1.0) ** 2 * sigma * sigma))


def implied_estimation_sigma(p: float, b: float,
                             kelly_fraction: float = KELLY_FRACTION) -> float:
    """sigma at which Baker-McHale shrinkage equals ``kelly_fraction``.

    From k = e^2 / (e^2 + (b+1)^2 sigma^2):  sigma = |e| sqrt(1/k - 1) / (b+1),
    e = (b+1)p - 1. Quarter-Kelly is optimal for a model whose probability
    standard error is this large.
    """
    _validate(p, b, 0.0)
    if not 0.0 < kelly_fraction <= 1.0:
        raise ValueError(f"kelly_fraction must be in (0, 1], got {kelly_fraction}")
    edge = (b + 1.0) * p - 1.0
    return abs(edge) * math.sqrt(1.0 / kelly_fraction - 1.0) / (b + 1.0)


def size_bet_under_uncertainty(p: float, offered_dec: float, sigma: float,
                               existing_event_units: float = 0.0, legs: int = 1,
                               bankroll_units: float = DEFAULT_BANKROLL_UNITS) -> dict:
    """Desk-mandate stake: min(quarter-Kelly, Baker-McHale), hard-capped.

    Returns a dict with ``units``, ``status`` (APPROVED / PENDING_CAPTAIN_HZ /
    REJECTED), ``reason`` and the intermediate fractions. ``bankroll_units``
    must be the fixed baseline, never scaled by recent P&L.
    """
    if not offered_dec > 1.0:
        raise ValueError(f"Decimal odds must be > 1.0, got {offered_dec}")
    b = offered_dec - 1.0
    _validate(p, b, sigma)
    out = {"units": 0.0, "status": HITL_REJECTED, "reason": "",
           "raw_kelly": raw_kelly_fraction(p, b),
           "shrinkage": calculate_baker_mchale_shrinkage(p, b, sigma),
           "baker_mchale_fraction": baker_mchale_stake_fraction(p, b, sigma),
           "quarter_kelly_fraction": 0.0}
    if legs != 1:
        out["reason"] = f"Straight singles only (got {legs} legs)"
        return out
    if offered_dec < MIN_DECIMAL_ODDS - 1e-12:
        out["reason"] = f"Odds {offered_dec:.4f} shorter than -130 floor"
        return out
    if out["raw_kelly"] <= 0.0:
        out["reason"] = "Non-positive EV"
        return out
    out["quarter_kelly_fraction"] = KELLY_FRACTION * out["raw_kelly"]
    raw_units = min(out["quarter_kelly_fraction"], out["baker_mchale_fraction"]) * bankroll_units
    single_cap = MAX_STAKE_FRACTION * bankroll_units
    remaining = round(MAX_EVENT_FRACTION * bankroll_units - max(existing_event_units, 0.0), 6)
    if remaining <= 0.0:
        out["reason"] = (f"Event exposure cap reached ({existing_event_units:.2f}u of "
                         f"{MAX_EVENT_FRACTION * bankroll_units:.2f}u)")
        return out
    units = round(min(raw_units, single_cap, remaining), 2)
    if units <= 0.0:
        out["reason"] = "Stake rounds to zero"
        return out
    notes: list[str] = []
    out["status"] = HITL_APPROVED
    if raw_units > single_cap:
        out["status"] = HITL_PENDING
        notes.append(f"Uncapped stake {raw_units:.2f}u exceeds {single_cap:.2f}u")
    if remaining < min(raw_units, single_cap):
        notes.append(f"Trimmed to {units:.2f}u by event cap")
    out["units"] = units
    out["reason"] = "; ".join(notes)
    return out
