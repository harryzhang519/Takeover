"""Exact quant pricing engine. Pure stdlib, deterministic, stateless.

Desk mandate encoded here (reports to Captain HZ):
    * Odds floor: -130 American (1.7692 decimal) or longer.
    * Fair value: POWER de-vig of sharp books is primary (Clarke, Kovalchik &
      Ingram 2017); odds-ratio, Shin, additive and multiplicative are kept as
      comparators. Optionally blended with Kalshi into a consensus.
    * Sizing: quarter-Kelly off a FIXED baseline bankroll, capped at 2.0% per
      single and 2.0u aggregate per event_id. Never scaled by recent P&L.

Odds conventions: American as int (-110, +140); decimal as float (> 1.0).
De-vig functions take DECIMAL odds for the two sides of one market.
"""

from __future__ import annotations

import math

MIN_AMERICAN_ODDS: int = -130
MIN_DECIMAL_ODDS: float = 1.0 + 100.0 / 130.0          # 1.769231
KELLY_FRACTION: float = 0.25
DEFAULT_BANKROLL_UNITS: float = 100.0
MAX_STAKE_FRACTION: float = 0.02                         # 2.0u on 100u, per single
MAX_EVENT_FRACTION: float = 0.02                         # 2.0u on 100u, per event_id
KALSHI_FEE_MULTIPLIER: float = 0.07
CONSENSUS_WEIGHT_PINNACLE: float = 0.75

POWER_ITERATIONS: int = 60
_BISECT_ITERATIONS: int = 200

HITL_APPROVED = "APPROVED"
HITL_PENDING = "PENDING_CAPTAIN_HZ"
HITL_REJECTED = "REJECTED"


# --------------------------------------------------------------------------
# Conversions and floor
# --------------------------------------------------------------------------
def american_to_decimal(odds: int | float) -> float:
    """American -> decimal. Raises ValueError for -100 < odds < +100."""
    if -100 < odds < 100:
        raise ValueError(f"Invalid American odds: {odds}")
    return 1.0 + odds / 100.0 if odds > 0 else 1.0 + 100.0 / abs(odds)


def decimal_to_american(dec: float) -> int:
    """Decimal -> nearest integer American. 2.0 maps to +100."""
    _check_decimal(dec)
    if dec >= 2.0:
        return int(round((dec - 1.0) * 100.0))
    return int(round(-100.0 / (dec - 1.0)))


def _check_decimal(dec: float) -> None:
    if not dec > 1.0:
        raise ValueError(f"Decimal odds must be > 1.0, got {dec}")


def validate_odds_floor(american_odds: int) -> bool:
    """True only for valid American odds at -130 or longer.

    -130..-100 and +100 upward pass; -131 and shorter fail; values strictly
    between -100 and +100 are not valid American odds and fail.
    """
    if -100 < american_odds < 100:
        return False
    return american_odds >= MIN_AMERICAN_ODDS


validate_odds_threshold = validate_odds_floor           # backward-compatible alias


# --------------------------------------------------------------------------
# De-vigging
# --------------------------------------------------------------------------
#
# n-outcome overround adjustment after Clarke, Kovalchik & Ingram (2017),
# "Adjusting Bookmaker's Odds to Allow for Overround". Notation: pi_i = 1/dec_i
# are bookmaker implied probabilities, booksum = sum(pi_i), p_i are true
# probabilities. Desk routing: POWER is the primary sharp de-vigger. For n = 2
# Shin reduces exactly to Additive (paper Appendix), and Power beat both on
# LogLoss and Brier RMSE in the paper's evaluation.

def _check_probs(probs: list[float] | tuple[float, ...], name: str) -> list[float]:
    ps = [float(p) for p in probs]
    if len(ps) < 2:
        raise ValueError(f"{name} needs at least 2 outcomes, got {len(ps)}")
    for p in ps:
        if not 0.0 < p < 1.0:
            raise ValueError(f"{name} must be in (0, 1), got {p}")
    return ps


def _solve_decreasing(f, df, lo: float, hi: float, x0: float, tol: float,
                      max_iter: int = 200) -> float:
    """Root of a strictly decreasing f on [lo, hi] with f(lo) > 0 > f(hi).

    Newton-Raphson, keeping the bracket; any step that leaves the bracket
    (or a flat derivative) falls back to bisection, so convergence is assured.
    """
    x = min(max(x0, lo), hi)
    for _ in range(max_iter):
        fx = f(x)
        if abs(fx) <= tol:
            return x
        if fx > 0.0:
            lo = x
        else:
            hi = x
        d = df(x)
        nxt = x - fx / d if d != 0.0 else lo - 1.0
        x = nxt if lo < nxt < hi else 0.5 * (lo + hi)
        if hi - lo <= tol * max(1.0, abs(x)):
            return x
    return x


def devig_power_n(implied_probs: list[float], tol: float = 1e-12) -> tuple[list[float], float]:
    """Power method: k > 0 with sum(pi_i ** k) = 1; returns ``(fair_probs, k)``.

    f(k) = sum(pi_i^k) - 1 is strictly decreasing (ln pi_i < 0) with f(0) =
    n - 1 > 0, so the root is unique. k > 1 for an overround (booksum > 1),
    k < 1 for an underround (e.g. a crossed exchange book), k = 1 at par.
    """
    pis = _check_probs(implied_probs, "implied_probs")
    logs = [math.log(p) for p in pis]

    def f(k: float) -> float:
        return sum(p ** k for p in pis) - 1.0

    def df(k: float) -> float:
        return sum((p ** k) * lp for p, lp in zip(pis, logs))

    hi = 2.0
    while f(hi) > 0.0:
        hi *= 2.0
        if hi > 1e6:
            raise ArithmeticError("power de-vig failed to bracket k")
    k = _solve_decreasing(f, df, 0.0, hi, 1.0, tol)
    return [p ** k for p in pis], k


def apply_vig_power_n(true_probs: list[float], target_booksum: float,
                      tol: float = 1e-12) -> list[float]:
    """Inverse power method: inject overround so sum(p_i ** (1/k)) = target_booksum.

    Solves c = 1/k in (0, 1). Each p_i ** c lies strictly in (p_i, 1) for
    c in (0, 1), so no bookmaker probability can reach or exceed 1, even for
    0.90+ favourites (where additive/multiplicative margins can break).
    Requires true_probs summing to 1 and 1 < target_booksum < n.
    """
    ps = _check_probs(true_probs, "true_probs")
    if abs(sum(ps) - 1.0) > 1e-9:
        raise ValueError(f"true_probs must sum to 1, got {sum(ps)}")
    if not 1.0 < target_booksum < len(ps):
        raise ValueError(f"target_booksum must be in (1, {len(ps)}), got {target_booksum}")
    logs = [math.log(p) for p in ps]

    def g(c: float) -> float:
        return sum(p ** c for p in ps) - target_booksum

    def dg(c: float) -> float:
        return sum((p ** c) * lp for p, lp in zip(ps, logs))

    c = _solve_decreasing(g, dg, 0.0, 1.0, 0.5, tol)
    return [p ** c for p in ps]


def devig_additive_n(implied_probs: list[float]) -> list[float]:
    """Additive, Clarke et al. Eq. (1): p_i = pi_i - (booksum - 1)/n.

    Baseline comparator only: removes the same margin from every outcome, so
    longshots can go NEGATIVE in wide-field markets (returned as-is).
    """
    pis = _check_probs(implied_probs, "implied_probs")
    excess = (sum(pis) - 1.0) / len(pis)
    return [p - excess for p in pis]


def devig_multiplicative_n(implied_probs: list[float]) -> list[float]:
    """Multiplicative (normalisation), Clarke et al. Eq. (2): p_i = pi_i / booksum."""
    pis = _check_probs(implied_probs, "implied_probs")
    booksum = sum(pis)
    return [p / booksum for p in pis]


def shin_probs(implied_probs: list[float], z: float) -> list[float]:
    """Shin Eq. (6): p_i = (sqrt(z^2 + 4(1-z) pi_i^2 / booksum) - z) / (2(1-z))."""
    pis = _check_probs(implied_probs, "implied_probs")
    if not 0.0 <= z < 1.0:
        raise ValueError(f"z must be in [0, 1), got {z}")
    booksum = sum(pis)
    return [(math.sqrt(z * z + 4.0 * (1.0 - z) * p * p / booksum) - z) / (2.0 * (1.0 - z))
            for p in pis]


def shin_z_root(implied_probs: list[float], tol: float = 1e-14) -> float:
    """Insider fraction z solving sum(shin_probs(pi, z)) = 1 directly (any n >= 2).

    Bisection on z in [0, 1): the sum equals sqrt(booksum) > 1 at z = 0 and
    falls as z grows. Used for n = 2, where the Eq. (5) fixed-point update
    divides by n - 2 = 0, and as a cross-check of the iteration for n > 2.
    """
    pis = _check_probs(implied_probs, "implied_probs")
    if sum(pis) <= 1.0:
        raise ValueError("Shin requires an overround (booksum > 1)")
    lo, hi = 0.0, 1.0 - 1e-12
    for _ in range(_BISECT_ITERATIONS):
        mid = 0.5 * (lo + hi)
        if sum(shin_probs(pis, mid)) > 1.0:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol:
            break
    return 0.5 * (lo + hi)


def devig_shin_n(implied_probs: list[float], tol: float = 1e-12,
                 max_iter: int = 10_000) -> tuple[list[float], float]:
    """Shin method -> ``(fair_probs, z)``.

    * n = 2: exact closed form from the paper's Appendix, Shin == Additive:
      p_i = pi_i - (booksum - 1)/2 = (pi_i - pi_j + 1)/2, always in (0, 1)
      for valid pi (the p <= 0 guard is defensive). z via :func:`shin_z_root`.
    * n > 2: fixed-point iteration of Eq. (5) from z = 0,
      z <- (sum(sqrt(z^2 + 4(1-z) pi_i^2/booksum)) - 2) / (n - 2),
      then p_i from Eq. (6).
    """
    pis = _check_probs(implied_probs, "implied_probs")
    n, booksum = len(pis), sum(pis)
    if booksum <= 1.0:
        raise ValueError(f"Shin requires an overround (booksum > 1), got {booksum}")
    if n == 2:
        probs = [p - (booksum - 1.0) / 2.0 for p in pis]
        if min(probs) <= 0.0:
            raise ValueError(f"Shin/additive n=2 gives non-positive probability: {probs}")
        return probs, shin_z_root(pis)
    z = 0.0
    for _ in range(max_iter):
        z_next = (sum(math.sqrt(z * z + 4.0 * (1.0 - z) * p * p / booksum) for p in pis)
                  - 2.0) / (n - 2.0)
        if not 0.0 <= z_next < 1.0:
            raise ArithmeticError(f"Shin z left [0, 1): {z_next}")
        if abs(z_next - z) <= tol:
            z = z_next
            break
        z = z_next
    else:
        raise ArithmeticError("Shin z iteration did not converge")
    return shin_probs(pis, z), z


def power_exponent(dec_a: float, dec_b: float, iterations: int = POWER_ITERATIONS) -> float:
    """Exponent k with (1/dec_a)^k + (1/dec_b)^k = 1 (two-way :func:`devig_power_n`).

    ``iterations`` is kept for backward compatibility and ignored; the solver
    is Newton-Raphson with a bisection fallback to 1e-12.
    """
    _check_decimal(dec_a)
    _check_decimal(dec_b)
    return devig_power_n([1.0 / dec_a, 1.0 / dec_b])[1]


def devig_power(dec_a: float, dec_b: float) -> tuple[float, float]:
    """PRIMARY desk de-vig for two-way NBA/NFL/Kalshi markets -> (fair_a, fair_b)."""
    _check_decimal(dec_a)
    _check_decimal(dec_b)
    (fa, fb), _ = devig_power_n([1.0 / dec_a, 1.0 / dec_b])
    return fa, fb


def devig_odds_ratio_n(implied_probs: list[float], tol: float = 1e-12) -> tuple[list[float], float]:
    """Odds-ratio / logit de-vig (Strumbelj 2014) -> ``(fair_probs, c)``.

    Solves f(c) = sum(pi_i / (c + pi_i (1 - c))) - 1 = 0 for c > 0 by
    Newton-Raphson with a bisection fallback, using
    f'(c) = sum(-pi_i (1 - pi_i) / (c + pi_i (1 - c))^2) < 0.
    Equivalent to dividing every outcome's odds pi/(1-pi) by c (a uniform
    shift in logit space). f(0+) = n - 1 > 0 and f -> -1 as c -> inf, so the
    root is unique; c > 1 for an overround, c < 1 for an underround.
    """
    pis = _check_probs(implied_probs, "implied_probs")

    def f(c: float) -> float:
        return sum(p / (c + p * (1.0 - c)) for p in pis) - 1.0

    def df(c: float) -> float:
        return sum(-p * (1.0 - p) / ((c + p * (1.0 - c)) ** 2) for p in pis)

    hi = 2.0
    while f(hi) > 0.0:
        hi *= 2.0
        if hi > 1e9:
            raise ArithmeticError("odds-ratio de-vig failed to bracket c")
    c = _solve_decreasing(f, df, 0.0, hi, 1.0, tol)
    return [p / (c + p * (1.0 - c)) for p in pis], c


def odds_ratio_constant(dec_a: float, dec_b: float) -> float:
    """Constant c with sum(imp_i / (c + imp_i - c*imp_i)) = 1 (two-way)."""
    _check_decimal(dec_a)
    _check_decimal(dec_b)
    return devig_odds_ratio_n([1.0 / dec_a, 1.0 / dec_b])[1]


def devig_odds_ratio(dec_a: float, dec_b: float) -> tuple[float, float]:
    """Odds-ratio (logit) de-vig of a two-way market -> (fair_a, fair_b)."""
    _check_decimal(dec_a)
    _check_decimal(dec_b)
    (pa, pb), _ = devig_odds_ratio_n([1.0 / dec_a, 1.0 / dec_b])
    s = pa + pb                                  # == 1 to solver tolerance
    return pa / s, pb / s


def get_sharp_fair_prob(dec_a: float, dec_b: float,
                        market_type: str = "h2h") -> tuple[float, float]:
    """Market-specific de-vig router for a sharp two-way line -> (fair_a, fair_b).

    * ``h2h`` (moneylines, favourite-longshot skew): power de-vig.
    * ``spreads`` / ``totals`` (near 50/50 by construction): 50/50 blend of
      power and odds-ratio, which agree closely there.
    """
    mt = market_type.lower()
    if mt == "h2h":
        return devig_power(dec_a, dec_b)
    if mt in ("spreads", "totals"):
        pa, pb = devig_power(dec_a, dec_b)
        oa, ob = devig_odds_ratio(dec_a, dec_b)
        return 0.5 * (pa + oa), 0.5 * (pb + ob)
    raise ValueError(f"unknown market_type {market_type!r} (h2h, spreads, totals)")


def devig_conservative(dec_a: float, dec_b: float) -> tuple[float, float, float]:
    """(fair_a, power_a, odds_ratio_a): fair_a is the lower of the two models.

    Comparator only; the pricing and settlement pipelines use :func:`devig_power`.
    """
    p_pow = devig_power(dec_a, dec_b)[0]
    p_or = devig_odds_ratio(dec_a, dec_b)[0]
    return min(p_pow, p_or), p_pow, p_or


def devig_one_sided(offered_dec: float, synthetic_hold: float = 0.05) -> float:
    """Fair probability of a side quoted alone, via a synthetic counterpart.

    With only ``q_1 = 1/offered_dec`` known, assume the market carries
    ``synthetic_hold`` and build the missing side as
    ``q_2 = (1 + synthetic_hold) - q_1``; then power de-vig the pair.
    Raises ValueError when ``q_2`` falls outside (0, 1) (the offered side is
    too short/long for that hold to be a coherent two-way market).
    """
    _check_decimal(offered_dec)
    if synthetic_hold < 0.0:
        raise ValueError(f"synthetic_hold must be >= 0, got {synthetic_hold}")
    q_2 = (1.0 + synthetic_hold) - 1.0 / offered_dec
    if not 0.0 < q_2 < 1.0:
        raise ValueError(f"synthetic counterpart q_2={q_2:.6f} outside (0, 1) for "
                         f"offered {offered_dec} at hold {synthetic_hold}")
    return devig_power(offered_dec, 1.0 / q_2)[0]


# --------------------------------------------------------------------------
# Exchange liquidity filtering
# --------------------------------------------------------------------------
MIN_CREDIBLE_LIQ_USD: float = 250.0
EXCHANGE_SYNTHETIC_HOLD: float = 0.04
LIQ_IMBALANCE_RATIO: float = 10.0

LIQ_ILLIQUID = "ILLIQUID_NOISE"
LIQ_ONE_SIDED = "ONE_SIDED_SYNTHETIC"
LIQ_SYNTHETIC_INVALID = "SYNTHETIC_INVALID"
LIQ_TWO_SIDED = "TWO_SIDED_LIQUID"
LIQ_POWER = LIQ_TWO_SIDED                               # backward-compatible alias


def compute_liquidity_weighted_prob(bid_prob: float, ask_prob: float, bid_liq_usd: float,
                                    ask_liq_usd: float,
                                    min_credible_liq: float = MIN_CREDIBLE_LIQ_USD,
                                    synthetic_hold: float = EXCHANGE_SYNTHETIC_HOLD
                                    ) -> tuple[float | None, str]:
    """YES fair probability from an exchange top of book, filtered by depth.

    ``bid_prob``/``ask_prob`` are the best YES bid/ask (0-1 prices);
    ``*_liq_usd`` the dollar depth resting at each. Returns ``(p_yes, method)``:

    * both sides < ``min_credible_liq``     -> ``(None, ILLIQUID_NOISE)``.
    * one-sided: exactly one side credible, or both credible but one side
      holds >= ``LIQ_IMBALANCE_RATIO`` (10x) the other's depth
      -> ``ONE_SIDED_SYNTHETIC``: only the deep side is priced, with
      :func:`devig_one_sided`. A deep ask is an offer to buy YES at ``1/ask``;
      a deep bid is an offer to buy NO at ``1/(1-bid)``, whose fair value is
      complemented. Incoherent synthetic counterpart -> ``(None, SYNTHETIC_INVALID)``.
    * both credible and balanced -> power de-vig of YES-ask vs NO-ask
      (``TWO_SIDED_LIQUID``).
    """
    if bid_liq_usd < 0.0 or ask_liq_usd < 0.0:
        raise ValueError("liquidity must be >= 0")
    bid_ok, ask_ok = bid_liq_usd >= min_credible_liq, ask_liq_usd >= min_credible_liq

    def _check(p: float, name: str) -> None:
        if not 0.0 < p < 1.0:
            raise ValueError(f"{name} must be in (0, 1), got {p}")

    if not bid_ok and not ask_ok:
        return None, LIQ_ILLIQUID
    if bid_ok and ask_ok:
        if ask_liq_usd >= LIQ_IMBALANCE_RATIO * bid_liq_usd:
            bid_ok = False
        elif bid_liq_usd >= LIQ_IMBALANCE_RATIO * ask_liq_usd:
            ask_ok = False
    if bid_ok != ask_ok:
        try:
            if ask_ok:
                _check(ask_prob, "ask_prob")
                return devig_one_sided(1.0 / ask_prob, synthetic_hold), LIQ_ONE_SIDED
            _check(bid_prob, "bid_prob")
            return 1.0 - devig_one_sided(1.0 / (1.0 - bid_prob), synthetic_hold), LIQ_ONE_SIDED
        except ArithmeticError:
            return None, LIQ_SYNTHETIC_INVALID
        except ValueError as exc:
            if "synthetic counterpart" in str(exc):
                return None, LIQ_SYNTHETIC_INVALID
            raise
    _check(bid_prob, "bid_prob")
    _check(ask_prob, "ask_prob")
    return devig_power(1.0 / ask_prob, 1.0 / (1.0 - bid_prob))[0], LIQ_TWO_SIDED


def compute_consensus_prob(pin_prob: float, kalshi_prob: float | None,
                           weight_pin: float = CONSENSUS_WEIGHT_PINNACLE) -> float:
    """Weighted true probability: Pinnacle-heavy blend with Kalshi when present."""
    for p in (pin_prob, kalshi_prob):
        if p is not None and not 0.0 < p < 1.0:
            raise ValueError(f"probability out of range: {p}")
    if not 0.0 <= weight_pin <= 1.0:
        raise ValueError("weight_pin must be in [0, 1]")
    if kalshi_prob is None:
        return pin_prob
    return weight_pin * pin_prob + (1.0 - weight_pin) * kalshi_prob


# --------------------------------------------------------------------------
# Kalshi fees
# --------------------------------------------------------------------------
def kalshi_taker_fee(price: float, fee_multiplier: float = KALSHI_FEE_MULTIPLIER,
                     contracts: int = 1) -> float:
    """Kalshi taker fee in dollars for an order: ceil-to-cent(m * C * P * (1-P)).

    Kalshi rounds the fee UP to the next whole cent per order, so the
    per-contract fee is highest on a 1-lot (conservative default).
    """
    if not 0.0 < price < 1.0:
        raise ValueError(f"price must be in (0, 1), got {price}")
    raw_cents = fee_multiplier * contracts * price * (1.0 - price) * 100.0
    return math.ceil(round(raw_cents, 9)) / 100.0


def kalshi_net_odds(ask_price_dollars: float, fee_multiplier: float = KALSHI_FEE_MULTIPLIER,
                    contracts: int = 1) -> float:
    """Effective decimal odds of buying YES at the ask, after the taker fee.

    A YES contract pays $1.00. All-in cost per contract is ask + fee/C, so
    the net decimal payout is 1 / (ask + fee/C).
    """
    fee = kalshi_taker_fee(ask_price_dollars, fee_multiplier, contracts)
    return 1.0 / (ask_price_dollars + fee / contracts)


# --------------------------------------------------------------------------
# EV and sizing
# --------------------------------------------------------------------------
def calculate_ev(fair_prob: float, offered_decimal: float) -> float:
    """EV per unit staked as a fraction: fair_prob * decimal - 1."""
    if not 0.0 <= fair_prob <= 1.0:
        raise ValueError(f"fair_prob must be in [0, 1], got {fair_prob}")
    _check_decimal(offered_decimal)
    return fair_prob * offered_decimal - 1.0


def full_kelly_fraction(fair_prob: float, offered_decimal: float) -> float:
    b = offered_decimal - 1.0
    return (b * fair_prob - (1.0 - fair_prob)) / b


def raw_quarter_kelly_units(fair_prob: float, offered_decimal: float,
                            bankroll_units: float = DEFAULT_BANKROLL_UNITS) -> float:
    """Uncapped quarter-Kelly units (0 if EV <= 0)."""
    if calculate_ev(fair_prob, offered_decimal) <= 0.0:
        return 0.0
    return KELLY_FRACTION * full_kelly_fraction(fair_prob, offered_decimal) * bankroll_units


def calculate_quarter_kelly(fair_prob: float, offered_decimal: float,
                            bankroll_units: float = DEFAULT_BANKROLL_UNITS,
                            cap_percent: float = MAX_STAKE_FRACTION) -> float:
    """Quarter-Kelly units capped at ``cap_percent`` of the baseline bankroll."""
    raw = raw_quarter_kelly_units(fair_prob, offered_decimal, bankroll_units)
    return round(min(raw, cap_percent * bankroll_units), 2) if raw > 0 else 0.0


def size_position(fair_prob: float, offered_dec: float, existing_event_units: float = 0.0,
                  bankroll_units: float = DEFAULT_BANKROLL_UNITS) -> tuple[float, str, str]:
    """Quarter-Kelly with the single cap and the per-event correlation cap.

    Returns ``(final_units, hitl_status, hitl_reason)``:
        * REJECTED  - EV <= 0, or the event's 2.0u exposure is already used.
        * PENDING_CAPTAIN_HZ - uncapped quarter-Kelly exceeds the 2.0u cap.
        * APPROVED  - otherwise. ``hitl_reason`` may still note an event trim.
    ``bankroll_units`` must be the fixed baseline (anti-house-money).
    """
    single_cap = MAX_STAKE_FRACTION * bankroll_units
    event_cap = MAX_EVENT_FRACTION * bankroll_units
    raw = raw_quarter_kelly_units(fair_prob, offered_dec, bankroll_units)
    if raw <= 0.0:
        return 0.0, HITL_REJECTED, "Non-positive EV"
    remaining = round(event_cap - max(existing_event_units, 0.0), 6)
    if remaining <= 0.0:
        return 0.0, HITL_REJECTED, (f"Event exposure cap reached ({existing_event_units:.2f}u "
                                    f"of {event_cap:.2f}u)")
    units = round(min(raw, single_cap, remaining), 2)
    if units <= 0.0:
        return 0.0, HITL_REJECTED, "Stake rounds to zero"
    notes: list[str] = []
    status = HITL_APPROVED
    if raw > single_cap:
        status = HITL_PENDING
        notes.append(f"Uncapped quarter-Kelly {raw:.2f}u exceeds {single_cap:.2f}u")
    if remaining < min(raw, single_cap):
        notes.append(f"Trimmed to {units:.2f}u by event cap ({existing_event_units:.2f}u "
                     f"already open)")
    return units, status, "; ".join(notes)


# --------------------------------------------------------------------------
# Self-tests:  python -m src.core.engine
# --------------------------------------------------------------------------
if __name__ == "__main__":
    def approx(a: float, b: float, tol: float = 1e-6) -> bool:
        return abs(a - b) <= tol

    results: list[tuple[str, bool]] = []

    def check(name: str, cond: bool) -> None:
        results.append((name, bool(cond)))

    # Conversions / floor
    check("american_to_decimal(-110)", approx(american_to_decimal(-110), 1.909091))
    check("american_to_decimal(+140)", approx(american_to_decimal(140), 2.4))
    check("american_to_decimal(-130) = MIN_DECIMAL", approx(american_to_decimal(-130), 1.769231))
    check("decimal_to_american(2.40) = +140", decimal_to_american(2.40) == 140)
    check("decimal_to_american(1.909091) = -110", decimal_to_american(1.909091) == -110)
    check("round-trip -125", decimal_to_american(american_to_decimal(-125)) == -125)
    for odds, ok in [(-130, True), (-125, True), (-110, True), (-100, True), (100, True),
                     (130, True), (-131, False), (-135, False), (-160, False), (50, False)]:
        check(f"validate_odds_floor({odds:+d}) is {ok}", validate_odds_floor(odds) is ok)

    # Power de-vig vs Pinnacle lines (references: independent Newton solve on k)
    for a, b, k_ref, fa_ref in [(-110, -110, None, 0.5), (-150, 130, 1.052970, 0.583983),
                                (-105, -115, 1.071116, 0.488396), (-250, 210, 1.062464, 0.699430),
                                (-118, 108, 1.032581, 0.530567)]:
        da, db = american_to_decimal(a), american_to_decimal(b)
        fa, fb = devig_power(da, db)
        check(f"power {a:+d}/{b:+d} fair_a={fa_ref}", approx(fa, fa_ref))
        check(f"power {a:+d}/{b:+d} sums to 1", approx(fa + fb, 1.0, 1e-12))
        if k_ref:
            check(f"power {a:+d}/{b:+d} k={k_ref}", approx(power_exponent(da, db), k_ref))

    # Odds-ratio de-vig (references: independent Newton solve on c)
    for a, b, c_ref, fa_ref in [(-150, 130, 1.074172, 0.582712), (-105, -115, 1.098863, 0.488630),
                                (-250, 210, 1.091089, 0.696168), (-118, 108, 1.045272, 0.530272)]:
        da, db = american_to_decimal(a), american_to_decimal(b)
        fa, fb = devig_odds_ratio(da, db)
        check(f"odds-ratio {a:+d}/{b:+d} fair_a={fa_ref}", approx(fa, fa_ref))
        check(f"odds-ratio {a:+d}/{b:+d} c={c_ref}", approx(odds_ratio_constant(da, db), c_ref))
        check(f"odds-ratio {a:+d}/{b:+d} sums to 1", approx(fa + fb, 1.0, 1e-12))
    check("odds-ratio -110/-110 = 0.5", approx(devig_odds_ratio(1.909091, 1.909091)[0], 0.5))

    # One-sided synthetic de-vig @ 5.0% hold
    # q1 = 0.525 -> q2 = 1.05 - 0.525 = 0.525: symmetric pair, fair exactly 0.5.
    check("one-sided: q1=0.525 @5% -> 0.5", approx(devig_one_sided(1 / 0.525, 0.05), 0.5))
    # -110 alone: q2 = 1.05 - 0.523810 = 0.526190 -> power de-vig of (1.909091, 1.900452).
    d110 = american_to_decimal(-110)
    p110 = devig_one_sided(d110, 0.05)
    check("one-sided -110 @5% = power(-110, 1/0.526190)",
          approx(p110, devig_power(d110, 1 / (1.05 - 1 / d110))[0], 1e-12))
    check("one-sided -110 @5% = 0.498780", approx(p110, 0.498780))
    check("one-sided fair < implied", p110 < 1 / d110)
    try:                                         # q1 = 0.04 -> q2 = 1.01, not a probability
        devig_one_sided(25.0, 0.05)
        check("one-sided q2 >= 1 raises", False)
    except ValueError:
        check("one-sided q2 >= 1 raises", True)

    # Exchange liquidity filter
    check("liq: $20/$100 -> ILLIQUID_NOISE",
          compute_liquidity_weighted_prob(0.50, 0.52, 20.0, 100.0) == (None, LIQ_ILLIQUID))
    check("liq: $249.99 both -> ILLIQUID_NOISE",
          compute_liquidity_weighted_prob(0.50, 0.52, 249.99, 249.99)[1] == LIQ_ILLIQUID)
    p, m = compute_liquidity_weighted_prob(0.50, 0.55, 20.0, 2500.0)
    check("liq: ask $2,500 / bid $20 -> ONE_SIDED_SYNTHETIC", m == LIQ_ONE_SIDED)
    check("liq: ask-anchored = devig_one_sided(1/0.55, 4%)",
          approx(p, devig_one_sided(1 / 0.55, 0.04), 1e-12))
    check("liq: ask 0.55 one-sided @4% -> 0.530577", approx(p, 0.530577))
    p, m = compute_liquidity_weighted_prob(0.50, 0.55, 2500.0, 20.0)
    check("liq: bid $2,500 / ask $20 -> ONE_SIDED_SYNTHETIC", m == LIQ_ONE_SIDED)
    check("liq: bid-anchored = 1 - devig_one_sided(1/0.50, 4%)",
          approx(p, 1 - devig_one_sided(1 / 0.50, 0.04), 1e-12))
    check("liq: bid 0.50 one-sided @4% -> 0.520384", approx(p, 0.520384))
    check("liq: incoherent synthetic -> SYNTHETIC_INVALID",
          compute_liquidity_weighted_prob(0.0, 0.02, 0.0, 900.0) == (None, LIQ_SYNTHETIC_INVALID))
    p, m = compute_liquidity_weighted_prob(0.54, 0.56, 1000.0, 1000.0)
    check("liq: both liquid -> TWO_SIDED_LIQUID", m == LIQ_TWO_SIDED)
    check("liq: power 0.54/0.56 = power(1/0.56, 1/0.46)",
          approx(p, devig_power(1 / 0.56, 1 / 0.46)[0], 1e-12))
    p, m = compute_liquidity_weighted_prob(0.50, 0.55, 300.0, 5000.0)
    check("liq: 10x imbalance -> ONE_SIDED_SYNTHETIC", m == LIQ_ONE_SIDED)

    # Consensus
    check("consensus 0.75*0.60+0.25*0.52", approx(compute_consensus_prob(0.60, 0.52), 0.58))
    check("consensus without Kalshi = Pinnacle", compute_consensus_prob(0.55, None) == 0.55)

    # Kalshi fees: 0.07*0.5*0.5 = 1.75c -> 2c; cost 0.52 -> 1.923077
    check("kalshi fee @0.50 1-lot = $0.02", approx(kalshi_taker_fee(0.50), 0.02))
    check("kalshi fee @0.50 100-lot = $1.75", approx(kalshi_taker_fee(0.50, contracts=100), 1.75))
    check("kalshi fee @0.95 1-lot = $0.01", approx(kalshi_taker_fee(0.95), 0.01))
    check("kalshi_net_odds(0.50) = 1/0.52", approx(kalshi_net_odds(0.50), 1 / 0.52))
    check("kalshi_net_odds(0.50, 100-lot) = 1/0.5175",
          approx(kalshi_net_odds(0.50, contracts=100), 1 / 0.5175))

    # Sizing
    check("size: -EV rejected", size_position(0.45, 2.0)[1] == HITL_REJECTED)
    u, s, r = size_position(0.53, 2.0)
    check("size: 0.53 @ 2.0 -> 1.50u APPROVED", u == 1.5 and s == HITL_APPROVED and r == "")
    u, s, r = size_position(0.55, 2.0)
    check("size: raw 2.5u -> 2.00u PENDING", u == 2.0 and s == HITL_PENDING and "2.50u" in r)
    u, s, r = size_position(0.53, 2.0, existing_event_units=1.2)
    check("size: event cap trims 1.50 -> 0.80u", u == 0.8 and s == HITL_APPROVED and "Trimmed" in r)
    u, s, r = size_position(0.53, 2.0, existing_event_units=2.0)
    check("size: event cap exhausted -> REJECTED", u == 0.0 and s == HITL_REJECTED)
    check("size: never exceeds 2.0u", size_position(0.80, 3.0)[0] == 2.0)

    failed = [n for n, ok in results if not ok]
    for n, ok in results:
        print(f"[{'PASS' if ok else 'FAIL'}] {n}")
    print(f"\n{len(results) - len(failed)}/{len(results)} engine tests passed.")
    raise SystemExit(1 if failed else 0)
