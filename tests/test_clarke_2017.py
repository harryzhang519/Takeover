"""Ground-truth tests against Clarke, Kovalchik & Ingram (2017),
"Adjusting Bookmaker's Odds to Allow for Overround".

Run: python -m pytest tests/test_clarke_2017.py -v
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.core.engine import (  # noqa: E402
    MIN_AMERICAN_ODDS,
    american_to_decimal,
    apply_vig_power_n,
    devig_additive_n,
    devig_multiplicative_n,
    devig_power,
    devig_power_n,
    devig_shin_n,
    shin_probs,
    shin_z_root,
    size_position,
    validate_odds_floor,
)

# --------------------------------------------------------------------------
# Table 2: six-runner market, booksum = 1.2496 (published as 1.25)
# --------------------------------------------------------------------------
TABLE2_PRICES = [1.15, 5.00, 10.00, 20.00, 50.00, 100.00]
TABLE2_PI = [1.0 / d for d in TABLE2_PRICES]
TABLE2_POWER = [0.825, 0.110, 0.042, 0.016, 0.005, 0.002]
# Published fair prices. An exact solve gives k = 1.373616; the paper's prices
# imply k ~= 1.37343 (k rounded before pricing), so the 100.00 runner prices
# at 558.77 here vs 558.47 published (0.054%). Prices are checked to 0.1% rel.
TABLE2_POWER_PRICES = [1.21, 9.12, 23.63, 61.23, 215.54, 558.47]
TABLE2_SHIN = [0.769, 0.148, 0.059, 0.020, 0.004, 0.001]
PROB_TOL = 1e-3          # table values are printed to 3 dp
PRICE_REL_TOL = 1e-3


def test_table2_booksum() -> None:
    assert sum(TABLE2_PI) == pytest.approx(1.25, abs=1e-3)


def test_table2_power_probabilities() -> None:
    probs, k = devig_power_n(TABLE2_PI)
    assert probs == pytest.approx(TABLE2_POWER, abs=PROB_TOL)
    assert sum(probs) == pytest.approx(1.0, abs=1e-12)
    assert k == pytest.approx(1.373616, abs=1e-6)
    assert k > 1.0                                    # overround -> k > 1


def test_table2_power_fair_prices() -> None:
    probs, _ = devig_power_n(TABLE2_PI)
    prices = [1.0 / p for p in probs]
    # 2-dp print rounding (1.2116 -> 1.21) or 0.1% k-rounding, whichever is larger.
    assert prices == pytest.approx(TABLE2_POWER_PRICES, rel=PRICE_REL_TOL, abs=0.005)


def test_table2_shin_probabilities() -> None:
    probs, z = devig_shin_n(TABLE2_PI)
    assert probs == pytest.approx(TABLE2_SHIN, abs=PROB_TOL)
    assert sum(probs) == pytest.approx(1.0, abs=1e-9)
    assert 0.0 <= z < 1.0
    # Eq. (5) fixed point agrees with a direct root solve of sum(p_i) = 1.
    assert z == pytest.approx(shin_z_root(TABLE2_PI), abs=1e-9)


def test_table2_additive_negative_longshots() -> None:
    probs = devig_additive_n(TABLE2_PI)
    assert probs[-2:] == pytest.approx([-0.022, -0.032], abs=PROB_TOL)
    assert all(p < 0.0 for p in probs[-2:])
    assert sum(probs) == pytest.approx(1.0, abs=1e-12)


def test_table2_multiplicative() -> None:
    booksum = sum(TABLE2_PI)
    probs = devig_multiplicative_n(TABLE2_PI)
    assert probs == pytest.approx([p / booksum for p in TABLE2_PI], abs=1e-15)
    assert sum(probs) == pytest.approx(1.0, abs=1e-12)


# --------------------------------------------------------------------------
# Table 3: reverse overround injection, target booksum = 1.25
# --------------------------------------------------------------------------
TABLE3_TRUE = [0.01, 0.015, 0.02, 0.025, 0.03, 0.90]
TABLE3_BOOK = [0.040, 0.053, 0.065, 0.076, 0.086, 0.929]
TABLE3_PRICES = [24.93, 18.78, 15.36, 13.15, 11.57, 1.08]


def test_table3_power_injection_probabilities() -> None:
    book = apply_vig_power_n(TABLE3_TRUE, 1.25)
    assert book == pytest.approx(TABLE3_BOOK, abs=PROB_TOL)
    assert sum(book) == pytest.approx(1.25, abs=1e-12)
    assert all(0.0 < b < 1.0 for b in book)            # zero probabilities > 1.0
    assert all(b > p for b, p in zip(book, TABLE3_TRUE))


def test_table3_power_injection_prices() -> None:
    prices = [1.0 / b for b in apply_vig_power_n(TABLE3_TRUE, 1.25)]
    assert prices == pytest.approx(TABLE3_PRICES, abs=0.005)


def test_table3_round_trip() -> None:
    book = apply_vig_power_n(TABLE3_TRUE, 1.25)
    recovered, _ = devig_power_n(book)
    assert recovered == pytest.approx(TABLE3_TRUE, abs=1e-10)


def test_table3_contrast_multiplicative_breaks_favourite() -> None:
    """Why power: scaling the 0.90 favourite by 1.25 would exceed 1.0."""
    assert TABLE3_TRUE[-1] * 1.25 > 1.0
    assert max(apply_vig_power_n(TABLE3_TRUE, 1.25)) < 1.0


@pytest.mark.parametrize("fav", [0.90, 0.95, 0.99])
def test_power_injection_extreme_favourites_stay_below_one(fav: float) -> None:
    for booksum in (1.02, 1.05, 1.25, 1.5):
        book = apply_vig_power_n([fav, 1.0 - fav], booksum)
        assert all(0.0 < b < 1.0 for b in book)
        assert sum(book) == pytest.approx(booksum, abs=1e-12)


# --------------------------------------------------------------------------
# Appendix: for n = 2, Shin == Additive
# --------------------------------------------------------------------------
def _random_two_way(rng: random.Random) -> list[float]:
    p = rng.uniform(0.15, 0.85)
    booksum = rng.uniform(1.005, 1.10)                 # max implied 0.935 < 1
    return [q * booksum for q in (p, 1.0 - p)]


def test_appendix_shin_equals_additive_n2_random() -> None:
    rng = random.Random(2017)
    for _ in range(500):
        pis = _random_two_way(rng)
        booksum = sum(pis)
        additive = [p - (booksum - 1.0) / 2.0 for p in pis]
        # Solve Shin's equations numerically (no closed form used) ...
        iterated = shin_probs(pis, shin_z_root(pis))
        assert iterated == pytest.approx(additive, abs=1e-10)
        # ... and the production closed form matches both.
        closed, _ = devig_shin_n(pis)
        assert closed == pytest.approx(additive, abs=1e-15)


@pytest.mark.parametrize("a,b", [(-110, -110), (-150, 130), (-250, 210), (-118, 108)])
def test_appendix_shin_equals_additive_real_lines(a: int, b: int) -> None:
    pis = [1.0 / american_to_decimal(a), 1.0 / american_to_decimal(b)]
    booksum = sum(pis)
    additive = [p - (booksum - 1.0) / 2.0 for p in pis]
    assert shin_probs(pis, shin_z_root(pis)) == pytest.approx(additive, abs=1e-10)
    assert devig_shin_n(pis)[0] == pytest.approx(additive, abs=1e-15)


def test_shin_n2_always_positive() -> None:
    """n = 2: p_2 = (pi_2 - pi_1 + 1)/2 > 0 whenever pi_1 < 1, even at huge holds."""
    probs, z = devig_shin_n([0.99, 0.60])
    assert all(p > 0.0 for p in probs)
    assert probs == pytest.approx([0.695, 0.305], abs=1e-12)
    assert 0.0 <= z < 1.0


def test_shin_rejects_underround() -> None:
    with pytest.raises(ValueError):
        devig_shin_n([0.48, 0.50])


# --------------------------------------------------------------------------
# Desk routing and mandate (unchanged by the Clarke methods)
# --------------------------------------------------------------------------
def test_primary_devig_is_power_two_way() -> None:
    da, db = american_to_decimal(-150), american_to_decimal(130)
    fa, fb = devig_power(da, db)
    (pa, pb), _ = devig_power_n([1.0 / da, 1.0 / db])
    assert (fa, fb) == pytest.approx((pa, pb), abs=1e-15)
    assert fa == pytest.approx(0.583983, abs=1e-6)       # existing engine reference


def test_odds_floor_minus_130() -> None:
    assert MIN_AMERICAN_ODDS == -130
    assert validate_odds_floor(-130) and validate_odds_floor(+120)
    assert not validate_odds_floor(-131)


def test_quarter_kelly_cap_two_units() -> None:
    units, status, _ = size_position(0.80, 3.0)
    assert units == 2.0 and status == "PENDING_CAPTAIN_HZ"
