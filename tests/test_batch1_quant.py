"""Batch 1 ground-truth tests: de-vigging, derivatives (Stern / Skellam), risk.

Run: python -m pytest tests/test_batch1_quant.py -v
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from statistics import NormalDist

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.core.derivatives import (  # noqa: E402
    asian_handicap_ev,
    compare_spread_offers,
    compute_skellam_lambdas,
    nfl_margin_pmf,
    nfl_spread_outcome_probs,
    nfl_spread_to_win_prob,
    poisson_totals_probs,
    skellam_cdf,
    skellam_distribution,
    skellam_handicap_probs,
    skellam_mean,
    skellam_outcome_probs,
    skellam_pmf,
    skellam_push_prob,
    skellam_variance,
    spread_bet_ev,
    value_half_point,
)
from src.core.engine import (  # noqa: E402
    LIQ_ILLIQUID,
    LIQ_ONE_SIDED,
    LIQ_TWO_SIDED,
    american_to_decimal,
    apply_vig_power_n,
    compute_liquidity_weighted_prob,
    devig_odds_ratio,
    devig_odds_ratio_n,
    devig_one_sided,
    devig_power,
    devig_power_n,
    devig_shin_n,
    get_sharp_fair_prob,
    odds_ratio_constant,
)
from src.core.risk import (  # noqa: E402
    baker_mchale_stake_fraction,
    calculate_baker_mchale_shrinkage,
    implied_estimation_sigma,
    raw_kelly_fraction,
    size_bet_under_uncertainty,
)


def _dec(a: int) -> float:
    return american_to_decimal(a)


# ==========================================================================
# PART 1 - De-vigging (Clarke et al. 2017, Strumbelj 2014)
# ==========================================================================
TABLE2_PI = [1.0 / d for d in [1.15, 5.0, 10.0, 20.0, 50.0, 100.0]]
TABLE2_POWER = [0.825, 0.110, 0.042, 0.016, 0.005, 0.002]
TABLE3_TRUE = [0.01, 0.015, 0.02, 0.025, 0.03, 0.90]
TABLE3_BOOK = [0.040, 0.053, 0.065, 0.076, 0.086, 0.929]


def test_clarke_table2_power() -> None:
    probs, k = devig_power_n(TABLE2_PI)
    assert probs == pytest.approx(TABLE2_POWER, abs=1e-3)
    assert sum(probs) == pytest.approx(1.0, abs=1e-12)
    assert k > 1.0


def test_clarke_table3_reverse_injection() -> None:
    book = apply_vig_power_n(TABLE3_TRUE, 1.25)
    assert book == pytest.approx(TABLE3_BOOK, abs=1e-3)
    assert sum(book) == pytest.approx(1.25, abs=1e-12)
    assert all(0.0 < b < 1.0 for b in book)
    assert devig_power_n(book)[0] == pytest.approx(TABLE3_TRUE, abs=1e-10)


def test_power_handles_underround() -> None:
    probs, k = devig_power_n([0.48, 0.50])
    assert sum(probs) == pytest.approx(1.0, abs=1e-12)
    assert k < 1.0


def test_odds_ratio_n_definition() -> None:
    """Fair odds equal implied odds divided by one constant c (logit shift)."""
    probs, c = devig_odds_ratio_n(TABLE2_PI)
    assert sum(probs) == pytest.approx(1.0, abs=1e-12)
    assert c > 1.0
    for pi, p in zip(TABLE2_PI, probs):
        assert (pi / (1 - pi)) / (p / (1 - p)) == pytest.approx(c, rel=1e-10)


@pytest.mark.parametrize("a,b,c_ref,fa_ref", [(-150, 130, 1.074172, 0.582712),
                                             (-105, -115, 1.098863, 0.488630),
                                             (-250, 210, 1.091089, 0.696168)])
def test_odds_ratio_two_way_references(a: int, b: int, c_ref: float, fa_ref: float) -> None:
    assert odds_ratio_constant(_dec(a), _dec(b)) == pytest.approx(c_ref, abs=1e-6)
    fa, fb = devig_odds_ratio(_dec(a), _dec(b))
    assert fa == pytest.approx(fa_ref, abs=1e-6)
    assert fa + fb == pytest.approx(1.0, abs=1e-12)


def test_odds_ratio_and_power_agree_on_balanced_spread() -> None:
    da, db = _dec(-108), _dec(-112)
    assert abs(devig_power(da, db)[0] - devig_odds_ratio(da, db)[0]) < 0.0005


def test_odds_ratio_and_power_diverge_on_heavy_moneyline() -> None:
    da, db = _dec(-300), _dec(250)
    p_pow, p_or = devig_power(da, db)[0], devig_odds_ratio(da, db)[0]
    assert abs(p_pow - p_or) > 0.001
    assert p_pow > p_or            # power leaves more of the margin on the longshot
    # ... and both disagree far more than on the balanced spread.
    bal = abs(devig_power(_dec(-108), _dec(-112))[0] - devig_odds_ratio(_dec(-108), _dec(-112))[0])
    assert abs(p_pow - p_or) > 5 * bal


def test_shin_n2_exact_additive_equivalence() -> None:
    pis = [1 / _dec(-150), 1 / _dec(130)]
    booksum = sum(pis)
    probs, _ = devig_shin_n(pis)
    assert probs == pytest.approx([p - (booksum - 1) / 2 for p in pis], abs=1e-15)


def test_one_sided_symmetric_at_5pct_hold() -> None:
    assert devig_one_sided(1 / 0.525, 0.05) == pytest.approx(0.5, abs=1e-12)


def test_one_sided_minus_110_at_5pct_hold() -> None:
    d = _dec(-110)
    p = devig_one_sided(d, 0.05)
    assert p == pytest.approx(devig_power(d, 1 / (1.05 - 1 / d))[0], abs=1e-12)
    assert p == pytest.approx(0.498780, abs=1e-6)
    assert p < 1 / d


def test_one_sided_incoherent_raises() -> None:
    with pytest.raises(ValueError):
        devig_one_sided(25.0, 0.05)


@pytest.mark.parametrize("bid_liq,ask_liq", [(0.0, 0.0), (20.0, 100.0), (249.99, 249.99)])
def test_liquidity_illiquid_noise(bid_liq: float, ask_liq: float) -> None:
    assert compute_liquidity_weighted_prob(0.50, 0.52, bid_liq, ask_liq) == (None, LIQ_ILLIQUID)


def test_liquidity_one_sided_ask() -> None:
    p, m = compute_liquidity_weighted_prob(0.50, 0.55, 20.0, 2500.0)
    assert m == LIQ_ONE_SIDED
    assert p == pytest.approx(devig_one_sided(1 / 0.55, 0.04), abs=1e-12)


def test_liquidity_one_sided_bid() -> None:
    p, m = compute_liquidity_weighted_prob(0.50, 0.55, 2500.0, 20.0)
    assert m == LIQ_ONE_SIDED
    assert p == pytest.approx(1 - devig_one_sided(1 / 0.50, 0.04), abs=1e-12)


def test_liquidity_10x_imbalance_is_one_sided() -> None:
    p, m = compute_liquidity_weighted_prob(0.50, 0.55, 300.0, 3000.0)
    assert m == LIQ_ONE_SIDED
    assert p == pytest.approx(devig_one_sided(1 / 0.55, 0.04), abs=1e-12)


def test_liquidity_two_sided_power() -> None:
    p, m = compute_liquidity_weighted_prob(0.54, 0.56, 1000.0, 1000.0)
    assert m == LIQ_TWO_SIDED == "TWO_SIDED_LIQUID"
    assert p == pytest.approx(devig_power(1 / 0.56, 1 / 0.46)[0], abs=1e-12)


def test_router_h2h_is_power() -> None:
    da, db = _dec(-300), _dec(250)
    assert get_sharp_fair_prob(da, db, "h2h") == pytest.approx(devig_power(da, db), abs=1e-15)


@pytest.mark.parametrize("market", ["spreads", "totals"])
def test_router_spreads_totals_blend(market: str) -> None:
    da, db = _dec(-108), _dec(-112)
    fa, fb = get_sharp_fair_prob(da, db, market)
    blend = 0.5 * (devig_power(da, db)[0] + devig_odds_ratio(da, db)[0])
    assert fa == pytest.approx(blend, abs=1e-15)
    assert fa + fb == pytest.approx(1.0, abs=1e-12)


def test_router_rejects_unknown_market() -> None:
    with pytest.raises(ValueError):
        get_sharp_fair_prob(1.9, 1.9, "props")


# ==========================================================================
# PART 2 - Stern (1991) NFL margin model and half-point valuation
# ==========================================================================
@pytest.mark.parametrize("spread", [-7.0, -3.0, 0.0, 3.0, 6.5, 14.0])
def test_stern_normal_win_prob(spread: float) -> None:
    assert nfl_spread_to_win_prob(spread) == pytest.approx(NormalDist(0, 13.86).cdf(spread),
                                                           abs=1e-12)


def test_stern_symmetry_and_pickem() -> None:
    assert nfl_spread_to_win_prob(0.0) == pytest.approx(0.5, abs=1e-15)
    assert nfl_spread_to_win_prob(3.0) + nfl_spread_to_win_prob(-3.0) == pytest.approx(1.0)
    assert nfl_spread_to_win_prob(3.0) == pytest.approx(0.5857, abs=1e-4)


def test_margin_pmf_is_distribution_with_key_spikes() -> None:
    pmf = nfl_margin_pmf(0.0)
    assert sum(pmf.values()) == pytest.approx(1.0, abs=1e-12)
    assert pmf[3] > pmf[2] and pmf[3] > pmf[4] and pmf[3] > pmf[1]
    assert pmf[7] > pmf[8] and pmf[7] > pmf[5]
    assert pmf[0] < 0.01                                   # ties are rare
    assert sum(m * p for m, p in pmf.items()) == pytest.approx(0.0, abs=1e-12)


def test_spread_outcomes_sum_to_one() -> None:
    for line in (-3.0, -3.5, 2.5, 7.0):
        assert sum(nfl_spread_outcome_probs(line, 3.0)) == pytest.approx(1.0, abs=1e-12)
    assert nfl_spread_outcome_probs(-3.5, 3.0)[1] == 0.0      # half lines never push


def test_half_point_onto_3_converts_losses_to_pushes() -> None:
    v = value_half_point(-3.5, -3.0)
    assert v["direction"] == "buy"
    assert v["margins_crossed"] == [3] and v["key_numbers_crossed"] == [3]
    assert v["delta_win"] == pytest.approx(0.0, abs=1e-12)
    assert v["delta_push"] > 0.0
    assert v["delta_loss"] == pytest.approx(-v["delta_push"], abs=1e-12)


def test_half_point_off_3_converts_pushes_to_wins() -> None:
    v = value_half_point(-3.0, -2.5, expected_margin=3.0)
    assert v["margins_crossed"] == [3]
    assert v["delta_push"] < 0.0
    assert v["delta_win"] == pytest.approx(-v["delta_push"], abs=1e-12)


def test_full_point_through_3() -> None:
    v = value_half_point(-3.5, -2.5)
    assert v["margins_crossed"] == [3]
    assert v["delta_win"] > 0.0 and v["delta_push"] == pytest.approx(0.0, abs=1e-12)


def test_key_number_half_points_worth_more() -> None:
    on3 = value_half_point(-3.5, -3.0)["delta_push"]
    on7 = value_half_point(-7.5, -7.0)["delta_push"]
    on5 = value_half_point(-5.5, -5.0)["delta_push"]
    assert on3 > on7 > on5
    sell = value_half_point(-3.0, -3.5)
    assert sell["direction"] == "sell" and sell["value_per_unit_even_money"] < 0.0


def test_retail_minus3_vs_pinnacle_minus3p5() -> None:
    cmp = compare_spread_offers(-3.0, _dec(-115), -3.5, _dec(-108), expected_margin=3.0)
    for side, line, dec in (("a", -3.0, _dec(-115)), ("b", -3.5, _dec(-108))):
        w, _, l = nfl_spread_outcome_probs(line, 3.0)
        assert cmp[side]["ev"] == pytest.approx(w * (dec - 1) - l, abs=1e-12)
        assert cmp[side]["ev"] == pytest.approx(spread_bet_ev(line, dec, 3.0), abs=1e-15)
    assert cmp["better"] in ("a", "b")
    assert cmp["a"]["probs"][1] > 0.0 and cmp["b"]["probs"][1] == 0.0


# ==========================================================================
# PART 3 - Skellam (Karlis & Ntzoufras 2009)
# ==========================================================================
def _poisson(k: int, lam: float) -> float:
    return math.exp(-lam + k * math.log(lam) - math.lgamma(k + 1)) if k >= 0 else 0.0


def _skellam_by_convolution(z: int, l1: float, l2: float) -> float:
    return sum(_poisson(y + z, l1) * _poisson(y, l2) for y in range(max(0, -z), 200))


@pytest.mark.parametrize("l1,l2", [(1.5, 1.1), (2.7, 0.4), (0.3, 3.2), (24.0, 21.0)])
@pytest.mark.parametrize("z", [-5, -1, 0, 1, 3, 7])
def test_skellam_pmf_matches_poisson_convolution(z: int, l1: float, l2: float) -> None:
    assert skellam_pmf(z, l1, l2) == pytest.approx(_skellam_by_convolution(z, l1, l2),
                                                   rel=1e-10, abs=1e-300)


def test_skellam_pmf_matches_scipy() -> None:
    stats = pytest.importorskip("scipy.stats")
    for z in range(-6, 7):
        assert skellam_pmf(z, 1.4, 1.05) == pytest.approx(stats.skellam.pmf(z, 1.4, 1.05),
                                                          rel=1e-9)


@pytest.mark.parametrize("l1,l2", [(1.5, 1.1), (2.7, 0.4), (24.0, 21.0)])
def test_skellam_moments(l1: float, l2: float) -> None:
    dist = skellam_distribution(l1, l2)
    mean = sum(z * p for z, p in dist.items())
    var = sum((z - mean) ** 2 * p for z, p in dist.items())
    assert sum(dist.values()) == pytest.approx(1.0, abs=1e-12)
    assert mean == pytest.approx(skellam_mean(l1, l2), abs=1e-9)
    assert var == pytest.approx(skellam_variance(l1, l2), abs=1e-8)


def test_skellam_cdf() -> None:
    assert skellam_cdf(0, 1.5, 1.1) == pytest.approx(
        sum(skellam_pmf(z, 1.5, 1.1) for z in range(-60, 1)), abs=1e-12)


def test_skellam_lambdas_log_linear() -> None:
    attack = {"ARS": 0.20, "CHE": -0.05, "EVE": -0.15}
    defense = {"ARS": -0.10, "CHE": 0.02, "EVE": 0.08}
    l1, l2 = compute_skellam_lambdas(0.1, 0.25, attack, defense, "ARS", "EVE")
    assert l1 == pytest.approx(math.exp(0.1 + 0.25 + 0.20 + 0.08), abs=1e-15)
    assert l2 == pytest.approx(math.exp(0.1 - 0.15 - 0.10), abs=1e-15)


def test_skellam_lambdas_zero_sum_constraint() -> None:
    with pytest.raises(ValueError):
        compute_skellam_lambdas(0.0, 0.2, {"A": 0.3, "B": 0.0}, {"A": 0.0, "B": 0.0}, "A", "B")
    l1, _ = compute_skellam_lambdas(0.0, 0.2, {"A": 0.3, "B": 0.0}, {"A": 0.0, "B": 0.0},
                                    "A", "B", enforce_zero_sum=False)
    assert l1 == pytest.approx(math.exp(0.5))


def test_skellam_1x2_and_handicaps() -> None:
    l1, l2 = 1.6, 1.1
    o = skellam_outcome_probs(l1, l2)
    assert sum(o.values()) == pytest.approx(1.0, abs=1e-12)
    assert o["draw"] == pytest.approx(skellam_pmf(0, l1, l2), abs=1e-15)
    # Level ball (0.0): draw refunds; -0.5 is the home win.
    assert skellam_handicap_probs(0.0, l1, l2) == pytest.approx((o["home"], o["draw"], o["away"]))
    assert skellam_handicap_probs(-0.5, l1, l2)[0] == pytest.approx(o["home"])
    # -1.0 pushes exactly on a one-goal home win.
    w, push, _ = skellam_handicap_probs(-1.0, l1, l2)
    assert push == pytest.approx(skellam_pmf(1, l1, l2), abs=1e-15)
    assert push == pytest.approx(skellam_push_prob(-1.0, l1, l2), abs=1e-15)
    assert w == pytest.approx(o["home"] - skellam_pmf(1, l1, l2))
    assert skellam_push_prob(-1.5, l1, l2) == 0.0


def test_asian_quarter_line_splits_stake() -> None:
    l1, l2, dec = 1.6, 1.1, 1.95
    q = asian_handicap_ev(-0.25, dec, l1, l2)
    halves = 0.5 * (asian_handicap_ev(-0.5, dec, l1, l2) + asian_handicap_ev(0.0, dec, l1, l2))
    assert q == pytest.approx(halves, abs=1e-15)
    w, _, l = skellam_handicap_probs(-0.5, l1, l2)
    assert asian_handicap_ev(-0.5, dec, l1, l2) == pytest.approx(w * (dec - 1) - l)


def test_poisson_totals() -> None:
    over, push, under = poisson_totals_probs(2.0, 1.6, 1.1)
    lam = 2.7
    assert push == pytest.approx(_poisson(2, lam), abs=1e-15)
    assert under == pytest.approx(_poisson(0, lam) + _poisson(1, lam), abs=1e-15)
    assert over + push + under == pytest.approx(1.0, abs=1e-12)
    assert poisson_totals_probs(2.5, 1.6, 1.1)[1] == 0.0


# ==========================================================================
# PART 5 - Baker & McHale (2013) shrinkage and desk mandate
# ==========================================================================
def test_shrinkage_no_uncertainty_is_full_kelly() -> None:
    assert calculate_baker_mchale_shrinkage(0.55, 1.0, 0.0) == 1.0
    assert baker_mchale_stake_fraction(0.55, 1.0, 0.0) == pytest.approx(
        raw_kelly_fraction(0.55, 1.0))


def test_shrinkage_formula_value() -> None:
    # p = 0.55, b = 1: e = 0.1; sigma = 0.05 -> 0.01 / (0.01 + 4 * 0.0025) = 0.5
    assert calculate_baker_mchale_shrinkage(0.55, 1.0, 0.05) == pytest.approx(0.5, abs=1e-15)
    assert baker_mchale_stake_fraction(0.55, 1.0, 0.05) == pytest.approx(0.05, abs=1e-15)


@pytest.mark.parametrize("p,b,sigma", [(0.55, 1.0, 0.03), (0.40, 1.8, 0.04), (0.62, 0.85, 0.02)])
def test_closed_form_equals_shrunk_kelly(p: float, b: float, sigma: float) -> None:
    assert baker_mchale_stake_fraction(p, b, sigma) == pytest.approx(
        calculate_baker_mchale_shrinkage(p, b, sigma) * raw_kelly_fraction(p, b), rel=1e-12)


def test_shrinkage_monotone_in_sigma() -> None:
    ks = [calculate_baker_mchale_shrinkage(0.55, 1.0, s) for s in (0.0, 0.01, 0.03, 0.1)]
    assert ks == sorted(ks, reverse=True)
    assert all(0.0 <= k <= 1.0 for k in ks)


def test_no_edge_no_stake() -> None:
    assert calculate_baker_mchale_shrinkage(0.45, 1.0, 0.02) == 0.0
    assert baker_mchale_stake_fraction(0.45, 1.0, 0.02) == 0.0


def test_quarter_kelly_implied_sigma_round_trip() -> None:
    sig = implied_estimation_sigma(0.55, 1.0, 0.25)
    assert calculate_baker_mchale_shrinkage(0.55, 1.0, sig) == pytest.approx(0.25, abs=1e-12)


def test_mandate_odds_floor() -> None:
    assert size_bet_under_uncertainty(0.70, _dec(-135), 0.01)["status"] == "REJECTED"
    assert size_bet_under_uncertainty(0.70, _dec(-130), 0.01)["status"] != "REJECTED"


def test_mandate_straight_singles_only() -> None:
    r = size_bet_under_uncertainty(0.55, 2.0, 0.01, legs=2)
    assert r["status"] == "REJECTED" and "singles" in r["reason"]


def test_mandate_takes_smaller_of_quarter_kelly_and_baker_mchale() -> None:
    # Low sigma: quarter-Kelly binds -> 0.25 * 0.06 * 100 = 1.50u.
    r = size_bet_under_uncertainty(0.53, 2.0, 0.001)
    assert r["units"] == 1.5 and r["status"] == "APPROVED"
    # High sigma: Baker-McHale binds below quarter-Kelly.
    r = size_bet_under_uncertainty(0.53, 2.0, 0.08)
    expected = round(baker_mchale_stake_fraction(0.53, 1.0, 0.08) * 100, 2)
    assert r["units"] == expected < 1.5


def test_mandate_caps_two_units_per_bet_and_event() -> None:
    r = size_bet_under_uncertainty(0.80, 3.0, 0.0)
    assert r["units"] == 2.0 and r["status"] == "PENDING_CAPTAIN_HZ"
    r = size_bet_under_uncertainty(0.53, 2.0, 0.001, existing_event_units=1.2)
    assert r["units"] == 0.8 and "event cap" in r["reason"]
    r = size_bet_under_uncertainty(0.53, 2.0, 0.001, existing_event_units=2.0)
    assert r["units"] == 0.0 and r["status"] == "REJECTED"


def test_mandate_rejects_negative_ev() -> None:
    assert size_bet_under_uncertainty(0.45, 2.0, 0.01)["status"] == "REJECTED"
