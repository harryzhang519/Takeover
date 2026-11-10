"""Batch 2 tests: Polymarket microstructure (Dubach 2026), nflWAR multilevel
models (Yurko et al. 2018), CLV / no-vig price / Brier (Buchdahl).

Run: python -m pytest tests/test_batch2.py -v
"""

from __future__ import annotations

import math
import random
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.core.backtester import (  # noqa: E402
    calculate_no_vig_price,
    clv_skill_test,
    compute_clv,
    evaluate_forecast_skill,
)
from src.core.engine import devig_power  # noqa: E402
from src.core.matchup_model import (  # noqa: E402
    decompose_pass_value,
    fit_air_model,
    fit_rush_model,
    fit_varying_intercepts,
    fit_yac_model,
    is_designed_non_qb_rush,
    points_added,
    side_gap_key,
)
from src.data.market_fetcher import (  # noqa: E402
    FeedSideHeuristicError,
    PolymarketBook,
    available_liquidity_usd,
    book_vwap,
    cumulative_depth,
    depth_ratio_l1_l10,
    effective_spread,
    filter_wash_trades,
    onchain_vwap,
    parse_order_filled,
    parse_polymarket_book,
    signed_volume,
    trade_sign_from_fill,
    volume_weighted_effective_spread,
)

TOKEN = 71321045679252212594626385532706912750332728571942532289631379312455583992563
SCALE = 1_000_000


def _event(maker="0xAAA", taker="0xBBB", block=100, log=0, *, taker_buys=True,
           shares=100.0, price=0.6, token=TOKEN):
    """OrderFilled args. Taker buy: maker gives tokens, taker gives USDC (takerAssetId 0)."""
    usdc, tok = int(round(shares * price * SCALE)), int(round(shares * SCALE))
    if taker_buys:
        args = {"makerAssetId": token, "takerAssetId": 0,
                "makerAmountFilled": tok, "takerAmountFilled": usdc}
    else:
        args = {"makerAssetId": 0, "takerAssetId": token,
                "makerAmountFilled": usdc, "takerAmountFilled": tok}
    args.update({"maker": maker, "taker": taker, "fee": 0})
    return {"transactionHash": f"0x{block:x}{log}", "logIndex": log, "blockNumber": block,
            "args": args}


# ==========================================================================
# PART 1: Polymarket microstructure
# ==========================================================================

class TestOnChainDirection:
    def test_taker_buy_when_taker_asset_is_collateral(self):
        f = parse_order_filled(_event(taker_buys=True, shares=100, price=0.6))
        assert trade_sign_from_fill(f) == +1 and f.sign == +1
        assert f.shares == pytest.approx(100.0)
        assert f.usdc == pytest.approx(60.0)
        assert f.price == pytest.approx(0.6)
        assert f.token_id == TOKEN

    def test_taker_sell_when_maker_asset_is_collateral(self):
        f = parse_order_filled(_event(taker_buys=False, shares=100, price=0.45))
        assert f.sign == -1
        assert f.price == pytest.approx(0.45)
        assert f.token_id == TOKEN

    def test_hex_asset_ids_and_flat_snake_case(self):
        f = parse_order_filled({"block_number": 5, "maker": "0xA", "taker": "0xB",
                                "maker_asset_id": hex(TOKEN), "taker_asset_id": "0x0",
                                "maker_amount": 10 * SCALE, "taker_amount": 3 * SCALE})
        assert f.sign == +1 and f.price == pytest.approx(0.3) and f.token_id == TOKEN

    def test_feed_side_heuristic_rejected(self):
        with pytest.raises(FeedSideHeuristicError):
            parse_order_filled({"change_side": "BUY", "price": "0.6", "size": "100",
                                "blockNumber": 1})

    def test_missing_legs_rejected(self):
        with pytest.raises(ValueError):
            parse_order_filled({"blockNumber": 1, "maker": "a", "taker": "b"})

    @pytest.mark.parametrize("m,t", [(0, 0), (TOKEN, TOKEN + 1)])
    def test_exactly_one_collateral_leg(self, m, t):
        ev = _event()
        ev["args"].update(makerAssetId=m, takerAssetId=t)
        with pytest.raises(ValueError):
            parse_order_filled(ev)

    def test_effective_spread_positive_with_onchain_signs(self):
        buy = parse_order_filled(_event(taker_buys=True, price=0.60))
        sell = parse_order_filled(_event(taker_buys=False, price=0.45, log=1))
        assert effective_spread(buy, 0.55) == pytest.approx(0.10)
        assert effective_spread(sell, 0.50) == pytest.approx(0.10)
        # A feed-side sign flip would have produced -0.10: the on-chain sign avoids it.
        assert -2.0 * buy.sign * (buy.price - 0.55) == pytest.approx(-0.10)

    def test_effective_spread_rejects_bad_mid(self):
        with pytest.raises(ValueError):
            effective_spread(parse_order_filled(_event()), 1.2)

    def test_vwap_and_signed_volume(self):
        fills = [parse_order_filled(_event(taker_buys=True, shares=100, price=0.60)),
                 parse_order_filled(_event(taker_buys=False, shares=100, price=0.45, log=1)),
                 parse_order_filled(_event(taker_buys=True, shares=50, price=0.50, log=2))]
        assert onchain_vwap(fills) == pytest.approx((60 + 45 + 25) / 250)
        assert signed_volume(fills) == pytest.approx(100 - 100 + 50)
        assert onchain_vwap([]) is None
        vw = volume_weighted_effective_spread(fills, [0.55, 0.50, 0.50])
        assert vw == pytest.approx((100 * 0.10 + 100 * 0.10 + 50 * 0.0) / 250)
        with pytest.raises(ValueError):
            volume_weighted_effective_spread(fills, [0.5])


class TestWashFilter:
    def test_self_match_removed(self):
        fills = [parse_order_filled(_event("0xAAA", "0xaaa")),
                 parse_order_filled(_event("0xAAA", "0xBBB", log=1))]
        clean, removed = filter_wash_trades(fills)
        assert len(clean) == 1 and len(removed) == 1
        assert removed[0].maker == removed[0].taker

    def test_reciprocal_round_trip_within_finality_removed(self):
        a = parse_order_filled(_event("0xA", "0xB", block=100))
        b = parse_order_filled(_event("0xB", "0xA", block=228, taker_buys=False))
        c = parse_order_filled(_event("0xC", "0xD", block=150))
        clean, removed = filter_wash_trades([a, c, b])
        assert clean == [c]
        assert set(removed) == {a, b}

    def test_round_trip_outside_buffer_kept(self):
        a = parse_order_filled(_event("0xA", "0xB", block=100))
        b = parse_order_filled(_event("0xB", "0xA", block=229))
        clean, removed = filter_wash_trades([a, b])
        assert clean == [a, b] and removed == []

    def test_different_token_not_paired(self):
        a = parse_order_filled(_event("0xA", "0xB", block=100))
        b = parse_order_filled(_event("0xB", "0xA", block=101, token=TOKEN + 7))
        assert filter_wash_trades([a, b])[1] == []

    def test_each_fill_pairs_once(self):
        a = parse_order_filled(_event("0xA", "0xB", block=100))
        b1 = parse_order_filled(_event("0xB", "0xA", block=101))
        b2 = parse_order_filled(_event("0xB", "0xA", block=102))
        clean, removed = filter_wash_trades([a, b1, b2])
        assert clean == [b2] and set(removed) == {a, b1}


class TestL2Depth:
    def _grid(self, n=12, size=100.0):
        bids = [(round(0.49 - 0.01 * i, 2), size) for i in range(n)]
        asks = [(round(0.51 + 0.01 * i, 2), size) for i in range(n)]
        return PolymarketBook(bids=bids, asks=asks)

    def test_parse_sorts_best_first_and_drops_bad_levels(self):
        book = parse_polymarket_book({
            "bids": [{"price": "0.40", "size": "10"}, {"price": "0.48", "size": "5"},
                     {"price": "0.45", "size": "0"}],
            "asks": [{"price": "0.60", "size": "7"}, {"price": "0.52", "size": "3"},
                     {"price": "1.00", "size": "9"}]})
        assert book.bids == [(0.48, 5.0), (0.40, 10.0)]
        assert book.asks == [(0.52, 3.0), (0.60, 7.0)]
        assert book.midpoint == pytest.approx(0.50)

    def test_depth_aggregates_exactly_ten_levels(self):
        book = self._grid()
        cum = cumulative_depth(book.asks)
        assert len(cum) == 10 and cum[-1] == pytest.approx(1000.0)
        assert depth_ratio_l1_l10(book.asks) == pytest.approx(0.1)
        bid_usd, ask_usd = available_liquidity_usd(book)
        assert bid_usd == pytest.approx(sum(p * 100 for p, _ in book.bids[:10]))
        assert ask_usd == pytest.approx(sum(p * 100 for p, _ in book.asks[:10]))
        assert ask_usd > book.asks[0][0] * book.asks[0][1] * 9     # far beyond top-of-book

    def test_empty_side(self):
        assert depth_ratio_l1_l10([]) is None
        assert available_liquidity_usd(PolymarketBook()) == (0.0, 0.0)
        assert PolymarketBook().midpoint is None

    def test_book_vwap_walks_levels(self):
        book = PolymarketBook(bids=[(0.48, 100), (0.47, 100)],
                              asks=[(0.50, 100), (0.52, 100), (0.55, 100)])
        vwap, filled = book_vwap(book, "buy", 250)
        assert filled == pytest.approx(250)
        assert vwap == pytest.approx((50 + 52 + 0.55 * 50) / 250)
        vwap, filled = book_vwap(book, "sell", 150)
        assert vwap == pytest.approx((48 + 0.47 * 50) / 150)

    def test_book_vwap_caps_at_ten_levels(self):
        vwap, filled = book_vwap(self._grid(), "buy", 2000)
        assert filled == pytest.approx(1000)
        assert vwap == pytest.approx(sum(0.51 + 0.01 * i for i in range(10)) / 10)

    def test_book_vwap_validation(self):
        with pytest.raises(ValueError):
            book_vwap(self._grid(), "buy", 0)
        with pytest.raises(ValueError):
            book_vwap(self._grid(), "hold", 10)


# ==========================================================================
# PART 2: nflWAR multilevel models
# ==========================================================================

def _corr(a, b):
    ma, mb = sum(a) / len(a), sum(b) / len(b)
    num = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    return num / math.sqrt(sum((x - ma) ** 2 for x in a) * sum((y - mb) ** 2 for y in b))


class TestMixedModel:
    def test_fixed_effects_only_is_ols(self):
        xs = [float(i) for i in range(20)]
        fit = fit_varying_intercepts([1.0 + 2.0 * x for x in xs], [[1.0, x] for x in xs],
                                     ["intercept", "x"], {})
        assert fit.fixed_effects["intercept"] == pytest.approx(1.0, abs=1e-6)
        assert fit.fixed_effects["x"] == pytest.approx(2.0, abs=1e-6)
        assert fit.converged

    def test_recovers_varying_intercepts(self):
        rng = random.Random(11)
        truth = {"g0": -1.0, "g1": -0.5, "g2": 0.0, "g3": 0.5, "g4": 1.0}
        y, x, lab = [], [], []
        for g, eff in truth.items():
            for _ in range(200):
                y.append(0.3 + eff + rng.gauss(0, 1))
                x.append([1.0])
                lab.append(g)
        fit = fit_varying_intercepts(y, x, ["intercept"], {"grp": lab}, tol=1e-7)
        est = fit.random_effects["grp"]
        for g, eff in truth.items():
            assert est[g] == pytest.approx(eff, abs=0.2)
        assert sorted(est, key=est.get) == list(truth)
        assert fit.fixed_effects["intercept"] == pytest.approx(0.3, abs=0.2)
        assert fit.sigma2 == pytest.approx(1.0, abs=0.15)
        assert fit.group_variances["grp"] > 0.2
        assert fit.predict({"intercept": 1.0}, {"grp": "g4"}) == pytest.approx(
            fit.fixed_effects["intercept"] + est["g4"])
        assert fit.predict({}, {"grp": "unseen"}) == pytest.approx(fit.fixed_effects["intercept"])

    def test_thin_samples_shrunk_toward_zero(self):
        rng = random.Random(3)
        y, lab = [], []
        for g in range(8):
            for _ in range(100):
                y.append(rng.gauss(0, 1) + (0.4 if g % 2 else -0.4))
                lab.append(f"p{g}")
        y += [3.0, 3.0]
        lab += ["rookie", "rookie"]
        fit = fit_varying_intercepts(y, [[1.0]] * len(y), ["intercept"], {"p": lab}, tol=1e-7)
        raw = 3.0 - fit.fixed_effects["intercept"]
        assert 0.0 < fit.random_effects["p"]["rookie"] < 0.5 * raw

    def test_input_validation(self):
        with pytest.raises(ValueError):
            fit_varying_intercepts([], [], ["intercept"], {})
        with pytest.raises(ValueError):
            fit_varying_intercepts([1.0, 2.0], [[1.0]], ["intercept"], {})
        with pytest.raises(ValueError):
            fit_varying_intercepts([1.0, 2.0], [[1.0], [1.0]], ["intercept"], {"g": ["a"]})


def _pass_plays(n=900, seed=5):
    rng = random.Random(seed)
    passers = {"QB_elite": 0.35, "QB_avg": 0.0, "QB_bad": -0.35}
    receivers = {"WR1": 0.25, "WR2": 0.0, "TE1": -0.1, "RB1": -0.15}
    pos = {"WR1": "WR", "WR2": "WR", "TE1": "TE", "RB1": "RB"}
    yac_rec = {"WR1": 0.1, "WR2": 0.0, "TE1": 0.05, "RB1": 0.3}
    defenses = {"D1": 0.1, "D2": -0.1}
    plays = []
    for _ in range(n):
        qb, rec, de = rng.choice(list(passers)), rng.choice(list(receivers)), rng.choice(list(defenses))
        ay = rng.uniform(-2, 25)
        loc = rng.choice(["left", "middle", "right"])
        sg = rng.random() < 0.6
        rush = rng.gauss(0, 1)
        air = (0.05 + passers[qb] + receivers[rec] + defenses[de] + 0.05 * rush
               + (0.1 if loc == "middle" else 0.0) + rng.gauss(0, 0.8))
        complete = rng.random() < 0.65
        play = {"passer": qb, "receiver": rec, "receiver_position": pos[rec], "defense": de,
                "air_yards": ay, "pass_location": loc, "shotgun": sg,
                "team_rush_strength": rush, "complete": complete}
        if complete:
            yac = 0.6 - 0.03 * ay + yac_rec[rec] + rng.gauss(0, 0.5)
            play.update(air_epa=air, epa=air + yac)
            play.update(air_wpa=air / 40, wpa=(air + yac) / 40)
        else:
            play.update(epa=air, wpa=air / 40)
        plays.append(play)
    return plays, passers


class TestPassingDecomposition:
    def test_decompose_complete_and_incomplete(self):
        assert decompose_pass_value({"complete": True, "epa": 2.0, "air_epa": 1.2}) == \
            pytest.approx((1.2, 0.8))
        assert decompose_pass_value({"complete": False, "epa": -0.5}) == (-0.5, 0.0)
        assert decompose_pass_value({"complete": True, "wpa": 0.05, "air_wpa": 0.02},
                                    value="wpa") == pytest.approx((0.02, 0.03))

    def test_air_model_ranks_passers(self):
        plays, passers = _pass_plays()
        fit = fit_air_model(plays, tol=1e-6)
        est = fit.random_effects["passer"]
        assert sorted(est, key=est.get) == ["QB_bad", "QB_avg", "QB_elite"]
        assert _corr([est[k] for k in passers], list(passers.values())) > 0.95
        assert est["QB_elite"] == pytest.approx(0.35, abs=0.15)
        assert fit.fixed_effects["loc_middle"] == pytest.approx(0.1, abs=0.12)
        assert set(fit.random_effects) == {"passer", "receiver", "defense"}
        assert fit.n_obs == len(plays)

    def test_yac_model_completions_only_and_air_yards_interaction(self):
        plays, _ = _pass_plays()
        fit = fit_yac_model(plays, tol=1e-6)
        assert fit.n_obs == sum(p["complete"] for p in plays)
        assert fit.fixed_effects["air_yards"] == pytest.approx(-0.03, abs=0.01)
        assert "air_yards:TE" in fit.fixed_effects and "air_yards:RB" in fit.fixed_effects

    def test_wpa_response(self):
        plays, _ = _pass_plays(n=400)
        fit = fit_air_model(plays, value="wpa", tol=1e-8)
        assert abs(fit.fixed_effects["intercept"]) < 0.05

    def test_yac_requires_completions(self):
        with pytest.raises(ValueError):
            fit_yac_model([{"complete": False, "epa": 0.1}])


def _rush_plays(n=900, seed=9):
    rng = random.Random(seed)
    rushers = {"RB_a": 0.15, "RB_b": -0.1, "WR_jet": 0.0}
    rpos = {"RB_a": "RB", "RB_b": "RB", "WR_jet": "WR"}
    gaps = {("KC", "left", "B"): 0.4, ("KC", "right", "C"): -0.2,
            ("BUF", "middle", "A"): 0.0, ("BUF", "left", "D"): -0.2}
    defenses = {"D1": 0.1, "D2": -0.1}
    plays = []
    for _ in range(n):
        r, (team, side, gap), d = (rng.choice(list(rushers)), rng.choice(list(gaps)),
                                   rng.choice(list(defenses)))
        epa = -0.05 + rushers[r] + gaps[(team, side, gap)] + defenses[d] + rng.gauss(0, 0.7)
        plays.append({"rusher": r, "rusher_position": rpos[r], "team": team, "run_side": side,
                      "run_gap": gap, "defense": d, "shotgun": rng.random() < 0.4,
                      "epa": epa, "wpa": epa / 40})
    # QB carries that must be excluded (huge values would distort the fit)
    plays.append({"rusher": "QB1", "rusher_position": "QB", "team": "KC", "run_side": "right",
                  "run_gap": "C", "defense": "D1", "epa": 50.0, "wpa": 1.0})
    plays.append({"rusher": "RB_a", "rusher_position": "RB", "qb_scramble": True, "team": "KC",
                  "run_side": "right", "run_gap": "C", "defense": "D1", "epa": 50.0, "wpa": 1.0})
    return plays, gaps


class TestRushingSideGap:
    def test_side_gap_key(self):
        assert side_gap_key("KC", "Left", "b") == "KC-left-B"
        with pytest.raises(ValueError):
            side_gap_key("KC", "up", "A")
        with pytest.raises(ValueError):
            side_gap_key("KC", "left", "E")

    def test_designed_non_qb_filter(self):
        assert is_designed_non_qb_rush({"rusher_position": "RB"})
        assert not is_designed_non_qb_rush({"rusher_position": "QB"})
        assert not is_designed_non_qb_rush({"rusher_position": "RB", "qb_scramble": True})

    def test_rush_model_side_gap_proxy(self):
        plays, gaps = _rush_plays()
        fit = fit_rush_model(plays, tol=1e-6)
        assert fit.n_obs == len(plays) - 2
        assert "QB1" not in fit.random_effects["rusher"]
        tpa = fit.random_effects["team_side_gap"]
        assert max(tpa, key=tpa.get) == "KC-left-B"
        keys = [side_gap_key(*k) for k in gaps]
        assert _corr([tpa[k] for k in keys], list(gaps.values())) > 0.95
        totals = points_added(fit, "team_side_gap")
        for k, b in tpa.items():
            assert totals[k] == pytest.approx(b * fit.group_counts["team_side_gap"][k])
        with pytest.raises(KeyError):
            points_added(fit, "passer")

    def test_rush_model_wpa(self):
        plays, _ = _rush_plays(n=300)
        fit = fit_rush_model(plays, value="wpa", tol=1e-9)
        assert fit.n_obs == 300


# ==========================================================================
# PART 3: CLV, no-vig price, Brier
# ==========================================================================

class TestNoVigPrice:
    @pytest.mark.parametrize("method", ["shin", "odds_ratio", "logit", "power"])
    def test_symmetric_market(self, method):
        assert calculate_no_vig_price(1.91, 1.91, method) == pytest.approx((0.5, 0.5))

    @pytest.mark.parametrize("method", ["shin", "odds_ratio", "power"])
    def test_sums_to_one(self, method):
        pa, pb = calculate_no_vig_price(1.5, 2.6, method)
        assert pa + pb == pytest.approx(1.0, abs=1e-12)
        assert pa > pb

    def test_shin_two_way_equals_additive(self):
        pis = [1 / 1.5, 1 / 2.6]
        over = sum(pis) - 1.0
        pa, pb = calculate_no_vig_price(1.5, 2.6)
        assert pa == pytest.approx(pis[0] - over / 2)
        assert pb == pytest.approx(pis[1] - over / 2)

    def test_power_matches_engine_and_beats_crude_on_longshot(self):
        pa, pb = calculate_no_vig_price(1.5, 2.6, "power")
        ea, eb = devig_power(1.5, 2.6)
        assert (pa, pb) == pytest.approx((ea, eb))
        crude_b = (1 / 2.6) / (1 / 1.5 + 1 / 2.6)
        assert pb < crude_b                          # favourite-longshot correction

    def test_rejects_crude_and_bad_inputs(self):
        with pytest.raises(ValueError):
            calculate_no_vig_price(1.9, 1.9, "multiplicative")
        with pytest.raises(ValueError):
            calculate_no_vig_price(1.0, 1.9)
        with pytest.raises(ValueError):
            calculate_no_vig_price(2.1, 2.1, "shin")    # underround


class TestCLV:
    def test_clv_ratio(self):
        assert compute_clv(2.10, 0.5) == pytest.approx(1.05)
        assert compute_clv(1.90, 0.5) == pytest.approx(0.95)

    def test_clv_equals_offered_over_fair_closing_decimal(self):
        pa, _ = calculate_no_vig_price(1.80, 2.08, "power")
        assert compute_clv(1.95, pa) == pytest.approx(1.95 / (1.0 / pa))

    def test_clv_validation(self):
        with pytest.raises(ValueError):
            compute_clv(1.0, 0.5)
        with pytest.raises(ValueError):
            compute_clv(2.0, 1.0)

    def test_skill_detectable_in_50_bets(self):
        clv = [1.03 + (0.1 if i % 2 else -0.1) for i in range(50)]
        t, p = clv_skill_test(clv)
        assert t > 2.0 and p < 0.05

    def test_no_skill_not_significant(self):
        clv = [1.0 + (0.1 if i % 2 else -0.1) for i in range(50)]
        t, p = clv_skill_test(clv)
        assert t == pytest.approx(0.0, abs=1e-9) and p == pytest.approx(0.5)
        with pytest.raises(ValueError):
            clv_skill_test([1.0])


class TestBrier:
    def test_values(self):
        assert evaluate_forecast_skill([1.0, 0.0], [1, 0]) == 0.0
        assert evaluate_forecast_skill([0.5] * 4, [1, 0, 1, 0]) == pytest.approx(0.25)
        assert evaluate_forecast_skill([0.7, 0.2], [1, 0]) == pytest.approx(0.065)

    def test_better_calibration_scores_lower(self):
        rng = random.Random(1)
        true_p = [rng.uniform(0.2, 0.8) for _ in range(2000)]
        ys = [int(rng.random() < p) for p in true_p]
        assert evaluate_forecast_skill(true_p, ys) < evaluate_forecast_skill([0.5] * 2000, ys)

    def test_validation(self):
        with pytest.raises(ValueError):
            evaluate_forecast_skill([0.5], [1, 0])
        with pytest.raises(ValueError):
            evaluate_forecast_skill([], [])
        with pytest.raises(ValueError):
            evaluate_forecast_skill([1.2], [1])
        with pytest.raises(ValueError):
            evaluate_forecast_skill([0.5], [2])
