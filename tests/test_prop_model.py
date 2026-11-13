"""Player-prop model maths: weights, shrinkage, pace, distributions, projections, gate.

Distribution checks run against scipy when it is installed (skipped otherwise).
Run: python -m pytest tests/test_prop_model.py -v
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.core import prop_model as pm  # noqa: E402
from src.core.prop_model import (  # noqa: E402
    DIVERGENCE_ALL,
    PACE_HARMONIC,
    PACE_MULTIPLICATIVE,
    Projection,
    blowout_weight,
    count_cdf,
    effective_n,
    evaluate_side,
    ewma_weights,
    expected_possessions,
    gamma_cdf,
    minutes_weight,
    negbin_pmf,
    opponent_factor,
    owens_t,
    poisson_pmf,
    project_nba,
    project_nfl,
    shrink_rate,
    side_probabilities,
    skew_normal_cdf,
    skew_normal_params,
    weighted_moments,
)


# ==========================================================================
# Configuration and markets
# ==========================================================================
def test_decided_thresholds() -> None:
    assert pm.PROP_SIGMA_THRESHOLD == 0.65
    assert pm.PROP_MIN_SHRUNK_EV == 0.04
    assert pm.PACE_HARMONIC == "harmonic"
    assert pm.DEFAULT_DIVERGENCE_MODE == pm.DIVERGENCE_ALL == "all"
    assert pm.MAX_PROP_UNITS == 1.0
    assert (pm.TEAMMATE_OUT_BOOST, pm.TEAMMATE_OUT_BOOST_CAP) == (0.04, 0.08)
    assert pm.PRIOR_FULL_GAMES == 20


def test_eight_markets_split_by_league() -> None:
    assert set(pm.MARKETS_BY_LEAGUE["NBA"]) == {"player_points", "player_rebounds",
                                                "player_assists", "player_threes"}
    assert set(pm.MARKETS_BY_LEAGUE["NFL"]) == {"player_pass_yds", "player_rush_yds",
                                                "player_reception_yds", "player_receptions"}
    kinds = {k: m.kind for k, m in pm.PROP_MARKETS.items()}
    assert kinds["player_points"] == "continuous" and kinds["player_threes"] == "count"
    assert kinds["player_receptions"] == "count" and kinds["player_rush_yds"] == "continuous"


# ==========================================================================
# Weights, moments, shrinkage, pace
# ==========================================================================
def test_ewma_half_life() -> None:
    w = ewma_weights(20, 8.0)
    assert w[0] == 1.0 and w[8] == pytest.approx(0.5) and w[16] == pytest.approx(0.25)
    with pytest.raises(ValueError):
        ewma_weights(3, 0)


@pytest.mark.parametrize("margin,w", [(None, 1.0), (4, 1.0), (-14.9, 1.0), (15, 0.75),
                                      (-19, 0.75), (20, 0.5), (-31, 0.5)])
def test_blowout_weight(margin, w) -> None:
    assert blowout_weight(margin) == w


def test_minutes_deviation_weight() -> None:
    assert minutes_weight(34, 35) == 1.0
    assert minutes_weight(20, 35) == 0.5            # < 60% of typical
    assert minutes_weight(10, 0) == 1.0


def test_weighted_moments_equal_weights_match_sample() -> None:
    xs = [2.0, 4.0, 4.0, 5.0, 9.0]
    m, v, _ = weighted_moments(xs, [1.0] * 5)
    assert m == pytest.approx(4.8)
    assert v == pytest.approx(sum((x - 4.8) ** 2 for x in xs) / 4)
    assert effective_n([1.0] * 5) == pytest.approx(5.0)
    assert effective_n([1.0, 0.0, 0.0]) == pytest.approx(1.0)


def test_shrink_rate_beta_binomial() -> None:
    assert shrink_rate(10, 15, 0.36, 300) == pytest.approx((10 + 300 * 0.36) / 315)
    assert shrink_rate(0, 0, 0.36, 300) == 0.36
    assert shrink_rate(10, 15, 0.36, 0) == pytest.approx(10 / 15)


def test_opponent_factor_shrinks_toward_one() -> None:
    raw = opponent_factor(120.0, 100.0, 1e9, 10)
    assert raw == pytest.approx(1.2, abs=1e-6)
    assert opponent_factor(120.0, 100.0, 10, 10) == pytest.approx(1.1)
    assert opponent_factor(None, 100.0, 10, 10) == 1.0


def test_pace_methods() -> None:
    h = expected_possessions(96.0, 104.0, 100.0, PACE_HARMONIC)
    assert h == pytest.approx(2 / (1 / 96 + 1 / 104))
    assert h < 100.0                                          # harmonic <= arithmetic
    m = expected_possessions(96.0, 104.0, 100.0, PACE_MULTIPLICATIVE)
    assert m == pytest.approx(96 * 104 / 100)
    assert expected_possessions(100, 100, 100, PACE_HARMONIC) == pytest.approx(100)
    with pytest.raises(ValueError):
        expected_possessions(100, 100, 100, "average")


# ==========================================================================
# Distributions
# ==========================================================================
@pytest.mark.parametrize("mean,var", [(2.4, 2.4), (2.4, 4.1), (8.5, 13.0), (0.7, 1.1)])
def test_negbin_is_a_distribution_with_matching_moments(mean, var) -> None:
    ks = range(0, 200)
    ps = [negbin_pmf(k, mean, var) for k in ks]
    assert sum(ps) == pytest.approx(1.0, abs=1e-10)
    mu = sum(k * p for k, p in zip(ks, ps))
    assert mu == pytest.approx(mean, rel=1e-8)
    assert sum((k - mu) ** 2 * p for k, p in zip(ks, ps)) == pytest.approx(var, rel=1e-6)


def test_negbin_falls_back_to_poisson() -> None:
    assert negbin_pmf(3, 2.5, 2.5) == pytest.approx(poisson_pmf(3, 2.5))
    assert negbin_pmf(3, 2.5, 2.0) == pytest.approx(poisson_pmf(3, 2.5))


def test_distributions_match_scipy() -> None:
    stats = pytest.importorskip("scipy.stats")
    special = pytest.importorskip("scipy.special")
    for k in range(0, 12):
        assert poisson_pmf(k, 3.7) == pytest.approx(stats.poisson.pmf(k, 3.7), rel=1e-10)
        r = 2.4 ** 2 / (4.1 - 2.4)
        assert negbin_pmf(k, 2.4, 4.1) == pytest.approx(
            stats.nbinom.pmf(k, r, r / (r + 2.4)), rel=1e-9)
    for x in (0.5, 10.0, 24.5, 31.0, 60.0):
        assert gamma_cdf(x, 26.0, 60.0) == pytest.approx(
            stats.gamma.cdf(x, 26.0 ** 2 / 60.0, scale=60.0 / 26.0), abs=1e-10)
    for h, a in ((0.3, 0.5), (1.2, 2.0), (-0.7, 4.0), (2.5, -1.3)):
        assert owens_t(h, a) == pytest.approx(special.owens_t(h, a), abs=1e-9)
    xi, omega, alpha = skew_normal_params(60.0, 25.0, 0.6)
    dist = stats.skewnorm(alpha, loc=xi, scale=omega)
    assert dist.mean() == pytest.approx(60.0, rel=1e-9)
    assert dist.std() == pytest.approx(25.0, rel=1e-9)
    assert float(dist.stats(moments="s")) == pytest.approx(0.6, rel=1e-6)
    for x in (20.0, 55.5, 80.0):
        assert skew_normal_cdf(x, 60.0, 25.0, 0.6) == pytest.approx(dist.cdf(x), abs=1e-8)


def test_gamma_cdf_both_branches_monotone() -> None:
    xs = [0.1 * i for i in range(1, 400)]
    cs = [gamma_cdf(x, 5.0, 4.0) for x in xs]
    assert all(b >= a for a, b in zip(cs, cs[1:]))
    assert cs[0] < 1e-6 and cs[-1] == pytest.approx(1.0, abs=1e-9)
    assert gamma_cdf(-1.0, 5.0, 4.0) == 0.0


@pytest.mark.parametrize("kind,dist", [("count", None), ("continuous", "gamma"),
                                       ("continuous", "skew_normal")])
@pytest.mark.parametrize("line", [2.5, 3.0, 24.5, 25.0])
def test_side_probabilities_partition(kind, dist, line) -> None:
    mean, var = (3.2, 4.0) if kind == "count" and line < 10 else (25.0, 56.0)
    sp = side_probabilities(line, mean, var, kind, dist or "gamma", 0.3)
    assert sp["over"] + sp["push"] + sp["under"] == pytest.approx(1.0, abs=1e-9)
    if line % 1:
        assert sp["push"] == 0.0
    else:
        assert sp["push"] > 0.0


def test_count_side_probabilities_exact() -> None:
    sp = side_probabilities(2.0, 2.4, 2.4, "count")
    assert sp["under"] == pytest.approx(count_cdf(1, 2.4, 2.4))
    assert sp["push"] == pytest.approx(poisson_pmf(2, 2.4))
    sp = side_probabilities(1.5, 2.4, 2.4, "count")
    assert sp["under"] == pytest.approx(poisson_pmf(0, 2.4) + poisson_pmf(1, 2.4))


def test_continuous_whole_line_uses_continuity_correction() -> None:
    sp = side_probabilities(25.0, 25.0, 49.0, "continuous", "gamma")
    assert sp["push"] == pytest.approx(gamma_cdf(25.5, 25, 49) - gamma_cdf(24.5, 25, 49))
    with pytest.raises(ValueError):
        side_probabilities(25.3, 25.0, 49.0, "continuous")


# ==========================================================================
# Projections
# ==========================================================================
def _nba_game(i: int, pts=22, fg3m=2, fg3a=6, minutes=34.0, margin=4.0, reb=6, ast=5):
    fg2m = (pts - 3 * fg3m - 4) // 2
    return {"minutes": minutes, "pts": pts, "reb": reb, "ast": ast, "fg3m": fg3m, "fg3a": fg3a,
            "fgm": fg2m + fg3m, "fga": fg2m * 2 + fg3a, "ftm": 4, "fta": 5, "tov": 2,
            "team_margin": margin, "game_poss": 100.0, "game_date": f"2026-11-{20 - i:02d}"}


TEAM = {"pace": 100.0, "drtg": 115.0, "games": 12, "allowed_reb_100": 44.0,
        "allowed_ast_100": 26.0, "allowed_fg3m_100": 13.0}
LEAGUE = {"pace": 100.0, "drtg": 115.0, "allowed_reb_100": 44.0, "allowed_ast_100": 26.0,
          "allowed_fg3m_100": 13.0}


def test_nba_steady_player_projects_near_average() -> None:
    logs = [_nba_game(i) for i in range(15)]
    p = project_nba(logs, logs, "reb", TEAM, TEAM, LEAGUE)
    assert p.mean == pytest.approx(6.0, rel=1e-9)        # same minutes, pace, neutral opp
    assert p.kind == "count" and p.var >= p.mean
    assert p.details["possessions"] == pytest.approx(100.0)


def test_nba_hot_streak_is_regressed() -> None:
    base = [_nba_game(i) for i in range(3, 18)]
    hot = [_nba_game(i, pts=40, fg3m=8, fg3a=12) for i in range(3)]
    logs = hot + base
    p = project_nba(logs, logs, "pts", TEAM, TEAM, LEAGUE)
    recent3 = sum(g["pts"] for g in hot) / 3
    assert p.mean < recent3 - 10                           # nowhere near the streak
    s = p.details["shooting"]["fg3"]
    assert s["regressed"] < s["recent"]                     # 3P% pulled back to baseline
    assert p.kind == "continuous" and p.sd >= 0.30 * p.mean - 1e-9   # CV floor


def test_nba_opponent_and_pace_adjustments() -> None:
    logs = [_nba_game(i) for i in range(15)]
    soft = {**TEAM, "allowed_reb_100": 50.0}
    assert project_nba(logs, logs, "reb", TEAM, soft, LEAGUE).mean > \
        project_nba(logs, logs, "reb", TEAM, TEAM, LEAGUE).mean
    # The two formulas coincide whenever (a + b) / 2 == league, so avoid that case.
    slow, fast = {**TEAM, "pace": 96.0}, {**TEAM, "pace": 108.0}
    h = project_nba(logs, logs, "ast", slow, fast, LEAGUE, PACE_HARMONIC)
    m = project_nba(logs, logs, "ast", slow, fast, LEAGUE, PACE_MULTIPLICATIVE)
    assert h.details["possessions"] == pytest.approx(2 / (1 / 96 + 1 / 108), abs=0.01)
    assert m.details["possessions"] == pytest.approx(96 * 108 / 100, abs=0.01)
    assert h.mean != pytest.approx(m.mean)


def test_nba_blowouts_downweighted() -> None:
    normal = [_nba_game(i, reb=6) for i in range(12)]
    blow = [_nba_game(i, reb=14, margin=28) for i in range(3)]
    p = project_nba(blow + normal, blow + normal, "reb", TEAM, TEAM, LEAGUE)
    q = project_nba([{**g, "team_margin": 2} for g in blow] + normal,
                    blow + normal, "reb", TEAM, TEAM, LEAGUE)
    assert p.mean < q.mean


def test_nba_needs_three_games() -> None:
    with pytest.raises(ValueError):
        project_nba([_nba_game(0), _nba_game(1)], [], "pts", TEAM, TEAM, LEAGUE)


NFL_TEAM = {"plays_pg": 64.0, "def_plays_pg": 64.0, "pass_pg": 36.0, "rush_pg": 28.0,
            "neutral_pass_rate": 0.56, "def_pass_ypa": 6.4, "def_rush_ypc": 4.3, "games": 9}
NFL_LEAGUE = {"plays_pg": 64.0, "def_plays_pg": 64.0, "neutral_pass_rate": 0.56,
              "def_pass_ypa": 6.4, "def_rush_ypc": 4.3}


def test_nfl_receiving_projection_is_volume_times_efficiency() -> None:
    weeks = [{"targets": 9, "receptions": 6, "rec_yds": 81, "target_share": 0.25,
              "air_yards_share": 0.3} for _ in range(8)]
    p = project_nfl(weeks, "rec_yds", "WR", NFL_TEAM, NFL_TEAM, NFL_LEAGUE)
    d = p.details
    assert d["team_plays"] == pytest.approx(64.0)
    assert d["targets"] == pytest.approx(64.0 * 0.56 * 0.25)
    assert p.mean == pytest.approx(d["targets"] * d["yards_per_target"] * d["opp_factor"],
                                   rel=1e-3)
    r = project_nfl(weeks, "receptions", "WR", NFL_TEAM, NFL_TEAM, NFL_LEAGUE)
    assert r.kind == "count" and r.mean == pytest.approx(d["targets"] * r.details["catch_rate"],
                                                         rel=1e-3)


def test_nfl_rush_and_pass_projections() -> None:
    rb = [{"carries": 16, "rush_yds": 70} for _ in range(6)]
    p = project_nfl(rb, "rush_yds", "RB", NFL_TEAM, {**NFL_TEAM, "def_rush_ypc": 5.0},
                    NFL_LEAGUE)
    assert p.details["opp_factor"] > 1.0 and p.mean > 0
    qb = [{"pass_att": 35, "pass_yds": 245} for _ in range(6)]
    q = project_nfl(qb, "pass_yds", "QB", NFL_TEAM, NFL_TEAM, NFL_LEAGUE)
    assert 150 < q.mean < 300


# ==========================================================================
# Pricing and the gate
# ==========================================================================
def _proj(mean=25.0, sd=7.5, se=1.0, kind="continuous") -> Projection:
    return Projection(mean, sd * sd, kind, se, 10.0)


def test_no_market_edge_never_qualifies() -> None:
    e = evaluate_side(_proj(30.0), 24.5, "Over", 1.87, 0.52)     # retail worse than market
    assert e.market_ev < 0 and not e.qualified
    assert any("Market EV" in r for r in e.reasons)


def test_sigma_trigger_alone_qualifies_in_any_mode() -> None:
    e = evaluate_side(_proj(29.5, sd=7.0), 24.5, "Over", 2.05, 0.50, mode="any")
    assert e.z >= 0.65 and e.market_ev > 0
    assert e.qualified and any(t.startswith("SIGMA") for t in e.triggers)


def test_all_mode_requires_both_triggers() -> None:
    e_any = evaluate_side(_proj(29.6, sd=7.0), 24.5, "Over", 1.95, 0.52, min_shrunk_ev=0.5,
                          mode="any")
    e_all = evaluate_side(_proj(29.6, sd=7.0), 24.5, "Over", 1.95, 0.52, min_shrunk_ev=0.5,
                          mode=DIVERGENCE_ALL)
    assert e_any.qualified and not e_all.qualified


def test_default_gate_is_and() -> None:
    """Shrunk EV clears 4% but z < 0.65: qualifies only under the OR mode."""
    kw = dict(line=24.5, side="Over", offered_dec=2.05, market_prob=0.50)
    e_default = evaluate_side(_proj(28.5, sd=7.5), **kw)
    assert e_default.z < 0.65 and e_default.shrunk_ev >= 0.04
    assert not e_default.qualified
    assert evaluate_side(_proj(28.5, sd=7.5), **kw, mode="any").qualified
    both = evaluate_side(_proj(30.0, sd=7.5), **kw)
    assert both.qualified and len(both.triggers) == 2


def test_model_must_favour_side() -> None:
    e = evaluate_side(_proj(20.0), 24.5, "Over", 2.10, 0.45)
    assert not e.qualified and any("does not favour" in r for r in e.reasons)
    u = evaluate_side(_proj(19.0), 24.5, "Under", 2.05, 0.50)
    assert u.z > 0.65 and u.qualified


# ==========================================================================
# Teammate absences and the prior-season blend
# ==========================================================================
@pytest.mark.parametrize("n,boost", [(0, 0.0), (1, 0.04), (2, 0.08), (3, 0.08)])
def test_absence_boost_capped(n, boost) -> None:
    assert pm.absence_boost(n) == pytest.approx(boost)


def test_usage_boost_keeps_distribution_shape() -> None:
    cont = Projection(20.0, 36.0, "continuous", 1.6, 10.0, details={"a": 1})
    b = pm.apply_usage_boost(cont, 0.04, "why")
    assert b.mean == pytest.approx(20.8) and b.sd / b.mean == pytest.approx(0.3)   # CV kept
    assert b.se_mean == pytest.approx(1.664)
    assert b.details["usage_boost"] == 0.04 and b.details["usage_boost_reason"] == "why"
    cnt = Projection(5.0, 7.5, "count", 0.6, 10.0)
    c = pm.apply_usage_boost(cnt, 0.08)
    assert c.var / c.mean == pytest.approx(1.5)                                    # dispersion
    assert pm.apply_usage_boost(cnt, 0.0) is cnt


PRIOR = {"season": "2025-26", "min": 2400.0, "reb": 600.0, "ast": 200.0, "pts": 1500.0,
         "fgm": 550.0, "fga": 1100.0, "fg3m": 200.0, "fg3a": 470.0, "ftm": 200.0, "fta": 240.0}


def test_prior_season_blends_small_samples_linearly() -> None:
    six = [_nba_game(i, reb=3) for i in range(6)]
    cur = project_nba(six, six, "reb", TEAM, TEAM, LEAGUE)
    with_prior = project_nba(six, six, "reb", TEAM, TEAM, LEAGUE, prior=PRIOR)
    assert with_prior.details["prior_weight"] == pytest.approx(1 - 6 / 20)
    assert with_prior.details["prior_season"] == "2025-26"
    prior_rate = 600.0 / 2400.0 * 48.0 / LEAGUE["pace"]                # per possession
    cur_rate = cur.details["rate_per_poss"]
    expect = 0.3 * cur_rate + 0.7 * prior_rate
    assert with_prior.details["rate_per_poss"] == pytest.approx(expect, abs=1e-5)
    assert cur.mean < with_prior.mean                                  # prior reb rate higher
    assert with_prior.details["minutes"] == cur.details["minutes"]     # role never blended


def test_prior_ignored_from_twenty_games() -> None:
    logs = [_nba_game(i, reb=3) for i in range(22)]
    a = project_nba(logs, logs, "reb", TEAM, TEAM, LEAGUE)
    b = project_nba(logs, logs, "reb", TEAM, TEAM, LEAGUE, prior=PRIOR)
    assert b.details["prior_weight"] == 0.0 and a.mean == pytest.approx(b.mean)


def test_prior_anchors_shooting_baseline() -> None:
    six = [_nba_game(i, pts=22, fg3m=2, fg3a=6) for i in range(6)]
    league_anchor = project_nba(six, six, "fg3m", TEAM, TEAM, LEAGUE)
    good_shooter = project_nba(six, six, "fg3m", TEAM, TEAM, LEAGUE,
                               prior={**PRIOR, "fg3m": 300.0, "fg3a": 700.0})    # 42.9%
    assert good_shooter.details["shooting"]["fg3"]["regressed"] > \
        league_anchor.details["shooting"]["fg3"]["regressed"]


def test_shrinkage_and_blend_ordering() -> None:
    e = evaluate_side(_proj(30.0, se=1.0), 24.5, "Over", 2.05, 0.50)
    assert e.market_prob < e.fair_prob <= e.blend_prob < e.model_prob
    assert 0.0 <= e.shrinkage <= 1.0
    noisy = evaluate_side(_proj(30.0, se=6.0), 24.5, "Over", 2.05, 0.50)
    assert noisy.shrinkage < e.shrinkage                 # more uncertainty, more shrink
    pure = evaluate_side(_proj(30.0, se=1e-9), 24.5, "Over", 2.05, 0.50, model_weight=1.0)
    assert pure.blend_prob == pytest.approx(pure.model_prob)
    assert pure.fair_prob == pytest.approx(pure.model_prob, abs=1e-6)


def test_push_handled_conditionally() -> None:
    p = _proj(3.0, sd=math.sqrt(3.0), kind="count")
    e = evaluate_side(p, 3.0, "Over", 2.2, 0.40)
    assert e.p_push > 0
    assert e.model_prob == pytest.approx(e.p_win / (e.p_win + e.p_loss))


def test_evaluate_side_validation() -> None:
    with pytest.raises(ValueError):
        evaluate_side(_proj(), 24.5, "Yes", 2.0, 0.5)
    with pytest.raises(ValueError):
        evaluate_side(_proj(), 24.5, "Over", 1.0, 0.5)
    with pytest.raises(ValueError):
        evaluate_side(_proj(), 24.5, "Over", 2.0, 0.5, model_weight=0.0)
