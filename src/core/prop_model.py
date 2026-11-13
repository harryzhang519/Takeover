"""Player-prop fundamental projections and fair pricing. Pure stdlib, stateless.

Pipeline for one (player, market, line):

    history  -> EWMA weights (recency x blowout x minutes-deviation)
    project  -> NBA: minutes/48 x expected possessions x per-possession rate
                     x opponent allowance (shrunk), shooting splits regressed
                     toward a season baseline (beta-binomial)
                NFL: team plays (neutral-script pace) x pass/rush rate x
                     player share x shrunk efficiency x opponent allowance
    price    -> count props: Negative Binomial (Poisson when not overdispersed)
                yardage/points: Gamma (default) or Skew-Normal, with a
                continuity correction because outcomes are integers
    gate     -> market EV > 0 vs the sharp/consensus de-vig AND
                z >= PROP_SIGMA_THRESHOLD AND shrunk EV >= PROP_MIN_SHRUNK_EV
                (DEFAULT_DIVERGENCE_MODE = "all"; "any" makes the last two an OR)

"Shrunk": the model probability is first blended with the market at
PROP_MODEL_WEIGHT (as the game pipeline blends its model with the sharp line),
then pulled further toward the market by Baker & McHale (2013)
k* = e^2 / (e^2 + (b+1)^2 sigma^2), sigma being the blend's standard error
(delta method on the projection's standard error). See :mod:`src.core.risk`.

Constants marked APPROX are practical defaults, not values from a paper;
recalibrate them against settled prop results.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

from src.core.risk import calculate_baker_mchale_shrinkage

# ==========================================================================
# Markets
# ==========================================================================


@dataclass(frozen=True)
class PropMarket:
    key: str            # The-Odds-API market key
    league: str         # NBA | NFL
    stat: str           # stat field in normalized logs / box scores
    label: str          # short display label
    kind: str           # "count" | "continuous"


PROP_MARKETS: dict[str, PropMarket] = {m.key: m for m in (
    PropMarket("player_points", "NBA", "pts", "PTS", "continuous"),
    PropMarket("player_rebounds", "NBA", "reb", "REB", "count"),
    PropMarket("player_assists", "NBA", "ast", "AST", "count"),
    PropMarket("player_threes", "NBA", "fg3m", "3PM", "count"),
    PropMarket("player_pass_yds", "NFL", "pass_yds", "Pass Yds", "continuous"),
    PropMarket("player_rush_yds", "NFL", "rush_yds", "Rush Yds", "continuous"),
    PropMarket("player_reception_yds", "NFL", "rec_yds", "Rec Yds", "continuous"),
    PropMarket("player_receptions", "NFL", "receptions", "Rec", "count"),
)}
MARKETS_BY_LEAGUE: dict[str, tuple[str, ...]] = {
    lg: tuple(k for k, m in PROP_MARKETS.items() if m.league == lg) for lg in ("NBA", "NFL")}


def is_prop_market(market_type: str | None) -> bool:
    return market_type in PROP_MARKETS


# ==========================================================================
# Configuration (all overridable from the pipeline / .env)
# ==========================================================================
PROP_SIGMA_THRESHOLD: float = 0.65        # |projection - line| / game sd
PROP_MIN_SHRUNK_EV: float = 0.04          # 4% EV after Baker-McHale shrinkage
PROP_MIN_EV: float = 0.015                # desk floor (same 1.5% as game picks)
DIVERGENCE_ANY = "any"                    # sigma OR shrunk-EV
DIVERGENCE_ALL = "all"                    # sigma AND shrunk-EV (desk default)
DEFAULT_DIVERGENCE_MODE = DIVERGENCE_ALL
MAX_PROP_UNITS: float = 1.0               # hard cap per prop pick (game cap stays 2.0u)
# Discord bankroll rule: every official prop that clears the AND gate is a
# standard play, same as a game pick. Raw Kelly is kept as a diagnostic only.
MIN_PROP_OFFICIAL_UNITS: float = 1.0


def official_prop_units(raw_kelly_units: float) -> float:
    """Official stake for a gate-passing prop: raw Kelly clamped to
    [MIN_PROP_OFFICIAL_UNITS, MAX_PROP_UNITS] (both 1.0u, so a flat 1.00u)."""
    return round(min(MAX_PROP_UNITS, max(MIN_PROP_OFFICIAL_UNITS, raw_kelly_units)), 2)

# Key-teammate absence (APPROX): volume bump per newly-Out key teammate, capped.
TEAMMATE_OUT_BOOST: float = 0.04
TEAMMATE_OUT_BOOST_CAP: float = 0.08

# Small-sample prior (NBA): below this many current-season games, per-possession
# rates are blended linearly with the prior season (weight on prior = 1 - n/20).
PRIOR_FULL_GAMES: int = 20
PACE_HARMONIC = "harmonic"
PACE_MULTIPLICATIVE = "multiplicative"
DIST_GAMMA = "gamma"
DIST_SKEW_NORMAL = "skew_normal"

# Weighting (APPROX)
NBA_HALF_LIFE_GAMES: float = 8.0
NBA_WINDOW_GAMES: int = 25
NFL_HALF_LIFE_WEEKS: float = 4.0
NFL_WINDOW_WEEKS: int = 10
BLOWOUT_MARGINS: tuple[tuple[float, float], ...] = ((20.0, 0.5), (15.0, 0.75))
MINUTES_DEVIATION_RATIO: float = 0.6      # < 60% of typical minutes -> half weight
MINUTES_DEVIATION_WEIGHT: float = 0.5

# Shooting regression (APPROX): pseudo-attempts of prior strength.
NBA_LEAGUE_PCT: dict[str, float] = {"fg2": 0.540, "fg3": 0.360, "ft": 0.785}
NBA_SEASON_PRIOR_K: dict[str, float] = {"fg2": 150.0, "fg3": 250.0, "ft": 100.0}
NBA_RECENT_PRIOR_K: dict[str, float] = {"fg2": 200.0, "fg3": 300.0, "ft": 150.0}
OPP_FACTOR_K_GAMES: dict[str, float] = {"NBA": 10.0, "NFL": 4.0}

# NFL efficiency priors by position (APPROX), with pseudo-sample strength.
NFL_YPT_PRIOR: dict[str, float] = {"WR": 8.2, "TE": 7.2, "RB": 5.8}
NFL_CATCH_PRIOR: dict[str, float] = {"WR": 0.63, "TE": 0.70, "RB": 0.78}
NFL_YPC_PRIOR: float = 4.3
NFL_YPA_PRIOR: float = 6.9
NFL_PRIOR_K: dict[str, float] = {"ypt": 30.0, "catch": 40.0, "ypc": 60.0, "ypa": 150.0}

# Distribution floors (APPROX): typical game-to-game coefficient of variation, so a
# short or unusually steady sample cannot produce an over-confident spike.
MIN_CV: dict[str, float] = {"pts": 0.30, "pass_yds": 0.26, "rush_yds": 0.45, "rec_yds": 0.60}
MIN_DISPERSION: float = 1.0               # count variance >= mean (Poisson floor)
MAX_ABS_SKEW: float = 0.95
# Projection-mean uncertainty floor (APPROX): role / minutes / game-script risk
# that game logs alone do not reveal, as a fraction of the projected mean.
MEAN_SE_FLOOR_FRAC: float = 0.08
# Model weight vs the de-vigged market before Baker-McHale shrinkage (APPROX).
# 1.0 = pure model (spec as written); the game pipeline uses 0.20 for its model.
PROP_MODEL_WEIGHT: float = 0.35


# ==========================================================================
# Weights and moments
# ==========================================================================
def ewma_weights(n: int, half_life: float) -> list[float]:
    """Recency weights for n observations ordered MOST RECENT FIRST."""
    if half_life <= 0:
        raise ValueError("half_life must be > 0")
    return [0.5 ** (i / half_life) for i in range(n)]


def blowout_weight(final_margin: float | None) -> float:
    """Down-weight lopsided games (garbage-time inflation / early hook)."""
    if final_margin is None:
        return 1.0
    m = abs(final_margin)
    for threshold, w in BLOWOUT_MARGINS:
        if m >= threshold:
            return w
    return 1.0


def minutes_weight(minutes: float, typical: float) -> float:
    """Half weight for games far below typical minutes (foul trouble, injury exit)."""
    if typical <= 0:
        return 1.0
    return MINUTES_DEVIATION_WEIGHT if minutes < MINUTES_DEVIATION_RATIO * typical else 1.0


def weighted_mean(xs: Sequence[float], ws: Sequence[float]) -> float:
    tw = sum(ws)
    if tw <= 0:
        raise ValueError("weights must have a positive sum")
    return sum(x * w for x, w in zip(xs, ws)) / tw


def weighted_moments(xs: Sequence[float], ws: Sequence[float]) -> tuple[float, float, float]:
    """(mean, unbiased reliability-weighted variance, skewness)."""
    m = weighted_mean(xs, ws)
    v1, v2 = sum(ws), sum(w * w for w in ws)
    denom = v1 - v2 / v1
    if denom <= 0:
        return m, 0.0, 0.0
    var = sum(w * (x - m) ** 2 for x, w in zip(xs, ws)) / denom
    if var <= 0:
        return m, 0.0, 0.0
    m3 = sum(w * (x - m) ** 3 for x, w in zip(xs, ws)) / v1
    return m, var, m3 / var ** 1.5


def effective_n(ws: Sequence[float]) -> float:
    """Kish effective sample size (sum w)^2 / sum w^2."""
    s2 = sum(w * w for w in ws)
    return (sum(ws) ** 2) / s2 if s2 > 0 else 0.0


def shrink_rate(made: float, attempts: float, prior: float, strength: float) -> float:
    """Beta-binomial posterior mean (made + k*prior) / (attempts + k)."""
    if strength < 0 or attempts < 0:
        raise ValueError("attempts and strength must be >= 0")
    if attempts + strength <= 0:
        return prior
    return (made + strength * prior) / (attempts + strength)


def opponent_factor(allowed: float | None, league: float | None, games: float | None,
                    k_games: float) -> float:
    """Opponent allowance multiplier, shrunk toward 1.0 for small samples."""
    if not allowed or not league or not games or league <= 0:
        return 1.0
    return 1.0 + (allowed / league - 1.0) * games / (games + k_games)


def expected_possessions(pace_a: float, pace_b: float, league_pace: float,
                         method: str = PACE_HARMONIC) -> float:
    """Expected game possessions (NBA) or team plays (NFL) from two paces.

    harmonic (default):   2 / (1/a + 1/b)
    multiplicative:       a * b / league
    """
    if min(pace_a, pace_b, league_pace) <= 0:
        raise ValueError("paces must be > 0")
    if method == PACE_HARMONIC:
        return 2.0 / (1.0 / pace_a + 1.0 / pace_b)
    if method == PACE_MULTIPLICATIVE:
        return pace_a * pace_b / league_pace
    raise ValueError(f"unknown pace method {method!r}")


# ==========================================================================
# Distributions (stdlib)
# ==========================================================================
def _phi(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def poisson_pmf(k: int, lam: float) -> float:
    if k < 0:
        return 0.0
    if lam <= 0:
        return 1.0 if k == 0 else 0.0
    return math.exp(k * math.log(lam) - lam - math.lgamma(k + 1))


def negbin_pmf(k: int, mean: float, var: float) -> float:
    """NB with given mean and variance (var > mean); Poisson otherwise."""
    if k < 0:
        return 0.0
    if var <= mean * (1.0 + 1e-9):
        return poisson_pmf(k, mean)
    r = mean * mean / (var - mean)
    p = r / (r + mean)
    return math.exp(math.lgamma(k + r) - math.lgamma(r) - math.lgamma(k + 1)
                    + r * math.log(p) + k * math.log1p(-p))


def count_cdf(k: int, mean: float, var: float) -> float:
    """P(X <= k)."""
    if k < 0:
        return 0.0
    return min(1.0, sum(negbin_pmf(i, mean, var) for i in range(k + 1)))


def _gamma_p(a: float, x: float) -> float:
    """Regularized lower incomplete gamma P(a, x) (Numerical Recipes 6.2)."""
    if x <= 0:
        return 0.0
    gln = math.lgamma(a)
    if x < a + 1.0:
        term = total = 1.0 / a
        ap = a
        for _ in range(10_000):
            ap += 1.0
            term *= x / ap
            total += term
            if abs(term) < abs(total) * 1e-15:
                break
        return total * math.exp(-x + a * math.log(x) - gln)
    tiny = 1e-300
    b = x + 1.0 - a
    c, d = 1.0 / tiny, 1.0 / b
    h = d
    for i in range(1, 10_000):
        an = -i * (i - a)
        b += 2.0
        d = an * d + b
        d = tiny if abs(d) < tiny else d
        c = b + an / c
        c = tiny if abs(c) < tiny else c
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-15:
            break
    return 1.0 - math.exp(-x + a * math.log(x) - gln) * h


def gamma_cdf(x: float, mean: float, var: float) -> float:
    """CDF of a Gamma matched to mean and variance."""
    if mean <= 0 or var <= 0:
        raise ValueError("gamma needs mean > 0 and var > 0")
    shape, scale = mean * mean / var, var / mean
    return _gamma_p(shape, x / scale) if x > 0 else 0.0


def owens_t(h: float, a: float, steps: int = 400) -> float:
    """Owen's T(h, a) = 1/(2pi) int_0^a exp(-h^2 (1+x^2)/2) / (1+x^2) dx (Simpson)."""
    if a == 0:
        return 0.0
    if a < 0:
        return -owens_t(h, -a, steps)
    n = steps + steps % 2
    dx = a / n

    def f(x: float) -> float:
        return math.exp(-0.5 * h * h * (1.0 + x * x)) / (1.0 + x * x)

    s = f(0.0) + f(a) + sum((4 if i % 2 else 2) * f(i * dx) for i in range(1, n))
    return s * dx / 3.0 / (2.0 * math.pi)


def skew_normal_params(mean: float, sd: float, skew: float) -> tuple[float, float, float]:
    """(xi, omega, alpha) matching mean, sd and skewness (|skew| clipped to < 0.995)."""
    if sd <= 0:
        raise ValueError("sd must be > 0")
    g = max(-MAX_ABS_SKEW, min(MAX_ABS_SKEW, skew))
    ga = abs(g) ** (2.0 / 3.0)
    delta = math.copysign(math.sqrt((math.pi / 2.0) * ga / (ga + ((4.0 - math.pi) / 2.0)
                                                            ** (2.0 / 3.0))), g)
    alpha = delta / math.sqrt(1.0 - delta * delta)
    omega = sd / math.sqrt(1.0 - 2.0 * delta * delta / math.pi)
    xi = mean - omega * delta * math.sqrt(2.0 / math.pi)
    return xi, omega, alpha


def skew_normal_cdf(x: float, mean: float, sd: float, skew: float) -> float:
    xi, omega, alpha = skew_normal_params(mean, sd, skew)
    z = (x - xi) / omega
    return min(1.0, max(0.0, _phi(z) - 2.0 * owens_t(z, alpha)))


def _is_whole(x: float) -> bool:
    return abs(x - round(x)) < 1e-9


def side_probabilities(line: float, mean: float, var: float, kind: str,
                       dist: str = DIST_GAMMA, skew: float = 0.0) -> dict[str, float]:
    """{'over', 'push', 'under'} for an integer-valued stat at a whole or half line."""
    if abs(line * 2.0 - round(line * 2.0)) > 1e-9:
        raise ValueError(f"line must be whole or half, got {line}")
    if mean <= 0:
        raise ValueError("projection mean must be > 0")
    if kind == "count":
        if _is_whole(line):
            k = int(round(line))
            under, push = count_cdf(k - 1, mean, var), negbin_pmf(k, mean, var)
        else:
            under, push = count_cdf(int(math.floor(line)), mean, var), 0.0
        return {"over": max(0.0, 1.0 - under - push), "push": push, "under": under}
    if dist == DIST_GAMMA:
        cdf = lambda x: gamma_cdf(x, mean, var)  # noqa: E731
    elif dist == DIST_SKEW_NORMAL:
        cdf = lambda x: skew_normal_cdf(x, mean, math.sqrt(var), skew)  # noqa: E731
    else:
        raise ValueError(f"unknown continuous distribution {dist!r}")
    if _is_whole(line):                     # continuity correction: push = [L-0.5, L+0.5)
        under, upper = cdf(line - 0.5), cdf(line + 0.5)
        return {"over": max(0.0, 1.0 - upper), "push": max(0.0, upper - under), "under": under}
    u = cdf(line)
    return {"over": max(0.0, 1.0 - u), "push": 0.0, "under": u}


# ==========================================================================
# Projections
# ==========================================================================
@dataclass
class Projection:
    mean: float
    var: float
    kind: str                       # count | continuous
    se_mean: float                  # standard error of the projected mean
    n_eff: float
    skew: float = 0.0
    details: dict[str, Any] = field(default_factory=dict)

    @property
    def sd(self) -> float:
        return math.sqrt(self.var)


def _game_weights(logs: Sequence[dict[str, Any]], half_life: float,
                  minutes_key: str | None, margin_key: str | None) -> list[float]:
    ws = ewma_weights(len(logs), half_life)
    if margin_key:
        ws = [w * blowout_weight(g.get(margin_key)) for w, g in zip(ws, logs)]
    if minutes_key:
        mins = [float(g.get(minutes_key) or 0.0) for g in logs]
        typical = weighted_mean(mins, ewma_weights(len(mins), half_life))
        ws = [w * minutes_weight(m, typical) for w, m in zip(ws, mins)]
    return ws


def _spread(values: list[float], ws: list[float], mean: float, kind: str,
            stat: str) -> tuple[float, float]:
    """(variance, skew) of the projected game outcome, from the history's shape."""
    hm, hvar, hskew = weighted_moments(values, ws)
    if kind == "count":
        dispersion = hvar / hm if hm > 0 else MIN_DISPERSION
        return mean * max(dispersion, MIN_DISPERSION), 0.0
    cv = math.sqrt(hvar) / hm if hm > 0 and hvar > 0 else 0.0
    cv = max(cv, MIN_CV.get(stat, 0.25))
    return (cv * mean) ** 2, hskew


def absence_boost(n_out: int) -> float:
    """Volume multiplier increment for ``n_out`` newly-Out key teammates (+4% each, max +8%)."""
    return min(TEAMMATE_OUT_BOOST * max(n_out, 0), TEAMMATE_OUT_BOOST_CAP)


def apply_usage_boost(proj: Projection, boost: float, note: str = "") -> Projection:
    """Scale projected volume by (1 + boost), keeping the distribution's shape.

    Count props keep their dispersion (var scales with the mean); continuous
    props keep their coefficient of variation (var scales with the mean squared).
    """
    if boost <= 0:
        return proj
    r = 1.0 + boost
    var = proj.var * (r if proj.kind == "count" else r * r)
    details = {**proj.details, "usage_boost": round(boost, 4)}
    if note:
        details["usage_boost_reason"] = note
    return Projection(proj.mean * r, var, proj.kind, proj.se_mean * r, proj.n_eff, proj.skew,
                      details)


def project_nba(logs: Sequence[dict[str, Any]], season_logs: Sequence[dict[str, Any]],
                stat: str, team: dict[str, Any], opp: dict[str, Any],
                league: dict[str, Any], pace_method: str = PACE_HARMONIC,
                prior: dict[str, Any] | None = None) -> Projection:
    """NBA projection: minutes/48 x expected possessions x per-possession rate x opp factor.

    ``logs`` are the player's games MOST RECENT FIRST with keys minutes, pts,
    reb, ast, fg3m, fg3a, fgm, fga, ftm, fta, team_margin, game_poss.
    ``season_logs`` (all season games) set the shooting baselines.
    ``team``/``opp``/``league`` hold pace, drtg and allowed_{reb,ast,fg3m}_100.
    ``prior`` (optional) is the player's PRIOR-season totals (min, pts, reb, ast,
    fgm, fga, fg3m, fg3a, ftm, fta). It anchors the shooting baselines in place
    of the league average, and below PRIOR_FULL_GAMES current games blends the
    per-possession rates linearly toward the prior season. Minutes are never
    blended: they describe the current role.
    """
    if stat not in ("pts", "reb", "ast", "fg3m"):
        raise ValueError(f"unsupported NBA stat {stat!r}")
    played = [g for g in logs if (g.get("minutes") or 0) > 0][:NBA_WINDOW_GAMES]
    if len(played) < 3:
        raise ValueError("need at least 3 games played")
    ws = _game_weights(played, NBA_HALF_LIFE_GAMES, "minutes", "team_margin")
    lg_pace = float(league["pace"])
    poss = [float(g.get("game_poss") or lg_pace) for g in played]
    floor_poss = [float(g["minutes"]) / 48.0 * p for g, p in zip(played, poss)]
    w_floor = sum(w * fp for w, fp in zip(ws, floor_poss))

    def per_poss(key: str) -> float:
        return sum(w * float(g.get(key) or 0.0) for w, g in zip(ws, played)) / w_floor

    proj_min = weighted_mean([float(g["minutes"]) for g in played], ws)
    exp_poss = expected_possessions(float(team.get("pace") or lg_pace),
                                    float(opp.get("pace") or lg_pace), lg_pace, pace_method)
    k_opp = OPP_FACTOR_K_GAMES["NBA"]
    details: dict[str, Any] = {"minutes": round(proj_min, 2), "possessions": round(exp_poss, 2),
                               "pace_method": pace_method, "games": len(played)}
    has_prior = bool(prior) and float(prior.get("min") or 0.0) > 0.0
    w_cur = 1.0 if not has_prior or len(played) >= PRIOR_FULL_GAMES \
        else len(played) / PRIOR_FULL_GAMES

    def prior_rate(key: str) -> float:
        """Prior-season stat per on-floor possession (per-minute x 48 / league pace)."""
        return float(prior.get(key) or 0.0) / float(prior["min"]) * 48.0 / lg_pace

    def anchor(k: str) -> float:
        """Shooting baseline anchor: prior season shrunk to league, else league."""
        if not has_prior:
            return NBA_LEAGUE_PCT[k]
        made_att = {"fg2": (float(prior.get("fgm") or 0) - float(prior.get("fg3m") or 0),
                            float(prior.get("fga") or 0) - float(prior.get("fg3a") or 0)),
                    "fg3": (float(prior.get("fg3m") or 0), float(prior.get("fg3a") or 0)),
                    "ft": (float(prior.get("ftm") or 0), float(prior.get("fta") or 0))}[k]
        return shrink_rate(*made_att, NBA_LEAGUE_PCT[k], NBA_SEASON_PRIOR_K[k])

    if has_prior:
        details["prior_weight"] = round(1.0 - w_cur, 4)
        if prior.get("season"):
            details["prior_season"] = prior["season"]

    if stat == "pts":
        def split(rows: Iterable[dict[str, Any]], wts: Iterable[float] | None = None
                  ) -> dict[str, tuple[float, float]]:
            rows = list(rows)
            wl = list(wts) if wts is not None else [1.0] * len(rows)
            g = lambda k: sum(w * float(r.get(k) or 0.0) for w, r in zip(wl, rows))  # noqa
            return {"fg2": (g("fgm") - g("fg3m"), g("fga") - g("fg3a")),
                    "fg3": (g("fg3m"), g("fg3a")), "ft": (g("ftm"), g("fta"))}
        season = split(season_logs or played)
        recent = split(played, ws)
        pct, raw = {}, {}
        for k in ("fg2", "fg3", "ft"):
            base = shrink_rate(*season[k], anchor(k), NBA_SEASON_PRIOR_K[k])
            pct[k] = shrink_rate(*recent[k], base, NBA_RECENT_PRIOR_K[k])
            raw[k] = recent[k][0] / recent[k][1] if recent[k][1] > 0 else None
        a2 = (recent["fg2"][1]) / w_floor
        a3 = (recent["fg3"][1]) / w_floor
        aft = (recent["ft"][1]) / w_floor
        rate = 2.0 * a2 * pct["fg2"] + 3.0 * a3 * pct["fg3"] + aft * pct["ft"]
        if w_cur < 1.0:
            rate = w_cur * rate + (1.0 - w_cur) * prior_rate("pts")
        opp_f = opponent_factor(opp.get("drtg"), league.get("drtg"), opp.get("games"), k_opp)
        details["shooting"] = {k: {"recent": None if raw[k] is None else round(raw[k], 4),
                                   "regressed": round(pct[k], 4)} for k in pct}
    elif stat == "fg3m":
        made, att = (sum(w * float(g.get(k) or 0.0) for w, g in zip(ws, played))
                     for k in ("fg3m", "fg3a"))
        s_made = sum(float(g.get("fg3m") or 0.0) for g in (season_logs or played))
        s_att = sum(float(g.get("fg3a") or 0.0) for g in (season_logs or played))
        base = shrink_rate(s_made, s_att, anchor("fg3"), NBA_SEASON_PRIOR_K["fg3"])
        p3 = shrink_rate(made, att, base, NBA_RECENT_PRIOR_K["fg3"])
        att_rate = att / w_floor
        if w_cur < 1.0:
            att_rate = w_cur * att_rate + (1.0 - w_cur) * prior_rate("fg3a")
        rate = att_rate * p3
        opp_f = opponent_factor(opp.get("allowed_fg3m_100"), league.get("allowed_fg3m_100"),
                                opp.get("games"), k_opp)
        details["shooting"] = {"fg3": {"recent": round(made / att, 4) if att else None,
                                       "regressed": round(p3, 4)}}
    else:
        rate = per_poss(stat)
        if w_cur < 1.0:
            rate = w_cur * rate + (1.0 - w_cur) * prior_rate(stat)
        key = f"allowed_{stat}_100"
        opp_f = opponent_factor(opp.get(key), league.get(key), opp.get("games"), k_opp)

    mean = proj_min / 48.0 * exp_poss * rate * opp_f
    details.update({"rate_per_poss": round(rate, 5), "opp_factor": round(opp_f, 4)})
    values = [float(g.get(stat) or 0.0) for g in played]
    kind = "continuous" if stat == "pts" else "count"
    var, skew = _spread(values, ws, mean, kind, stat)
    n_eff = effective_n(ws)
    return Projection(mean, var, kind, _se_mean(var, n_eff, mean), n_eff, skew, details)


def _se_mean(var: float, n_eff: float, mean: float) -> float:
    """Sampling error of the mean, floored at MEAN_SE_FLOOR_FRAC of the projection."""
    return max(math.sqrt(var / max(n_eff, 1.0)), MEAN_SE_FLOOR_FRAC * mean)


def project_nfl(weeks: Sequence[dict[str, Any]], stat: str, position: str,
                team: dict[str, Any], opp: dict[str, Any], league: dict[str, Any],
                pace_method: str = PACE_HARMONIC) -> Projection:
    """NFL projection from neutral-script team volume x player share x efficiency.

    ``weeks`` MOST RECENT FIRST with keys targets, receptions, rec_yds,
    target_share, air_yards_share, carries, rush_yds, pass_att, completions,
    pass_yds. ``team``/``opp``/``league``: plays_pg, def_plays_pg,
    neutral_pass_rate, def_pass_ypa, def_rush_ypc, games.
    """
    if stat not in ("rec_yds", "receptions", "rush_yds", "pass_yds"):
        raise ValueError(f"unsupported NFL stat {stat!r}")
    rows = list(weeks)[:NFL_WINDOW_WEEKS]
    if len(rows) < 2:
        raise ValueError("need at least 2 weeks")
    ws = _game_weights(rows, NFL_HALF_LIFE_WEEKS, None, None)
    s = lambda k: sum(w * float(r.get(k) or 0.0) for w, r in zip(ws, rows))  # noqa: E731
    lg_plays = float(league["plays_pg"])
    plays = expected_possessions(float(team.get("plays_pg") or lg_plays),
                                 float(opp.get("def_plays_pg") or lg_plays), lg_plays,
                                 pace_method)
    pass_rate = float(team.get("neutral_pass_rate") or league.get("neutral_pass_rate") or 0.58)
    team_pass, team_rush = plays * pass_rate, plays * (1.0 - pass_rate)
    k_opp = OPP_FACTOR_K_GAMES["NFL"]
    pos = position.upper() if position else "WR"
    details: dict[str, Any] = {"team_plays": round(plays, 2), "pace_method": pace_method,
                               "neutral_pass_rate": round(pass_rate, 4), "weeks": len(rows)}
    pass_f = opponent_factor(opp.get("def_pass_ypa"), league.get("def_pass_ypa"),
                             opp.get("games"), k_opp)

    if stat in ("rec_yds", "receptions"):
        share = weighted_mean([float(r.get("target_share") or 0.0) for r in rows], ws)
        targets = team_pass * share
        prior_pos = pos if pos in NFL_YPT_PRIOR else "WR"
        details.update({"target_share": round(share, 4), "targets": round(targets, 2),
                        "air_yards_share": round(weighted_mean(
                            [float(r.get("air_yards_share") or 0.0) for r in rows], ws), 4)})
        if stat == "rec_yds":
            ypt = shrink_rate(s("rec_yds"), s("targets"), NFL_YPT_PRIOR[prior_pos],
                              NFL_PRIOR_K["ypt"])
            mean = targets * ypt * pass_f
            details.update({"yards_per_target": round(ypt, 3), "opp_factor": round(pass_f, 4)})
        else:
            catch = shrink_rate(s("receptions"), s("targets"), NFL_CATCH_PRIOR[prior_pos],
                                NFL_PRIOR_K["catch"])
            mean = targets * catch
            details.update({"catch_rate": round(catch, 4)})
    elif stat == "rush_yds":
        team_rush_hist = float(team.get("rush_pg") or team_rush)
        share = weighted_mean([float(r.get("carries") or 0.0) for r in rows], ws) / team_rush_hist
        carries = team_rush * share
        ypc = shrink_rate(s("rush_yds"), s("carries"), NFL_YPC_PRIOR, NFL_PRIOR_K["ypc"])
        rush_f = opponent_factor(opp.get("def_rush_ypc"), league.get("def_rush_ypc"),
                                 opp.get("games"), k_opp)
        mean = carries * ypc * rush_f
        details.update({"rush_share": round(share, 4), "carries": round(carries, 2),
                        "yards_per_carry": round(ypc, 3), "opp_factor": round(rush_f, 4)})
    else:
        team_pass_hist = float(team.get("pass_pg") or team_pass)
        share = min(1.0, weighted_mean([float(r.get("pass_att") or 0.0) for r in rows], ws)
                    / team_pass_hist)
        attempts = team_pass * share
        ypa = shrink_rate(s("pass_yds"), s("pass_att"), NFL_YPA_PRIOR, NFL_PRIOR_K["ypa"])
        mean = attempts * ypa * pass_f
        details.update({"pass_attempts": round(attempts, 2), "yards_per_attempt": round(ypa, 3),
                        "opp_factor": round(pass_f, 4)})
    if mean <= 0:
        raise ValueError("projection is zero (no usage)")
    values = [float(r.get(stat) or 0.0) for r in rows]
    kind = "count" if stat == "receptions" else "continuous"
    var, skew = _spread(values, ws, mean, kind, stat)
    n_eff = effective_n(ws)
    return Projection(mean, var, kind, _se_mean(var, n_eff, mean), n_eff, skew, details)


# ==========================================================================
# Pricing and the divergence gate
# ==========================================================================
@dataclass
class PropEvaluation:
    side: str
    p_win: float
    p_push: float
    p_loss: float
    model_prob: float               # P(win | action), raw model
    model_prob_sigma: float         # standard error of model_prob
    market_prob: float              # de-vigged market P(win | action)
    blend_prob: float               # market + w (model - market)
    shrinkage: float                # Baker-McHale k* on the blend
    fair_prob: float                # market + k* (blend - market): the probability used
    market_ev: float                # market_prob * dec - 1
    model_ev: float                 # model_prob * dec - 1
    shrunk_ev: float                # fair_prob * dec - 1
    z: float                        # (projection - line) / sd, signed toward this side
    triggers: list[str]
    qualified: bool
    reasons: list[str]


def evaluate_side(proj: Projection, line: float, side: str, offered_dec: float,
                  market_prob: float, dist: str = DIST_GAMMA,
                  sigma_threshold: float = PROP_SIGMA_THRESHOLD,
                  min_shrunk_ev: float = PROP_MIN_SHRUNK_EV,
                  mode: str = DEFAULT_DIVERGENCE_MODE,
                  model_weight: float = PROP_MODEL_WEIGHT) -> PropEvaluation:
    """Price one side and apply the model-vs-market gate.

    Probabilities are conditional on action (pushes refund), so they compare
    directly with a de-vigged two-way market and with ``offered_dec``.

        blend = market + w (model - market)            w = ``model_weight``
        k*    = Baker-McHale(blend, b, w * sigma_model)
        fair  = market + k* (blend - market)           -> EV, sizing
    """
    if not 0.0 < model_weight <= 1.0:
        raise ValueError("model_weight must be in (0, 1]")
    if side not in ("Over", "Under"):
        raise ValueError("side must be Over or Under")
    if not offered_dec > 1.0:
        raise ValueError("offered_dec must be > 1")
    if not 0.0 < market_prob < 1.0:
        raise ValueError("market_prob must be in (0, 1)")

    def model_p(mean: float) -> tuple[float, float, float]:
        m = max(mean, 1e-6)
        ratio = m / proj.mean           # count: keep dispersion; continuous: keep CV
        var = proj.var * (ratio if proj.kind == "count" else ratio * ratio)
        sp = side_probabilities(line, m, var, proj.kind, dist, proj.skew)
        win, loss = (sp["over"], sp["under"]) if side == "Over" else (sp["under"], sp["over"])
        return win, sp["push"], loss

    p_win, p_push, p_loss = model_p(proj.mean)
    action = p_win + p_loss
    p_model = p_win / action if action > 0 else 0.5
    lo, hi = model_p(proj.mean - proj.se_mean), model_p(proj.mean + proj.se_mean)
    cond = lambda t: t[0] / (t[0] + t[2]) if t[0] + t[2] > 0 else 0.5  # noqa: E731
    p_sigma = abs(cond(hi) - cond(lo)) / 2.0

    b = offered_dec - 1.0
    blend = market_prob + model_weight * (p_model - market_prob)
    k = calculate_baker_mchale_shrinkage(min(max(blend, 1e-9), 1 - 1e-9), b,
                                         model_weight * p_sigma)
    fair = market_prob + k * (blend - market_prob) if blend > market_prob else blend
    z = (proj.mean - line) / proj.sd if proj.sd > 0 else 0.0
    z_side = z if side == "Over" else -z

    market_ev, model_ev, shrunk_ev = (market_prob * offered_dec - 1.0,
                                      p_model * offered_dec - 1.0, fair * offered_dec - 1.0)
    triggers = []
    if z_side >= sigma_threshold:
        triggers.append(f"SIGMA {z_side:.2f}>={sigma_threshold:g}")
    if shrunk_ev >= min_shrunk_ev:
        triggers.append(f"SHRUNK_EV {shrunk_ev * 100:.2f}%>={min_shrunk_ev * 100:g}%")
    reasons = []
    if market_ev <= 0:
        reasons.append(f"Market EV {market_ev * 100:+.2f}% (retail not beating the market)")
    if z_side <= 0 or p_model <= market_prob:
        reasons.append("Model does not favour this side")
    need = 2 if mode == DIVERGENCE_ALL else 1
    if len(triggers) < need:
        reasons.append(f"Divergence below threshold (z {z_side:+.2f}, shrunk EV "
                       f"{shrunk_ev * 100:+.2f}%)")
    if shrunk_ev < PROP_MIN_EV:
        reasons.append(f"Shrunk EV {shrunk_ev * 100:.2f}% below {PROP_MIN_EV * 100:.1f}%")
    return PropEvaluation(side, p_win, p_push, p_loss, p_model, p_sigma, market_prob, blend, k,
                          fair, market_ev, model_ev, shrunk_ev, z_side, triggers, not reasons,
                          reasons)
