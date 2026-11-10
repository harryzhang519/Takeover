"""Derivative pricing: point spreads, half-point valuation and Skellam margins.

Pure stdlib, deterministic, stateless.

* NFL margin model after Stern, H. (1991), "On the Probability of Winning a
  Football Game as a Function of the Point Spread": the favourite's margin of
  victory is approximately Normal(spread, 13.86).
* Key-number push pricing: the Normal is discretised with a continuity
  correction and re-weighted so the long-run frequency of each key margin
  (3, 7, 6, 10, 4) matches its empirical NFL frequency. The frequencies in
  ``NFL_KEY_MARGIN_FREQ`` are approximate long-run league values, not a table
  reproduced from Stern (1991); recalibrate them from settled results.
* Low-scoring margins after Karlis, D. & Ntzoufras, I. (2009), "Bayesian
  modelling of football outcomes: using the Skellam's distribution for the
  goal difference": Z = X - Y with X ~ Poisson(lambda1), Y ~ Poisson(lambda2).

Line convention: a bettor's ``line`` is the handicap added to their side's
margin (team score minus opponent score). -3.0 wins when the margin > 3,
pushes at exactly 3; +2.5 wins when the margin > -2.5.
"""

from __future__ import annotations

import math

NFL_MARGIN_SIGMA: float = 13.86
NFL_MAX_MARGIN: int = 80

# Approximate long-run share of NFL games decided by exactly |margin| = k.
NFL_KEY_MARGIN_FREQ: dict[int, float] = {3: 0.150, 7: 0.093, 6: 0.058, 10: 0.058, 4: 0.050}
NFL_KEY_NUMBERS: tuple[int, ...] = (3, 7, 6, 10, 4)
NFL_TIE_FREQ: float = 0.003                     # overtime makes ties rare


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def _phi(x: float) -> float:
    """Standard normal CDF."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _check_decimal(dec: float) -> None:
    if not dec > 1.0:
        raise ValueError(f"Decimal odds must be > 1.0, got {dec}")


def _check_line(line: float) -> None:
    if abs(line * 2.0 - round(line * 2.0)) > 1e-9:
        raise ValueError(f"line must be a whole or half number, got {line}")


def _is_whole(x: float) -> bool:
    return abs(x - round(x)) < 1e-9


# --------------------------------------------------------------------------
# NFL: Stern (1991) normal margin model
# --------------------------------------------------------------------------
def nfl_spread_to_win_prob(spread: float, sigma: float = NFL_MARGIN_SIGMA) -> float:
    """P(win) = Phi(spread / sigma) (Stern 1991).

    ``spread`` is the team's expected margin, i.e. the points by which it is
    favoured (+3.0 for a 3-point favourite, -3.0 for a 3-point underdog).
    """
    if sigma <= 0.0:
        raise ValueError(f"sigma must be > 0, got {sigma}")
    return _phi(spread / sigma)


def _normal_mass(m: int, mu: float, sigma: float) -> float:
    """P(m - 0.5 < X < m + 0.5) for X ~ N(mu, sigma): Stern's continuity correction."""
    return _phi((m + 0.5 - mu) / sigma) - _phi((m - 0.5 - mu) / sigma)


def _margin_weights(sigma: float) -> dict[int, float]:
    """Per-|margin| multipliers so key margins hit their empirical frequency at mu = 0."""
    w: dict[int, float] = {}
    for k, freq in NFL_KEY_MARGIN_FREQ.items():
        w[k] = freq / (_normal_mass(k, 0.0, sigma) + _normal_mass(-k, 0.0, sigma))
    w[0] = NFL_TIE_FREQ / _normal_mass(0, 0.0, sigma)
    return w


def nfl_margin_pmf(expected_margin: float, sigma: float = NFL_MARGIN_SIGMA) -> dict[int, float]:
    """Key-number-weighted pmf of the integer final margin, summing to 1."""
    if sigma <= 0.0:
        raise ValueError(f"sigma must be > 0, got {sigma}")
    w = _margin_weights(sigma)
    raw = {m: _normal_mass(m, expected_margin, sigma) * w.get(abs(m), 1.0)
           for m in range(-NFL_MAX_MARGIN, NFL_MAX_MARGIN + 1)}
    total = sum(raw.values())
    return {m: v / total for m, v in raw.items()}


def nfl_spread_outcome_probs(line: float, expected_margin: float,
                             sigma: float = NFL_MARGIN_SIGMA) -> tuple[float, float, float]:
    """(win, push, loss) for a bettor laying/taking ``line`` on a side."""
    _check_line(line)
    pmf = nfl_margin_pmf(expected_margin, sigma)
    win = push = loss = 0.0
    for m, p in pmf.items():
        r = m + line
        if abs(r) < 1e-9:
            push += p
        elif r > 0.0:
            win += p
        else:
            loss += p
    return win, push, loss


def spread_bet_ev(line: float, offered_dec: float, expected_margin: float,
                  sigma: float = NFL_MARGIN_SIGMA) -> float:
    """EV per unit staked; a push returns the stake (EV 0 on that branch)."""
    _check_decimal(offered_dec)
    win, _, loss = nfl_spread_outcome_probs(line, expected_margin, sigma)
    return win * (offered_dec - 1.0) - loss


def value_half_point(current_spread: float, target_spread: float,
                     expected_margin: float | None = None,
                     sigma: float = NFL_MARGIN_SIGMA) -> dict:
    """Marginal value of moving a spread from ``current_spread`` to ``target_spread``.

    Buying points (target > current, e.g. -3.5 -> -3.0) improves the bettor's
    position; selling points worsens it. Every integer margin the move passes
    over or lands on changes a loss into a push or a push into a win (or the
    reverse). ``expected_margin`` defaults to the market-implied margin
    ``-current_spread`` (the current line is taken as fair).

    Returns win/push/loss at both lines, their deltas, the integer margins
    crossed and which of those are key numbers.
    """
    _check_line(current_spread)
    _check_line(target_spread)
    mu = -current_spread if expected_margin is None else expected_margin
    w0, p0, l0 = nfl_spread_outcome_probs(current_spread, mu, sigma)
    w1, p1, l1 = nfl_spread_outcome_probs(target_spread, mu, sigma)
    def _outcome(r: float) -> int:                  # 1 win, 0 push, -1 loss
        return 0 if abs(r) < 1e-9 else (1 if r > 0.0 else -1)

    lo, hi = sorted((-current_spread, -target_spread))
    crossed = [m for m in range(math.floor(lo) - 1, math.ceil(hi) + 2)
               if _outcome(m + current_spread) != _outcome(m + target_spread)]
    if target_spread > current_spread:
        direction = "buy"
    elif target_spread < current_spread:
        direction = "sell"
    else:
        direction = "none"
    return {
        "current_spread": current_spread,
        "target_spread": target_spread,
        "expected_margin": mu,
        "direction": direction,
        "margins_crossed": crossed,
        "key_numbers_crossed": [m for m in crossed if abs(m) in NFL_KEY_NUMBERS],
        "win_current": w0, "push_current": p0, "loss_current": l0,
        "win_target": w1, "push_target": p1, "loss_target": l1,
        "delta_win": w1 - w0, "delta_push": p1 - p0, "delta_loss": l1 - l0,
        # Expected-stake value of the move at even money: d(win) - d(loss).
        "value_per_unit_even_money": (w1 - w0) - (l1 - l0),
    }


def compare_spread_offers(line_a: float, dec_a: float, line_b: float, dec_b: float,
                          expected_margin: float, sigma: float = NFL_MARGIN_SIGMA) -> dict:
    """Price two spread offers on the same side with exact push probabilities.

    Example: retail -3.0 at -115 vs Pinnacle -3.5 at -108.
    """
    ev_a = spread_bet_ev(line_a, dec_a, expected_margin, sigma)
    ev_b = spread_bet_ev(line_b, dec_b, expected_margin, sigma)
    return {
        "a": {"line": line_a, "dec": dec_a, "ev": ev_a,
              "probs": nfl_spread_outcome_probs(line_a, expected_margin, sigma)},
        "b": {"line": line_b, "dec": dec_b, "ev": ev_b,
              "probs": nfl_spread_outcome_probs(line_b, expected_margin, sigma)},
        "better": "a" if ev_a > ev_b else "b",
        "ev_edge": abs(ev_a - ev_b),
    }


# --------------------------------------------------------------------------
# Skellam distribution (Karlis & Ntzoufras 2009)
# --------------------------------------------------------------------------
def _check_lambdas(lambda1: float, lambda2: float) -> None:
    if not (lambda1 > 0.0 and lambda2 > 0.0):
        raise ValueError(f"lambdas must be > 0, got {lambda1}, {lambda2}")


def _log_bessel_i(n: int, x: float) -> float:
    """log I_n(x) for integer n >= 0, x > 0, from the power series in log space.

    I_n(x) = sum_m (x/2)^(2m+n) / (m! (m+n)!).
    """
    log_half = math.log(0.5 * x)
    terms: list[float] = []
    m = 0
    while True:
        t = (2 * m + n) * log_half - math.lgamma(m + 1) - math.lgamma(m + n + 1)
        terms.append(t)
        # Terms peak near m ~ x/2; stop once well past it and negligible.
        if m > x and t < max(terms) - 40.0:
            break
        m += 1
        if m > 100_000:
            raise ArithmeticError("Bessel series did not converge")
    top = max(terms)
    return top + math.log(sum(math.exp(t - top) for t in terms))


def skellam_pmf(z: int, lambda1: float, lambda2: float) -> float:
    """P(Z = z), Z = X - Y, X ~ Poisson(lambda1), Y ~ Poisson(lambda2).

    P = exp(-(l1 + l2)) (l1/l2)^(z/2) I_|z|(2 sqrt(l1 l2)).
    """
    _check_lambdas(lambda1, lambda2)
    z = int(z)
    x = 2.0 * math.sqrt(lambda1 * lambda2)
    log_p = (-(lambda1 + lambda2) + 0.5 * z * math.log(lambda1 / lambda2)
             + _log_bessel_i(abs(z), x))
    return math.exp(log_p)


def skellam_mean(lambda1: float, lambda2: float) -> float:
    """E(Z) = lambda1 - lambda2."""
    _check_lambdas(lambda1, lambda2)
    return lambda1 - lambda2


def skellam_variance(lambda1: float, lambda2: float) -> float:
    """Var(Z) = lambda1 + lambda2."""
    _check_lambdas(lambda1, lambda2)
    return lambda1 + lambda2


def _skellam_support(lambda1: float, lambda2: float) -> range:
    sd = math.sqrt(lambda1 + lambda2)
    half = int(math.ceil(abs(lambda1 - lambda2) + 15.0 * sd + 20.0))
    return range(-half, half + 1)


def skellam_distribution(lambda1: float, lambda2: float) -> dict[int, float]:
    """Truncated Skellam pmf over all margins with non-negligible mass."""
    _check_lambdas(lambda1, lambda2)
    return {z: skellam_pmf(z, lambda1, lambda2) for z in _skellam_support(lambda1, lambda2)}


def skellam_cdf(z: int, lambda1: float, lambda2: float) -> float:
    """P(Z <= z)."""
    return sum(p for k, p in skellam_distribution(lambda1, lambda2).items() if k <= z)


def compute_skellam_lambdas(mu: float, home_adv: float, attack: dict[str, float],
                            defense: dict[str, float], home_team: str, away_team: str,
                            enforce_zero_sum: bool = True,
                            tol: float = 1e-9) -> tuple[float, float]:
    """Log-linear Poisson rates for one game -> ``(lambda1_home, lambda2_away)``.

        log(lambda1) = mu + H + A[home] + D[away]
        log(lambda2) = mu + A[away] + D[home]

    ``attack``/``defense`` hold every team's parameter; with
    ``enforce_zero_sum`` the identifiability constraints sum(A) = 0 and
    sum(D) = 0 are checked to ``tol``.
    """
    for name, params in (("attack", attack), ("defense", defense)):
        for team in (home_team, away_team):
            if team not in params:
                raise KeyError(f"{team!r} missing from {name} parameters")
        if enforce_zero_sum and abs(sum(params.values())) > tol:
            raise ValueError(f"{name} parameters must sum to 0, got {sum(params.values())}")
    log_l1 = mu + home_adv + attack[home_team] + defense[away_team]
    log_l2 = mu + attack[away_team] + defense[home_team]
    return math.exp(log_l1), math.exp(log_l2)


def skellam_outcome_probs(lambda1: float, lambda2: float) -> dict[str, float]:
    """1X2 probabilities: home win (Z > 0), draw (Z = 0), away win (Z < 0)."""
    dist = skellam_distribution(lambda1, lambda2)
    return {"home": sum(p for z, p in dist.items() if z > 0),
            "draw": dist[0],
            "away": sum(p for z, p in dist.items() if z < 0)}


def skellam_handicap_probs(line: float, lambda1: float,
                           lambda2: float) -> tuple[float, float, float]:
    """(win, push, loss) for the HOME side at a whole or half handicap ``line``.

    Home wins when Z + line > 0 and pushes when Z + line == 0 (whole lines
    only), e.g. -1.0 pushes on a one-goal home win. For the away side use
    the negated line with (lambda2, lambda1).
    """
    _check_line(line)
    win = push = loss = 0.0
    for z, p in skellam_distribution(lambda1, lambda2).items():
        r = z + line
        if abs(r) < 1e-9:
            push += p
        elif r > 0.0:
            win += p
        else:
            loss += p
    return win, push, loss


def skellam_push_prob(line: float, lambda1: float, lambda2: float) -> float:
    """Exact push probability P(Z = -line); 0 for half lines."""
    _check_line(line)
    if not _is_whole(line):
        return 0.0
    return skellam_pmf(-int(round(line)), lambda1, lambda2)


def asian_handicap_ev(line: float, offered_dec: float, lambda1: float, lambda2: float) -> float:
    """EV per unit for the HOME side on an Asian handicap, including quarter lines.

    Quarter lines (e.g. -0.25, -0.75) split the stake equally across the two
    neighbouring half/whole lines; whole lines refund on a push.
    """
    _check_decimal(offered_dec)
    if abs(line * 4.0 - round(line * 4.0)) > 1e-9:
        raise ValueError(f"Asian line must be a multiple of 0.25, got {line}")
    q = round(line * 4.0) % 2
    legs = [line - 0.25, line + 0.25] if q == 1 else [line]
    ev = 0.0
    for leg in legs:
        win, _, loss = skellam_handicap_probs(leg, lambda1, lambda2)
        ev += (win * (offered_dec - 1.0) - loss) / len(legs)
    return ev


def poisson_totals_probs(line: float, lambda1: float,
                         lambda2: float) -> tuple[float, float, float]:
    """(over, push, under) for total goals; X + Y ~ Poisson(lambda1 + lambda2)."""
    _check_lambdas(lambda1, lambda2)
    _check_line(line)
    if line < 0.0:
        raise ValueError(f"total line must be >= 0, got {line}")
    lam = lambda1 + lambda2
    under = push = 0.0
    for k in range(0, int(math.floor(line)) + 1):
        pk = math.exp(-lam + k * math.log(lam) - math.lgamma(k + 1))
        if abs(k - line) < 1e-9:
            push += pk
        else:
            under += pk
    return max(0.0, 1.0 - under - push), push, under
