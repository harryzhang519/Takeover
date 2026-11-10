"""Fundamental matchup model and Bayesian pricing blender. Fully deterministic.

Every number comes from team_ratings / team_schedule / injury_reports (ESPN
harvest) or from an explicit fixture in mock mode. No judgement calls, no
language-model opinions about teams.

NBA projection (home perspective):
    pace    = home_pace * away_pace / league_avg_pace
    margin  = (home_adjNet - away_adjNet) * pace / 100 + 2.3 HCA + situational
    P(win)  = Phi(margin / 11.5)
NFL projection:
    margin  = home_SRS - away_SRS + 1.8 HFA + situational,  P(win) = Phi(margin / 13.5)

Ratings are shrunk toward league average (0) by games / (games + K) with
K = 10 (NBA) and 4 (NFL), so a 3-game sample cannot dominate the price.

Bayesian blend: 0.65 sharp de-vig anchor + 0.15 Kalshi mid + 0.20 model,
renormalised over whichever inputs exist.
"""

from __future__ import annotations

import math
import sqlite3
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Any

from src.core import database as db
from src.core.engine import HITL_APPROVED, HITL_PENDING
from src.data.team_registry import REGISTRY, try_resolve_team

# ---- Parameters (desk-tunable, documented) --------------------------------
HOME_ADVANTAGE = {"NBA": 2.3, "NFL": 1.8}
SIGMA = {"NBA": 11.5, "NFL": 13.5}
SHRINK_K = {"NBA": 10.0, "NFL": 4.0}
NBA_AWAY_B2B = 2.1               # pts credited to home when the away team is on a B2B
NBA_HOME_B2B = 1.4               # pts debited from home when the home team is on a B2B
NBA_ALTITUDE = 0.8               # DEN/UTA home vs sea-level visitor
NFL_ALTITUDE = 0.5               # DEN home or Mexico City neutral site
NFL_POST_BYE = 1.0               # team coming off a bye (>= 12 days rest)
NFL_SHORT_WEEK = 1.0             # team on <= 4 days rest vs a normally rested opponent
TRAVEL_PER_TZ = 0.3              # away penalty per time zone crossed beyond the first
STALE_RATINGS_H = 36

WEIGHTS = {"sharp": 0.65, "kalshi": 0.15, "model": 0.20}
MODEL_DIVERGENCE = 0.05
QUESTIONABLE_STATUSES = frozenset({"questionable", "day-to-day", "game-time decision", "gtd"})


def normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


# ==========================================================================
# Situational inputs
# ==========================================================================
@dataclass
class RestInfo:
    days_rest: int | None
    back_to_back: bool = False
    three_in_four: bool = False
    last_game: str | None = None

    def label(self) -> str:
        if self.days_rest is None:
            return "rest n/a"
        tags = [f"{self.days_rest}d rest"]
        if self.back_to_back:
            tags.append("B2B")
        if self.three_in_four:
            tags.append("3-in-4")
        return " ".join(tags)


def _et_date(ts: str):
    return (db.parse_ts(ts) - timedelta(hours=5)).date()


def rest_info(conn: sqlite3.Connection, canonical_id: str, game_time: str) -> RestInfo:
    """Days of rest before ``game_time`` from team_schedule (US-Eastern dates).

    0 days = back-to-back. 3-in-4 = this game is the third in four nights.
    """
    target = _et_date(game_time)
    cutoff = db.iso(db.parse_ts(game_time) - timedelta(hours=3))
    rows = conn.execute("SELECT game_time FROM team_schedule WHERE canonical_id = ? AND "
                        "game_time < ? ORDER BY game_time DESC LIMIT 4",
                        (canonical_id, cutoff)).fetchall()
    if not rows:
        return RestInfo(None)
    last = _et_date(rows[0]["game_time"])
    days = max((target - last).days - 1, 0)
    recent = sum(1 for r in rows if 1 <= (target - _et_date(r["game_time"])).days <= 3)
    return RestInfo(days, days == 0, recent >= 2, rows[0]["game_time"])


def load_ratings(conn: sqlite3.Connection, canonical_id: str) -> dict[str, Any] | None:
    row = conn.execute("SELECT * FROM team_ratings WHERE canonical_id = ?",
                       (canonical_id,)).fetchone()
    return dict(row) if row else None


def league_avg_pace(conn: sqlite3.Connection) -> float | None:
    row = conn.execute("SELECT AVG(pace) FROM team_ratings WHERE sport = 'NBA' AND pace > 0"
                       ).fetchone()
    return float(row[0]) if row and row[0] else None


def key_injuries(conn: sqlite3.Connection, canonical_id: str) -> list[dict[str, Any]]:
    return [dict(r) for r in conn.execute(
        "SELECT player, position, status, is_key, key_reason FROM injury_reports WHERE "
        "canonical_id = ? AND status NOT IN ('Active') ORDER BY is_key DESC, player",
        (canonical_id,))]


# ==========================================================================
# Projections
# ==========================================================================
def _first(row: dict[str, Any], *keys: str) -> float | None:
    """First non-null value among ``keys`` (a NULL column must not mask the fallback)."""
    return next((row[k] for k in keys if row.get(k) is not None), None)


def _shrunk(value: float | None, games: int | None, sport: str) -> float:
    if value is None or not games:
        return 0.0
    return value * games / (games + SHRINK_K[sport])


def _travel(home_id: str, away_id: str) -> tuple[float, str | None]:
    tz = abs(REGISTRY[home_id].utc_offset_std - REGISTRY[away_id].utc_offset_std)
    if tz < 2:
        return 0.0, None
    pts = TRAVEL_PER_TZ * (tz - 1)
    return pts, f"{REGISTRY[away_id].abbr} travels {tz} time zones"


def _ratings_pair(home_id: str, away_id: str, conn: sqlite3.Connection | None,
                  ratings: dict[str, Any] | None) -> tuple[dict, dict]:
    if ratings is not None:
        h, a = ratings.get(home_id), ratings.get(away_id)
    elif conn is not None:
        h, a = load_ratings(conn, home_id), load_ratings(conn, away_id)
    else:
        raise ValueError("pass conn or ratings")
    if not h or not a:
        missing = [t for t, r in ((home_id, h), (away_id, a)) if not r]
        raise LookupError(f"no ratings for {', '.join(missing)} (run --update-stats)")
    return h, a


def _finish(sport: str, home_id: str, away_id: str, base: float,
            adjustments: list[tuple[str, float]], extra: dict[str, Any]) -> dict[str, Any]:
    margin = base + sum(p for _, p in adjustments)
    p_home = normal_cdf(margin / SIGMA[sport])
    return {"sport": sport, "home_id": home_id, "away_id": away_id,
            "base_margin": round(base, 3), "adjustments": [(l, round(p, 2)) for l, p in adjustments],
            "projected_margin": round(margin, 3), "projected_home_spread": round(-margin, 1),
            "sigma": SIGMA[sport], "home_win_prob": p_home, "away_win_prob": 1.0 - p_home,
            **extra}


def project_nba_matchup(home_id: str, away_id: str, rest_home: int | None,
                        rest_away: int | None, *, conn: sqlite3.Connection | None = None,
                        ratings: dict[str, Any] | None = None,
                        league_pace: float | None = None,
                        venue_altitude_ft: int | None = None) -> dict[str, Any]:
    """Project an NBA game (home perspective). See module docstring for the formula."""
    h, a = _ratings_pair(home_id, away_id, conn, ratings)
    lg = league_pace or (ratings or {}).get("_league_avg_pace") or (
        league_avg_pace(conn) if conn else None) or 99.0
    pace = (h["pace"] or lg) * (a["pace"] or lg) / lg
    net_h = _shrunk(_first(h, "adj_net_rtg", "net_rtg"), h.get("games"), "NBA")
    net_a = _shrunk(_first(a, "adj_net_rtg", "net_rtg"), a.get("games"), "NBA")
    base = (net_h - net_a) * pace / 100.0
    adj: list[tuple[str, float]] = [("Home court", HOME_ADVANTAGE["NBA"])]
    if rest_away == 0:
        adj.append((f"{REGISTRY[away_id].abbr} on back-to-back", NBA_AWAY_B2B))
    if rest_home == 0:
        adj.append((f"{REGISTRY[home_id].abbr} on back-to-back", -NBA_HOME_B2B))
    home_alt = venue_altitude_ft if venue_altitude_ft is not None \
        else REGISTRY[home_id].arena_altitude_ft
    if home_alt >= 4000 and not REGISTRY[away_id].high_altitude:
        adj.append((f"Altitude ({home_alt:,} ft)", NBA_ALTITUDE))
    travel, note = _travel(home_id, away_id)
    if travel:
        adj.append((note, travel))                          # type: ignore[arg-type]
    return _finish("NBA", home_id, away_id, base, adj, {
        "expected_pace": round(pace, 2), "league_avg_pace": round(lg, 2),
        "home_rating": round(net_h, 2), "away_rating": round(net_a, 2),
        "rating_delta": round(net_h - net_a, 2), "rating_label": "Adj NetRtg (shrunk)",
        "games": (h.get("games"), a.get("games")), "updated_at": h.get("updated_at"),
        "rest": (rest_home, rest_away)})


def project_nfl_matchup(home_id: str, away_id: str, rest_home: int | None,
                        rest_away: int | None, *, conn: sqlite3.Connection | None = None,
                        ratings: dict[str, Any] | None = None,
                        venue_altitude_ft: int | None = None) -> dict[str, Any]:
    """Project an NFL game: SRS delta + HFA + bye / short-week / altitude / travel."""
    h, a = _ratings_pair(home_id, away_id, conn, ratings)
    net_h = _shrunk(_first(h, "srs", "point_diff_pg"), h.get("games"), "NFL")
    net_a = _shrunk(_first(a, "srs", "point_diff_pg"), a.get("games"), "NFL")
    adj: list[tuple[str, float]] = [("Home field", HOME_ADVANTAGE["NFL"])]
    for tid, rest, sign in ((home_id, rest_home, 1.0), (away_id, rest_away, -1.0)):
        other = rest_away if sign > 0 else rest_home
        if rest is not None and rest >= 12:
            adj.append((f"{REGISTRY[tid].abbr} off bye", sign * NFL_POST_BYE))
        elif rest is not None and rest <= 4 and (other is None or other >= 6):
            adj.append((f"{REGISTRY[tid].abbr} on short week", -sign * NFL_SHORT_WEEK))
    alt = venue_altitude_ft if venue_altitude_ft is not None \
        else REGISTRY[home_id].arena_altitude_ft
    if alt >= 4000 and not REGISTRY[away_id].high_altitude:
        adj.append((f"Altitude ({alt:,} ft)", NFL_ALTITUDE))
    travel, note = _travel(home_id, away_id)
    if travel:
        adj.append((note, travel))                          # type: ignore[arg-type]
    return _finish("NFL", home_id, away_id, net_h - net_a, adj, {
        "home_rating": round(net_h, 2), "away_rating": round(net_a, 2),
        "rating_delta": round(net_h - net_a, 2), "rating_label": "SRS pts/game (shrunk)",
        "games": (h.get("games"), a.get("games")), "updated_at": h.get("updated_at"),
        "rest": (rest_home, rest_away)})


def side_win_prob(proj: dict[str, Any], team_id: str) -> float:
    return proj["home_win_prob"] if team_id == proj["home_id"] else proj["away_win_prob"]


def side_cover_prob(proj: dict[str, Any], team_id: str, point: float) -> float:
    """P(team margin + point > 0) under the projection's normal margin model."""
    mu = proj["projected_margin"] if team_id == proj["home_id"] else -proj["projected_margin"]
    return normal_cdf((mu + point) / proj["sigma"])


# ==========================================================================
# Bayesian blend
# ==========================================================================
def model_gate_reasons(sharp_market_prob: float, model_prob: float | None,
                       key_questionable: list[str] | tuple[str, ...] = ()) -> list[str]:
    reasons = []
    if model_prob is not None and abs(model_prob - sharp_market_prob) > MODEL_DIVERGENCE:
        reasons.append(f"Model/sharp divergence {abs(model_prob - sharp_market_prob) * 100:.1f}%"
                       f" > {MODEL_DIVERGENCE * 100:.0f}%")
    if key_questionable:
        reasons.append("Key player Questionable: " + ", ".join(key_questionable))
    return reasons


def compute_bayesian_true_prob(sharp_market_prob: float, kalshi_prob: float | None,
                               model_prob: float | None, *,
                               key_questionable: list[str] | tuple[str, ...] = (),
                               context: dict[str, Any] | None = None) -> tuple[float, str, str]:
    """Blend sharp anchor (0.65), Kalshi mid (0.15) and model (0.20).

    Missing inputs drop out and the remaining weights are renormalised.
    Returns ``(true_prob, hitl_status, explanation)``; the explanation states
    why the price has value and, when gated, why it is PENDING_CAPTAIN_HZ.

    ``context`` (optional): offered_prob, anchor (e.g. "Pinnacle"), side
    (selection label), model_side_margin, model_drivers (list[str]).
    """
    parts = [("sharp", sharp_market_prob), ("kalshi", kalshi_prob), ("model", model_prob)]
    used = [(k, p) for k, p in parts if p is not None]
    total_w = sum(WEIGHTS[k] for k, _ in used)
    prob = sum(WEIGHTS[k] * p for k, p in used) / total_w
    ctx = context or {}
    why: list[str] = []
    offered = ctx.get("offered_prob")
    anchor = ctx.get("anchor", "sharp")
    if offered is not None:
        gap = (sharp_market_prob - offered) * 100
        why.append(f"Retail price needs {offered * 100:.1f}% vs {anchor} anchor "
                   f"{sharp_market_prob * 100:.1f}% ({gap:+.1f} pts: "
                   + ("retail lagging the sharp line" if gap > 0 else
                      "price is WORSE than the sharp anchor; any edge comes only from the "
                      "Kalshi/model blend") + ")")
    if kalshi_prob is not None:
        rel = "confirms" if abs(kalshi_prob - sharp_market_prob) <= 0.02 else (
            "leans further our way" if kalshi_prob > sharp_market_prob else "leans against")
        why.append(f"Kalshi mid {kalshi_prob * 100:.1f}% {rel}")
    if model_prob is not None:
        side = ctx.get("side", "our side")
        lean = "above" if model_prob >= sharp_market_prob else "below"
        drivers = ctx.get("model_drivers") or []
        txt = (f"Model rates {side} at {model_prob * 100:.1f}% ({lean} the sharp "
               f"{sharp_market_prob * 100:.1f}%)")
        if ctx.get("model_side_margin") is not None:
            txt += f" (projected margin {ctx['model_side_margin']:+.1f})"
        if drivers:
            txt += " due to " + ", ".join(drivers)
        why.append(txt)
    else:
        why.append("No model input for this market (sharp/exchange only)")
    gates = model_gate_reasons(sharp_market_prob, model_prob, key_questionable)
    status = HITL_PENDING if gates else HITL_APPROVED
    explanation = " AND ".join(why)
    weights_txt = " + ".join(f"{WEIGHTS[k] / total_w:.2f}x{k}" for k, _ in used)
    explanation += f" | blend {weights_txt} = {prob * 100:.2f}%"
    if gates:
        explanation += " | HITL: " + "; ".join(gates)
    return prob, status, explanation


# ==========================================================================
# Game context (used by the scanner)
# ==========================================================================
@dataclass
class GameContext:
    sport: str
    home_id: str | None
    away_id: str | None
    projection: dict[str, Any] | None = None
    rest_home: RestInfo = field(default_factory=lambda: RestInfo(None))
    rest_away: RestInfo = field(default_factory=lambda: RestInfo(None))
    injuries: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    @property
    def key_questionable(self) -> list[str]:
        out = []
        for cid, rows in self.injuries.items():
            for r in rows:
                if r.get("is_key") and str(r.get("status", "")).lower() in QUESTIONABLE_STATUSES:
                    out.append(f"{REGISTRY[cid].abbr} {r['player']} ({r.get('position') or '?'},"
                               f" {r['status']})")
        return out

    def model_prob_for(self, market_type: str, selection: str, point: float | None
                       ) -> tuple[float | None, float | None]:
        """(model probability, model margin from the selection's side) or (None, None)."""
        if not self.projection or market_type == "totals":
            return None, None
        tid = try_resolve_team(selection, self.sport)
        if tid not in (self.home_id, self.away_id):
            return None, None
        side_margin = self.projection["projected_margin"] * (1 if tid == self.home_id else -1)
        if market_type == "spreads" and point is not None:
            return side_cover_prob(self.projection, tid, point), side_margin
        return side_win_prob(self.projection, tid), side_margin

    def drivers_for(self, selection: str) -> list[str]:
        if not self.projection:
            return []
        tid = try_resolve_team(selection, self.sport)
        sign = 1 if tid == self.home_id else -1
        out = [f"{self.projection['rating_label']} delta "
               f"{sign * self.projection['rating_delta']:+.1f}"]
        out += [f"{label} {sign * pts:+.1f}" for label, pts in self.projection["adjustments"]]
        return out


def build_game_context(conn: sqlite3.Connection | None, sport: str, home_name: str,
                       away_name: str, commence_time: str,
                       fixture: dict[str, Any] | None = None) -> GameContext:
    """Assemble ratings, rest, injuries and the projection for one game.

    ``fixture`` (mock mode) supplies {"ratings", "rest", "injuries"} in memory
    so synthetic data never touches the live team tables.
    """
    home_id, away_id = try_resolve_team(home_name, sport), try_resolve_team(away_name, sport)
    ctx = GameContext(sport, home_id, away_id)
    if not home_id or not away_id:
        ctx.notes.append("teams not resolved; model skipped")
        return ctx
    if fixture is not None:
        rest = fixture.get("rest", {})
        ctx.rest_home = RestInfo(rest.get(home_id), rest.get(home_id) == 0)
        ctx.rest_away = RestInfo(rest.get(away_id), rest.get(away_id) == 0)
        ctx.injuries = {cid: fixture.get("injuries", {}).get(cid, []) for cid in (home_id, away_id)}
        ratings = fixture.get("ratings")
    else:
        if conn is None:
            raise ValueError("conn required without a fixture")
        ctx.rest_home = rest_info(conn, home_id, commence_time)
        ctx.rest_away = rest_info(conn, away_id, commence_time)
        ctx.injuries = {cid: key_injuries(conn, cid) for cid in (home_id, away_id)}
        ratings = None
    project = project_nba_matchup if sport == "NBA" else project_nfl_matchup
    try:
        ctx.projection = project(home_id, away_id, ctx.rest_home.days_rest,
                                 ctx.rest_away.days_rest, conn=conn, ratings=ratings)
    except LookupError as exc:
        ctx.notes.append(f"model unavailable: {exc}")
        return ctx
    upd = ctx.projection.get("updated_at")
    if upd and fixture is None and db.utcnow() - db.parse_ts(upd) > timedelta(hours=STALE_RATINGS_H):
        ctx.notes.append(f"ratings stale (updated {upd})")
    if ctx.rest_home.three_in_four or ctx.rest_away.three_in_four:
        ctx.notes.append("3-in-4 schedule spot (tracked, not priced)")
    return ctx


# ==========================================================================
# NFL multilevel player evaluation -- Yurko, Ventura & Horowitz (2018), nflWAR
# ==========================================================================
# Varying-intercept mixed-effects models fit by penalized likelihood (the lme4
# equivalent, pure stdlib):
#     delta_i = x_i' beta + sum_g b_{g, l(i)} + eps_i,
#     b_{g,l} ~ N(0, tau_g^2),  eps ~ N(0, sigma^2)
# Random intercepts for crossed groups (passer, receiver, rusher, defense,
# team-side-gap) are the BLUPs b = sum(resid) / (n + sigma^2 / tau_g^2), i.e. a
# ridge penalty sigma^2/tau_g^2 that shrinks thin samples toward 0. Variance
# components are re-estimated by EM until convergence.
#
# Response delta is play value added: EPA (default) or WPA (``value="wpa"``).
#   Passing:  delta_air (value to the catch point) + delta_yac (after the catch)
#   Rushing:  non-QB designed rushes; team-side-gap intercepts proxy O-line play.

NFL_RUN_SIDES = ("left", "middle", "right")
NFL_RUN_GAPS = ("A", "B", "C", "D")
PASS_LOCATIONS = ("left", "middle", "right")
RECEIVER_POSITIONS = ("WR", "TE", "RB")
AIR_FIXED = ("intercept", "team_rush_strength", "shotgun", "loc_middle", "loc_right")
YAC_FIXED = ("intercept", "air_yards", "pos_TE", "pos_RB", "air_yards:TE", "air_yards:RB",
             "shotgun", "team_rush_strength")
RUSH_FIXED = ("intercept", "shotgun")
_TAU_FLOOR = 1e-10


def _solve_linear(a: list[list[float]], b: list[float]) -> list[float]:
    """Gaussian elimination with partial pivoting (a is square, copied)."""
    n = len(b)
    m = [row[:] + [b[i]] for i, row in enumerate(a)]
    for c in range(n):
        piv = max(range(c, n), key=lambda r: abs(m[r][c]))
        if abs(m[piv][c]) < 1e-300:
            raise ArithmeticError("singular system")
        m[c], m[piv] = m[piv], m[c]
        for r in range(c + 1, n):
            f = m[r][c] / m[c][c]
            if f:
                for k in range(c, n + 1):
                    m[r][k] -= f * m[c][k]
    x = [0.0] * n
    for r in range(n - 1, -1, -1):
        x[r] = (m[r][n] - sum(m[r][k] * x[k] for k in range(r + 1, n))) / m[r][r]
    return x


def _ols(x: list[list[float]], y: list[float], ridge: float = 1e-9) -> list[float]:
    p = len(x[0])
    xtx = [[sum(row[i] * row[j] for row in x) + (ridge if i == j else 0.0) for j in range(p)]
           for i in range(p)]
    xty = [sum(row[i] * yi for row, yi in zip(x, y)) for i in range(p)]
    return _solve_linear(xtx, xty)


@dataclass
class MixedModelFit:
    """Fitted varying-intercept model."""
    fixed_effects: dict[str, float]
    random_effects: dict[str, dict[Any, float]]
    group_variances: dict[str, float]
    sigma2: float
    group_counts: dict[str, dict[Any, int]]
    n_obs: int
    iterations: int
    converged: bool

    def predict(self, x: dict[str, float], levels: dict[str, Any] | None = None) -> float:
        """x' beta + matching intercepts; unseen levels contribute 0 (the group mean)."""
        out = sum(b * float(x.get(k, 1.0 if k == "intercept" else 0.0))
                  for k, b in self.fixed_effects.items())
        for g, lvl in (levels or {}).items():
            out += self.random_effects.get(g, {}).get(lvl, 0.0)
        return out


def fit_varying_intercepts(y: list[float], x: list[list[float]], fixed_names: list[str],
                           groups: dict[str, list[Any]], max_iter: int = 5000,
                           tol: float = 1e-9) -> MixedModelFit:
    """Penalized-likelihood fit of y = X beta + sum_g Z_g b_g + eps (crossed random intercepts).

    Alternates (i) GLS-free OLS for beta on y minus random parts, (ii) BLUP
    backfitting per group with penalty sigma^2/tau_g^2, (iii) EM updates
        tau_g^2 = mean_l(b_l^2 + v_l),  sigma^2 = (RSS + sum n_l v_l) / N,
    with v_l = sigma^2 / (n_l + sigma^2/tau_g^2) the conditional variance.
    """
    n = len(y)
    if n == 0:
        raise ValueError("no observations")
    if len(x) != n or any(len(r) != len(fixed_names) for r in x):
        raise ValueError("x must be n rows of len(fixed_names)")
    for g, lv in groups.items():
        if len(lv) != n:
            raise ValueError(f"group {g!r} has {len(lv)} labels for {n} observations")
    ys = [float(v) for v in y]
    mean = sum(ys) / n
    var_y = max(sum((v - mean) ** 2 for v in ys) / max(n - 1, 1), 1e-12)

    idx: dict[str, dict[Any, list[int]]] = {}
    for g, labels in groups.items():
        d: dict[Any, list[int]] = {}
        for i, lvl in enumerate(labels):
            d.setdefault(lvl, []).append(i)
        idx[g] = d
    b = {g: {lvl: 0.0 for lvl in d} for g, d in idx.items()}
    tau2 = {g: 0.5 * var_y / max(len(groups), 1) for g in groups}
    sigma2 = 0.5 * var_y
    beta = [0.0] * len(fixed_names)

    def rand_part(i: int, skip: str | None = None) -> float:
        return sum(b[g][groups[g][i]] for g in groups if g != skip)

    converged, it = False, 0
    for it in range(1, max_iter + 1):
        old = (beta[:], dict(tau2), sigma2)
        beta = _ols(x, [ys[i] - rand_part(i) for i in range(n)])
        fitted = [sum(c * v for c, v in zip(beta, x[i])) for i in range(n)]
        cond_var_sum = 0.0
        for g, d in idx.items():
            lam = sigma2 / tau2[g]
            ss = 0.0
            for lvl, rows in d.items():
                s = sum(ys[i] - fitted[i] - rand_part(i, skip=g) for i in rows)
                nl = len(rows)
                b[g][lvl] = s / (nl + lam)
                v = sigma2 / (nl + lam)
                ss += b[g][lvl] ** 2 + v
                cond_var_sum += nl * v
            tau2[g] = max(ss / len(d), _TAU_FLOOR)
        rss = sum((ys[i] - fitted[i] - rand_part(i)) ** 2 for i in range(n))
        sigma2 = max((rss + cond_var_sum) / n, 1e-12)
        delta = max([abs(a - c) for a, c in zip(beta, old[0])]
                    + [abs(tau2[g] - old[1][g]) for g in tau2] + [abs(sigma2 - old[2])])
        if delta < tol:
            converged = True
            break
    return MixedModelFit(
        fixed_effects=dict(zip(fixed_names, beta)),
        random_effects={g: dict(v) for g, v in b.items()},
        group_variances=tau2, sigma2=sigma2,
        group_counts={g: {lvl: len(r) for lvl, r in d.items()} for g, d in idx.items()},
        n_obs=n, iterations=it, converged=converged)


def decompose_pass_value(play: dict[str, Any], value: str = "epa") -> tuple[float, float]:
    """(delta_air, delta_yac) for a pass play.

    Completion: air = air_<value> (value to the catch point), yac = total - air.
    Incompletion / interception: the whole play value is air; yac = 0.
    """
    total = float(play[value])
    if not play.get("complete"):
        return total, 0.0
    air = float(play[f"air_{value}"])
    return air, total - air


def _air_row(p: dict[str, Any]) -> list[float]:
    loc = str(p.get("pass_location", "left")).lower()
    if loc not in PASS_LOCATIONS:
        raise ValueError(f"pass_location must be one of {PASS_LOCATIONS}, got {loc!r}")
    return [1.0, float(p.get("team_rush_strength", 0.0)), float(bool(p.get("shotgun"))),
            float(loc == "middle"), float(loc == "right")]


def _yac_row(p: dict[str, Any]) -> list[float]:
    pos = str(p.get("receiver_position", "WR")).upper()
    if pos not in RECEIVER_POSITIONS:
        raise ValueError(f"receiver_position must be one of {RECEIVER_POSITIONS}, got {pos!r}")
    ay = float(p["air_yards"])
    te, rb = float(pos == "TE"), float(pos == "RB")
    return [1.0, ay, te, rb, ay * te, ay * rb, float(bool(p.get("shotgun"))),
            float(p.get("team_rush_strength", 0.0))]


def fit_air_model(plays: list[dict[str, Any]], value: str = "epa", **kw: Any) -> MixedModelFit:
    """Delta_air model on all pass attempts: passer + receiver + defense intercepts,
    controlling for team rush strength, formation (shotgun) and pass location."""
    if not plays:
        raise ValueError("no pass plays")
    y = [decompose_pass_value(p, value)[0] for p in plays]
    return fit_varying_intercepts(
        y, [_air_row(p) for p in plays], list(AIR_FIXED),
        {"passer": [p["passer"] for p in plays], "receiver": [p["receiver"] for p in plays],
         "defense": [p["defense"] for p in plays]}, **kw)


def fit_yac_model(plays: list[dict[str, Any]], value: str = "epa", **kw: Any) -> MixedModelFit:
    """Delta_yac model on completions: air yards x receiver-position interactions
    plus passer + receiver + defense intercepts."""
    comp = [p for p in plays if p.get("complete")]
    if not comp:
        raise ValueError("no completed passes")
    y = [decompose_pass_value(p, value)[1] for p in comp]
    return fit_varying_intercepts(
        y, [_yac_row(p) for p in comp], list(YAC_FIXED),
        {"passer": [p["passer"] for p in comp], "receiver": [p["receiver"] for p in comp],
         "defense": [p["defense"] for p in comp]}, **kw)


def side_gap_key(team: str, side: str, gap: str) -> str:
    """tau = team-side-gap label, e.g. 'KC-left-B'."""
    s, g = str(side).lower(), str(gap).upper()
    if s not in NFL_RUN_SIDES:
        raise ValueError(f"run side must be one of {NFL_RUN_SIDES}, got {side!r}")
    if g not in NFL_RUN_GAPS:
        raise ValueError(f"run gap must be one of {NFL_RUN_GAPS}, got {gap!r}")
    return f"{team}-{s}-{g}"


def is_designed_non_qb_rush(play: dict[str, Any]) -> bool:
    return (str(play.get("rusher_position", "")).upper() != "QB"
            and not play.get("qb_scramble", False))


def fit_rush_model(plays: list[dict[str, Any]], value: str = "epa", **kw: Any) -> MixedModelFit:
    """Non-QB designed rushes: rusher + team-side-gap (O-line proxy) + defense intercepts."""
    rush = [p for p in plays if is_designed_non_qb_rush(p)]
    if not rush:
        raise ValueError("no non-QB designed rushes")
    return fit_varying_intercepts(
        [float(p[value]) for p in rush],
        [[1.0, float(bool(p.get("shotgun")))] for p in rush], list(RUSH_FIXED),
        {"rusher": [p["rusher"] for p in rush],
         "team_side_gap": [side_gap_key(p["team"], p["run_side"], p["run_gap"]) for p in rush],
         "defense": [p["defense"] for p in rush]}, **kw)


def points_added(fit: MixedModelFit, group: str) -> dict[Any, float]:
    """Individual (or tPA for team_side_gap) points added: intercept x attempts."""
    if group not in fit.random_effects:
        raise KeyError(f"no group {group!r} in fit")
    return {lvl: b * fit.group_counts[group][lvl] for lvl, b in fit.random_effects[group].items()}
