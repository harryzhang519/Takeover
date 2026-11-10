"""Game analysis cards: the four-layer breakdown behind every +EV single.

    1. RETAIL  offered line (must be -130 or longer)
    2. SHARP   de-vigged benchmark (power / odds-ratio) and Kalshi mid
    3. MODEL   fundamental projection: spread, rating delta, rest, injuries
    4. BLEND   Bayesian fair probability, EV %, quarter-Kelly units (max 2.0u)

Plain ASCII so it renders identically in a terminal and a Discord code block.
"""

from __future__ import annotations

import textwrap
from typing import Any

from src.core.engine import MIN_AMERICAN_ODDS, validate_odds_floor

WIDTH = 100


def _pct(p: float | None, d: int = 2) -> str:
    return "n/a" if p is None else f"{p * 100:.{d}f}%"


def _abbr(cid: str | None) -> str:
    return cid.split("_", 1)[1] if cid else "?"


def render_analysis_card(opp: dict[str, Any], width: int = WIDTH) -> str:
    """Render one opportunity (from live_candidates.json) as an ASCII card."""
    m = opp.get("model") or {}
    text_w = width - 14                     # "| " + 9-char tag + " " + text + " |"
    lines: list[str] = []

    def row(tag: str, text: str) -> None:
        wrapped = textwrap.wrap(text, text_w) or [""]
        for i, chunk in enumerate(wrapped):
            lines.append(f"| {tag if i == 0 else '':<9} {chunk:<{text_w}} |")

    bar, thin = "+" + "=" * (width - 2) + "+", "+" + "-" * (width - 2) + "+"
    head = (f"GAME ANALYSIS  {opp['event']}  ({opp.get('league', '')})  "
            f"tip {opp.get('commence_time', 'n/a')}")
    pick = (f"PICK  {opp['selection']} [{opp['market_type']}] @ {opp.get('retail_book', '?')} "
            f"{opp['offered_american']:+d}")
    lines += [bar, f"| {head:<{width - 4}} |",
              f"| {pick:<{width - 26}}{('[' + opp['hitl_status'] + ']'):>22} |", thin]

    floor = "PASS" if validate_odds_floor(opp["offered_american"]) else "FAIL"
    row("1 RETAIL", f"{opp['offered_american']:+d}  implied "
                    f"{_pct(opp.get('offered_implied_probability'))}  |  floor "
                    f"{MIN_AMERICAN_ODDS}: {floor}")
    sharp = (f"{opp['benchmark_source']} {opp['benchmark_american']:+d}/"
             f"{opp.get('benchmark_opp_american', 0):+d} -> power "
             f"{_pct(opp.get('fair_probability_power'))} | odds-ratio "
             f"{_pct(opp.get('fair_probability_odds_ratio'))} -> sharp fair "
             f"{_pct(opp.get('pinnacle_fair_probability'))} "
             f"({opp.get('sharp_fair_american', 0):+d})")
    row("2 SHARP", sharp)
    if opp.get("exchange_probability") is not None:
        row("", f"Kalshi {opp.get('exchange_ticker', '')} mid "
                f"{_pct(opp['exchange_probability'], 1)} (bid {_pct(opp.get('exchange_yes_bid'), 0)}"
                f" / ask {_pct(opp.get('exchange_yes_ask'), 0)}), divergence vs sharp "
                f"{_pct(opp.get('exchange_divergence'), 1)}")
    else:
        row("", "Kalshi: no matched market for this selection")

    if m.get("available"):
        h, a = _abbr(m.get("home_id")), _abbr(m.get("away_id"))
        fav = h if m["projected_home_spread"] <= 0 else a
        row("3 MODEL", f"Projected spread {fav} -{abs(m['projected_home_spread']):.1f} "
                       f"(home margin {m['projected_margin_home']:+.2f}, sigma "
                       f"{m['sigma']}) | {h} win prob {_pct(m['home_win_prob'], 1)}")
        pace = f" | exp. pace {m['expected_pace']}" if m.get("expected_pace") else ""
        row("", f"{m['rating_label']}: {h} {m['home_rating']:+.2f} vs {a} "
                f"{m['away_rating']:+.2f} (delta {m['rating_delta']:+.2f}, base margin "
                f"{m['base_margin']:+.2f}){pace} | games {m['games'][0]}/{m['games'][1]}")
        row("", "Adjustments: " + ("; ".join(m.get("adjustments") or []) or "none"))
        row("", f"Rest: {h} {m['rest']['home']} | {a} {m['rest']['away']}")
        row("", "Injuries: " + ("; ".join(m.get("injuries") or []) or "none reported"))
        sel_p = m.get("model_probability")
        row("", f"Model P({opp['selection']}) = {_pct(sel_p)}" if sel_p is not None
                else "Model does not price this market type (totals): sharp/exchange only")
    else:
        row("3 MODEL", "Unavailable: " + ("; ".join(m.get("notes") or []) or "no ratings"))
    for note in m.get("notes") or []:
        if m.get("available"):
            row("", f"Note: {note}")

    row("4 BLEND", f"Bayesian fair {_pct(opp['fair_probability'])} "
                   f"({opp.get('fair_american', 0):+d})  |  EV +{opp['ev_percent']:.2f}%  |  "
                   f"Quarter-Kelly {opp['quarter_kelly_units']:.2f}u (uncapped "
                   f"{opp.get('raw_quarter_kelly_units', 0):.2f}u; caps 2.0u/bet, 2.0u/game)")
    row("WHY", (opp.get("value_thesis") or "").split(" | HITL:")[0])   # gates shown below
    row("HITL", f"{opp['hitl_status']}: {opp.get('hitl_reason') or 'all gates clear'}")
    lines.append(thin)
    return "\n".join(lines)


def render_cards_markdown(opps: list[dict[str, Any]]) -> str:
    """All cards for a Discord/markdown brief, each in a text code block."""
    if not opps:
        return ""
    out = ["## Game Analysis Cards", ""]
    for o in opps:
        out += ["```text", render_analysis_card(o), "```", ""]
    return "\n".join(out)
