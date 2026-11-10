"""Discord brief generators.

Turns a schema-validated opportunity list into markdown:
    * ``generate_free_pick``          -> one public Daily Free Pick
    * ``generate_vip_syndicate_card`` -> the full institutional slate card
    * ``write_discord_briefs``        -> writes both to the Discord OUTBOX

All copy is driven by the numbers in each opportunity dict, so the text
cannot drift from the math.
"""

from __future__ import annotations

from pathlib import Path

from templates.analysis_cards import render_cards_markdown

from src.core.engine import (
    MAX_STAKE_FRACTION,
    DEFAULT_BANKROLL_UNITS,
    american_to_decimal,
)

CAP_UNITS = MAX_STAKE_FRACTION * DEFAULT_BANKROLL_UNITS


def _am(odds: int | None) -> str:
    """Format American odds with an explicit sign."""
    return "n/a" if odds is None else f"{odds:+d}"


def _pct(p: float | None, digits: int = 2) -> str:
    return "n/a" if p is None else f"{p * 100:.{digits}f}%"


def select_free_pick(opps: list[dict]) -> dict | None:
    """Choose the free pick: the highest-EV APPROVED single (ties -> event name)."""
    approved = [o for o in opps if o["hitl_status"] == "APPROVED"]
    if not approved:
        return None
    return sorted(approved, key=lambda o: (-o["ev_percent"], o["event"]))[0]


def generate_free_pick(opp: dict) -> str:
    """Format the Daily Free Pick with its full mathematical justification."""
    offered_dec = american_to_decimal(opp["offered_american"])
    implied = 1.0 / offered_dec
    edge = opp["fair_probability"] - implied
    lines = [
        f"## Daily Free Pick - {opp.get('league', '')} {opp['market_type']}".rstrip(),
        "",
        f"**{opp['selection']}** at **{_am(opp['offered_american'])}** "
        f"({opp.get('retail_book', 'retail')}) - {opp['event']}",
        f"**Stake: {opp['quarter_kelly_units']:.2f}u** (quarter-Kelly, hard cap {CAP_UNITS:.1f}u). "
        "Straight single.",
        "",
        "### The math",
        "| Metric | Value |",
        "|---|---|",
        f"| Sharp benchmark ({opp['benchmark_source']}) | "
        f"{_am(opp['benchmark_american'])} / {_am(opp.get('benchmark_opp_american'))} |",
        f"| Fair prob - power de-vig | {_pct(opp.get('fair_probability_power'))} |",
        f"| Fair prob - odds-ratio de-vig | {_pct(opp.get('fair_probability_odds_ratio'))} |",
        f"| Sharp fair (conservative of the two) | "
        f"{_pct(opp.get('pinnacle_fair_probability', opp['fair_probability']))} |",
        f"| Fair prob used (75/25 sharp/Kalshi consensus) | {_pct(opp['fair_probability'])} |",
        f"| Fair line | {_am(opp.get('fair_american'))} |",
        f"| Offered price / implied prob | {_am(opp['offered_american'])} / {_pct(implied)} |",
        f"| Probability edge | {edge * 100:+.2f} pts |",
        f"| Expected value | +{opp['ev_percent']:.2f}% |",
        f"| Exchange check ({opp.get('exchange_source', 'n/a')}) | "
        f"{_pct(opp.get('exchange_probability'), 1)} |",
        "",
        "### Why the price is the pick",
        f"The sharp market, stripped of its margin, says this outcome lands "
        f"{_pct(opp['fair_probability'], 1)} of the time. The retail price only needs "
        f"{_pct(implied, 1)}. That gap is the entire thesis. There is no narrative, "
        "no matchup story, no \"lock\".",
        "",
        "### No locks. No parlays.",
        "- Anyone selling a \"lock\" is selling certainty that does not exist. A +EV bet "
        "still loses often; this one is expected to lose about "
        f"{_pct(1 - opp['fair_probability'], 0)} of the time.",
        "- Edge shows up over hundreds of positions, not one ticket.",
        "- Straight singles only. Every parlay leg multiplies the book's margin against you.",
        f"- Sizing is fixed off the baseline bankroll. No upsizing after wins, no chasing losses.",
    ]
    return "\n".join(lines)


def generate_vip_syndicate_card(opps: list[dict]) -> str:
    """Format the full institutional syndicate card for the slate."""
    approved = [o for o in opps if o["hitl_status"] == "APPROVED"]
    pending = [o for o in opps if o["hitl_status"] != "APPROVED"]
    exposure = sum(o["quarter_kelly_units"] for o in approved)
    ordered = sorted(opps, key=lambda o: -o["ev_percent"])

    out = [
        "## VIP Syndicate Card",
        "",
        f"Positions: {len(opps)} | Approved: {len(approved)} | "
        f"Pending Captain HZ: {len(pending)} | Approved exposure: {exposure:.2f}u",
        "",
        "### Full slate - fair lines",
        "| # | League | Event | Selection | Mkt | Offered | Benchmark | Fair % | Fair line "
        "| EV % | Units | HITL |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for i, o in enumerate(ordered, 1):
        out.append(
            f"| {i} | {o.get('league', '')} | {o['event']} | {o['selection']} | "
            f"{o['market_type']} | {_am(o['offered_american'])} | "
            f"{o['benchmark_source']} {_am(o['benchmark_american'])} | "
            f"{_pct(o['fair_probability'])} | {_am(o.get('fair_american'))} | "
            f"+{o['ev_percent']:.2f} | {o['quarter_kelly_units']:.2f} | {o['hitl_status']} |"
        )

    out += [
        "",
        "### Exchange liquidity comparison",
        "| Selection | Retail no-vig % | Sharp fair % | Exchange | Exchange % | Liquidity / volume "
        "| Exchange divergence |",
        "|---|---|---|---|---|---|---|",
    ]
    for o in ordered:
        liq, vol = o.get("exchange_liquidity_usd"), o.get("exchange_volume")
        depth = (f"${liq:,.0f}" if liq is not None
                 else f"{vol:,.0f} contracts" if vol is not None else "n/a")
        out.append(
            f"| {o['selection']} | {_pct(o.get('retail_novig_probability'))} | "
            f"{_pct(o['fair_probability'])} | {o.get('exchange_source', 'n/a')} | "
            f"{_pct(o.get('exchange_probability'))} | {depth} | "
            f"{_pct(o.get('exchange_divergence'))} |"
        )

    out += ["", "### HITL approval queue"]
    if pending:
        for o in pending:
            out.append(
                f"- **PENDING_CAPTAIN_HZ** - {o['selection']} ({o['event']}): {o['hitl_reason']}"
            )
    else:
        out.append("- None. All positions cleared automated gates.")

    out += [
        "",
        "### Desk rules",
        "- Straight singles only. Prices -130 or longer only.",
        "- Fair value = conservative of power and odds-ratio de-vig on Pinnacle/Circa, blended "
        "75/25 with the Kalshi mid when available.",
        f"- Quarter-Kelly off a fixed {DEFAULT_BANKROLL_UNITS:.0f}u baseline, hard cap "
        f"{CAP_UNITS:.1f}u per position and {CAP_UNITS:.1f}u per event. No escalation on "
        "heaters, no Martingale on drawdowns.",
        "- Pending positions are not to be placed until Captain HZ signs off.",
    ]
    return "\n".join(out)


def write_discord_briefs(opps: list[dict], slate_date: str, out_dir: str | Path,
                         banner: str = "") -> Path:
    """Write the free pick and VIP card to ``<out_dir>/<slate_date>_drop.md``."""
    pick = select_free_pick(opps)
    free = generate_free_pick(pick) if pick else (
        "## Daily Free Pick\n\nNo approved +EV single today. No price, no play."
    )
    body = "\n\n".join(filter(None, [
        f"# Discord Drop - {slate_date}",
        banner,
        free,
        "---",
        generate_vip_syndicate_card(opps),
        render_cards_markdown([o for o in opps if "model" in o]),
        "_21+. Betting involves risk of loss. Nothing here is a guarantee._",
    ]))
    path = Path(out_dir) / f"{slate_date}_drop.md"
    path.write_text(body + "\n", encoding="utf-8")
    return path
