"""Discord brief generators.

Turns schema-validated game opportunities and player props into markdown:
    * ``generate_slate_summary``      -> both pillars side by side
    * ``generate_free_pick``          -> one public Daily Free Pick
    * ``generate_vip_syndicate_card`` -> the full institutional game-line card
    * ``generate_prop_card``          -> official 1.00u player props
    * ``write_discord_briefs``        -> writes the unified drop to the Discord OUTBOX
    * ``discord_post``                -> one APPROVED pick as a Discord post (header + write-up)
    * ``render_approved_drop``        -> the approved-only card (<date>_approved_drop.md)

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
from src.core.ledger import MAX_EVENT_UNITS, MAX_GAME_EVENT_UNITS, MAX_PROP_EVENT_UNITS
from src.core.prop_model import MIN_PROP_OFFICIAL_UNITS

CAP_UNITS = MAX_STAKE_FRACTION * DEFAULT_BANKROLL_UNITS
EXPOSURE_RULE = (f"Per game: game lines up to {MAX_GAME_EVENT_UNITS:.1f}u, player props up to "
                 f"{MAX_PROP_EVENT_UNITS:.1f}u in their own bucket ({MIN_PROP_OFFICIAL_UNITS:.2f}u "
                 f"each), {MAX_EVENT_UNITS:.1f}u combined ceiling.")


def _am(odds: int | None) -> str:
    """Format American odds with an explicit sign."""
    return "n/a" if odds is None else f"{odds:+d}"


def _pct(p: float | None, digits: int = 2) -> str:
    return "n/a" if p is None else f"{p * 100:.{digits}f}%"


def prop_units(p: dict) -> float:
    """Official stake of a prop (``quarter_kelly_units`` for pre-flat-sizing scans)."""
    return float(p.get("official_units", p.get("quarter_kelly_units")) or 0.0)


def generate_slate_summary(opps: list[dict], props: list[dict] | None) -> str:
    """Both pillars side by side: plays, units and the Captain HZ queue."""
    def row(name: str, plays: int, units: float, pending: int) -> str:
        return f"| {name} | {plays} | {units:.2f}u | {pending} |"
    g_units = sum(o["quarter_kelly_units"] for o in opps)
    g_pend = sum(o["hitl_status"] != "APPROVED" for o in opps)
    out = ["## Today's Card", "", "| Pillar | Plays | Units | Pending Captain HZ |",
           "|---|---|---|---|", row("Game lines", len(opps), g_units, g_pend)]
    if props is None:
        out.append("| Player props | off | - | - |")
        return "\n".join(out)
    p_units = sum(prop_units(p) for p in props)
    p_pend = sum(p["hitl_status"] != "APPROVED" for p in props)
    out += [row("Player props", len(props), p_units, p_pend),
            row("**Total**", len(opps) + len(props), g_units + p_units, g_pend + p_pend),
            "", EXPOSURE_RULE]
    return "\n".join(out)


def generate_prop_card(props: list[dict], note: str = "") -> str:
    """Official player props: projection, line, model edge and 1.00u sizing."""
    out = ["## Official Player Props", ""]
    if note:
        out += [f"> {note}", ""]
    if not props:
        out.append("No player prop cleared the model gate today (market EV > 0, z >= 0.65 "
                   "sigma and shrunk EV >= 4%). No edge, no play.")
        return "\n".join(out)
    ordered = sorted(props, key=lambda p: (-p["ev_percent"], p["event"]))
    units = sum(prop_units(p) for p in ordered)
    pending = [p for p in ordered if p["hitl_status"] != "APPROVED"]
    out += [
        f"Props: {len(ordered)} | Exposure: {units:.2f}u | Pending Captain HZ: {len(pending)} | "
        f"Sizing: flat {MIN_PROP_OFFICIAL_UNITS:.2f}u standard play",
        "",
        "| # | League | Event | Pick | Projection | Line | Delta | Offered | Model % "
        "| Market fair % | Edge (EV %) | Units | HITL |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for i, p in enumerate(ordered, 1):
        out.append(
            f"| {i} | {p.get('league', '')} | {p['event']} | {p['selection']} | "
            f"{p['model_projection']:.2f} | {p['side']} {p['line']:g} | {p['delta']:+.2f} | "
            f"{_am(p['offered_american'])} {p['retail_book']} | "
            f"{_pct(p['model_fair_probability'])} | {_pct(p['market_fair_probability'])} | "
            f"+{p['ev_percent']:.2f} | {prop_units(p):.2f} | {p['hitl_status']} |")
    out += ["", "### Why each prop"]
    for p in ordered:
        raw_k = p.get("raw_kelly_units")
        z = p.get("z_score")
        out.append(
            f"- **{p['selection']}** ({p['event']}): model projects "
            f"{p['model_projection']:.2f} vs the {p['line']:g} line ({p['delta']:+.2f}"
            + (f", z {z:.2f}" if z is not None else "") + "). Model "
            f"{_pct(p['model_fair_probability'], 1)} vs market "
            f"{_pct(p['market_fair_probability'], 1)}; edge +{p['ev_percent']:.2f}% EV after "
            f"shrinkage. **{prop_units(p):.2f}u**"
            + (f" (raw Kelly {raw_k:.2f}u, diagnostic)" if raw_k is not None else "") + ".")
    if pending:
        out += ["", "### Prop HITL approval queue"]
        out += [f"- **{p['hitl_status']}** - {p['selection']} ({p['event']}): {p['hitl_reason']}"
                for p in pending]
    return "\n".join(out)


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
        "## VIP Syndicate Card - Game Lines",
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
        f"- Game lines: quarter-Kelly off a fixed {DEFAULT_BANKROLL_UNITS:.0f}u baseline, hard "
        f"cap {CAP_UNITS:.1f}u per position. Player props: flat {MIN_PROP_OFFICIAL_UNITS:.2f}u "
        "standard play. No escalation on heaters, no Martingale on drawdowns.",
        f"- {EXPOSURE_RULE}",
        "- Pending positions are not to be placed until Captain HZ signs off.",
    ]
    return "\n".join(out)


def write_discord_briefs(opps: list[dict], slate_date: str, out_dir: str | Path,
                         banner: str = "", props: list[dict] | None = None,
                         props_note: str = "") -> Path:
    """Write the unified drop to ``<out_dir>/<slate_date>_drop.md``.

    Game picks (free pick + VIP card) and official player props sit side by
    side. ``props=None`` means props were switched off (--no-props); an empty
    list means the prop scan ran and nothing qualified.
    """
    pick = select_free_pick(opps)
    free = generate_free_pick(pick) if pick else (
        "## Daily Free Pick\n\nNo approved +EV single today. No price, no play."
    )
    body = "\n\n".join(filter(None, [
        f"# Discord Drop - {slate_date}",
        banner,
        generate_slate_summary(opps, props),
        free,
        "---",
        generate_vip_syndicate_card(opps),
        None if props is None else "---",
        None if props is None else generate_prop_card(props, props_note),
        render_cards_markdown([o for o in opps if "model" in o]),
        "_21+. Betting involves risk of loss. Nothing here is a guarantee._",
    ]))
    path = Path(out_dir) / f"{slate_date}_drop.md"
    path.write_text(body + "\n", encoding="utf-8")
    return path


# ==========================================================================
# Approved-pick Discord posts (<date>_approved_drop.md and the web desk preview)
# ==========================================================================
#
#   8:15 PM ET
#   🏀 **Suns -6.5** (1.57u)
#   FanDuel: +105 / 2.05
#
#   <2-5 sentence capper write-up> @everyone
SPORT_EMOJI = {"NBA": "🏀", "NFL": "🏈"}       # basketball, american football
PROP_POST_LABELS = {"PTS": "Pts", "REB": "Reb", "AST": "Ast", "3PM": "Threes",
                    "Pass Yds": "Pass Yds", "Rush Yds": "Rush Yds", "Rec Yds": "Rec Yds",
                    "Rec": "Receptions"}
EASTERN = "America/New_York"


def decimal_odds_str(american: int | float) -> str:
    """American -> decimal odds, two places: -108 -> '1.93', +105 -> '2.05'."""
    return f"{american_to_decimal(int(american)):.2f}"


def eastern_start(commence_iso: str | None) -> str:
    """'2026-11-16T01:15:00Z' -> '8:15 PM ET' (EST or EDT as the date requires)."""
    if not commence_iso:
        return "Time TBD"
    from datetime import datetime
    from zoneinfo import ZoneInfo
    t = datetime.fromisoformat(commence_iso.replace("Z", "+00:00")).astimezone(ZoneInfo(EASTERN))
    return f"{t.strftime('%I:%M %p').lstrip('0')} ET"


def team_nickname(full_name: str, league: str) -> str:
    """'Jacksonville Jaguars' -> 'Jaguars', 'Portland Trail Blazers' -> 'Trail Blazers'."""
    from src.data.team_registry import REGISTRY, try_resolve_team
    cid = try_resolve_team(full_name, league) if league in ("NBA", "NFL") else None
    name = REGISTRY[cid].full_name if cid else full_name
    for two_word in ("Trail Blazers",):
        if name.endswith(two_word):
            return two_word
    return name.rsplit(" ", 1)[-1] if " " in name else name


def post_selection(pick: dict) -> str:
    """Bet-slip wording for the post header: 'Suns -6.5', 'Jayson Tatum Over 27.5 Pts'."""
    from src.core.prop_model import PROP_MARKETS
    mt, sel, pt = pick.get("market_type"), pick.get("selection") or "", pick.get("line_point")
    league = pick.get("league") or ""
    if mt in PROP_MARKETS:
        label = PROP_POST_LABELS.get(PROP_MARKETS[mt].label, PROP_MARKETS[mt].label)
        return f"{sel}{'' if pt is None else f' {pt:g}'} {label}"
    if mt == "totals":
        away, _, home = (pick.get("event") or pick.get("event_name") or "").partition(" @ ")
        teams = f"{team_nickname(away, league)}/{team_nickname(home, league)} " if home else ""
        return f"{teams}{sel}{'' if pt is None else f' {pt:g}'}"
    nick = team_nickname(sel, league)
    if mt == "spreads" and pt is not None:
        return f"{nick} {pt:+g}"
    return f"{nick} ML"


def discord_post_header(pick: dict) -> str:
    """Time, emoji + bold selection + units, and 'Book: American / Decimal'."""
    emoji = SPORT_EMOJI.get(pick.get("league") or "", "")
    units = float(pick.get("units", pick.get("kelly_units")) or 0.0)
    odds = int(pick["offered_american"])
    return "\n".join([
        eastern_start(pick.get("commence_time")),
        f"{emoji} **{post_selection(pick)}** ({units:.2f}u)".strip(),
        f"{pick.get('bookmaker') or 'Book'}: {odds:+d} / {decimal_odds_str(odds)}",
    ])


def discord_post(pick: dict, writeup: str | None = None) -> str:
    """Full Discord post for one APPROVED pick: header, blank line, write-up."""
    body = (writeup if writeup is not None else pick.get("writeup")) or ""
    return discord_post_header(pick) + ("\n\n" + body.strip() if body.strip() else "")


def render_approved_drop(picks: list[dict], day_label: str, mock: bool = False) -> str:
    """The approved-only card: every signed-off pick as a Discord post, in start-time order."""
    out = [f"# Breeze Bets - Official Card - {day_label}"]
    if mock:
        out += ["", "> **MOCK DESK - SYNTHETIC DATA. Do not post or bet.**"]
    if not picks:
        out += ["", "No official plays signed off yet. No edge, no play."]
        return "\n".join(out) + "\n"
    for p in sorted(picks, key=lambda x: (x.get("commence_time") or "", x.get("pick_id") or "")):
        out += ["", discord_post(p)]
    return "\n".join(out) + "\n"
