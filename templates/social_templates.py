"""Educational social content generators.

    * ``generate_x_thread``          -> 5-post thread on the House Money Fallacy
                                        and the Parlay Tax (no emojis, no hype)
    * ``generate_short_form_script`` -> 45-second vertical video script
    * ``write_social_content``       -> writes both to the social OUTBOX

Parlay Tax figures are computed from the engine at render time, never
hard-coded.
"""

from __future__ import annotations

from pathlib import Path

from src.core.engine import american_to_decimal, calculate_ev

MAX_POST_CHARS = 280
VALID_FOCUS = ("both", "parlay_tax", "house_money")


def parlay_ev(legs: int, leg_american: int = -110, leg_fair_prob: float = 0.5) -> float:
    """EV of an n-leg parlay of identical, independent legs (fraction)."""
    payout = american_to_decimal(leg_american) ** legs
    return calculate_ev(leg_fair_prob ** legs, payout)


def _assert_clean(posts: list[str]) -> None:
    """Enforce the 280-char limit and the zero-emoji (ASCII-only) voice."""
    for i, post in enumerate(posts, 1):
        if len(post) > MAX_POST_CHARS:
            raise ValueError(f"Post {i} is {len(post)} chars (> {MAX_POST_CHARS})")
        if not post.isascii():
            raise ValueError(f"Post {i} contains non-ASCII characters (emoji guard)")


def generate_x_thread(focus: str = "both") -> str:
    """Return a 5-post educational thread on two structural betting traps.

    Args:
        focus: ``"both"`` (default), ``"parlay_tax"`` or ``"house_money"``;
            controls which trap leads the thread. Both are always covered.
    """
    if focus not in VALID_FOCUS:
        raise ValueError(f"focus must be one of {VALID_FOCUS}")

    single = parlay_ev(1) * 100
    two, four, six = (parlay_ev(n) * 100 for n in (2, 4, 6))

    parlay = [
        "Trap: the Parlay Tax. A -110 single on a true coin flip carries an expected return of "
        f"{single:.2f}%. Chain legs and the margin compounds multiplicatively: "
        f"2 legs {two:.1f}%, 4 legs {four:.1f}%, 6 legs {six:.1f}%.",
        "The payout grows, so the ticket looks attractive. What actually grows is the expected "
        "loss per dollar. Books market parlays heavily because their hold on them is a multiple "
        "of their hold on singles. Correlation boosts rarely close the gap.",
    ]
    house = [
        "Trap: the House Money Fallacy. After a winning run, bettors size up because profit feels like "
        "free capital. It is not. A unit of profit has the same value as a unit of deposit, and "
        "the market does not know which dollar you are risking.",
        "Sizing should be a function of edge and baseline bankroll only. Recent results are not "
        "an input. No escalation after wins, no Martingale after losses. A drawdown is variance, "
        "not a debt to be recovered at higher stakes.",
    ]
    close = (
        "Our desk rules: straight singles only. Prices -130 or longer. Fair value from de-vigged "
        "sharp benchmarks. Quarter-Kelly sizing, hard cap 2 units of 100. The edge is in the "
        "price and the process, not the narrative."
    )

    middle = house + parlay if focus == "house_money" else parlay + house
    posts = [*middle, close]
    if len(posts) != 5:
        raise AssertionError(f"Thread must be exactly 5 posts, got {len(posts)}")
    posts = [f"{i}/5 {p}" for i, p in enumerate(posts, 1)]
    _assert_clean(posts)
    return "\n\n".join(posts)


def generate_short_form_script(pick_data: dict) -> str:
    """Return a 45-second retention script breaking down today's +EV spot."""
    offered = pick_data["offered_american"]
    implied = 1.0 / american_to_decimal(offered)
    fair = pick_data["fair_probability"]
    sel = pick_data["selection"]
    beats = [
        ("0:00-0:03",
         "Fast push-in on a single betting slip. Hard cut to black.",
         f"\"{sel} {offered:+d}\" in bold, white on black",
         f"This is the only bet we like today. And it is not because of a hunch."),
        ("0:03-0:10",
         "Split screen: retail sportsbook price left, sharp book price right.",
         f"RETAIL {offered:+d} | {pick_data['benchmark_source'].upper()} "
         f"{pick_data['benchmark_american']:+d}",
         f"The sharpest market in the world prices this at {pick_data['benchmark_american']:+d}. "
         f"A retail book is dealing {offered:+d}."),
        ("0:10-0:20",
         "Animated bar: implied probability bar shrinking as the vig is stripped.",
         f"FAIR PROBABILITY {fair * 100:.1f}%  vs  PRICE NEEDS {implied * 100:.1f}%",
         "Strip the bookmaker's margin out with a power and odds-ratio de-vig, and the true win rate "
         f"comes out near {fair * 100:.1f} percent. The price only needs {implied * 100:.1f}."),
        ("0:20-0:30",
         "Counter ticks up to the EV number. Then a stack of 100 chips, two slide forward.",
         f"EV +{pick_data['ev_percent']:.2f}%  |  STAKE {pick_data['quarter_kelly_units']:.2f}u",
         f"That gap is plus {pick_data['ev_percent']:.1f} percent expected value. We size it at "
         "quarter-Kelly, capped at two percent of bankroll. Never more."),
        ("0:30-0:40",
         "Parlay slip gets crossed out with a red line.",
         "STRAIGHT SINGLE. NO PARLAY.",
         "One straight bet. No parlay. Every extra leg stacks the house margin against you. "
         f"And this still loses about {(1 - fair) * 100:.0f} percent of the time. That is fine."),
        ("0:40-0:45",
         "Slow zoom out on the desk. Logo end card.",
         "PRICE > PICKS. Follow for the daily sheet.",
         "We do not sell locks. We sell prices. Follow for tomorrow's number."),
    ]
    out = [f"### 45s Short-Form Script - {sel}", ""]
    for ts, visual, graphic, vo in beats:
        out += [
            f"**{ts}**",
            f"[VISUAL ACTION] {visual}",
            f"[GRAPHIC OVERLAY] {graphic}",
            f"[VOICEOVER] {vo}",
            "",
        ]
    return "\n".join(out).rstrip()


def write_social_content(pick: dict | None, slate_date: str, out_dir: str | Path,
                         focus: str = "both", banner: str = "") -> Path:
    """Write the X thread and short-form script to ``<slate_date>_content.md``."""
    script = (generate_short_form_script(pick) if pick
              else "### 45s Short-Form Script\n\nNo approved +EV spot today. No script.")
    body = "\n\n".join(filter(None, [
        f"# Social Distribution - {slate_date}",
        banner,
        "## X Thread",
        generate_x_thread(focus),
        "---",
        "## Short-Form Video",
        script,
    ]))
    path = Path(out_dir) / f"{slate_date}_content.md"
    path.write_text(body + "\n", encoding="utf-8")
    return path
