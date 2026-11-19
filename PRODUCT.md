# Product

<!-- impeccable:product-schema 1 -->

## Platform

web

## Users

- **Captain HZ (the operator):** the only person who runs the desk. Each morning around the 9:00 AM scan, they sit at a desktop with a keyboard, look over the day's NBA and NFL game lines and player props, approve or decline each pick, edit the write-ups, and copy the approved card to Discord.
- **Discord members:** they see what the desk produces: one post per approved pick, in start-time order, ending with `@everyone`, plus the educational social content. Today most of them follow the **free plays**. The goal is to convert enough of them to a paid **premium membership** with its own picks.
- **The public (planned):** social media followers and prospective members. They will see a public-facing version of Breeze Bets that shows the record and the free plays and sells the membership. This audience is the reason the desk is being turned into something public.

## Go-to-Market

Breeze Bets will be marketed on social media as a team of AI agents working together in a 3D office (Agent Office, which lives in this repo) to find valuable straight singles on sportsbooks and prediction markets. Captain HZ chooses which of the desk's picks go to Discord. The funnel today is social media to free plays in Discord, and from there to premium.

**Framing (decided):** "Built and run by an AI agent team, every pick signed off by a human." That is literally true. A deterministic Python pipeline produces candidate picks, a local LLM drafts the write-ups, the agents in the office build and run that system, and Captain HZ chooses every pick. Marketing must not claim the agents choose the picks unless the agents are actually given that job. Each Discord card ends with this line (`HONEST_TAGLINE` in `templates/discord_templates.py`).

## Product Purpose

Breeze Bets is a quantitative sports-betting desk. A Python pipeline (`run_pipeline.py`) pulls odds and stats, de-vigs sharp prices, sizes stakes in units with capped Kelly, and puts candidate picks in a queue for human sign-off. The local web desk (`web_server.py` with `static/`) turns that queue into a morning workflow: scan, review, sign off, publish, settle.

Success means a fast, mistake-free morning sign-off, a Discord card that's ready to paste, and a ledger whose record can be trusted.

## Positioning

Verified and math-first. The edge comes from the math: sharp de-vigging, fair-probability models and stakes sized and capped by exposure rules. The record counts only picks Captain HZ signed off, in units, with pushes and voids shown openly. It stands against hype cappers and parlay culture, and the educational content explains the Parlay Tax and the House Money Fallacy.

## Operating Context

- **Daily rhythm:** at 9:00 AM the desk settles finished games, then runs a stats update and scan. The run comes from the server's scheduler or from Windows Task Scheduler (`scripts/register_breeze_9am.ps1`).
- **Live / Mock:** a toggle switches the desk between the live ledger and the mock one. Mock runs make no network calls and write separate `_mock` outputs.
- **Sign-off lifecycle:** a pick moves from `PENDING_CAPTAIN_HZ` to APPROVED or DECLINED. Undo is allowed until the pick is graded or dispatched. Grading is WIN / LOSS / PUSH / VOID.
- **Outputs:**
  - **Discord cards:** the free card goes to `OUTBOX/discord_briefs/<date>_free_card.md` and the premium card to `<date>_premium_card.md`.
  - **Social content:** `OUTBOX/social_distribution/`.
  - **Scans:** `OUTBOX/market_scans/`.
- **Keyboard-first desk:** `J`/`K` move between picks, `A` approves as a free play, `P` approves for premium, `T` switches a signed pick's card, `D` declines and `Z` undoes the last one.
- **Local only:** the desk binds to `127.0.0.1`, has no login, and every POST must carry the `X-Breeze-Desk` header.

## Capabilities and Constraints

- **Two sports, two pillars:** NBA and NFL. Game lines and player props are equal pillars.
- **Exposure rules:** each game has caps for game lines, for player props and for both combined. Approving a pick runs a bankroll check, the same filters as `run_pipeline.py --approve`, and the event caps.
- **Write-ups:** a local Ollama writes the first draft, with a deterministic template as the fallback. The text uses only grounded facts and never mentions the math (no models, Kelly, probabilities or percentages). Each write-up is 2 to 5 sentences, has no em dashes and ends with `@everyone`.
- **Verified Ledger:** it counts only signed-off APPROVED picks and reports units risked, net units, ROI, win rate with its sample size, W-L-V, and average closing line value (CLV) with the share of picks that beat the close. It breaks all of that down by free vs premium, by sport, and by game lines vs player props.
- **Stack:** the desk is Python standard library only, with no build step. The page is plain HTML, CSS and JS in `static/`, and the data lives in SQLite at `data/takeover.db`.
- **Membership tiers:** free plays and a paid premium membership.
  - **At sign-off:** every pick is tagged FREE (the default) or PREMIUM.
  - **Lock:** the tier can be switched until game time, then it's fixed, so the two records stay honest.
  - **Still undecided:** premium pricing, delivery and what it includes.
- **Undecided:** the shape of the public surface and where it is hosted. The operator desk has no login and binds to localhost, so it can't be made public as it is.

## Brand Commitments

- **Name and role:** the product is "Breeze Bets", and the operator is "Captain HZ".
- **Social content voice:** ASCII only, no emojis, no hype, at most 280 characters per post, and every figure computed by the engine.
- **Discord post format:** fixed. Start time, then a sport emoji with the bold pick and units, then the book and price, then the write-up, then `@everyone`. See `templates/discord_templates.py`.

## Evidence on Hand

- **Real records:** the signed-off ledger in `data/takeover.db`, the backtest summary and equity curve in `OUTBOX/market_scans/`, and the daily drops and content in `OUTBOX/`.
- **Method sources:** the papers cited in `specs/batch_*.md` (Clarke et al. 2017, Štrumbelj 2014, Stern 1991, Yurko et al. 2018, Dubach 2026).
- **Things to avoid fabricating:** there are no testimonials, member counts, pricing or third-party endorsements. Never invent records, ROI figures or win rates. Every number shown comes from the ledger.

## Product Principles

1. **The record is the product.** Only signed-off picks count, every result is shown, and no number appears without a source in the ledger or engine.
2. **A human signs off on every pick.** The scanner proposes and Captain HZ decides. Every decision can be reversed until the game is graded.
3. **Math in the engine, plain talk in the post.** Members get grounded reasons in a capper's voice, not model internals.
4. **Discipline over action.** Exposure caps, unit sizing and anti-parlay education come before volume and hype.
5. **The morning run should be fast.** The desk serves a keyboard-driven routine that takes a few minutes and should be impossible to get wrong.

## Accessibility & Inclusion

The desk is desktop and keyboard-first and is used in a morning session. Phone use isn't a target. Everything must still work fully from the keyboard, with visible focus, and W/L/V status must never be shown by color alone.
