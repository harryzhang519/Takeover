# Backtest Summary

Generated 2026-11-10T20:37:50+00:00 | Operating authority: Captain HZ | Baseline bankroll 100u (fixed; stakes never compound P&L)

> **Synthetic data.** This run includes seeded verification trades. It is not a real track record and must not be presented as one.

Scope: SQLite ledger_picks. Origins: {'SYNTHETIC_SEED': 245}. Excluded: {'unsettled': 6}.

## Core Performance

| Metric | Flat 1.0u | Quarter-Kelly |
|---|---:|---:|
| Sample size (N) | 245 | 245 |
| Record (W-L-P) | 125-118-2 | 125-118-2 |
| Win % (ex. pushes) | 51.44% | 51.44% |
| Units staked | 245.00 | 273.64 |
| Net units | +7.87 | +1.39 |
| ROI | 3.21% | 0.51% |
| Yield (mean per-bet) | 3.21% | 3.21% |
| Profit factor | 1.067 | 1.010 |
| Final equity | 107.87 | 101.39 |

## Risk

| Metric | Flat 1.0u | Quarter-Kelly |
|---|---:|---:|
| Max drawdown (u) | 21.11 | 24.91 |
| Max drawdown (% peak) | 18.75% | 22.05% |
| Max DD duration (bets) | 178 | 182 |
| Longest win streak | 8 | 8 |
| Longest loss streak | 7 | 7 |
| Sharpe (per bet) | 0.0316 | 0.0048 |
| Sortino (per bet) | 0.0463 | 0.0068 |

## Edge Validation (By Staking)

| Metric | Flat 1.0u | Quarter-Kelly |
|---|---:|---:|
| Expected EV % (stake-wtd) | 4.36% | 4.88% |
| Realised ROI % | 3.21% | 0.51% |
| t-stat (H0: no skill) | 0.504 | 0.076 |
| p-value (one-sided) | 0.3073 | 0.4696 |

## Edge Validation (Pick-Level)

| Metric | Value |
|---|---:|
| CLV average (n=245) | 2.28% |
| CLV beat rate | 71.02% |
| Brier: model / market | 0.2529 / 0.2524 |
| Log-loss: model / market | 0.6990 / 0.6981 |
| Mean fair prob / hit rate | 0.5261 / 0.5144 |

## Odds Buckets

| Offered odds | N | W-L-P | Win % | Flat net | Flat ROI | QK net | QK ROI | Avg EV | Avg CLV |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| -130 to -101 | 146 | 75-70-1 | 51.72% | -4.54 | -3.11% | -6.81 | -3.82% | 4.25% | 2.47% |
| +100 to +140 | 80 | 39-40-1 | 49.37% | +1.88 | 2.35% | +0.80 | 1.00% | 4.30% | 2.25% |
| +141 and up | 19 | 11-8-0 | 57.89% | +10.53 | 55.42% | +7.40 | 48.34% | 5.44% | 0.93% |

## Rule Quarantine (Excluded From All Kpis)

| Pick | Offered | Orig. result | Reason |
|---|---:|---:|---:|
| SEED-0024 | -145 | LOSS | Offered -145 worse than -130 floor |
| SEED-0072 | -145 | WIN | Offered -145 worse than -130 floor |
| SEED-0119 | -145 | LOSS | Offered -145 worse than -130 floor |
| SEED-0167 | -145 | LOSS | Offered -145 worse than -130 floor |
| SEED-0215 | -145 | WIN | Offered -145 worse than -130 floor |

## Methodology

- Universe: graded (WIN/LOSS/PUSH), APPROVED, compliant singles. PENDING picks never approved by Captain HZ are excluded.
- Flat: 1.0u per bet. Quarter-Kelly: the recorded stake (0.25x Kelly on consensus fair probability, capped at 2.0u per single and 2.0u per event).
- ROI = net / staked. Yield = mean of per-bet return on stake. Profit factor = gross won / gross lost.
- Sharpe = mean / stdev of per-bet P&L; Sortino uses downside deviation. Per bet, not annualised.
- CLV = offered decimal x closing de-vigged sharp probability - 1. Beat rate = share of bets with CLV > 0.
- Brier / log-loss compare the model's fair probability with the offered price's raw implied probability on the same bets (lower is better).
- Skill test: under H0 (win prob = 1/decimal) each bet's unit return has mean 0 and variance (decimal - 1); t = sum(stake x R) / sqrt(sum(stake^2 x (decimal - 1))). One-sided normal p-value.
- Picks priced worse than -130 are quarantined to `rule_violations`.
