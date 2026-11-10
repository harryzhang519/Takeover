# BATCH 2 IMPLEMENTATION SPECIFICATION
**Execution Rule for Claude Code:** Implement all modules and unit tests specified in this file across src/data/market_fetcher.py and src/core/engine.py. Write all file modifications first and run pytest ONLY ONCE at the very end to conserve execution tokens.

---

## PART 1: PHASE 2 & 7 — POLYGON / POLYMARKET EXCHANGE MICROSTRUCTURE (src/data/market_fetcher.py)
*Source: Dubach, P. D. (2026), "The Anatomy of a Decentralized Prediction Market: Microstructure Evidence from the Polymarket Order Book".*

### 1.1 Key Architectural Rules to Implement in src/data/market_fetcher.py

1. **On-Chain Trade Direction Ground Truth (The Lee-Ready Invalidation)**:
   - **Critical Rule:** As Dubach (2026) proves, inferring prediction market trade direction from public WebSocket feeds (change_side) yields only ~59% sign agreement with on-chain ground truth, causing severe 60-67% sign-flips in effective spreads and Kyle's lambda.
   - **Implementation:** Never rely on quote-side feed heuristics for trade direction. All trade-based microstructure measures (effective spreads, real-time volume weighting, VWAP) MUST source trade-direction and aggressor signs directly from on-chain OrderFilled contract events (where makerAssetId == 0 vs 	akerAssetId == 0).

2. **L2 Depth Profile & Geometric Grid Normalization**:
   - Prediction market books do not follow a pure top-of-book sparse pattern; their cumulative depth profile ( / L_{10}$ ratio median ~0.14) is closer to a uniform geometric grid.
   - **Implementation:** When computing VWAP and available liquidity across Polymarket order books, aggregate depth across all 10 levels ($ through {10}$) rather than just top-of-book.

3. **Wash Trading & Self-Counterparty Filtering**:
   - Implement direct self-match and immediate-roundtrip wash filters (maker == taker or reciprocal maker/taker swaps within the 128-block finality buffer) to strip out self-counterparty noise from order book volume metrics.

---

## PART 2: PHASE 4 — NFL MATCHUP EFFICIENCY & MULTILEVEL PLAYER EVALUATION (src/core/matchup_model.py)
*Source: Yurko, R., Ventura, S., and Horowitz, M. (2018), "nflWAR: A Reproducible Method for Offensive Player Evaluation in Football".*

### 2.1 Multilevel Modeling & Division of Credit Architecture

1. **Multilevel Varying-Intercept Structure (src/core/matchup_model.py)**:
   - To account for positional group structures and observation dependence in football plays, fit varying-intercept mixed-effects models using penalized likelihood (lme4 equivalent in Python/Statsmodels/PyMC)[cite: 74].
   - Response variable: Play value added ($\delta_{f,i}$ as Expected Points Added [EPA] or Win Probability Added [WPA])[cite: 72].

2. **Passing Play Decomposition ($\Delta_{air}$ and $\Delta_{yac}$)**:
   - Separate passing value into two components using air yards and yards after catch[cite: 76]:
     * **Air Model ($\Delta_{air}$):** Credits passers and receivers for the play value achieved through the air up to the catch point, controlling for team rush strength, formation, and pass location[cite: 76, 77].
     * **YAC Model ($\Delta_{yac}$):** Credits contributors for yards gained after the catch, incorporating air yards and receiver position interactions[cite: 77].

3. **Rushing Play & Offensive Line Proxy ($tPA_{rush,side-gap}$)**:
   - For non-QB designed rushes, model play values using team-side-gap indicators ($\tau \in \{\text{team}-\text{gap}-\text{side}\}$ across A/B/C/D gaps) as a statistical proxy for offensive line performance while controlling for rusher ability and opposing defense[cite: 77, 78].

---

## PART 3: PHASE 5 — CLOSING LINE VALUE (CLV) & BACKTESTER SKILL TESTING (src/core/backtester.py)
*Source: Buchdahl, Joseph, "What is the Closing Line?", "What is the Closing Line Value?", and "Making the closing odds fair".*

### 3.1 CLV & No-Vig Price (NVP) Mathematics to Implement in src/core/backtester.py

1. **Fair Closing Odds / No-Vig Price (calculate_no_vig_price(closing_odds_a: float, closing_odds_b: float, method: str = "shin") -> tuple[float, float])**:
   - Removes bookmaker vig from closing lines to establish true outcome probabilities, utilizing advanced de-vigging methods (Shin, Odds-Ratio/Logit, or Power) rather than crude equal margin division.

2. **Closing Line Value Calculation (compute_clv(offered_dec_odds: float, closing_nvp_prob: float) -> float)**:
   - Computes exact CLV as the bettor's offered odds ratio against the fair closing probability: $\text{CLV} = \text{Offered Dec Odds} \times \text{Closing NVP Prob}$ (or $\frac{\text{Offered Dec}}{\text{Fair Closing Dec}}$).
   - Serves as the primary statistical proxy for expected value and forecasting skill. Because CLV variance is significantly smaller than raw profit/loss variance ($\sigma_{\text{CLV}} \approx 0.1$ vs. $\sigma_{\text{P/L}} \approx 1.0$), statistical significance can be established in as few as 50 bets.

3. **Brier Score & Skill Testing (evaluate_forecast_skill(predictions: list[float], outcomes: list[int]) -> float)**:
   - Implements Brier scoring rules to evaluate forecast calibration and track whether model edges persist against Pinnacle's benchmark closing line.
