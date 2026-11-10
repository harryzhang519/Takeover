# BATCH 3 IMPLEMENTATION SPECIFICATION
**Execution Rule for Claude Code:** Implement all modules and unit tests specified in this file across src/core/ledger.py, src/core/orchestrator.py, and src/data/settler.py. Write all file modifications first and run pytest ONLY ONCE at the very end to conserve execution tokens.

---

## PART 1: PHASE 8 — LEDGER & BANKROLL STATE MANAGER (src/core/ledger.py)

### 1.1 Core Requirements
1. **Ledger Schema & SQLite Persistence**:
   - Manage the ledger_picks table inside data/takeover.db operating in WAL mode.
   - Track active positions, stake sizing, offered odds, closing lines, status (PENDING, WON, LOST, PUSH, CANCELLED), and final payout metrics.
2. **Bankroll & Exposure Controls**:
   - Enforce bankroll limits, tracking available cash vs. active exposure.
   - Ensure the strict desk mandates are hard-coded: **odds floor $\ge -130$** (reject any play priced worse than $-130$), straight singles only, **max .0\text{u}$ per bet**, and **max .0\text{u}$ total exposure per unique event_id**.

---

## PART 2: PHASE 9 — CAPTAIN HZ HITL GATES & ORCHESTRATION (src/core/orchestrator.py)

### 2.1 Core Requirements
1. **Human-In-The-Loop (HITL) Sign-Off Gate**:
   - Build a mandatory checkpoint function captain_hz_approval_gate(pick: dict) -> bool that intercepts qualified +EV bets meeting all quantitative filters.
   - Requires explicit operator sign-off (Captain HZ approval) before any order payload is dispatched to exchange/sportsbook APIs.
2. **Execution Pipeline Coordination**:
   - Tie together market_fetcher.py (live prices), derivatives.py / matchup_model.py (edge detection), isk.py (Quarter-Kelly sizing), and ledger.py (bankroll recording) into a unified trading floor pipeline loop.

---

## PART 3: PHASE 10 — AUTOMATED SETTLEMENT & RECONCILIATION (src/data/settler.py)

### 2.1 Core Requirements
1. **Game Result Scraping & Polling**:
   - Fetch final scores and game statuses from unauthenticated ESPN public endpoints (site.api.espn.com) and data packages (
ba_api, 
fl_data_py).
2. **Automated Ledger Reconciliation**:
   - Automatically reconcile PENDING bets against final outcomes.
   - Calculate profit/loss, update the ledger_picks table status, and credit/debit the bankroll state in ledger.py.
