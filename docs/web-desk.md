# Breeze Bets web desk

A local browser desk over `data/takeover.db` and `run_pipeline.py`. It uses only the Python standard library, has no build step and needs no new packages.

## Start it

```powershell
cd C:\Projects\Breeze
.\venv\Scripts\python.exe web_server.py
```

Open http://localhost:8000. Options:

| Flag | What it does |
|---|---|
| `--port 8010` | Listen on another port |
| `--no-scheduler` | Don't run the 9:00 AM job while the server is up |
| `--db path\to\other.db` | Use another ledger (handy for a scratch mock desk) |
| `--daily-run [--mock\|--live]` | Run the 9:00 job once and exit (what Task Scheduler calls) |

The desk binds to `127.0.0.1` and has no login. It answers only requests addressed to localhost, and every POST must carry the `X-Breeze-Desk` header, so another website can't approve picks through your browser.

## The daily flow

1. **9:00 AM:** the desk settles finished games, then runs `--update-stats --scan` for NBA and NFL, covering game lines and player props. In mock mode it runs `--settle --mock` and then `--mock`, which make no network calls.
   - The run happens while the server is up.
   - With the server off, the run happens through Task Scheduler if you register it (see below).
   - **Run 9:00 scan now** and **Settle final scores** trigger the same jobs on demand.
2. **Desk slate:**
   - **Which picks:** every open pick for today, grouped under NBA and NFL. A sport with no picks is hidden entirely.
   - **Columns:** inside each sport, game lines and player props sit side by side.
   - **Older picks:** open picks from earlier days aren't listed. A notice tells you to settle them instead.
3. **Sign-off:**
   - **Approve** runs the same sign-off as `run_pipeline.py --approve`, with the same quantitative filters, event caps and bankroll check.
   - **Free or premium:** every sign-off also picks the Discord card the pick goes on.
     - **Approve free** puts it on the free card; this is the default.
     - **Approve premium** puts it on the premium card. From the CLI: `run_pipeline.py --approve PICK_ID --tier premium`.
   - **Switching cards:** a signed pick shows its card (**Free play** or **Premium**) and a **Move to … card** button. From the CLI: `run_pipeline.py --set-tier PICK_ID --tier free|premium`.
   - **The card locks at game time.** Once a game starts, or the pick is graded or dispatched, its card can't change, so the free and premium records can't be rearranged after the fact.
   - **Decline** sets `hitl_status` to `DECLINED` (also available as `run_pipeline.py --decline PICK_ID`). The pick drops out of the queue, the exposure caps, the tracker and both Discord cards.
   - **Approve all qualified** signs off only picks that pass every filter, all as free plays, and reports the rest.
   - **Undo:** every approve or decline shows an 8-second toast with an **Undo** button, and `Z` undoes your most recent one. Signed picks on the slate and in **Open official plays** keep an **Undo** button until they're settled, and **Declined today** at the bottom of the slate lists the day's passes with **Undo** on each. Undo sends the pick back to `PENDING_CAPTAIN_HZ`; it's also `run_pipeline.py --undo PICK_ID`.
   - **What undo won't do:** it refuses a pick that's already graded (WIN, LOSS, PUSH or VOID) or was dispatched to a book. A declined pick returns only if it still fits the event caps and available cash.
   - **Bankroll effect:** undoing an approval releases the committed stake back to cash. The pick goes back to holding a pending reservation, the same as before you approved it. The journal is append-only, so the reversal is a new `STAKE_UNCOMMITTED` row that cancels the `STAKE_COMMITTED` row, and approving again leaves one net commitment.
   - **Undo and the card:** undo also clears the pick's card, so the next approval chooses again.
   - **Keyboard:**
     - `J`/`K` move between picks.
     - `A` approves as a free play and `P` approves for premium.
     - `T` switches a signed pick between the free and premium cards.
     - `D` declines.
     - `Z` undoes the last approve or decline.
4. **Discord cards:**
   - **When they update:** every approve, card switch, decline, undo, write-up edit and pipeline job rewrites both cards in `OUTBOX/discord_briefs/`, which hold signed-off open picks only:
     - `<date>_free_card.md`: the free plays, headed `Breeze Bets - Free Plays`.
     - `<date>_premium_card.md`: the paid membership card, headed `Breeze Bets - Premium Card`.
   - **Format:** each pick is a ready-to-paste Discord post, in start-time order:
     ```text
     8:15 PM ET
     🏈 **Jaguars -9.5** (1.50u)
     BetMGM: -108 / 1.93

     <2-5 sentence write-up> @everyone
     ```
   - **Footer:** each card ends with the line *Built and run by an AI agent team. Every pick signed off by Captain HZ.* That describes how picks are actually made: the agents build and run the desk, and a human chooses every pick. Keep it true if the process changes.
   - **Mock mode:** a mock desk writes `<date>_free_card_mock.md` and `<date>_premium_card_mock.md`, so it never overwrites the live cards.
   - **Copying:** **Copy free card** and **Copy premium card** put a whole card on the clipboard; the ledger's **Discord cards** section shows both with their own **Copy** buttons. Each approved pick also has **Copy post**.
5. **Write-ups:** a pick gets its write-up when you approve it, through Approve, Approve all qualified or `run_pipeline.py --approve`. Pending and declined picks never get one, and Undo clears it.
   - **Where the facts come from:**
     - **The pick itself:** its ledger row and analysis, plus `live_candidates.json` / `prop_candidates.json`.
     - **Schedule spots:** back-to-backs, NFL short weeks, hangover, letdown, bounce-back, lookahead and sandwich spots.
     - **Injuries:** key players Out or Doubtful, with QBs first in the NFL.
     - **Matchup and pace:** net rating / SRS ranks, pace ranks, and the opponent's defensive allowance ranks for props.
     - **Player form:** recent games against the line.
     - **Market:** the game total and its juice, the sharp vs retail price, and sharp line movement.
     - **Only facts that support your side are used.** A sport or team with no data just gets fewer facts.
   - **Who writes it:**
     - **Ollama first:** a local Ollama (`BREEZE_OLLAMA_URL`, default `http://localhost:11434`; `BREEZE_OLLAMA_MODEL`, default `qwen2.5-coder:7b`; 4-second timeout) gets those facts plus the three voice examples.
     - **Checks on its answer:** it is rejected if it uses a number that isn't in the facts, talks about the math (models, Kelly, probabilities, percentages) or isn't 2 to 5 sentences. Em dashes become commas, and every post ends with `@everyone`.
     - **Fallback:** when Ollama is off, slow or rejected, a deterministic template writes the post in the same voice from the strongest three facts. After one failed call Ollama is skipped for 60 seconds, so a batch approve never stalls. `BREEZE_WRITEUP_LLM=off` turns Ollama off.
   - **Editing:** each approved pick shows its post with **Edit** (`Ctrl+Enter` saves, `Esc` cancels), **Regenerate** and **Copy post**. A tag shows whether the text came from Ollama, the desk template or your edit.
6. **Verified Ledger:**
   - **What counts:** only picks that are APPROVED and signed off by Captain HZ. A scanner auto-approval is not a sign-off.
   - **Units risked:** stakes on wins and losses. A push or a void (game cancelled, player DNP or 0 snaps) hands the stake back.
   - **ROI:** net units ÷ units risked.
   - **Win rate:** W ÷ (W + L).
   - **Record:** W-L-V, where V counts pushes and voids.
   - **CLV (closing line value):** settlement records each pick's price against the sharp closing line. The desk shows the average CLV % and how many picks beat the close. Positive CLV over a decent sample is the best early sign of a real edge, well before the win-loss record means much. Picks without a closing line are left out.
   - **Sample size:** the win rate shows how many settled picks it's based on.
   - **Breakdowns:** free plays vs premium, NBA vs NFL, and game lines vs player props, each with its own record, units, ROI and CLV. A pick signed off before cards existed counts as a free play.

The **Live / Mock** toggle chooses which ledger origin the desk shows and which mode the scheduled run uses. It is saved in `data/web_desk.json`.

## Run at 9:00 AM without the server

```powershell
powershell -ExecutionPolicy Bypass -File scripts\register_breeze_9am.ps1          # follows the desk toggle
powershell -ExecutionPolicy Bypass -File scripts\register_breeze_9am.ps1 -Mock    # force mock
powershell -ExecutionPolicy Bypass -File scripts\register_breeze_9am.ps1 -Unregister
```

The task runs `web_server.py --daily-run`. If the PC is asleep at 9:00, it runs as soon as the PC wakes. The task and the server share `data/.desk_run.lock`, so two pipeline runs never overlap.

## API

| Method | Path | Does |
|---|---|---|
| GET | `/api/status` | Mode, next 9:00 run, current and last job |
| GET | `/api/slate[?origin=LIVE\|MOCK]` | Today's open picks by sport (empty sports omitted) |
| GET | `/api/tracker[?origin=]` | Unit P/L, ROI, W-L-V, curve, open and settled official picks |
| GET | `/api/discord-drop[?origin=]` | Today's signed-off free and premium cards: `{"cards": [{"tier", "label", "markdown", "path"}, ...]}` |
| POST | `/api/picks/{id}/approve` | Captain HZ sign-off; `{"tier": "free" \| "premium"}`, default free |
| POST | `/api/picks/{id}/tier` | `{"tier": "free" \| "premium"}` moves a signed-off open pick (refused once the game starts) |
| POST | `/api/picks/{id}/decline` | Mark DECLINED |
| POST | `/api/picks/{id}/undo` | APPROVED or DECLINED back to PENDING_CAPTAIN_HZ (open picks only) |
| POST | `/api/picks/{id}/writeup` | `{"text": "..."}` saves your edit, `{"regenerate": true}` writes a fresh one (approved picks only) |
| POST | `/api/picks/approve-all` | Sign off every qualified unsigned pick (`{"tier": ...}` optional, default free) |
| POST | `/api/scan` | Background stats + scan (`{"mode": "mock"}` optional) |
| POST | `/api/settle` | Background settlement |
| POST | `/api/settings` | `{"mode": "live" \| "mock"}` |

The code lives in `web_server.py` (HTTP and routes), `src/web/desk.py` (slate, tracker and actions), `src/core/tiers.py` (free / premium cards and the game-time lock), `src/web/jobs.py` (pipeline jobs, the lock and the scheduler), `templates/writeup_generator.py` (facts, Ollama and the template voice), `templates/discord_templates.py` (the post format) and `static/` (the page). The tests are in `tests/test_web_desk.py`, `tests/test_writeups.py` and `tests/test_tiers.py`.
