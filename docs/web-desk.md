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
   - **Decline** sets `hitl_status` to `DECLINED` (also available as `run_pipeline.py --decline PICK_ID`). The pick drops out of the queue, the exposure caps, the tracker and the Discord drop.
   - **Approve all qualified** signs off only picks that pass every filter, and reports the rest.
   - **Keyboard:** `J`/`K` move between picks, `A` approves and `D` declines.
4. **Discord drop:**
   - **When it updates:** every approve, decline and pipeline job rewrites `OUTBOX/discord_briefs/<date>_approved_drop.md`, which holds signed-off open picks only.
   - **Mock mode:** a mock desk writes `<date>_approved_drop_mock.md`, so it never overwrites the live card.
   - **Copying:** **Copy Discord drop** puts the card on the clipboard.
5. **Verified Ledger:**
   - **What counts:** only picks that are APPROVED and signed off by Captain HZ. A scanner auto-approval is not a sign-off.
   - **Units risked:** stakes on wins and losses. A push or a void (game cancelled, player DNP or 0 snaps) hands the stake back.
   - **ROI:** net units ÷ units risked.
   - **Win rate:** W ÷ (W + L).
   - **Record:** W-L-V, where V counts pushes and voids.
   - **Breakdowns:** by NBA vs NFL and by game lines vs player props.

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
| GET | `/api/discord-drop[?origin=]` | Markdown for today's signed-off card |
| POST | `/api/picks/{id}/approve` | Captain HZ sign-off |
| POST | `/api/picks/{id}/decline` | Mark DECLINED |
| POST | `/api/picks/approve-all` | Sign off every qualified unsigned pick |
| POST | `/api/scan` | Background stats + scan (`{"mode": "mock"}` optional) |
| POST | `/api/settle` | Background settlement |
| POST | `/api/settings` | `{"mode": "live" \| "mock"}` |

The code lives in `web_server.py` (HTTP and routes), `src/web/desk.py` (slate, tracker and actions), `src/web/jobs.py` (pipeline jobs, the lock and the scheduler) and `static/` (the page). The tests are in `tests/test_web_desk.py`.
