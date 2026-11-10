You are the Principal Quantitative Architect for our workspace. 
The user wants you to learn from this YouTube URL and apply it directly to our codebase: $ARGUMENTS

Execute this exact workflow:

1. **Extract Transcript Only (Zero Video Download)**:
   - Create a temporary folder .tmp_yt if it does not exist (and ensure .tmp_yt/ is in .gitignore).
   - Run yt-dlp --skip-download --write-auto-subs --write-subs --sub-lang en --sub-format vtt -o ".tmp_yt/video" ""
   - Read the generated .vtt file inside .tmp_yt/, strip out duplicate timestamp lines, and parse the full spoken transcript.
   - Delete the .tmp_yt/ contents once parsed so the repo stays clean.

2. **Filter Through Our Desk Mandate**:
   - Extract all mathematical formulas, API patterns, data structures, or UI/agent concepts from the transcript.
   - Reject anything that violates our core rules (must maintain -130 minimum odds floor, straight singles only, power/odds-ratio de-vigging, quarter-Kelly 2.0u cap, anti-house money discipline, and Captain HZ HITL gates).

3. **Direct Codebase Integration**:
   - Inspect our current Breeze directory structure.
   - Map the learned concepts directly to our existing files (e.g., src/core/engine.py, src/core/backtester.py, src/data/market_fetcher.py, or the Agent Office frontend/server).
   - Write, refactor, and test the code immediately. Do not just summarize the video—implement what improves our system and run tests to verify it works.
