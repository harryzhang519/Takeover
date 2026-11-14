"""Pipeline jobs for the web desk and the daily 09:00 schedule.

A job is one or more ``run_pipeline.py`` invocations run as subprocesses with
the active interpreter (the same pattern as ``infinite_monitor.py``), so a
slow ESPN call or a crash never takes the web server down with it.

    settle  run_pipeline.py --settle [--mock]
    scan    run_pipeline.py --update-stats --scan --sport NBA,NFL   (live)
            run_pipeline.py --mock --sport NBA,NFL                  (mock: no network)
    daily   settle, then scan  (yesterday's games are graded before today's card)

Only one job runs at a time across processes: the server and the Windows
Task Scheduler entry share ``data/.desk_run.lock``.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
from collections import deque
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).resolve().parents[2]
PIPELINE = ROOT / "run_pipeline.py"
DESK_SPORTS = "basketball_nba,americanfootball_nfl"
JOB_KINDS = ("scan", "settle", "daily")
MODES = ("live", "mock")
DEFAULT_TIMEOUT_S = 900
LOCK_STALE_S = 2 * 3600
TAIL_LINES = 40

# runner(args) -> (returncode or None, combined output)
Runner = Callable[[list[str]], tuple[int | None, str]]


class JobBusy(RuntimeError):
    """Another pipeline job is already running (here or in another process)."""


def commands_for(kind: str, mode: str, db_path: Path | None = None) -> list[list[str]]:
    """run_pipeline.py argument lists for a job, in order."""
    if kind not in JOB_KINDS:
        raise ValueError(f"unknown job kind {kind!r}")
    if mode not in MODES:
        raise ValueError(f"unknown mode {mode!r}")
    mock = mode == "mock"
    db_args = ["--db", str(db_path)] if db_path else []
    settle = ["--settle", *(["--mock"] if mock else []), *db_args]
    scan = (["--mock", "--sport", DESK_SPORTS] if mock
            else ["--update-stats", "--scan", "--sport", DESK_SPORTS]) + db_args
    return {"settle": [settle], "scan": [scan], "daily": [settle, scan]}[kind]


def subprocess_runner(timeout: float = DEFAULT_TIMEOUT_S) -> Runner:
    def run(args: list[str]) -> tuple[int | None, str]:
        env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1")
        try:
            cp = subprocess.run([sys.executable, str(PIPELINE), *args], cwd=ROOT, env=env,
                                capture_output=True, text=True, encoding="utf-8",
                                errors="replace", timeout=timeout)
            return cp.returncode, (cp.stdout or "") + (cp.stderr or "")
        except subprocess.TimeoutExpired:
            return None, f"timed out after {timeout:.0f}s"
        except OSError as exc:
            return None, f"could not start pipeline: {exc}"
    return run


# ==========================================================================
# Cross-process lock
# ==========================================================================
class RunLock:
    """Exclusive lock file; a lock older than ``stale_s`` is treated as abandoned."""

    def __init__(self, path: Path, stale_s: float = LOCK_STALE_S) -> None:
        self.path, self.stale_s = path, stale_s

    def acquire(self) -> bool:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        for _ in range(2):
            try:
                fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            except FileExistsError:
                try:
                    age = time.time() - self.path.stat().st_mtime
                except FileNotFoundError:
                    continue
                if age < self.stale_s:
                    return False
                self.path.unlink(missing_ok=True)
                continue
            with os.fdopen(fd, "w") as f:
                f.write(json.dumps({"pid": os.getpid(), "at": time.time()}))
            return True
        return False

    def release(self) -> None:
        self.path.unlink(missing_ok=True)


# ==========================================================================
# Jobs
# ==========================================================================
@dataclass
class Job:
    id: int
    kind: str
    mode: str
    trigger: str                                     # manual | schedule | cli
    started_at: str
    status: str = "running"                          # running | ok | failed
    finished_at: str | None = None
    steps: list[dict] = field(default_factory=list)

    def as_dict(self) -> dict:
        return asdict(self)


def _now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _tail(text: str, n: int = TAIL_LINES) -> str:
    return "\n".join(text.strip().splitlines()[-n:])


class JobRunner:
    """Runs one pipeline job at a time on a background thread."""

    def __init__(self, lock_path: Path, db_path: Path | None = None,
                 runner: Runner | None = None,
                 on_finish: Callable[[Job], None] | None = None) -> None:
        self.lock = RunLock(lock_path)
        self.db_path = db_path
        self.runner = runner or subprocess_runner()
        self.on_finish = on_finish
        self._mutex = threading.Lock()
        self._current: Job | None = None
        self._history: deque[Job] = deque(maxlen=20)
        self._next_id = 1

    def status(self) -> dict:
        with self._mutex:
            return {"current": self._current.as_dict() if self._current else None,
                    "last": self._history[-1].as_dict() if self._history else None}

    def _start(self, kind: str, mode: str, trigger: str) -> Job:
        commands_for(kind, mode)                              # validate early
        with self._mutex:
            if self._current is not None or not self.lock.acquire():
                raise JobBusy("a pipeline run is already in progress")
            job = Job(self._next_id, kind, mode, trigger, _now_iso())
            self._next_id += 1
            self._current = job
        return job

    def _execute(self, job: Job) -> Job:
        try:
            for args in commands_for(job.kind, job.mode, self.db_path):
                t0 = time.monotonic()
                rc, out = self.runner(args)
                job.steps.append({"args": args, "returncode": rc, "seconds":
                                  round(time.monotonic() - t0, 1), "tail": _tail(out)})
            job.status = "ok" if all(s["returncode"] == 0 for s in job.steps) else "failed"
        except Exception as exc:                              # never kill the server thread
            job.steps.append({"args": [], "returncode": None, "seconds": 0.0,
                              "tail": f"{type(exc).__name__}: {exc}"})
            job.status = "failed"
        finally:
            job.finished_at = _now_iso()
            self.lock.release()
            with self._mutex:
                self._current = None
                self._history.append(job)
        if self.on_finish:
            try:
                self.on_finish(job)
            except Exception:                                 # noqa: BLE001 - best effort
                pass
        return job

    def submit(self, kind: str, mode: str, trigger: str = "manual") -> Job:
        """Start a job in the background; raises JobBusy if one is running."""
        job = self._start(kind, mode, trigger)
        threading.Thread(target=self._execute, args=(job,), daemon=True,
                         name=f"desk-job-{job.id}").start()
        return job

    def run(self, kind: str, mode: str, trigger: str = "cli") -> Job:
        """Run a job on the calling thread (Task Scheduler / tests)."""
        return self._execute(self._start(kind, mode, trigger))


# ==========================================================================
# Daily schedule
# ==========================================================================
def next_run_at(now: datetime, hour: int = 9, minute: int = 0) -> datetime:
    """Next local ``hour:minute`` strictly after ``now`` (``now`` is local and aware)."""
    target = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
    return target if target > now else target + timedelta(days=1)


class DailyScheduler(threading.Thread):
    """Calls ``fire()`` every day at hour:minute local time while the server runs."""

    def __init__(self, fire: Callable[[], None], hour: int = 9, minute: int = 0,
                 clock: Callable[[], datetime] | None = None) -> None:
        super().__init__(daemon=True, name="desk-scheduler")
        self.fire, self.hour, self.minute = fire, hour, minute
        self.clock = clock or (lambda: datetime.now().astimezone())
        self._stop = threading.Event()

    def next_run(self) -> datetime:
        return next_run_at(self.clock(), self.hour, self.minute)

    def stop(self) -> None:
        self._stop.set()

    def run(self) -> None:
        while not self._stop.is_set():
            due = self.next_run()
            # wake at least every 60s so sleep/hibernate or clock changes cannot skip a day
            while not self._stop.is_set() and self.clock() < due:
                wait = (due - self.clock()).total_seconds()
                self._stop.wait(max(0.5, min(60.0, wait)))
            if self._stop.is_set():
                return
            try:
                self.fire()
            except Exception:                                 # noqa: BLE001
                pass
