"""Breeze Bets web desk: a local, zero-dependency web UI over data/takeover.db.

    python web_server.py                       # http://127.0.0.1:8000, 09:00 scheduler on
    python web_server.py --port 8010 --no-scheduler
    python web_server.py --daily-run           # one 09:00 run (settle + scan), then exit;
                                               # this is what Windows Task Scheduler calls

API (JSON):
    GET  /api/status                  mode, next 09:00 run, current/last pipeline job
    GET  /api/slate[?origin=]         today's open picks grouped NBA / NFL (empty sports omitted)
    GET  /api/tracker[?origin=]       official (signed-off) unit P/L, ROI, W-L-V, curve, history
    GET  /api/discord-drop[?origin=]  markdown for today's signed-off card only
    POST /api/picks/{id}/approve      Captain HZ sign-off (run_pipeline.py --approve)
    POST /api/picks/{id}/decline      -> DECLINED (run_pipeline.py --decline)
    POST /api/picks/{id}/undo         APPROVED / DECLINED -> PENDING (run_pipeline.py --undo)
    POST /api/picks/{id}/writeup      {"text": "..."} saves an edit, {"regenerate": true} redoes it
    POST /api/picks/approve-all       sign off every qualified unsigned pick on today's slate
    POST /api/scan                    background: --update-stats --scan (or --mock)
    POST /api/settle                  background: --settle (grades WIN / LOSS / PUSH / VOID)
    POST /api/settings                {"mode": "live" | "mock"}

The server binds to 127.0.0.1 and has no login, so it also refuses requests
whose Host is not localhost (DNS rebinding) and POSTs without the
``X-Breeze-Desk`` header (cross-site form posts cannot set it).
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import re
import sqlite3
import sys
import threading
from datetime import datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable
from urllib.parse import parse_qs, urlsplit

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.core import database as db  # noqa: E402
from src.core.config import load_env  # noqa: E402
from src.web import desk  # noqa: E402
from src.web.jobs import DailyScheduler, Job, JobBusy, JobRunner, next_run_at  # noqa: E402

STATIC_DIR = ROOT / "static"
SETTINGS_PATH = ROOT / "data" / "web_desk.json"
LOCK_PATH = ROOT / "data" / ".desk_run.lock"
DISCORD_DIR = ROOT / "OUTBOX" / "discord_briefs"
CSRF_HEADER = "X-Breeze-Desk"
LOCAL_HOSTS = frozenset({"localhost", "127.0.0.1", "[::1]"})
MAX_BODY = 16 * 1024
SCHEDULE_HOUR, SCHEDULE_MINUTE = 9, 0
PICK_ACTION = re.compile(r"^/api/picks/(PK-[0-9a-f]{6,40})/(approve|decline|undo|writeup)$")


# ==========================================================================
# Settings
# ==========================================================================
def load_settings(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        data = {}
    mode = data.get("mode") if isinstance(data, dict) else None
    return {"mode": mode if mode in ("live", "mock") else "live"}


def save_settings(path: Path, settings: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(settings, indent=2), encoding="utf-8")


def origin_for(mode: str) -> str:
    return db.ORIGIN_MOCK if mode == "mock" else db.ORIGIN_LIVE


# ==========================================================================
# Application (transport-free so it is easy to test)
# ==========================================================================
class HttpError(Exception):
    def __init__(self, status: int, message: str) -> None:
        super().__init__(message)
        self.status = status


class DeskApp:
    def __init__(self, db_path: Path = db.DB_PATH, settings_path: Path = SETTINGS_PATH,
                 lock_path: Path = LOCK_PATH, discord_dir: Path = DISCORD_DIR,
                 static_dir: Path = STATIC_DIR, runner=None,
                 clock: Callable[[], datetime] | None = None) -> None:
        self.db_path, self.settings_path = db_path, settings_path
        self.discord_dir, self.static_dir = discord_dir, static_dir
        self.clock = clock or (lambda: datetime.now().astimezone())
        self._settings_lock = threading.Lock()
        self.jobs = JobRunner(lock_path, db_path=db_path, runner=runner,
                              on_finish=self._after_job)
        self.scheduler: DailyScheduler | None = None
        self.connect().close()                      # create / migrate the schema up front

    # ---- plumbing --------------------------------------------------------
    def connect(self) -> sqlite3.Connection:
        return db.connect(self.db_path)

    @property
    def mode(self) -> str:
        return load_settings(self.settings_path)["mode"]

    def _origin(self, query: dict[str, list[str]]) -> str:
        raw = (query.get("origin") or [""])[0].upper()
        return raw if raw in desk.ORIGINS else origin_for(self.mode)

    def refresh_drop(self, origin: str) -> Path:
        conn = self.connect()
        try:
            return desk.write_discord_drop(conn, origin, self.discord_dir)
        finally:
            conn.close()

    def _after_job(self, job: Job) -> None:
        self.refresh_drop(origin_for(job.mode))

    def start_scheduler(self) -> DailyScheduler:
        def fire() -> None:
            try:
                self.jobs.submit("daily", self.mode, trigger="schedule")
            except JobBusy:
                pass
        self.scheduler = DailyScheduler(fire, SCHEDULE_HOUR, SCHEDULE_MINUTE, self.clock)
        self.scheduler.start()
        return self.scheduler

    # ---- routes ----------------------------------------------------------
    def status(self) -> dict[str, Any]:
        now = self.clock()
        nxt = next_run_at(now, SCHEDULE_HOUR, SCHEDULE_MINUTE)
        return {"mode": self.mode, "origin": origin_for(self.mode),
                "server_time": now.isoformat(timespec="seconds"),
                "scheduler": self.scheduler is not None and self.scheduler.is_alive(),
                "next_run": nxt.isoformat(timespec="seconds"),
                "seconds_to_next_run": int((nxt - now).total_seconds()),
                "jobs": self.jobs.status()}

    def handle(self, method: str, path: str, query: dict[str, list[str]],
               body: dict[str, Any]) -> tuple[int, dict[str, Any]]:
        if method == "GET":
            if path == "/api/status":
                return 200, self.status()
            conn = self.connect()
            try:
                if path == "/api/slate":
                    return 200, desk.slate(conn, self._origin(query))
                if path == "/api/tracker":
                    return 200, desk.tracker(conn, self._origin(query))
                if path == "/api/discord-drop":
                    origin = self._origin(query)
                    return 200, {"origin": origin, "markdown": desk.discord_drop(conn, origin),
                                 "path": str(desk.drop_path(self.discord_dir, origin))}
            finally:
                conn.close()
            raise HttpError(404, f"no route GET {path}")
        if method != "POST":
            raise HttpError(405, "method not allowed")
        m = PICK_ACTION.match(path)
        if m:
            pick_id, action = m.groups()
            conn = self.connect()
            try:
                if action == "writeup":
                    view = self._writeup(conn, pick_id, body)
                else:
                    act = {"approve": desk.approve, "decline": desk.decline,
                           "undo": desk.undo}[action]
                    view = act(conn, pick_id)
            finally:
                conn.close()
            drop = self.refresh_drop(view["origin"])
            return 200, {"pick": view, "drop_path": str(drop)}
        if path == "/api/picks/approve-all":
            origin = self._origin(query)
            conn = self.connect()
            try:
                out = desk.approve_all_qualified(conn, origin)
            finally:
                conn.close()
            out["drop_path"] = str(self.refresh_drop(origin))
            return 200, out
        if path in ("/api/scan", "/api/settle"):
            mode = body.get("mode") or self.mode
            if mode not in ("live", "mock"):
                raise HttpError(400, f"unknown mode {mode!r}")
            kind = "scan" if path == "/api/scan" else "settle"
            try:
                job = self.jobs.submit(kind, mode)
            except JobBusy as exc:
                raise HttpError(409, str(exc)) from exc
            return 202, {"job": job.as_dict()}
        if path == "/api/settings":
            mode = body.get("mode")
            if mode not in ("live", "mock"):
                raise HttpError(400, "mode must be 'live' or 'mock'")
            with self._settings_lock:
                save_settings(self.settings_path, {**load_settings(self.settings_path),
                                                   "mode": mode})
            return 200, self.status()
        raise HttpError(404, f"no route POST {path}")

    @staticmethod
    def _writeup(conn: sqlite3.Connection, pick_id: str, body: dict[str, Any]) -> dict[str, Any]:
        if body.get("regenerate"):
            return desk.regenerate_writeup(conn, pick_id)
        text = body.get("text")
        if not isinstance(text, str):
            raise HttpError(400, 'send {"text": "..."} or {"regenerate": true}')
        return desk.save_writeup(conn, pick_id, text)

    def static_file(self, path: str) -> tuple[bytes, str] | None:
        rel = "index.html" if path in ("", "/") else path.lstrip("/")
        if rel.startswith("static/"):
            rel = rel[len("static/"):]
        target = (self.static_dir / rel).resolve()
        if not target.is_relative_to(self.static_dir.resolve()) or not target.is_file():
            return None
        ctype = mimetypes.guess_type(target.name)[0] or "application/octet-stream"
        if ctype.startswith("text/") or ctype in ("application/javascript",):
            ctype += "; charset=utf-8"
        return target.read_bytes(), ctype


# ==========================================================================
# HTTP transport
# ==========================================================================
def make_handler(app: DeskApp) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        server_version = "BreezeDesk/1.0"

        def log_message(self, fmt: str, *args: Any) -> None:     # quiet by default
            if getattr(self.server, "verbose", False):
                super().log_message(fmt, *args)

        def _send(self, status: int, payload: bytes, ctype: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Referrer-Policy", "no-referrer")
            self.end_headers()
            self.wfile.write(payload)

        def _json(self, status: int, data: Any) -> None:
            self._send(status, json.dumps(data).encode("utf-8"), "application/json")

        def _host_ok(self) -> bool:
            host = (self.headers.get("Host") or "").strip().lower()
            name = host.rsplit(":", 1)[0] if not host.startswith("[") else host.split("]")[0] + "]"
            return name in LOCAL_HOSTS

        def _dispatch(self, method: str) -> None:
            if not self._host_ok():
                self._json(403, {"error": "desk only answers on localhost"})
                return
            url = urlsplit(self.path)
            if method == "GET" and not url.path.startswith("/api/"):
                found = app.static_file(url.path)
                if found is None:
                    self._json(404, {"error": "not found"})
                else:
                    self._send(200, *found)
                return
            body: dict[str, Any] = {}
            if method == "POST":
                if self.headers.get(CSRF_HEADER) != "1":
                    self._json(403, {"error": f"missing {CSRF_HEADER} header"})
                    return
                length = int(self.headers.get("Content-Length") or 0)
                if length > MAX_BODY:
                    self._json(413, {"error": "body too large"})
                    return
                raw = self.rfile.read(length) if length else b""
                if raw:
                    try:
                        body = json.loads(raw)
                    except ValueError:
                        self._json(400, {"error": "body is not JSON"})
                        return
                    if not isinstance(body, dict):
                        self._json(400, {"error": "body must be a JSON object"})
                        return
            try:
                status, data = app.handle(method, url.path, parse_qs(url.query), body)
            except HttpError as exc:
                status, data = exc.status, {"error": str(exc)}
            except desk.DeskError as exc:
                status, data = exc.status, {"error": str(exc)}
            except Exception as exc:                               # noqa: BLE001
                status, data = 500, {"error": f"{type(exc).__name__}: {exc}"}
            self._json(status, data)

        def do_GET(self) -> None:                                  # noqa: N802
            self._dispatch("GET")

        def do_POST(self) -> None:                                 # noqa: N802
            self._dispatch("POST")

    return Handler


def make_server(app: DeskApp, host: str = "127.0.0.1", port: int = 8000) -> ThreadingHTTPServer:
    server = ThreadingHTTPServer((host, port), make_handler(app))
    server.daemon_threads = True
    return server


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--host", default="127.0.0.1",
                    help="bind address (default 127.0.0.1; the desk has no login)")
    ap.add_argument("--db", type=Path, default=db.DB_PATH, help="SQLite path")
    ap.add_argument("--no-scheduler", action="store_true", help="disable the 09:00 daily run")
    ap.add_argument("--daily-run", action="store_true",
                    help="run the 09:00 job (settle, then stats + scan) once and exit")
    ap.add_argument("--mock", action="store_true", help="with --daily-run: force mock mode")
    ap.add_argument("--live", action="store_true", help="with --daily-run: force live mode")
    ap.add_argument("--verbose", action="store_true", help="log every request")
    args = ap.parse_args(argv)
    load_env()                       # .env settings (Ollama write-ups, API keys) like the CLI
    app = DeskApp(db_path=args.db)

    if args.daily_run:
        mode = "mock" if args.mock else "live" if args.live else app.mode
        try:
            job = app.jobs.run("daily", mode, trigger="task-scheduler")
        except JobBusy as exc:
            print(f"[desk] skipped: {exc}")
            return 0
        for step in job.steps:
            print(f"$ run_pipeline.py {' '.join(step['args'])}  -> rc {step['returncode']}")
            print(step["tail"])
        app.refresh_drop(origin_for(mode))
        return 0 if job.status == "ok" else 1

    server = make_server(app, args.host, args.port)
    server.verbose = args.verbose  # type: ignore[attr-defined]
    if not args.no_scheduler:
        app.start_scheduler()
    host, port = server.server_address[:2]
    print(f"[desk] Breeze Bets desk on http://{'localhost' if host == '127.0.0.1' else host}:"
          f"{port}  | mode {app.mode} | db {args.db} | 09:00 scheduler "
          f"{'off' if args.no_scheduler else 'on'}  (Ctrl+C to stop)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[desk] stopped")
    finally:
        if app.scheduler:
            app.scheduler.stop()
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
