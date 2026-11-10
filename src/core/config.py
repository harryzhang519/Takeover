"""Environment configuration, credential setup, permissions and API pings.

* ``.env`` (git-ignored) is loaded with python-dotenv when installed, else a
  minimal built-in parser.
* ``setup_env`` merges keys into ``.env`` atomically, restricts the file to
  the current user (chmod 600 on POSIX, icacls ACL on Windows), validates the
  result, and pings both APIs.
* Secrets are never printed: only masked forms are shown.

CLI:
    python -m src.core.config --setup-env [--odds-key K] [--kalshi-key-id ID]
                                          [--kalshi-key-path PATH] [--bankroll 100]
    python -m src.core.config --check
"""

from __future__ import annotations

import argparse
import base64
import getpass
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.http_client import HttpError, get_json  # noqa: E402

ENV_PATH = ROOT / ".env"
ODDS_API_BASE = "https://api.the-odds-api.com/v4"
KALSHI_BASE_DEFAULT = "https://external-api.kalshi.com/trade-api/v2"
DEFAULT_BANKROLL = 100.0

# key -> (prompt label, is_secret)
ENV_FIELDS: dict[str, tuple[str, bool]] = {
    "ODDS_API_KEY": ("The-Odds-API key", True),
    "KALSHI_API_KEY_ID": ("Kalshi API key ID", False),
    "KALSHI_PRIVATE_KEY_PATH": ("Kalshi private key PEM path (blank to skip)", False),
    "BANKROLL_UNITS": ("Baseline bankroll units", False),
}


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------
def _parse_env_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        return values
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        val = val.strip()
        if len(val) >= 2 and val[0] == val[-1] and val[0] in "\"'":
            val = val[1:-1]
        values[key.strip()] = val
    return values


def load_env(path: Path = ENV_PATH, override: bool = False) -> dict[str, str]:
    """Load ``.env`` into ``os.environ``; return the file's values."""
    try:
        from dotenv import dotenv_values, load_dotenv
    except ImportError:
        values = _parse_env_file(path)
        for k, v in values.items():
            if override or k not in os.environ:
                os.environ[k] = v
        return values
    load_dotenv(path, override=override)
    return {k: v for k, v in dotenv_values(path).items() if v is not None}


def get_setting(name: str, default: str | None = None) -> str | None:
    val = os.environ.get(name, "").strip()
    return val or default


def bankroll_units() -> float:
    """Fixed baseline bankroll from BANKROLL_UNITS (default 100).

    This is a configured constant, never derived from P&L (anti-house-money).
    """
    raw = get_setting("BANKROLL_UNITS", str(DEFAULT_BANKROLL))
    try:
        value = float(raw)  # type: ignore[arg-type]
    except ValueError:
        raise ValueError(f"BANKROLL_UNITS must be a number, got {raw!r}") from None
    if value <= 0:
        raise ValueError("BANKROLL_UNITS must be positive")
    return value


def kalshi_base_url() -> str:
    return (get_setting("KALSHI_BASE_URL") or KALSHI_BASE_DEFAULT).rstrip("/")


def mask(secret: str | None) -> str:
    if not secret:
        return "<unset>"
    return "****" + secret[-4:] if len(secret) > 8 else "****"


# --------------------------------------------------------------------------
# Writing and permissions
# --------------------------------------------------------------------------
def _quote(value: str) -> str:
    if value == "" or re.fullmatch(r"[A-Za-z0-9_./:@+\\-]+", value):
        return value
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'


def secure_file(path: Path) -> None:
    """Restrict ``path`` to the current user (best effort, never raises)."""
    try:
        if os.name == "nt":
            user = os.environ.get("USERNAME", "")
            if user:
                subprocess.run(["icacls", str(path), "/inheritance:r", "/grant:r", f"{user}:F"],
                               capture_output=True, check=False, timeout=15)
        else:
            os.chmod(path, 0o600)
    except (OSError, subprocess.SubprocessError):
        pass


def check_file_permissions(path: Path) -> tuple[bool, str]:
    """Verify that only the owner (plus SYSTEM/Administrators on Windows) can read ``path``."""
    if not path.exists():
        return False, f"{path.name} does not exist"
    if os.name != "nt":
        mode = path.stat().st_mode & 0o777
        ok = mode & 0o077 == 0
        return ok, f"mode {oct(mode)}" + ("" if ok else " (group/other access; run chmod 600)")
    try:
        out = subprocess.run(["icacls", str(path)], capture_output=True, text=True,
                             check=False, timeout=15).stdout
    except (OSError, subprocess.SubprocessError) as exc:
        return False, f"could not read ACL: {exc}"
    user = os.environ.get("USERNAME", "").lower()
    allowed = {"nt authority\\system", "builtin\\administrators"}
    principals = []
    for line in out.replace(str(path), "", 1).splitlines():
        line = line.strip()
        if ":(" in line:
            principals.append(line.split(":(", 1)[0].strip())
    others = [p for p in principals
              if p.lower() not in allowed and p.lower().rsplit("\\", 1)[-1] != user]
    if not principals:
        return False, "no ACL entries parsed"
    if others:
        return False, f"also readable by: {', '.join(others)}"
    return True, f"ACL restricted to {', '.join(principals)}"


def check_gitignored(path: Path) -> tuple[bool, str]:
    """Confirm ``path`` is git-ignored (git check-ignore, else .gitignore scan)."""
    try:
        res = subprocess.run(["git", "check-ignore", "-q", str(path)], cwd=ROOT,
                             capture_output=True, check=False, timeout=15)
        if res.returncode in (0, 1):
            return res.returncode == 0, "git check-ignore"
    except (OSError, subprocess.SubprocessError):
        pass
    gi = ROOT / ".gitignore"
    lines = gi.read_text(encoding="utf-8").splitlines() if gi.exists() else []
    ok = path.name in [ln.strip() for ln in lines]
    return ok, ".gitignore scan"


def write_env(updates: dict[str, str], path: Path = ENV_PATH) -> None:
    """Merge ``updates`` into ``path`` atomically; keep other lines; restrict access."""
    lines = path.read_text(encoding="utf-8").splitlines() if path.exists() else [
        "# Breeze credentials - git-ignored. Do not commit."]
    pending = dict(updates)
    for i, line in enumerate(lines):
        key = line.split("=", 1)[0].strip()
        if key in pending and not line.lstrip().startswith("#"):
            lines[i] = f"{key}={_quote(pending.pop(key))}"
    lines += [f"{k}={_quote(v)}" for k, v in pending.items()]
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text("\n".join(lines) + "\n", encoding="utf-8")
    os.replace(tmp, path)
    secure_file(path)


# --------------------------------------------------------------------------
# Kalshi request signing (RSA-PSS SHA-256, or Ed25519)
# --------------------------------------------------------------------------
def normalize_pem(pem: str) -> str:
    """Rebuild a PEM pasted on one line (spaces instead of newlines)."""
    pem = pem.strip().replace("\\n", "\n")
    m = re.match(r"-----BEGIN ([A-Z ]+)-----(.*)-----END \1-----", pem, re.S)
    if not m:
        raise ValueError("Not a PEM block")
    body = re.sub(r"\s+", "", m.group(2))
    wrapped = "\n".join(body[i:i + 64] for i in range(0, len(body), 64))
    return f"-----BEGIN {m.group(1)}-----\n{wrapped}\n-----END {m.group(1)}-----\n"


def load_kalshi_private_key(key_path: str | None = None, inline_pem: str | None = None) -> Any:
    """Load an RSA or Ed25519 private key from a PEM path (preferred) or inline PEM."""
    try:
        from cryptography.hazmat.primitives import serialization
        from cryptography.hazmat.primitives.asymmetric import ed25519, rsa
    except ImportError as exc:
        raise RuntimeError("Install 'cryptography' for Kalshi signed requests") from exc
    if key_path:
        p = Path(key_path)
        p = p if p.is_absolute() else ROOT / p
        if not p.exists():
            raise RuntimeError(f"KALSHI_PRIVATE_KEY_PATH not found: {p}")
        pem = p.read_text(encoding="utf-8")
    elif inline_pem:
        pem = inline_pem
    else:
        raise RuntimeError("No Kalshi private key configured")
    key = serialization.load_pem_private_key(normalize_pem(pem).encode(), password=None)
    if not isinstance(key, (rsa.RSAPrivateKey, ed25519.Ed25519PrivateKey)):
        raise RuntimeError(f"Unsupported key type {type(key).__name__}; need RSA or Ed25519")
    return key


def configured_kalshi_key() -> Any | None:
    """Key from KALSHI_PRIVATE_KEY_PATH or legacy inline KALSHI_API_SECRET, else None."""
    path, inline = get_setting("KALSHI_PRIVATE_KEY_PATH"), get_setting("KALSHI_API_SECRET")
    if not (path or inline):
        return None
    return load_kalshi_private_key(path, inline)


def kalshi_auth_headers(method: str, path: str, key_id: str, private_key: Any) -> dict[str, str]:
    """Kalshi v2 auth headers. Signs ``timestamp_ms + METHOD + path`` (path without query).

    RSA keys use PSS (MGF1-SHA256, digest-length salt); Ed25519 keys sign raw.
    Kalshi's documentation specifies RSA-PSS; Ed25519 is supported here for
    forward compatibility and will be rejected by Kalshi if unsupported.
    """
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.asymmetric import ed25519, padding

    ts = str(int(time.time() * 1000))
    msg = f"{ts}{method.upper()}{path.split('?', 1)[0]}".encode()
    if isinstance(private_key, ed25519.Ed25519PrivateKey):
        sig = private_key.sign(msg)
    else:
        sig = private_key.sign(msg, padding.PSS(mgf=padding.MGF1(hashes.SHA256()),
                                                salt_length=padding.PSS.DIGEST_LENGTH),
                               hashes.SHA256())
    return {"KALSHI-ACCESS-KEY": key_id, "KALSHI-ACCESS-TIMESTAMP": ts,
            "KALSHI-ACCESS-SIGNATURE": base64.b64encode(sig).decode()}


def kalshi_headers_for(url_path: str) -> dict[str, str]:
    """Signed headers if a key ID and private key are configured, else {}."""
    key_id = get_setting("KALSHI_API_KEY_ID")
    if not key_id:
        return {}
    try:
        key = configured_kalshi_key()
    except (RuntimeError, ValueError):
        return {}
    return kalshi_auth_headers("GET", url_path, key_id, key) if key else {}


# --------------------------------------------------------------------------
# Pings
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class PingResult:
    service: str
    ok: bool
    detail: str


def ping_odds_api(api_key: str | None) -> PingResult:
    """GET /v4/sports (free: does not consume quota)."""
    if not api_key:
        return PingResult("The-Odds-API", False, "ODDS_API_KEY not set")
    try:
        data, hdrs = get_json(f"{ODDS_API_BASE}/sports", {"apiKey": api_key}, retries=1)
    except HttpError as exc:
        return PingResult("The-Odds-API", False, str(exc))
    return PingResult("The-Odds-API", True, f"{len(data)} sports; quota remaining "
                      f"{hdrs.get('x-requests-remaining', 'n/a')}, used "
                      f"{hdrs.get('x-requests-used', 'n/a')}")


def ping_kalshi() -> PingResult:
    """Public exchange status, then a signed balance call if a key is configured."""
    base = kalshi_base_url()
    try:
        status, _ = get_json(f"{base}/exchange/status", retries=1)
    except HttpError as exc:
        return PingResult("Kalshi", False, str(exc))
    public = f"exchange_active={status.get('exchange_active')}"
    key_id = get_setting("KALSHI_API_KEY_ID")
    if not key_id:
        return PingResult("Kalshi", True, f"{public}; public data only (no key ID)")
    try:
        key = configured_kalshi_key()
    except (RuntimeError, ValueError) as exc:
        return PingResult("Kalshi", False, f"{public}; key not loadable: {exc}")
    if key is None:
        return PingResult("Kalshi", True, f"{public}; public data only (no private key)")
    path = "/trade-api/v2/portfolio/balance"
    try:
        get_json(f"{base}/portfolio/balance",
                 headers=kalshi_auth_headers("GET", path, key_id, key), retries=0)
    except HttpError as exc:
        return PingResult("Kalshi", False, f"{public}; signed auth rejected "
                          f"({type(key).__name__.lstrip('_')}): {exc}")
    return PingResult("Kalshi", True, f"{public}; signed auth OK")


def ping_apis() -> list[PingResult]:
    return [ping_odds_api(get_setting("ODDS_API_KEY")), ping_kalshi()]


# --------------------------------------------------------------------------
# Setup
# --------------------------------------------------------------------------
def setup_env(odds_key: str | None = None, kalshi_key_id: str | None = None,
              private_key_path: str | None = None, bankroll: float | None = None,
              interactive: bool | None = None, ping: bool = True,
              path: Path = ENV_PATH) -> bool:
    """Write keys to ``.env``, lock down permissions, validate, then ping.

    Precedence per key: argument > existing value > interactive prompt.
    Returns True when permissions, git-ignore and all pings check out.
    """
    existing = load_env(path)
    if interactive is None:
        interactive = sys.stdin.isatty()
    provided = {"ODDS_API_KEY": odds_key, "KALSHI_API_KEY_ID": kalshi_key_id,
                "KALSHI_PRIVATE_KEY_PATH": private_key_path,
                "BANKROLL_UNITS": None if bankroll is None else f"{float(bankroll):.2f}"}
    updates: dict[str, str] = {}
    for key, (label, secret) in ENV_FIELDS.items():
        value = (provided[key] or "").strip()
        if not value and existing.get(key, "").strip():
            continue
        if (not value and key == "KALSHI_PRIVATE_KEY_PATH"
                and existing.get("KALSHI_API_SECRET", "").strip()):
            continue                                    # legacy inline key already configured
        if not value and key == "BANKROLL_UNITS":
            value = f"{DEFAULT_BANKROLL:.2f}"
        elif not value and interactive:
            try:
                value = (getpass.getpass if secret else input)(f"{label}: ").strip()
            except (EOFError, KeyboardInterrupt):
                print(f"\n[env] no input for {key}; leaving unset")
                value = ""
        if value:
            updates[key] = value
    if updates.get("KALSHI_PRIVATE_KEY_PATH"):
        load_kalshi_private_key(updates["KALSHI_PRIVATE_KEY_PATH"])     # fail fast on a bad key

    write_env(updates, path) if (updates or not path.exists()) else secure_file(path)
    load_env(path, override=True)
    bankroll_units()                                                    # validates

    print(f"[env] {path.name}: " + ", ".join(
        f"{k}={mask(get_setting(k)) if secret else (get_setting(k) or '<unset>')}"
        for k, (_, secret) in ENV_FIELDS.items()))
    ok_perm, perm = check_file_permissions(path)
    ok_git, how = check_gitignored(path)
    print(f"[env] permissions: {'OK  ' if ok_perm else 'WARN'} {perm}")
    print(f"[env] git-ignored: {'OK  ' if ok_git else 'FAIL'} ({how})")
    all_ok = ok_perm and ok_git
    if ping:
        for r in ping_apis():
            print(f"[ping] {r.service:<12} {'OK  ' if r.ok else 'FAIL'} {r.detail}")
            all_ok &= r.ok
    return all_ok


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Breeze environment setup")
    ap.add_argument("--setup-env", action="store_true", help="Write keys to .env and ping APIs")
    ap.add_argument("--check", action="store_true", help="Validate .env and ping without writing")
    ap.add_argument("--odds-key")
    ap.add_argument("--kalshi-key-id")
    ap.add_argument("--kalshi-key-path")
    ap.add_argument("--bankroll", type=float)
    ap.add_argument("--no-ping", action="store_true")
    args = ap.parse_args(argv)
    if args.check:
        load_env()
        ok_perm, perm = check_file_permissions(ENV_PATH)
        print(f"[env] permissions: {'OK' if ok_perm else 'WARN'} {perm}")
        results = ping_apis()
        for r in results:
            print(f"[ping] {r.service:<12} {'OK  ' if r.ok else 'FAIL'} {r.detail}")
        return 0 if ok_perm and all(r.ok for r in results) else 1
    if args.setup_env:
        return 0 if setup_env(args.odds_key, args.kalshi_key_id, args.kalshi_key_path,
                              args.bankroll, ping=not args.no_ping) else 1
    ap.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
