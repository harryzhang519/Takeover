"""Minimal stdlib JSON-over-HTTP client (no ``requests`` dependency).

Retries transient failures (network errors, 429, 5xx) with short linear
backoff, and never echoes query strings in error messages so API keys
passed as URL parameters cannot leak into logs.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Mapping

DEFAULT_TIMEOUT_S: float = 15.0
USER_AGENT = "breeze-quant/1.0"


class HttpError(RuntimeError):
    """Raised for non-2xx responses or network failures after retries."""

    def __init__(self, message: str, status: int | None = None) -> None:
        super().__init__(message)
        self.status = status


def redact_url(url: str) -> str:
    """Strip the query string so secrets in parameters are never logged."""
    parts = urllib.parse.urlsplit(url)
    return urllib.parse.urlunsplit((parts.scheme, parts.netloc, parts.path, "", ""))


def get_json(
    url: str,
    params: Mapping[str, Any] | None = None,
    headers: Mapping[str, str] | None = None,
    timeout: float = DEFAULT_TIMEOUT_S,
    retries: int = 2,
) -> tuple[Any, dict[str, str]]:
    """GET ``url`` and decode JSON.

    Returns:
        (payload, response_headers) with header names lower-cased.

    Raises:
        HttpError: on non-retryable status, or once retries are exhausted.
    """
    full = url + ("?" + urllib.parse.urlencode(params) if params else "")
    req_headers = {"Accept": "application/json", "User-Agent": USER_AGENT, **(headers or {})}
    last_err: HttpError | None = None

    for attempt in range(retries + 1):
        req = urllib.request.Request(full, headers=req_headers, method="GET")
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                body = resp.read()
                hdrs = {k.lower(): v for k, v in resp.headers.items()}
                return json.loads(body.decode("utf-8")), hdrs
        except urllib.error.HTTPError as exc:
            detail = exc.read()[:300].decode("utf-8", "replace")
            last_err = HttpError(f"HTTP {exc.code} from {redact_url(url)}: {detail}", exc.code)
            if exc.code != 429 and exc.code < 500:
                raise last_err from None
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            reason = getattr(exc, "reason", exc)
            last_err = HttpError(f"Network error reaching {redact_url(url)}: {reason}")
        except json.JSONDecodeError:
            raise HttpError(f"Non-JSON response from {redact_url(url)}") from None
        if attempt < retries:
            time.sleep(1.0 + attempt)
    assert last_err is not None
    raise last_err
