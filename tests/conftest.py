"""Shared pytest setup: never call a real Ollama from the test suite.

Write-up tests that exercise the Ollama path patch the HTTP call explicitly.
"""

import pytest


@pytest.fixture(autouse=True)
def _no_live_ollama(monkeypatch):
    monkeypatch.setenv("BREEZE_WRITEUP_LLM", "off")
    try:
        from templates import writeup_generator
    except ImportError:                       # pragma: no cover - module always present
        yield
        return
    writeup_generator.reset_ollama_cooldown()
    yield
    writeup_generator.reset_ollama_cooldown()
