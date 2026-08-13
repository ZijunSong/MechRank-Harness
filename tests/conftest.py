"""Pytest configuration."""

from __future__ import annotations

import pytest


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "integration: tests that call a real LLM endpoint")
    config.addinivalue_line("markers", "e2e: end-to-end server tests")
