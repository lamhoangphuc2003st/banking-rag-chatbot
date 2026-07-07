"""Shared pytest fixtures.

The RAG settings are read from process environment variables via
``pydantic-settings``. Some third-party clients (e.g. litellm) mutate
``os.environ`` as a side effect of a call, which previously leaked an
``OPENAI_API_KEY`` from one test into the next and made the suite
order-dependent. The autouse fixture below snapshots and restores the
environment (and the cached ``Settings`` singleton) around every test so
each test observes a clean, deterministic configuration.
"""

from __future__ import annotations

import os
from collections.abc import Iterator

import pytest

from apps.api.app.core.config import get_settings


@pytest.fixture(autouse=True)
def isolate_settings_environment() -> Iterator[None]:
    saved_environment = dict(os.environ)
    get_settings.cache_clear()
    try:
        yield
    finally:
        os.environ.clear()
        os.environ.update(saved_environment)
        get_settings.cache_clear()
