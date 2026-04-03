"""Shared test fixtures."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from bist_predict.config import Config, DataConfig


@pytest.fixture
def tmp_db_path(tmp_path: Path) -> Path:
    """Return a temporary database path."""
    return tmp_path / "test.db"


@pytest.fixture
def config(tmp_db_path: Path) -> Config:
    """Return a test config with temporary database."""
    return Config(
        data=DataConfig(fetch_retries=1, rate_limit_delay=0.0),
        db_path=tmp_db_path,
    )
