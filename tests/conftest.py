"""Shared test fixtures."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from bist_predict.config import Config, DataConfig

ROOT = Path(__file__).resolve().parents[1]
_COMMITTED_RUN_ID = re.compile(r"^COMMITTED_RUN_ID \?= (?P<run_id>\S+)$", re.MULTILINE)


def accepted_run_directory() -> Path:
    """Return the run bundle the published results are read out of.

    The identifier lives in the Makefile, which is what every documented command
    uses, so the tests and the commands cannot disagree about which run is the
    accepted one.
    """
    match = _COMMITTED_RUN_ID.search((ROOT / "Makefile").read_text(encoding="utf-8"))
    if match is None:
        raise RuntimeError("the Makefile does not declare COMMITTED_RUN_ID")
    return ROOT / "runs" / match.group("run_id")


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
