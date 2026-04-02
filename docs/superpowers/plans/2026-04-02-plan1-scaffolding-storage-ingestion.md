# Plan 1: Project Scaffolding, Storage & Data Ingestion

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Set up the project structure, SQLite storage layer, and multi-source data ingestion pipeline (Is Yatirim, Yahoo Finance, TCMB EVDS, sentiment sources) for BIST-100 stocks.

**Architecture:** Python package (`bist_predict`) with Click CLI entry point, SQLite database for persistence, and modular data collectors with fallback logic. Each collector implements a common interface. A scheduler orchestrates fetching from all sources with rate limiting and incremental updates.

**Tech Stack:** Python 3.12+, uv (package manager), Click (CLI), httpx (async HTTP), yfinance, feedparser, SQLite, pytest

**Design spec:** `docs/superpowers/specs/2026-04-02-bist-predictor-design.md`

---

## File Structure

```
bist-predictor/
├── pyproject.toml                      # Python project config (uv)
├── Cargo.toml                          # Rust workspace root (placeholder for Plan 2)
├── config.toml                         # Default configuration
├── .gitignore
│
├── src/
│   └── bist_predict/
│       ├── __init__.py                 # Package init, version
│       ├── cli.py                      # Click CLI — fetch command
│       ├── config.py                   # Load/validate config.toml
│       │
│       ├── storage/
│       │   ├── __init__.py
│       │   ├── database.py             # SQLite connection, schema init
│       │   └── migrations.py           # Schema versioning
│       │
│       └── ingest/
│           ├── __init__.py
│           ├── types.py                # OHLCVBar dataclass, collector protocol
│           ├── scheduler.py            # Orchestrates all collectors
│           ├── isyatirim.py            # Is Yatirim API client
│           ├── yahoo.py               # Yahoo Finance fallback
│           ├── tcmb.py                # TCMB EVDS macro data
│           ├── sentiment.py           # News & social sentiment
│           └── quality.py             # OHLCV validation rules
│
├── tests/
│   ├── conftest.py                    # Shared fixtures (tmp db, mock data)
│   ├── test_storage/
│   │   ├── __init__.py
│   │   └── test_database.py
│   └── test_ingest/
│       ├── __init__.py
│       ├── test_types.py
│       ├── test_quality.py
│       ├── test_isyatirim.py
│       ├── test_yahoo.py
│       ├── test_tcmb.py
│       ├── test_sentiment.py
│       └── test_scheduler.py
│
└── data/                              # Created at runtime, gitignored
```

---

### Task 1: Project Scaffolding

**Files:**
- Create: `pyproject.toml`
- Create: `.gitignore`
- Create: `src/bist_predict/__init__.py`
- Create: `config.toml`
- Create: `src/bist_predict/config.py`
- Create: `tests/conftest.py`
- Create: `Cargo.toml`

- [ ] **Step 1: Initialize git repo**

```bash
cd /Users/dmr/Development/Projects/PersonalProjects/quant/BIST-Predictorcl
git init
```

- [ ] **Step 2: Create .gitignore**

```gitignore
# Python
__pycache__/
*.py[cod]
*.egg-info/
dist/
build/
.venv/
*.so

# Rust
target/

# Data
data/
*.db

# IDE
.idea/
.vscode/
*.swp

# OS
.DS_Store

# Project
.superpowers/
config.toml
```

Note: `config.toml` is gitignored because it may contain API keys. We'll create a `config.example.toml` for reference.

- [ ] **Step 3: Create pyproject.toml**

```toml
[project]
name = "bist-predict"
version = "0.1.0"
description = "BIST-100 Stock Market Prediction System"
requires-python = ">=3.12"
dependencies = [
    "click>=8.1",
    "httpx>=0.27",
    "yfinance>=0.2",
    "feedparser>=6.0",
    "tomli>=2.0; python_version < '3.11'",
]

[project.scripts]
bist-predict = "bist_predict.cli:main"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/bist_predict"]

[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["src"]

[dependency-groups]
dev = [
    "pytest>=8.0",
    "pytest-asyncio>=0.23",
    "respx>=0.21",
]
```

- [ ] **Step 4: Create Cargo.toml workspace placeholder**

```toml
[workspace]
members = ["rust/bist_features"]
resolver = "2"
```

- [ ] **Step 5: Create src/bist_predict/__init__.py**

```python
"""BIST-100 Stock Market Prediction System."""

__version__ = "0.1.0"
```

- [ ] **Step 6: Create config.toml and config.example.toml**

Create `config.example.toml` (tracked in git):

```toml
[data]
tcmb_api_key = ""
fetch_retries = 3
rate_limit_delay = 1.0

[signals]
min_confidence = 0.70
lookback_days = 30

[models]
retrain_interval = "monthly"
ensemble_weights = "learned"

[quant]
hmm_states = 3
kelly_fraction = 0.25
hurst_window = 252

[backtest]
commission = 0.001
slippage = 0.0005
```

Create `config.toml` (gitignored, local copy):

Same content as `config.example.toml`.

- [ ] **Step 7: Create src/bist_predict/config.py**

```python
"""Configuration management — loads and validates config.toml."""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.toml"
DEFAULT_DB_PATH = PROJECT_ROOT / "data" / "bist.db"


@dataclass(frozen=True)
class DataConfig:
    tcmb_api_key: str = ""
    fetch_retries: int = 3
    rate_limit_delay: float = 1.0


@dataclass(frozen=True)
class SignalsConfig:
    min_confidence: float = 0.70
    lookback_days: int = 30


@dataclass(frozen=True)
class ModelsConfig:
    retrain_interval: str = "monthly"
    ensemble_weights: str = "learned"


@dataclass(frozen=True)
class QuantConfig:
    hmm_states: int = 3
    kelly_fraction: float = 0.25
    hurst_window: int = 252


@dataclass(frozen=True)
class BacktestConfig:
    commission: float = 0.001
    slippage: float = 0.0005


@dataclass(frozen=True)
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    signals: SignalsConfig = field(default_factory=SignalsConfig)
    models: ModelsConfig = field(default_factory=ModelsConfig)
    quant: QuantConfig = field(default_factory=QuantConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)
    db_path: Path = DEFAULT_DB_PATH


def load_config(path: Path = DEFAULT_CONFIG_PATH) -> Config:
    """Load configuration from a TOML file. Returns defaults if file missing."""
    if not path.exists():
        return Config()

    with open(path, "rb") as f:
        raw = tomllib.load(f)

    return Config(
        data=DataConfig(**raw.get("data", {})),
        signals=SignalsConfig(**raw.get("signals", {})),
        models=ModelsConfig(**raw.get("models", {})),
        quant=QuantConfig(**raw.get("quant", {})),
        backtest=BacktestConfig(**raw.get("backtest", {})),
    )
```

- [ ] **Step 8: Create tests/conftest.py**

```python
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
```

- [ ] **Step 9: Install dependencies and verify**

```bash
cd /Users/dmr/Development/Projects/PersonalProjects/quant/BIST-Predictorcl
uv sync
uv run python -c "from bist_predict.config import load_config; print(load_config())"
```

Expected: prints the default Config object with no errors.

- [ ] **Step 10: Commit**

```bash
git add .gitignore pyproject.toml Cargo.toml config.example.toml \
    src/bist_predict/__init__.py src/bist_predict/config.py \
    tests/conftest.py CLAUDE.md docs/
git commit -m "feat: project scaffolding with config, pyproject, and test fixtures"
```

---

### Task 2: SQLite Storage Layer

**Files:**
- Create: `src/bist_predict/storage/__init__.py`
- Create: `src/bist_predict/storage/database.py`
- Create: `src/bist_predict/storage/migrations.py`
- Create: `tests/test_storage/__init__.py`
- Create: `tests/test_storage/test_database.py`

- [ ] **Step 1: Create storage package init**

`src/bist_predict/storage/__init__.py`:

```python
"""SQLite storage layer."""
```

- [ ] **Step 2: Write failing tests for database**

`tests/test_storage/__init__.py`: empty file.

`tests/test_storage/test_database.py`:

```python
"""Tests for SQLite database layer."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from bist_predict.storage.database import Database


class TestDatabaseInit:
    def test_creates_db_file(self, tmp_db_path: Path) -> None:
        db = Database(tmp_db_path)
        db.initialize()
        assert tmp_db_path.exists()

    def test_creates_data_directory(self, tmp_path: Path) -> None:
        db_path = tmp_path / "subdir" / "test.db"
        db = Database(db_path)
        db.initialize()
        assert db_path.exists()

    def test_creates_raw_prices_table(self, tmp_db_path: Path) -> None:
        db = Database(tmp_db_path)
        db.initialize()
        with db.connect() as conn:
            cursor = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='raw_prices'"
            )
            assert cursor.fetchone() is not None

    def test_creates_macro_data_table(self, tmp_db_path: Path) -> None:
        db = Database(tmp_db_path)
        db.initialize()
        with db.connect() as conn:
            cursor = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='macro_data'"
            )
            assert cursor.fetchone() is not None

    def test_creates_sentiment_data_table(self, tmp_db_path: Path) -> None:
        db = Database(tmp_db_path)
        db.initialize()
        with db.connect() as conn:
            cursor = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='sentiment_data'"
            )
            assert cursor.fetchone() is not None

    def test_creates_predictions_table(self, tmp_db_path: Path) -> None:
        db = Database(tmp_db_path)
        db.initialize()
        with db.connect() as conn:
            cursor = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='predictions'"
            )
            assert cursor.fetchone() is not None

    def test_creates_schema_version_table(self, tmp_db_path: Path) -> None:
        db = Database(tmp_db_path)
        db.initialize()
        with db.connect() as conn:
            cursor = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='schema_version'"
            )
            assert cursor.fetchone() is not None

    def test_idempotent_initialize(self, tmp_db_path: Path) -> None:
        db = Database(tmp_db_path)
        db.initialize()
        db.initialize()  # Should not raise
        with db.connect() as conn:
            cursor = conn.execute(
                "SELECT COUNT(*) FROM sqlite_master WHERE type='table'"
            )
            count = cursor.fetchone()[0]
            assert count >= 5


class TestDatabaseOperations:
    def test_insert_and_query_raw_prices(self, tmp_db_path: Path) -> None:
        db = Database(tmp_db_path)
        db.initialize()
        with db.connect() as conn:
            conn.execute(
                """INSERT INTO raw_prices
                   (ticker, date, open, high, low, close, adj_close, volume, source)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                ("THYAO", "2026-04-01", 310.0, 315.0, 308.0, 312.5, 312.5, 1000000, "isyatirim"),
            )
            conn.commit()
            row = conn.execute(
                "SELECT * FROM raw_prices WHERE ticker = ? AND date = ?",
                ("THYAO", "2026-04-01"),
            ).fetchone()
            assert row is not None
            assert row[1] == "THYAO"  # ticker
            assert row[5] == 312.5    # close

    def test_unique_constraint_ticker_date(self, tmp_db_path: Path) -> None:
        db = Database(tmp_db_path)
        db.initialize()
        with db.connect() as conn:
            conn.execute(
                """INSERT INTO raw_prices
                   (ticker, date, open, high, low, close, adj_close, volume, source)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                ("THYAO", "2026-04-01", 310.0, 315.0, 308.0, 312.5, 312.5, 1000000, "isyatirim"),
            )
            conn.commit()
            with pytest.raises(sqlite3.IntegrityError):
                conn.execute(
                    """INSERT INTO raw_prices
                       (ticker, date, open, high, low, close, adj_close, volume, source)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    ("THYAO", "2026-04-01", 311.0, 316.0, 309.0, 313.0, 313.0, 1100000, "yahoo"),
                )

    def test_get_latest_date_for_ticker(self, tmp_db_path: Path) -> None:
        db = Database(tmp_db_path)
        db.initialize()
        with db.connect() as conn:
            for date in ["2026-03-28", "2026-03-31", "2026-04-01"]:
                conn.execute(
                    """INSERT INTO raw_prices
                       (ticker, date, open, high, low, close, adj_close, volume, source)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    ("THYAO", date, 310.0, 315.0, 308.0, 312.5, 312.5, 1000000, "isyatirim"),
                )
            conn.commit()
        latest = db.get_latest_date("THYAO")
        assert latest == "2026-04-01"

    def test_get_latest_date_no_data(self, tmp_db_path: Path) -> None:
        db = Database(tmp_db_path)
        db.initialize()
        latest = db.get_latest_date("THYAO")
        assert latest is None
```

- [ ] **Step 3: Run tests to verify they fail**

```bash
uv run pytest tests/test_storage/test_database.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'bist_predict.storage.database'`

- [ ] **Step 4: Implement database.py**

`src/bist_predict/storage/database.py`:

```python
"""SQLite database connection and schema management."""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Generator

SCHEMA_VERSION = 1

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS schema_version (
    version INTEGER NOT NULL,
    applied_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS raw_prices (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker TEXT NOT NULL,
    date TEXT NOT NULL,
    open REAL NOT NULL,
    high REAL NOT NULL,
    low REAL NOT NULL,
    close REAL NOT NULL,
    adj_close REAL NOT NULL,
    volume INTEGER NOT NULL,
    source TEXT NOT NULL,
    fetched_at TEXT NOT NULL DEFAULT (datetime('now')),
    UNIQUE(ticker, date)
);

CREATE INDEX IF NOT EXISTS idx_raw_prices_ticker_date ON raw_prices(ticker, date);
CREATE INDEX IF NOT EXISTS idx_raw_prices_ticker ON raw_prices(ticker);

CREATE TABLE IF NOT EXISTS macro_data (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    indicator TEXT NOT NULL,
    date TEXT NOT NULL,
    value REAL NOT NULL,
    source TEXT NOT NULL,
    fetched_at TEXT NOT NULL DEFAULT (datetime('now')),
    UNIQUE(indicator, date)
);

CREATE INDEX IF NOT EXISTS idx_macro_data_indicator_date ON macro_data(indicator, date);

CREATE TABLE IF NOT EXISTS sentiment_data (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker TEXT NOT NULL,
    date TEXT NOT NULL,
    source TEXT NOT NULL,
    headline TEXT,
    sentiment_score REAL,
    raw_text TEXT,
    fetched_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_sentiment_ticker_date ON sentiment_data(ticker, date);

CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker TEXT NOT NULL,
    prediction_date TEXT NOT NULL,
    target_date TEXT NOT NULL,
    direction TEXT NOT NULL,
    confidence REAL NOT NULL,
    predicted_pct_move REAL NOT NULL,
    actual_pct_move REAL,
    model_version TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    UNIQUE(ticker, target_date, model_version)
);

CREATE INDEX IF NOT EXISTS idx_predictions_ticker_target ON predictions(ticker, target_date);

CREATE TABLE IF NOT EXISTS features (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker TEXT NOT NULL,
    date TEXT NOT NULL,
    feature_name TEXT NOT NULL,
    value REAL,
    version INTEGER NOT NULL DEFAULT 1,
    UNIQUE(ticker, date, feature_name, version)
);

CREATE INDEX IF NOT EXISTS idx_features_ticker_date ON features(ticker, date);

CREATE TABLE IF NOT EXISTS model_registry (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    model_name TEXT NOT NULL,
    version TEXT NOT NULL,
    model_path TEXT NOT NULL,
    metrics_json TEXT,
    trained_at TEXT NOT NULL DEFAULT (datetime('now')),
    is_active INTEGER NOT NULL DEFAULT 0,
    UNIQUE(model_name, version)
);
"""


class Database:
    """SQLite database for BIST predictor data."""

    def __init__(self, db_path: Path) -> None:
        self._db_path = db_path

    @property
    def path(self) -> Path:
        return self._db_path

    def initialize(self) -> None:
        """Create database file, parent directories, and schema."""
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as conn:
            conn.executescript(SCHEMA_SQL)
            # Record schema version if not already set
            existing = conn.execute("SELECT version FROM schema_version ORDER BY version DESC LIMIT 1").fetchone()
            if existing is None:
                conn.execute("INSERT INTO schema_version (version) VALUES (?)", (SCHEMA_VERSION,))
            conn.commit()

    @contextmanager
    def connect(self) -> Generator[sqlite3.Connection, None, None]:
        """Yield a SQLite connection with WAL mode and foreign keys enabled."""
        conn = sqlite3.connect(str(self._db_path))
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        try:
            yield conn
        finally:
            conn.close()

    def get_latest_date(self, ticker: str) -> str | None:
        """Return the most recent date for a ticker in raw_prices, or None."""
        with self.connect() as conn:
            row = conn.execute(
                "SELECT MAX(date) FROM raw_prices WHERE ticker = ?",
                (ticker,),
            ).fetchone()
            return row[0] if row and row[0] else None
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
uv run pytest tests/test_storage/test_database.py -v
```

Expected: all tests PASS.

- [ ] **Step 6: Create migrations.py stub**

`src/bist_predict/storage/migrations.py`:

```python
"""Schema versioning and migrations.

Migrations are applied in order. Each migration is a SQL string that transforms
the schema from version N to N+1. The current schema version is stored in the
schema_version table.
"""

from __future__ import annotations

import sqlite3

# Map from version -> SQL to migrate TO that version.
# Version 1 is the initial schema (created by database.py SCHEMA_SQL).
MIGRATIONS: dict[int, str] = {}


def get_current_version(conn: sqlite3.Connection) -> int:
    """Return the current schema version."""
    row = conn.execute(
        "SELECT MAX(version) FROM schema_version"
    ).fetchone()
    return row[0] if row and row[0] else 0


def apply_pending_migrations(conn: sqlite3.Connection) -> int:
    """Apply any pending migrations. Returns the final schema version."""
    current = get_current_version(conn)
    for version in sorted(MIGRATIONS.keys()):
        if version > current:
            conn.executescript(MIGRATIONS[version])
            conn.execute(
                "INSERT INTO schema_version (version) VALUES (?)",
                (version,),
            )
            conn.commit()
            current = version
    return current
```

- [ ] **Step 7: Commit**

```bash
git add src/bist_predict/storage/ tests/test_storage/
git commit -m "feat: SQLite storage layer with schema, indices, and migrations"
```

---

### Task 3: Data Types and Collector Protocol

**Files:**
- Create: `src/bist_predict/ingest/__init__.py`
- Create: `src/bist_predict/ingest/types.py`
- Create: `tests/test_ingest/__init__.py`
- Create: `tests/test_ingest/test_types.py`

- [ ] **Step 1: Create ingest package init**

`src/bist_predict/ingest/__init__.py`:

```python
"""Data ingestion module — multi-source market data collection."""
```

- [ ] **Step 2: Write failing tests for types**

`tests/test_ingest/__init__.py`: empty file.

`tests/test_ingest/test_types.py`:

```python
"""Tests for data types and validation."""

from __future__ import annotations

from datetime import date

import pytest

from bist_predict.ingest.types import OHLCVBar, MacroDataPoint, SentimentRecord


class TestOHLCVBar:
    def test_create_valid_bar(self) -> None:
        bar = OHLCVBar(
            ticker="THYAO",
            date=date(2026, 4, 1),
            open=310.0,
            high=315.0,
            low=308.0,
            close=312.5,
            adj_close=312.5,
            volume=1_000_000,
            source="isyatirim",
        )
        assert bar.ticker == "THYAO"
        assert bar.close == 312.5

    def test_date_str(self) -> None:
        bar = OHLCVBar(
            ticker="THYAO",
            date=date(2026, 4, 1),
            open=310.0,
            high=315.0,
            low=308.0,
            close=312.5,
            adj_close=312.5,
            volume=1_000_000,
            source="isyatirim",
        )
        assert bar.date_str == "2026-04-01"


class TestMacroDataPoint:
    def test_create_macro_point(self) -> None:
        point = MacroDataPoint(
            indicator="USD_TRY",
            date=date(2026, 4, 1),
            value=38.45,
            source="tcmb",
        )
        assert point.indicator == "USD_TRY"
        assert point.value == 38.45


class TestSentimentRecord:
    def test_create_sentiment_record(self) -> None:
        record = SentimentRecord(
            ticker="THYAO",
            date=date(2026, 4, 1),
            source="google_news",
            headline="THY hisseleri yükseldi",
            sentiment_score=0.72,
            raw_text="THY hisseleri yükseldi",
        )
        assert record.sentiment_score == 0.72
        assert record.source == "google_news"
```

- [ ] **Step 3: Run tests to verify they fail**

```bash
uv run pytest tests/test_ingest/test_types.py -v
```

Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 4: Implement types.py**

`src/bist_predict/ingest/types.py`:

```python
"""Data types for the ingestion layer."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Protocol, Sequence


@dataclass(frozen=True)
class OHLCVBar:
    """A single OHLCV price bar for one ticker on one date."""

    ticker: str
    date: date
    open: float
    high: float
    low: float
    close: float
    adj_close: float
    volume: int
    source: str

    @property
    def date_str(self) -> str:
        return self.date.isoformat()


@dataclass(frozen=True)
class MacroDataPoint:
    """A single macro-economic data point."""

    indicator: str
    date: date
    value: float
    source: str

    @property
    def date_str(self) -> str:
        return self.date.isoformat()


@dataclass(frozen=True)
class SentimentRecord:
    """A single sentiment observation for a ticker."""

    ticker: str
    date: date
    source: str
    headline: str | None
    sentiment_score: float | None
    raw_text: str | None

    @property
    def date_str(self) -> str:
        return self.date.isoformat()


class PriceCollector(Protocol):
    """Protocol for price data collectors."""

    async def fetch(
        self, ticker: str, start_date: date, end_date: date
    ) -> list[OHLCVBar]: ...


class MacroCollector(Protocol):
    """Protocol for macro data collectors."""

    async def fetch(
        self, indicator: str, start_date: date, end_date: date
    ) -> list[MacroDataPoint]: ...


class SentimentCollector(Protocol):
    """Protocol for sentiment data collectors."""

    async def fetch(
        self, ticker: str, start_date: date, end_date: date
    ) -> list[SentimentRecord]: ...
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
uv run pytest tests/test_ingest/test_types.py -v
```

Expected: all tests PASS.

- [ ] **Step 6: Commit**

```bash
git add src/bist_predict/ingest/ tests/test_ingest/
git commit -m "feat: data types and collector protocols for ingestion layer"
```

---

### Task 4: Data Quality Validation

**Files:**
- Create: `src/bist_predict/ingest/quality.py`
- Create: `tests/test_ingest/test_quality.py`

- [ ] **Step 1: Write failing tests for quality checks**

`tests/test_ingest/test_quality.py`:

```python
"""Tests for OHLCV data quality validation."""

from __future__ import annotations

from datetime import date

import pytest

from bist_predict.ingest.quality import validate_bar, ValidationError
from bist_predict.ingest.types import OHLCVBar


def make_bar(**overrides) -> OHLCVBar:
    """Helper to create a bar with defaults."""
    defaults = dict(
        ticker="THYAO",
        date=date(2026, 4, 1),
        open=310.0,
        high=315.0,
        low=308.0,
        close=312.5,
        adj_close=312.5,
        volume=1_000_000,
        source="isyatirim",
    )
    defaults.update(overrides)
    return OHLCVBar(**defaults)


class TestValidateBar:
    def test_valid_bar_passes(self) -> None:
        bar = make_bar()
        assert validate_bar(bar) is True

    def test_high_below_low_fails(self) -> None:
        bar = make_bar(high=300.0, low=308.0)
        with pytest.raises(ValidationError, match="high .* below low"):
            validate_bar(bar)

    def test_open_above_high_fails(self) -> None:
        bar = make_bar(open=320.0, high=315.0)
        with pytest.raises(ValidationError, match="open .* above high"):
            validate_bar(bar)

    def test_close_above_high_fails(self) -> None:
        bar = make_bar(close=320.0, high=315.0)
        with pytest.raises(ValidationError, match="close .* above high"):
            validate_bar(bar)

    def test_open_below_low_fails(self) -> None:
        bar = make_bar(open=305.0, low=308.0)
        with pytest.raises(ValidationError, match="open .* below low"):
            validate_bar(bar)

    def test_close_below_low_fails(self) -> None:
        bar = make_bar(close=305.0, low=308.0)
        with pytest.raises(ValidationError, match="close .* below low"):
            validate_bar(bar)

    def test_negative_volume_fails(self) -> None:
        bar = make_bar(volume=-100)
        with pytest.raises(ValidationError, match="volume"):
            validate_bar(bar)

    def test_zero_volume_passes(self) -> None:
        bar = make_bar(volume=0)
        assert validate_bar(bar) is True

    def test_negative_price_fails(self) -> None:
        bar = make_bar(close=-1.0, low=-2.0, open=-1.5, high=0.5)
        with pytest.raises(ValidationError, match="negative"):
            validate_bar(bar)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/test_ingest/test_quality.py -v
```

Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement quality.py**

`src/bist_predict/ingest/quality.py`:

```python
"""OHLCV data quality validation rules."""

from __future__ import annotations

from bist_predict.ingest.types import OHLCVBar


class ValidationError(Exception):
    """Raised when a data quality check fails."""


def validate_bar(bar: OHLCVBar) -> bool:
    """Validate an OHLCV bar. Returns True if valid, raises ValidationError otherwise."""
    # Check for negative prices
    for field_name, value in [("open", bar.open), ("high", bar.high), ("low", bar.low), ("close", bar.close)]:
        if value < 0:
            raise ValidationError(f"{bar.ticker} {bar.date_str}: {field_name} is negative ({value})")

    # High must be >= Low
    if bar.high < bar.low:
        raise ValidationError(
            f"{bar.ticker} {bar.date_str}: high ({bar.high}) below low ({bar.low})"
        )

    # Open must be within [low, high]
    if bar.open > bar.high:
        raise ValidationError(
            f"{bar.ticker} {bar.date_str}: open ({bar.open}) above high ({bar.high})"
        )
    if bar.open < bar.low:
        raise ValidationError(
            f"{bar.ticker} {bar.date_str}: open ({bar.open}) below low ({bar.low})"
        )

    # Close must be within [low, high]
    if bar.close > bar.high:
        raise ValidationError(
            f"{bar.ticker} {bar.date_str}: close ({bar.close}) above high ({bar.high})"
        )
    if bar.close < bar.low:
        raise ValidationError(
            f"{bar.ticker} {bar.date_str}: close ({bar.close}) below low ({bar.low})"
        )

    # Volume must be non-negative
    if bar.volume < 0:
        raise ValidationError(
            f"{bar.ticker} {bar.date_str}: volume is negative ({bar.volume})"
        )

    return True
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/test_ingest/test_quality.py -v
```

Expected: all tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/bist_predict/ingest/quality.py tests/test_ingest/test_quality.py
git commit -m "feat: OHLCV data quality validation with comprehensive checks"
```

---

### Task 5: Is Yatirim API Client

**Files:**
- Create: `src/bist_predict/ingest/isyatirim.py`
- Create: `tests/test_ingest/test_isyatirim.py`

- [ ] **Step 1: Write failing tests**

`tests/test_ingest/test_isyatirim.py`:

```python
"""Tests for Is Yatirim API client."""

from __future__ import annotations

from datetime import date

import httpx
import pytest
import respx

from bist_predict.ingest.isyatirim import IsYatirimClient
from bist_predict.ingest.types import OHLCVBar


SAMPLE_RESPONSE = {
    "value": [
        {
            "HGDG_HS_KODU": "THYAO",
            "HGDG_TARIH": "2026-04-01T00:00:00",
            "HGDG_ACILIS": 310.0,
            "HGDG_KAPANIS": 312.5,
            "HGDG_BIRINCISI": 315.0,
            "HGDG_SONUNCUSU": 308.0,
            "HGDG_HACIMLOT": 1000000,
        },
        {
            "HGDG_HS_KODU": "THYAO",
            "HGDG_TARIH": "2026-03-31T00:00:00",
            "HGDG_ACILIS": 305.0,
            "HGDG_KAPANIS": 310.0,
            "HGDG_BIRINCISI": 311.0,
            "HGDG_SONUNCUSU": 303.0,
            "HGDG_HACIMLOT": 900000,
        },
    ]
}


class TestIsYatirimClient:
    @respx.mock
    @pytest.mark.asyncio
    async def test_fetch_returns_ohlcv_bars(self) -> None:
        respx.get("https://www.isyatirim.com.tr/_layouts/15/Jeyjey.Yatirim/Jeyjey.Yatirim.Module.GecmisVeriler/GecmisVeriler.aspx/HisseSenetleriGecmisVeriler").mock(
            return_value=httpx.Response(200, json=SAMPLE_RESPONSE)
        )

        client = IsYatirimClient()
        bars = await client.fetch("THYAO", date(2026, 3, 31), date(2026, 4, 1))

        assert len(bars) == 2
        assert all(isinstance(b, OHLCVBar) for b in bars)
        assert bars[0].ticker == "THYAO"
        assert bars[0].source == "isyatirim"

    @respx.mock
    @pytest.mark.asyncio
    async def test_fetch_parses_prices_correctly(self) -> None:
        respx.get("https://www.isyatirim.com.tr/_layouts/15/Jeyjey.Yatirim/Jeyjey.Yatirim.Module.GecmisVeriler/GecmisVeriler.aspx/HisseSenetleriGecmisVeriler").mock(
            return_value=httpx.Response(200, json=SAMPLE_RESPONSE)
        )

        client = IsYatirimClient()
        bars = await client.fetch("THYAO", date(2026, 3, 31), date(2026, 4, 1))

        apr1 = bars[0]
        assert apr1.open == 310.0
        assert apr1.high == 315.0
        assert apr1.low == 308.0
        assert apr1.close == 312.5
        assert apr1.volume == 1_000_000

    @respx.mock
    @pytest.mark.asyncio
    async def test_fetch_empty_response(self) -> None:
        respx.get("https://www.isyatirim.com.tr/_layouts/15/Jeyjey.Yatirim/Jeyjey.Yatirim.Module.GecmisVeriler/GecmisVeriler.aspx/HisseSenetleriGecmisVeriler").mock(
            return_value=httpx.Response(200, json={"value": []})
        )

        client = IsYatirimClient()
        bars = await client.fetch("THYAO", date(2026, 3, 31), date(2026, 4, 1))
        assert bars == []

    @respx.mock
    @pytest.mark.asyncio
    async def test_fetch_http_error_raises(self) -> None:
        respx.get("https://www.isyatirim.com.tr/_layouts/15/Jeyjey.Yatirim/Jeyjey.Yatirim.Module.GecmisVeriler/GecmisVeriler.aspx/HisseSenetleriGecmisVeriler").mock(
            return_value=httpx.Response(500)
        )

        client = IsYatirimClient()
        with pytest.raises(httpx.HTTPStatusError):
            await client.fetch("THYAO", date(2026, 3, 31), date(2026, 4, 1))
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/test_ingest/test_isyatirim.py -v
```

Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement isyatirim.py**

`src/bist_predict/ingest/isyatirim.py`:

```python
"""Is Yatirim API client for BIST historical price data."""

from __future__ import annotations

from datetime import date, datetime

import httpx

from bist_predict.ingest.types import OHLCVBar

BASE_URL = (
    "https://www.isyatirim.com.tr/_layouts/15/Jeyjey.Yatirim/"
    "Jeyjey.Yatirim.Module.GecmisVeriler/GecmisVeriler.aspx/"
    "HisseSenetleriGecmisVeriler"
)


class IsYatirimClient:
    """Fetches historical OHLCV data from Is Yatirim's public API."""

    def __init__(self, timeout: float = 30.0) -> None:
        self._timeout = timeout

    async def fetch(
        self, ticker: str, start_date: date, end_date: date
    ) -> list[OHLCVBar]:
        """Fetch OHLCV bars for a ticker between start_date and end_date."""
        params = {
            "hession": ticker,
            "startdate": start_date.strftime("%d-%m-%Y"),
            "enddate": end_date.strftime("%d-%m-%Y"),
        }

        async with httpx.AsyncClient(timeout=self._timeout) as client:
            response = await client.get(BASE_URL, params=params)
            response.raise_for_status()

        data = response.json()
        rows = data.get("value", [])

        bars: list[OHLCVBar] = []
        for row in rows:
            dt = datetime.fromisoformat(row["HGDG_TARIH"]).date()
            bar = OHLCVBar(
                ticker=ticker,
                date=dt,
                open=float(row["HGDG_ACILIS"]),
                high=float(row["HGDG_BIRINCISI"]),
                low=float(row["HGDG_SONUNCUSU"]),
                close=float(row["HGDG_KAPANIS"]),
                adj_close=float(row["HGDG_KAPANIS"]),  # Is Yatirim doesn't provide adj close separately
                volume=int(row["HGDG_HACIMLOT"]),
                source="isyatirim",
            )
            bars.append(bar)

        return bars
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/test_ingest/test_isyatirim.py -v
```

Expected: all tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/bist_predict/ingest/isyatirim.py tests/test_ingest/test_isyatirim.py
git commit -m "feat: Is Yatirim API client for BIST historical price data"
```

---

### Task 6: Yahoo Finance Fallback Client

**Files:**
- Create: `src/bist_predict/ingest/yahoo.py`
- Create: `tests/test_ingest/test_yahoo.py`

- [ ] **Step 1: Write failing tests**

`tests/test_ingest/test_yahoo.py`:

```python
"""Tests for Yahoo Finance fallback client."""

from __future__ import annotations

from datetime import date
from unittest.mock import MagicMock, patch

import pytest

from bist_predict.ingest.yahoo import YahooFinanceClient
from bist_predict.ingest.types import OHLCVBar


class TestYahooFinanceClient:
    def test_ticker_suffix(self) -> None:
        client = YahooFinanceClient()
        assert client._bist_ticker("THYAO") == "THYAO.IS"
        assert client._bist_ticker("GARAN") == "GARAN.IS"

    @patch("bist_predict.ingest.yahoo.yf.download")
    def test_fetch_returns_ohlcv_bars(self, mock_download: MagicMock) -> None:
        import pandas as pd

        mock_df = pd.DataFrame(
            {
                "Open": [310.0, 305.0],
                "High": [315.0, 311.0],
                "Low": [308.0, 303.0],
                "Close": [312.5, 310.0],
                "Adj Close": [312.5, 310.0],
                "Volume": [1_000_000, 900_000],
            },
            index=pd.DatetimeIndex([
                pd.Timestamp("2026-04-01"),
                pd.Timestamp("2026-03-31"),
            ], name="Date"),
        )
        mock_download.return_value = mock_df

        client = YahooFinanceClient()
        bars = client.fetch_sync("THYAO", date(2026, 3, 31), date(2026, 4, 1))

        assert len(bars) == 2
        assert all(isinstance(b, OHLCVBar) for b in bars)
        assert bars[0].ticker == "THYAO"
        assert bars[0].source == "yahoo"

    @patch("bist_predict.ingest.yahoo.yf.download")
    def test_fetch_empty_dataframe(self, mock_download: MagicMock) -> None:
        import pandas as pd

        mock_download.return_value = pd.DataFrame()

        client = YahooFinanceClient()
        bars = client.fetch_sync("THYAO", date(2026, 3, 31), date(2026, 4, 1))
        assert bars == []
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/test_ingest/test_yahoo.py -v
```

Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement yahoo.py**

`src/bist_predict/ingest/yahoo.py`:

```python
"""Yahoo Finance fallback client for BIST price data."""

from __future__ import annotations

import asyncio
from datetime import date

import yfinance as yf

from bist_predict.ingest.types import OHLCVBar


class YahooFinanceClient:
    """Fetches historical OHLCV data from Yahoo Finance as a fallback source."""

    def _bist_ticker(self, ticker: str) -> str:
        """Convert a BIST ticker to Yahoo Finance format."""
        return f"{ticker}.IS"

    def fetch_sync(
        self, ticker: str, start_date: date, end_date: date
    ) -> list[OHLCVBar]:
        """Synchronous fetch — wraps yfinance which is sync-only."""
        yahoo_ticker = self._bist_ticker(ticker)
        df = yf.download(
            yahoo_ticker,
            start=start_date.isoformat(),
            end=end_date.isoformat(),
            progress=False,
            auto_adjust=False,
        )

        if df.empty:
            return []

        bars: list[OHLCVBar] = []
        for idx, row in df.iterrows():
            bar = OHLCVBar(
                ticker=ticker,
                date=idx.date(),
                open=float(row["Open"]),
                high=float(row["High"]),
                low=float(row["Low"]),
                close=float(row["Close"]),
                adj_close=float(row["Adj Close"]),
                volume=int(row["Volume"]),
                source="yahoo",
            )
            bars.append(bar)

        return bars

    async def fetch(
        self, ticker: str, start_date: date, end_date: date
    ) -> list[OHLCVBar]:
        """Async wrapper — runs yfinance in a thread to avoid blocking."""
        return await asyncio.to_thread(self.fetch_sync, ticker, start_date, end_date)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/test_ingest/test_yahoo.py -v
```

Expected: all tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/bist_predict/ingest/yahoo.py tests/test_ingest/test_yahoo.py
git commit -m "feat: Yahoo Finance fallback client for BIST price data"
```

---

### Task 7: TCMB EVDS Macro Data Client

**Files:**
- Create: `src/bist_predict/ingest/tcmb.py`
- Create: `tests/test_ingest/test_tcmb.py`

- [ ] **Step 1: Write failing tests**

`tests/test_ingest/test_tcmb.py`:

```python
"""Tests for TCMB EVDS macro data client."""

from __future__ import annotations

from datetime import date

import httpx
import pytest
import respx

from bist_predict.ingest.tcmb import TcmbClient, INDICATORS
from bist_predict.ingest.types import MacroDataPoint


SAMPLE_EVDS_RESPONSE = {
    "items": [
        {"Tarih": "01-04-2026", "TP_DK_USD_A_YTL": "38.4500"},
        {"Tarih": "31-03-2026", "TP_DK_USD_A_YTL": "38.3200"},
    ],
    "totalCount": 2,
}


class TestTcmbClient:
    @respx.mock
    @pytest.mark.asyncio
    async def test_fetch_returns_macro_points(self) -> None:
        respx.get("https://evds2.tcmb.gov.tr/service/evds/series=TP.DK.USD.A.YTL").mock(
            return_value=httpx.Response(200, json=SAMPLE_EVDS_RESPONSE)
        )

        client = TcmbClient(api_key="test-key")
        points = await client.fetch("USD_TRY", date(2026, 3, 31), date(2026, 4, 1))

        assert len(points) == 2
        assert all(isinstance(p, MacroDataPoint) for p in points)
        assert points[0].indicator == "USD_TRY"
        assert points[0].source == "tcmb"
        assert points[0].value == 38.45

    @respx.mock
    @pytest.mark.asyncio
    async def test_fetch_empty_response(self) -> None:
        respx.get("https://evds2.tcmb.gov.tr/service/evds/series=TP.DK.USD.A.YTL").mock(
            return_value=httpx.Response(200, json={"items": [], "totalCount": 0})
        )

        client = TcmbClient(api_key="test-key")
        points = await client.fetch("USD_TRY", date(2026, 3, 31), date(2026, 4, 1))
        assert points == []

    def test_indicators_mapping_exists(self) -> None:
        assert "USD_TRY" in INDICATORS
        assert "EUR_TRY" in INDICATORS
        assert "GOLD_TRY" in INDICATORS
        assert "POLICY_RATE" in INDICATORS

    @respx.mock
    @pytest.mark.asyncio
    async def test_fetch_missing_api_key_raises(self) -> None:
        client = TcmbClient(api_key="")
        with pytest.raises(ValueError, match="API key"):
            await client.fetch("USD_TRY", date(2026, 3, 31), date(2026, 4, 1))
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/test_ingest/test_tcmb.py -v
```

Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement tcmb.py**

`src/bist_predict/ingest/tcmb.py`:

```python
"""TCMB EVDS client for Turkish macro-economic data."""

from __future__ import annotations

from datetime import date, datetime

import httpx

from bist_predict.ingest.types import MacroDataPoint

BASE_URL = "https://evds2.tcmb.gov.tr/service/evds"

# Map our indicator names to EVDS series codes and their JSON field names.
INDICATORS: dict[str, tuple[str, str]] = {
    "USD_TRY": ("TP.DK.USD.A.YTL", "TP_DK_USD_A_YTL"),
    "EUR_TRY": ("TP.DK.EUR.A.YTL", "TP_DK_EUR_A_YTL"),
    "GOLD_TRY": ("TP.DK.ALT.A.YTL", "TP_DK_ALT_A_YTL"),
    "POLICY_RATE": ("TP.PO.FAIZ.ON", "TP_PO_FAIZ_ON"),
    "CPI": ("TP.FG.J0", "TP_FG_J0"),
    "BOND_2Y": ("TP.GS.DT02", "TP_GS_DT02"),
}


class TcmbClient:
    """Fetches macro-economic data from TCMB EVDS API."""

    def __init__(self, api_key: str, timeout: float = 30.0) -> None:
        self._api_key = api_key
        self._timeout = timeout

    async def fetch(
        self, indicator: str, start_date: date, end_date: date
    ) -> list[MacroDataPoint]:
        """Fetch macro data for an indicator between start_date and end_date."""
        if not self._api_key:
            raise ValueError("TCMB EVDS API key is required. Register free at evds2.tcmb.gov.tr")

        series_code, field_name = INDICATORS[indicator]

        params = {
            "series": series_code,
            "startDate": start_date.strftime("%d-%m-%Y"),
            "endDate": end_date.strftime("%d-%m-%Y"),
            "type": "json",
            "key": self._api_key,
        }

        url = f"{BASE_URL}/series={series_code}"

        async with httpx.AsyncClient(timeout=self._timeout) as client:
            response = await client.get(url, params=params)
            response.raise_for_status()

        data = response.json()
        items = data.get("items", [])

        points: list[MacroDataPoint] = []
        for item in items:
            raw_value = item.get(field_name)
            if raw_value is None or raw_value == "":
                continue

            dt = datetime.strptime(item["Tarih"], "%d-%m-%Y").date()
            point = MacroDataPoint(
                indicator=indicator,
                date=dt,
                value=float(raw_value),
                source="tcmb",
            )
            points.append(point)

        return points
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/test_ingest/test_tcmb.py -v
```

Expected: all tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/bist_predict/ingest/tcmb.py tests/test_ingest/test_tcmb.py
git commit -m "feat: TCMB EVDS client for Turkish macro-economic data"
```

---

### Task 8: Sentiment Collector

**Files:**
- Create: `src/bist_predict/ingest/sentiment.py`
- Create: `tests/test_ingest/test_sentiment.py`

- [ ] **Step 1: Write failing tests**

`tests/test_ingest/test_sentiment.py`:

```python
"""Tests for sentiment data collectors."""

from __future__ import annotations

from datetime import date

import httpx
import pytest
import respx

from bist_predict.ingest.sentiment import GoogleNewsSentiment
from bist_predict.ingest.types import SentimentRecord

SAMPLE_RSS = """<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0">
  <channel>
    <title>THYAO - Google News</title>
    <item>
      <title>THY hisseleri güçlü yükseldi</title>
      <pubDate>Tue, 01 Apr 2026 10:00:00 GMT</pubDate>
      <description>Türk Hava Yolları hisseleri bugün güçlü yükseliş gösterdi.</description>
    </item>
    <item>
      <title>THYAO bilançosu beklentilerin üzerinde</title>
      <pubDate>Mon, 31 Mar 2026 14:00:00 GMT</pubDate>
      <description>THY bilançosu beklentilerin üzerinde geldi.</description>
    </item>
  </channel>
</rss>"""


class TestGoogleNewsSentiment:
    @respx.mock
    @pytest.mark.asyncio
    async def test_fetch_returns_sentiment_records(self) -> None:
        respx.get("https://news.google.com/rss/search").mock(
            return_value=httpx.Response(200, text=SAMPLE_RSS)
        )

        collector = GoogleNewsSentiment()
        records = await collector.fetch("THYAO", date(2026, 3, 31), date(2026, 4, 1))

        assert len(records) == 2
        assert all(isinstance(r, SentimentRecord) for r in records)
        assert records[0].ticker == "THYAO"
        assert records[0].source == "google_news"
        assert records[0].headline is not None

    @respx.mock
    @pytest.mark.asyncio
    async def test_fetch_empty_feed(self) -> None:
        empty_rss = """<?xml version="1.0" encoding="UTF-8"?>
        <rss version="2.0"><channel><title>Empty</title></channel></rss>"""
        respx.get("https://news.google.com/rss/search").mock(
            return_value=httpx.Response(200, text=empty_rss)
        )

        collector = GoogleNewsSentiment()
        records = await collector.fetch("THYAO", date(2026, 3, 31), date(2026, 4, 1))
        assert records == []

    @respx.mock
    @pytest.mark.asyncio
    async def test_fetch_http_error_returns_empty(self) -> None:
        respx.get("https://news.google.com/rss/search").mock(
            return_value=httpx.Response(429)
        )

        collector = GoogleNewsSentiment()
        records = await collector.fetch("THYAO", date(2026, 3, 31), date(2026, 4, 1))
        assert records == []
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/test_ingest/test_sentiment.py -v
```

Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement sentiment.py**

`src/bist_predict/ingest/sentiment.py`:

```python
"""Sentiment data collectors — Google News RSS, Turkish finance RSS, etc."""

from __future__ import annotations

import logging
from datetime import date, datetime, timezone
from email.utils import parsedate_to_datetime

import feedparser
import httpx

from bist_predict.ingest.types import SentimentRecord

logger = logging.getLogger(__name__)


class GoogleNewsSentiment:
    """Fetches news headlines from Google News RSS for sentiment analysis."""

    def __init__(self, timeout: float = 15.0) -> None:
        self._timeout = timeout

    async def fetch(
        self, ticker: str, start_date: date, end_date: date
    ) -> list[SentimentRecord]:
        """Fetch news headlines for a BIST ticker from Google News RSS."""
        query = f"{ticker} borsa hisse"
        url = "https://news.google.com/rss/search"
        params = {"q": query, "hl": "tr", "gl": "TR", "ceid": "TR:tr"}

        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                response = await client.get(url, params=params)
                response.raise_for_status()
        except (httpx.HTTPStatusError, httpx.RequestError) as e:
            logger.warning("Google News fetch failed for %s: %s", ticker, e)
            return []

        feed = feedparser.parse(response.text)
        records: list[SentimentRecord] = []

        for entry in feed.entries:
            pub_date = _parse_rss_date(entry.get("published", ""))
            if pub_date is None:
                continue

            if not (start_date <= pub_date <= end_date):
                continue

            record = SentimentRecord(
                ticker=ticker,
                date=pub_date,
                source="google_news",
                headline=entry.get("title"),
                sentiment_score=None,  # Scored later by NLP model
                raw_text=entry.get("description"),
            )
            records.append(record)

        return records


class TurkishFinanceRSS:
    """Fetches headlines from Turkish financial news RSS feeds."""

    FEEDS = [
        ("https://www.bloomberght.com/rss", "bloomberght"),
        ("https://bigpara.hurriyet.com.tr/rss/", "bigpara"),
    ]

    def __init__(self, timeout: float = 15.0) -> None:
        self._timeout = timeout

    async def fetch(
        self, ticker: str, start_date: date, end_date: date
    ) -> list[SentimentRecord]:
        """Fetch headlines mentioning the ticker from Turkish finance RSS feeds."""
        records: list[SentimentRecord] = []
        ticker_lower = ticker.lower()

        for feed_url, source_name in self.FEEDS:
            try:
                async with httpx.AsyncClient(timeout=self._timeout) as client:
                    response = await client.get(feed_url)
                    response.raise_for_status()
            except (httpx.HTTPStatusError, httpx.RequestError) as e:
                logger.warning("RSS fetch failed for %s from %s: %s", ticker, source_name, e)
                continue

            feed = feedparser.parse(response.text)
            for entry in feed.entries:
                title = entry.get("title", "")
                description = entry.get("description", "")
                combined = f"{title} {description}".lower()

                if ticker_lower not in combined:
                    continue

                pub_date = _parse_rss_date(entry.get("published", ""))
                if pub_date is None:
                    continue

                if not (start_date <= pub_date <= end_date):
                    continue

                record = SentimentRecord(
                    ticker=ticker,
                    date=pub_date,
                    source=source_name,
                    headline=title,
                    sentiment_score=None,
                    raw_text=description,
                )
                records.append(record)

        return records


def _parse_rss_date(date_str: str) -> date | None:
    """Parse an RSS pubDate string to a date. Returns None on failure."""
    if not date_str:
        return None
    try:
        return parsedate_to_datetime(date_str).date()
    except (ValueError, TypeError):
        pass
    try:
        return datetime.fromisoformat(date_str).date()
    except (ValueError, TypeError):
        return None
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/test_ingest/test_sentiment.py -v
```

Expected: all tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/bist_predict/ingest/sentiment.py tests/test_ingest/test_sentiment.py
git commit -m "feat: Google News and Turkish finance RSS sentiment collectors"
```

---

### Task 9: Data Ingestion Scheduler

**Files:**
- Create: `src/bist_predict/ingest/scheduler.py`
- Create: `tests/test_ingest/test_scheduler.py`

- [ ] **Step 1: Write failing tests**

`tests/test_ingest/test_scheduler.py`:

```python
"""Tests for the data ingestion scheduler."""

from __future__ import annotations

import sqlite3
from datetime import date
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from bist_predict.config import Config, DataConfig
from bist_predict.ingest.scheduler import IngestionScheduler
from bist_predict.ingest.types import OHLCVBar, MacroDataPoint, SentimentRecord
from bist_predict.storage.database import Database


@pytest.fixture
def db(tmp_db_path: Path) -> Database:
    db = Database(tmp_db_path)
    db.initialize()
    return db


def _make_bar(ticker: str = "THYAO", d: date = date(2026, 4, 1)) -> OHLCVBar:
    return OHLCVBar(
        ticker=ticker, date=d, open=310.0, high=315.0,
        low=308.0, close=312.5, adj_close=312.5,
        volume=1_000_000, source="isyatirim",
    )


def _make_macro(d: date = date(2026, 4, 1)) -> MacroDataPoint:
    return MacroDataPoint(indicator="USD_TRY", date=d, value=38.45, source="tcmb")


def _make_sentiment(ticker: str = "THYAO", d: date = date(2026, 4, 1)) -> SentimentRecord:
    return SentimentRecord(
        ticker=ticker, date=d, source="google_news",
        headline="Test headline", sentiment_score=None, raw_text="Test text",
    )


class TestIngestionScheduler:
    @pytest.mark.asyncio
    async def test_store_price_bars(self, db: Database, config: Config) -> None:
        mock_primary = AsyncMock(return_value=[_make_bar()])
        mock_fallback = AsyncMock(return_value=[])

        scheduler = IngestionScheduler(
            db=db,
            config=config,
            price_primary=mock_primary,
            price_fallback=mock_fallback,
        )
        stored = await scheduler.store_prices([_make_bar()])
        assert stored == 1

        with db.connect() as conn:
            row = conn.execute("SELECT ticker, close FROM raw_prices").fetchone()
            assert row[0] == "THYAO"
            assert row[1] == 312.5

    @pytest.mark.asyncio
    async def test_store_macro_data(self, db: Database, config: Config) -> None:
        scheduler = IngestionScheduler(db=db, config=config)
        stored = await scheduler.store_macro([_make_macro()])
        assert stored == 1

        with db.connect() as conn:
            row = conn.execute("SELECT indicator, value FROM macro_data").fetchone()
            assert row[0] == "USD_TRY"
            assert row[1] == 38.45

    @pytest.mark.asyncio
    async def test_store_sentiment(self, db: Database, config: Config) -> None:
        scheduler = IngestionScheduler(db=db, config=config)
        stored = await scheduler.store_sentiment([_make_sentiment()])
        assert stored == 1

        with db.connect() as conn:
            row = conn.execute("SELECT ticker, source FROM sentiment_data").fetchone()
            assert row[0] == "THYAO"
            assert row[1] == "google_news"

    @pytest.mark.asyncio
    async def test_fetch_with_fallback(self, db: Database, config: Config) -> None:
        mock_primary = AsyncMock(side_effect=Exception("API down"))
        mock_fallback = AsyncMock(return_value=[_make_bar(d=date(2026, 4, 1))])

        scheduler = IngestionScheduler(
            db=db,
            config=config,
            price_primary=mock_primary,
            price_fallback=mock_fallback,
        )
        bars = await scheduler.fetch_prices("THYAO", date(2026, 4, 1), date(2026, 4, 1))

        assert len(bars) == 1
        mock_primary.assert_called_once()
        mock_fallback.assert_called_once()

    @pytest.mark.asyncio
    async def test_duplicate_bars_ignored(self, db: Database, config: Config) -> None:
        scheduler = IngestionScheduler(db=db, config=config)
        bar = _make_bar()
        await scheduler.store_prices([bar])
        stored = await scheduler.store_prices([bar])  # duplicate
        assert stored == 0

        with db.connect() as conn:
            count = conn.execute("SELECT COUNT(*) FROM raw_prices").fetchone()[0]
            assert count == 1
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/test_ingest/test_scheduler.py -v
```

Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement scheduler.py**

`src/bist_predict/ingest/scheduler.py`:

```python
"""Ingestion scheduler — orchestrates data collection from all sources."""

from __future__ import annotations

import asyncio
import logging
import sqlite3
from datetime import date
from typing import Any, Callable, Coroutine, Sequence

from bist_predict.config import Config
from bist_predict.ingest.quality import ValidationError, validate_bar
from bist_predict.ingest.types import MacroDataPoint, OHLCVBar, SentimentRecord
from bist_predict.storage.database import Database

logger = logging.getLogger(__name__)

# Type alias for async fetch callables
PriceFetcher = Callable[[str, date, date], Coroutine[Any, Any, list[OHLCVBar]]]


class IngestionScheduler:
    """Orchestrates data fetching from all sources with fallback and storage."""

    def __init__(
        self,
        db: Database,
        config: Config,
        price_primary: PriceFetcher | None = None,
        price_fallback: PriceFetcher | None = None,
    ) -> None:
        self._db = db
        self._config = config
        self._price_primary = price_primary
        self._price_fallback = price_fallback

    async def fetch_prices(
        self, ticker: str, start_date: date, end_date: date
    ) -> list[OHLCVBar]:
        """Fetch price data with fallback. Tries primary source first."""
        if self._price_primary is not None:
            try:
                bars = await self._price_primary(ticker, start_date, end_date)
                if bars:
                    return bars
            except Exception as e:
                logger.warning("Primary source failed for %s: %s", ticker, e)

        if self._price_fallback is not None:
            try:
                return await self._price_fallback(ticker, start_date, end_date)
            except Exception as e:
                logger.warning("Fallback source failed for %s: %s", ticker, e)

        return []

    async def store_prices(self, bars: Sequence[OHLCVBar]) -> int:
        """Validate and store price bars. Returns count of newly stored bars."""
        stored = 0
        with self._db.connect() as conn:
            for bar in bars:
                try:
                    validate_bar(bar)
                except ValidationError as e:
                    logger.warning("Skipping invalid bar: %s", e)
                    continue

                try:
                    conn.execute(
                        """INSERT INTO raw_prices
                           (ticker, date, open, high, low, close, adj_close, volume, source)
                           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                        (
                            bar.ticker, bar.date_str, bar.open, bar.high,
                            bar.low, bar.close, bar.adj_close, bar.volume, bar.source,
                        ),
                    )
                    stored += 1
                except sqlite3.IntegrityError:
                    logger.debug("Duplicate bar skipped: %s %s", bar.ticker, bar.date_str)

            conn.commit()
        return stored

    async def store_macro(self, points: Sequence[MacroDataPoint]) -> int:
        """Store macro data points. Returns count of newly stored points."""
        stored = 0
        with self._db.connect() as conn:
            for point in points:
                try:
                    conn.execute(
                        """INSERT INTO macro_data (indicator, date, value, source)
                           VALUES (?, ?, ?, ?)""",
                        (point.indicator, point.date_str, point.value, point.source),
                    )
                    stored += 1
                except sqlite3.IntegrityError:
                    logger.debug("Duplicate macro point skipped: %s %s", point.indicator, point.date_str)

            conn.commit()
        return stored

    async def store_sentiment(self, records: Sequence[SentimentRecord]) -> int:
        """Store sentiment records. Returns count of newly stored records."""
        stored = 0
        with self._db.connect() as conn:
            for record in records:
                conn.execute(
                    """INSERT INTO sentiment_data
                       (ticker, date, source, headline, sentiment_score, raw_text)
                       VALUES (?, ?, ?, ?, ?, ?)""",
                    (
                        record.ticker, record.date_str, record.source,
                        record.headline, record.sentiment_score, record.raw_text,
                    ),
                )
                stored += 1

            conn.commit()
        return stored
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/test_ingest/test_scheduler.py -v
```

Expected: all tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/bist_predict/ingest/scheduler.py tests/test_ingest/test_scheduler.py
git commit -m "feat: ingestion scheduler with fallback logic and data storage"
```

---

### Task 10: CLI — fetch Command

**Files:**
- Create: `src/bist_predict/cli.py`

- [ ] **Step 1: Implement cli.py with the fetch command**

`src/bist_predict/cli.py`:

```python
"""CLI entry point for bist-predict."""

from __future__ import annotations

import asyncio
import logging
from datetime import date, timedelta

import click

from bist_predict.config import load_config
from bist_predict.ingest.isyatirim import IsYatirimClient
from bist_predict.ingest.scheduler import IngestionScheduler
from bist_predict.ingest.sentiment import GoogleNewsSentiment, TurkishFinanceRSS
from bist_predict.ingest.tcmb import TcmbClient, INDICATORS
from bist_predict.ingest.yahoo import YahooFinanceClient
from bist_predict.storage.database import Database

# BIST-100 tickers (representative subset — full list loaded from DB or config in future)
BIST_100_SAMPLE = [
    "THYAO", "GARAN", "AKBNK", "EREGL", "SISE", "TUPRS", "TCELL", "TOASO",
    "VESTL", "SAHOL", "KCHOL", "HEKTS", "BIMAS", "ASELS", "SASA", "KOZAL",
    "PETKM", "DOHOL", "FROTO", "ENKAI", "ARCLK", "ISCTR", "YKBNK", "VAKBN",
    "HALKB", "TAVHL", "TTKOM", "EKGYO", "PGSUS", "MGROS",
]


@click.group()
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose logging")
def main(verbose: bool) -> None:
    """BIST-100 Stock Market Prediction System."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(levelname)s: %(message)s")


@main.command()
@click.option("--days", default=30, help="Number of days of history to fetch")
@click.option("--ticker", default=None, help="Fetch a single ticker instead of all BIST-100")
def fetch(days: int, ticker: str | None) -> None:
    """Fetch latest market data from all sources."""
    asyncio.run(_fetch(days, ticker))


async def _fetch(days: int, ticker: str | None) -> None:
    config = load_config()
    db = Database(config.db_path)
    db.initialize()

    is_client = IsYatirimClient()
    yahoo_client = YahooFinanceClient()

    scheduler = IngestionScheduler(
        db=db,
        config=config,
        price_primary=is_client.fetch,
        price_fallback=yahoo_client.fetch,
    )

    end_date = date.today()
    start_date = end_date - timedelta(days=days)

    tickers = [ticker] if ticker else BIST_100_SAMPLE

    # Fetch prices
    total_bars = 0
    for t in tickers:
        latest = db.get_latest_date(t)
        fetch_start = start_date
        if latest:
            fetch_start = max(start_date, date.fromisoformat(latest) + timedelta(days=1))
            if fetch_start > end_date:
                click.echo(f"  {t}: up to date")
                continue

        click.echo(f"  {t}: fetching {fetch_start} → {end_date}...")
        bars = await scheduler.fetch_prices(t, fetch_start, end_date)
        stored = await scheduler.store_prices(bars)
        total_bars += stored

        # Rate limiting
        await asyncio.sleep(config.data.rate_limit_delay)

    click.echo(f"\nStored {total_bars} new price bars.")

    # Fetch macro data
    if config.data.tcmb_api_key:
        tcmb = TcmbClient(api_key=config.data.tcmb_api_key)
        total_macro = 0
        for indicator in INDICATORS:
            click.echo(f"  Macro: {indicator}...")
            try:
                points = await tcmb.fetch(indicator, start_date, end_date)
                stored = await scheduler.store_macro(points)
                total_macro += stored
            except Exception as e:
                click.echo(f"    Warning: {e}")
        click.echo(f"Stored {total_macro} new macro data points.")
    else:
        click.echo("Skipping macro data (no TCMB API key in config.toml)")

    # Fetch sentiment
    google_news = GoogleNewsSentiment()
    total_sentiment = 0
    for t in tickers[:10]:  # Limit sentiment fetch to top 10 to avoid rate limits
        click.echo(f"  Sentiment: {t}...")
        records = await google_news.fetch(t, start_date, end_date)
        stored = await scheduler.store_sentiment(records)
        total_sentiment += stored
        await asyncio.sleep(config.data.rate_limit_delay)

    click.echo(f"Stored {total_sentiment} new sentiment records.")
    click.echo("\nFetch complete.")


@main.command()
def stocks() -> None:
    """List tracked BIST-100 stocks."""
    click.echo("BIST-100 Tracked Stocks:")
    click.echo("=" * 40)
    for i, ticker in enumerate(BIST_100_SAMPLE, 1):
        click.echo(f"  {i:3d}. {ticker}")
    click.echo(f"\nTotal: {len(BIST_100_SAMPLE)} stocks")


@main.command()
def config() -> None:
    """Show current configuration."""
    cfg = load_config()
    click.echo("Current Configuration:")
    click.echo("=" * 40)
    click.echo(f"  Database: {cfg.db_path}")
    click.echo(f"  TCMB API key: {'set' if cfg.data.tcmb_api_key else 'not set'}")
    click.echo(f"  Fetch retries: {cfg.data.fetch_retries}")
    click.echo(f"  Rate limit delay: {cfg.data.rate_limit_delay}s")
    click.echo(f"  Min confidence: {cfg.signals.min_confidence}")
    click.echo(f"  Backtest commission: {cfg.backtest.commission}")
    click.echo(f"  Backtest slippage: {cfg.backtest.slippage}")
```

- [ ] **Step 2: Verify CLI works**

```bash
uv run bist-predict --help
uv run bist-predict stocks
uv run bist-predict config
```

Expected: help text, stock list, and config output display correctly.

- [ ] **Step 3: Commit**

```bash
git add src/bist_predict/cli.py
git commit -m "feat: CLI entry point with fetch, stocks, and config commands"
```

---

### Task 11: Integration Test — Full Fetch Pipeline

**Files:**
- Create: `tests/test_ingest/test_integration.py`

- [ ] **Step 1: Write integration test**

`tests/test_ingest/test_integration.py`:

```python
"""Integration test for the full fetch pipeline using mocked HTTP."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from bist_predict.config import Config, DataConfig
from bist_predict.ingest.scheduler import IngestionScheduler
from bist_predict.ingest.types import OHLCVBar, MacroDataPoint, SentimentRecord
from bist_predict.storage.database import Database


@pytest.fixture
def db(tmp_db_path: Path) -> Database:
    db = Database(tmp_db_path)
    db.initialize()
    return db


class TestFullPipeline:
    @pytest.mark.asyncio
    async def test_fetch_store_query_cycle(self, db: Database, config: Config) -> None:
        """Test: fetch → validate → store → query roundtrip."""
        bars = [
            OHLCVBar("THYAO", date(2026, 4, 1), 310.0, 315.0, 308.0, 312.5, 312.5, 1_000_000, "isyatirim"),
            OHLCVBar("THYAO", date(2026, 3, 31), 305.0, 311.0, 303.0, 310.0, 310.0, 900_000, "isyatirim"),
            OHLCVBar("GARAN", date(2026, 4, 1), 85.0, 87.0, 84.5, 86.5, 86.5, 5_000_000, "isyatirim"),
        ]

        mock_primary = AsyncMock(return_value=bars)
        scheduler = IngestionScheduler(
            db=db, config=config,
            price_primary=mock_primary,
        )

        fetched = await scheduler.fetch_prices("THYAO", date(2026, 3, 31), date(2026, 4, 1))
        assert len(fetched) == 3

        stored = await scheduler.store_prices(fetched)
        assert stored == 3

        # Verify data is queryable
        latest = db.get_latest_date("THYAO")
        assert latest == "2026-04-01"

        latest_garan = db.get_latest_date("GARAN")
        assert latest_garan == "2026-04-01"

        # Verify incremental fetch would skip
        stored_again = await scheduler.store_prices(fetched)
        assert stored_again == 0  # All duplicates

    @pytest.mark.asyncio
    async def test_invalid_bars_filtered_out(self, db: Database, config: Config) -> None:
        """Invalid bars should be skipped, valid ones stored."""
        bars = [
            OHLCVBar("THYAO", date(2026, 4, 1), 310.0, 315.0, 308.0, 312.5, 312.5, 1_000_000, "isyatirim"),
            OHLCVBar("BAD", date(2026, 4, 1), 310.0, 300.0, 308.0, 312.5, 312.5, 1_000_000, "isyatirim"),  # high < low
        ]

        scheduler = IngestionScheduler(db=db, config=config)
        stored = await scheduler.store_prices(bars)
        assert stored == 1

    @pytest.mark.asyncio
    async def test_macro_and_sentiment_storage(self, db: Database, config: Config) -> None:
        """Test macro and sentiment data storage."""
        scheduler = IngestionScheduler(db=db, config=config)

        macros = [
            MacroDataPoint("USD_TRY", date(2026, 4, 1), 38.45, "tcmb"),
            MacroDataPoint("EUR_TRY", date(2026, 4, 1), 41.20, "tcmb"),
        ]
        stored_macro = await scheduler.store_macro(macros)
        assert stored_macro == 2

        sentiments = [
            SentimentRecord("THYAO", date(2026, 4, 1), "google_news", "THY yükseldi", None, "text"),
        ]
        stored_sent = await scheduler.store_sentiment(sentiments)
        assert stored_sent == 1

        # Verify counts
        with db.connect() as conn:
            macro_count = conn.execute("SELECT COUNT(*) FROM macro_data").fetchone()[0]
            sent_count = conn.execute("SELECT COUNT(*) FROM sentiment_data").fetchone()[0]
            assert macro_count == 2
            assert sent_count == 1
```

- [ ] **Step 2: Run all tests**

```bash
uv run pytest tests/ -v
```

Expected: all tests PASS.

- [ ] **Step 3: Commit**

```bash
git add tests/test_ingest/test_integration.py
git commit -m "test: integration test for full fetch-store-query pipeline"
```

---

### Task 12: Update CLAUDE.md

**Files:**
- Modify: `CLAUDE.md`

- [ ] **Step 1: Add project-specific guidance to CLAUDE.md**

Append to the existing `CLAUDE.md`:

```markdown

## Project: BIST Predictor

### Quick Start
```bash
uv sync                        # Install dependencies
uv run bist-predict --help     # CLI usage
uv run pytest tests/ -v        # Run all tests
```

### Architecture
Modular pipeline: Data Ingest → Feature Engine (Rust) → Quant Alpha → ML Ensemble → Evaluation
- Design spec: `docs/superpowers/specs/2026-04-02-bist-predictor-design.md`
- Implementation plans: `docs/superpowers/plans/`

### Key Conventions
- All data types in `src/bist_predict/ingest/types.py` — use these, don't create ad-hoc dicts
- Database access via `Database` class only — never raw sqlite3 outside storage/
- Collectors implement `PriceCollector`, `MacroCollector`, or `SentimentCollector` protocols
- All async I/O — use `httpx.AsyncClient`, wrap sync libs with `asyncio.to_thread`
- Tests use `respx` for HTTP mocking, `tmp_db_path` fixture for database tests
- Config loaded via `load_config()` — never hardcode paths or API keys

### Testing
```bash
uv run pytest tests/ -v                        # All tests
uv run pytest tests/test_ingest/ -v             # Ingest tests only
uv run pytest tests/test_storage/ -v            # Storage tests only
uv run pytest tests/path/test_file.py::test_fn  # Single test
```
```

- [ ] **Step 2: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: update CLAUDE.md with project-specific guidance"
```

---

## Plan Summary

| Task | Component | Tests |
|------|-----------|-------|
| 1 | Project scaffolding, config, pyproject | - |
| 2 | SQLite storage layer | 9 tests |
| 3 | Data types & collector protocols | 4 tests |
| 4 | OHLCV quality validation | 9 tests |
| 5 | Is Yatirim API client | 4 tests |
| 6 | Yahoo Finance fallback | 3 tests |
| 7 | TCMB EVDS macro client | 4 tests |
| 8 | Sentiment collectors | 3 tests |
| 9 | Ingestion scheduler | 5 tests |
| 10 | CLI fetch command | manual verify |
| 11 | Integration test | 3 tests |
| 12 | Update CLAUDE.md | - |

**Total: 12 tasks, ~44 tests, 12 commits**

**Subsequent plans (written after this plan completes):**
- Plan 2: Rust Feature Engine (PyO3, technical indicators, patterns, correlations)
- Plan 3: Quantitative Alpha Layer (factors, HMM, GARCH, Kalman, Kelly, wavelets)
- Plan 4: Model Layer (XGBoost, LightGBM, LSTM, Transformer, ensemble, calibration)
- Plan 5: Evaluation & Backtesting (walk-forward, metrics, live tracker)
- Plan 6: CLI Polish (signals, backtest, accuracy commands, detailed output)
