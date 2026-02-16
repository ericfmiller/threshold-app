"""Shared test fixtures for Threshold.

Provides reusable fixtures for database, config, price data, and SA data
across all test modules.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from threshold.config.schema import ThresholdConfig
from threshold.engine.context import ScoringContext
from threshold.storage.database import Database
from threshold.storage.migrations import ensure_schema

# ---------------------------------------------------------------------------
# Core infrastructure
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_dir(tmp_path: Path) -> Path:
    """Temporary directory for test artifacts."""
    return tmp_path


@pytest.fixture
def test_config(tmp_path: Path) -> ThresholdConfig:
    """Minimal config with temp database path."""
    return ThresholdConfig(
        database={"path": str(tmp_path / "test.db")},
        output={
            "score_history_dir": str(tmp_path / "history"),
            "dashboard_dir": str(tmp_path / "dashboards"),
            "narrative_dir": str(tmp_path / "narratives"),
        },
    )


@pytest.fixture
def test_db(tmp_path: Path) -> Database:
    """Database with schema applied, using temp file."""
    db = Database(tmp_path / "test.db")
    ensure_schema(db)
    yield db
    db.close()


@pytest.fixture
def memory_db() -> Database:
    """In-memory database for fast unit tests."""
    db = Database(":memory:")
    # In-memory DBs need direct schema application since the migration
    # runner reads files. Apply the SQL directly.
    migration_path = Path(__file__).parent.parent / "threshold" / "migrations" / "001_initial.sql"
    sql = migration_path.read_text()
    db.executescript(sql)
    yield db
    db.close()


# ---------------------------------------------------------------------------
# Price series generators (deterministic)
# ---------------------------------------------------------------------------

@pytest.fixture
def uptrend_252() -> pd.DataFrame:
    """252-bar uptrend with Close and Volume (seed=42)."""
    np.random.seed(42)
    n = 252
    close = 100 * np.cumprod(1 + np.random.normal(0.0005, 0.01, n))
    volume = np.random.uniform(500_000, 2_000_000, n)
    return pd.DataFrame({"Close": close, "Volume": volume})


@pytest.fixture
def downtrend_252() -> pd.DataFrame:
    """252-bar downtrend with Close and Volume (seed=42)."""
    np.random.seed(42)
    n = 252
    close = 100 * np.cumprod(1 + np.random.normal(-0.001, 0.015, n))
    volume = np.random.uniform(500_000, 2_000_000, n)
    return pd.DataFrame({"Close": close, "Volume": volume})


@pytest.fixture
def oversold_252() -> pd.DataFrame:
    """252-bar series with sharp selloff in last 30 bars (seed=42)."""
    np.random.seed(42)
    n = 252
    stable = 100 * np.cumprod(1 + np.random.normal(0.0003, 0.008, n - 30))
    crash = stable[-1] * np.cumprod(1 + np.random.normal(-0.008, 0.012, 30))
    close = np.concatenate([stable, crash])
    volume = np.random.uniform(500_000, 3_000_000, n)
    return pd.DataFrame({"Close": close, "Volume": volume})


@pytest.fixture
def spy_252() -> pd.Series:
    """252-bar SPY close series (seed=99)."""
    np.random.seed(99)
    return pd.Series(450 * np.cumprod(1 + np.random.normal(0.0004, 0.008, 252)))


# ---------------------------------------------------------------------------
# SA data profiles
# ---------------------------------------------------------------------------

@pytest.fixture
def strong_sa_data() -> dict:
    """SA data for a high-quality stock (Quant ~4.8)."""
    return {
        "quantScore": 4.8,
        "momentum": "A",
        "profitability": "A-",
        "revisions": "A-",
        "growth": "B+",
        "valuation": "B",
    }


@pytest.fixture
def average_sa_data() -> dict:
    """SA data for an average stock (Quant ~3.5)."""
    return {
        "quantScore": 3.5,
        "momentum": "B",
        "profitability": "B",
        "revisions": "C+",
        "growth": "C",
        "valuation": "C+",
    }


@pytest.fixture
def weak_sa_data() -> dict:
    """SA data for a weak stock (Quant ~1.5)."""
    return {
        "quantScore": 1.5,
        "momentum": "D",
        "profitability": "D-",
        "revisions": "F",
        "growth": "D",
        "valuation": "D+",
    }


# ---------------------------------------------------------------------------
# Scoring contexts
# ---------------------------------------------------------------------------

@pytest.fixture
def normal_scoring_ctx(spy_252) -> ScoringContext:
    """ScoringContext in NORMAL regime."""
    return ScoringContext(
        market_regime_score=0.55,
        vix_regime="NORMAL",
        spy_close=spy_252,
    )


@pytest.fixture
def fear_scoring_ctx(spy_252) -> ScoringContext:
    """ScoringContext in FEAR regime with drawdown classifications."""
    return ScoringContext(
        market_regime_score=0.65,
        vix_regime="FEAR",
        spy_close=spy_252,
        drawdown_classifications={
            "HEDGE_TICKER": {"classification": "HEDGE", "downside_capture": -0.85},
            "DEFENSIVE_TICKER": {"classification": "DEFENSIVE", "downside_capture": 0.15},
            "AMP_TICKER": {"classification": "AMPLIFIER", "downside_capture": 1.78},
        },
    )
