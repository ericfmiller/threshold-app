"""Shared test fixtures for Threshold."""

from __future__ import annotations

import sqlite3
import tempfile
from pathlib import Path

import pytest

from threshold.config.schema import ThresholdConfig
from threshold.storage.database import Database
from threshold.storage.migrations import ensure_schema


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
