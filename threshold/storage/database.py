"""SQLite database connection manager."""

from __future__ import annotations

import logging
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Generator

logger = logging.getLogger(__name__)


class Database:
    """SQLite database with WAL mode and foreign key enforcement."""

    def __init__(self, path: str | Path):
        self.path = Path(path).expanduser().resolve()
        self._conn: sqlite3.Connection | None = None

    def _ensure_dir(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def connect(self) -> sqlite3.Connection:
        """Open the database connection with optimal settings."""
        if self._conn is not None:
            return self._conn

        self._ensure_dir()
        self._conn = sqlite3.connect(str(self.path))
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode = WAL")
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._conn.execute("PRAGMA busy_timeout = 5000")
        logger.debug("Connected to database: %s", self.path)
        return self._conn

    @property
    def conn(self) -> sqlite3.Connection:
        return self.connect()

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None
            logger.debug("Closed database connection")

    @contextmanager
    def transaction(self) -> Generator[sqlite3.Cursor, None, None]:
        """Context manager for a database transaction."""
        conn = self.connect()
        cursor = conn.cursor()
        try:
            yield cursor
            conn.commit()
        except Exception:
            conn.rollback()
            raise

    def execute(self, sql: str, params: tuple = ()) -> sqlite3.Cursor:
        """Execute a single SQL statement."""
        return self.conn.execute(sql, params)

    def executemany(self, sql: str, params_seq: list[tuple]) -> sqlite3.Cursor:
        """Execute a SQL statement against multiple parameter sets."""
        return self.conn.executemany(sql, params_seq)

    def executescript(self, sql: str) -> None:
        """Execute a multi-statement SQL script."""
        self.conn.executescript(sql)

    def fetchone(self, sql: str, params: tuple = ()) -> sqlite3.Row | None:
        """Execute and fetch one result."""
        return self.conn.execute(sql, params).fetchone()

    def fetchall(self, sql: str, params: tuple = ()) -> list[sqlite3.Row]:
        """Execute and fetch all results."""
        return self.conn.execute(sql, params).fetchall()

    def schema_version(self) -> int:
        """Get the current schema version. Returns 0 if no schema exists."""
        try:
            row = self.fetchone(
                "SELECT MAX(version) as v FROM _schema_version"
            )
            return row["v"] if row and row["v"] is not None else 0
        except sqlite3.OperationalError:
            return 0

    def __enter__(self) -> "Database":
        self.connect()
        return self

    def __exit__(self, *args) -> None:
        self.close()

    def __repr__(self) -> str:
        return f"Database({self.path})"
