"""Database migration runner.

Migrations are SQL files in threshold/migrations/ named NNN_description.sql.
Each migration is applied exactly once, tracked in the _schema_version table.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

from threshold.storage.database import Database

logger = logging.getLogger(__name__)

MIGRATION_PATTERN = re.compile(r"^(\d{3})_.*\.sql$")


def _discover_migrations() -> list[tuple[int, str, str]]:
    """Find all migration files and return (version, name, sql) tuples."""
    migrations = []

    migration_dir = Path(__file__).parent.parent / "migrations"
    if not migration_dir.exists():
        logger.warning("Migration directory not found: %s", migration_dir)
        return migrations

    for sql_file in sorted(migration_dir.glob("*.sql")):
        match = MIGRATION_PATTERN.match(sql_file.name)
        if match:
            version = int(match.group(1))
            sql = sql_file.read_text()
            migrations.append((version, sql_file.name, sql))

    return migrations


def apply_migrations(db: Database) -> int:
    """Apply all pending migrations. Returns the new schema version."""
    current = db.schema_version()
    migrations = _discover_migrations()
    applied = 0

    for version, name, sql in migrations:
        if version > current:
            logger.info("Applying migration %s (v%d -> v%d)", name, current, version)
            try:
                db.executescript(sql)
                applied += 1
                current = version
            except Exception as e:
                logger.error("Migration %s failed: %s", name, e)
                raise RuntimeError(f"Migration {name} failed: {e}") from e

    if applied:
        logger.info("Applied %d migration(s). Schema version: %d", applied, current)
    else:
        logger.debug("Schema up to date (version %d)", current)

    return current


def ensure_schema(db: Database) -> int:
    """Ensure the database schema is up to date. Returns schema version."""
    return apply_migrations(db)
