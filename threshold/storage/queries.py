"""Named query functions for database operations."""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any

from threshold.storage.database import Database

# ---------------------------------------------------------------------------
# Tickers
# ---------------------------------------------------------------------------

def upsert_ticker(
    db: Database,
    symbol: str,
    *,
    name: str | None = None,
    type: str | None = None,
    sector: str | None = None,
    sector_detail: str | None = None,
    yf_symbol: str | None = None,
    alden_category: str = "Other",
    is_gold: bool = False,
    is_hard_money: bool = False,
    is_crypto: bool = False,
    is_crypto_exempt: bool = False,
    is_cash: bool = False,
    is_war_chest: bool = False,
    is_international: bool = False,
    is_amplifier_trim: bool = False,
    is_defensive_add: bool = False,
    dd_override: str | None = None,
    verified_at: str | None = None,
    needs_review: bool = False,
    notes: str | None = None,
) -> None:
    """Insert or update a ticker in the database."""
    db.execute(
        """INSERT INTO tickers (
            symbol, name, type, sector, sector_detail, yf_symbol,
            alden_category, is_gold, is_hard_money, is_crypto,
            is_crypto_exempt, is_cash, is_war_chest, is_international,
            is_amplifier_trim, is_defensive_add, dd_override,
            verified_at, needs_review, notes, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))
        ON CONFLICT(symbol) DO UPDATE SET
            name=excluded.name, type=excluded.type, sector=excluded.sector,
            sector_detail=excluded.sector_detail, yf_symbol=excluded.yf_symbol,
            alden_category=excluded.alden_category, is_gold=excluded.is_gold,
            is_hard_money=excluded.is_hard_money, is_crypto=excluded.is_crypto,
            is_crypto_exempt=excluded.is_crypto_exempt, is_cash=excluded.is_cash,
            is_war_chest=excluded.is_war_chest, is_international=excluded.is_international,
            is_amplifier_trim=excluded.is_amplifier_trim,
            is_defensive_add=excluded.is_defensive_add,
            dd_override=excluded.dd_override, verified_at=excluded.verified_at,
            needs_review=excluded.needs_review, notes=excluded.notes,
            updated_at=datetime('now')
        """,
        (
            symbol, name, type, sector, sector_detail, yf_symbol,
            alden_category, int(is_gold), int(is_hard_money), int(is_crypto),
            int(is_crypto_exempt), int(is_cash), int(is_war_chest),
            int(is_international), int(is_amplifier_trim), int(is_defensive_add),
            dd_override, verified_at, int(needs_review), notes,
        ),
    )
    db.conn.commit()


def get_ticker(db: Database, symbol: str) -> dict[str, Any] | None:
    """Get a single ticker's full metadata."""
    row = db.fetchone("SELECT * FROM tickers WHERE symbol = ?", (symbol,))
    return dict(row) if row else None


def list_tickers(db: Database, *, needs_review: bool | None = None) -> list[dict[str, Any]]:
    """List all tickers, optionally filtering by review status."""
    if needs_review is not None:
        rows = db.fetchall(
            "SELECT * FROM tickers WHERE needs_review = ? ORDER BY symbol",
            (int(needs_review),),
        )
    else:
        rows = db.fetchall("SELECT * FROM tickers ORDER BY symbol")
    return [dict(r) for r in rows]


def delete_ticker(db: Database, symbol: str) -> bool:
    """Delete a ticker. Returns True if a row was deleted."""
    cursor = db.execute("DELETE FROM tickers WHERE symbol = ?", (symbol,))
    db.conn.commit()
    return cursor.rowcount > 0


def get_ticker_count(db: Database) -> int:
    """Get the number of registered tickers."""
    row = db.fetchone("SELECT COUNT(*) as cnt FROM tickers")
    return row["cnt"] if row else 0


# ---------------------------------------------------------------------------
# Accounts
# ---------------------------------------------------------------------------

def upsert_account(
    db: Database,
    id: str,
    name: str,
    type: str,
    institution: str = "Fidelity",
    tax_treatment: str = "taxable",
    sa_export_prefix: str = "",
    sa_export_prefix_old: str = "",
) -> None:
    """Insert or update an account."""
    db.execute(
        """INSERT INTO accounts (id, name, type, institution, tax_treatment,
            sa_export_prefix, sa_export_prefix_old)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(id) DO UPDATE SET
            name=excluded.name, type=excluded.type,
            institution=excluded.institution,
            tax_treatment=excluded.tax_treatment,
            sa_export_prefix=excluded.sa_export_prefix,
            sa_export_prefix_old=excluded.sa_export_prefix_old
        """,
        (id, name, type, institution, tax_treatment, sa_export_prefix, sa_export_prefix_old),
    )
    db.conn.commit()


def list_accounts(db: Database) -> list[dict[str, Any]]:
    """List all accounts."""
    rows = db.fetchall("SELECT * FROM accounts WHERE is_active = 1 ORDER BY name")
    return [dict(r) for r in rows]


# ---------------------------------------------------------------------------
# Scoring Runs
# ---------------------------------------------------------------------------

def insert_scoring_run(db: Database, run_id: str, **kwargs: Any) -> str:
    """Insert a new scoring run. Returns run_id."""
    cols = ["run_id", "started_at"] + list(kwargs.keys())
    placeholders = ", ".join(["?"] * len(cols))
    col_names = ", ".join(cols)
    values = [run_id, datetime.now().isoformat()] + list(kwargs.values())
    db.execute(f"INSERT INTO scoring_runs ({col_names}) VALUES ({placeholders})", tuple(values))
    db.conn.commit()
    return run_id


def update_scoring_run(db: Database, run_id: str, **kwargs: Any) -> None:
    """Update a scoring run with results."""
    sets = ", ".join(f"{k} = ?" for k in kwargs)
    values = list(kwargs.values()) + [run_id]
    db.execute(f"UPDATE scoring_runs SET {sets} WHERE run_id = ?", tuple(values))
    db.conn.commit()


def get_latest_scoring_run(db: Database) -> dict[str, Any] | None:
    """Get the most recent scoring run."""
    row = db.fetchone(
        "SELECT * FROM scoring_runs ORDER BY started_at DESC LIMIT 1"
    )
    return dict(row) if row else None


def list_scoring_runs(db: Database, limit: int = 20) -> list[dict[str, Any]]:
    """List recent scoring runs."""
    rows = db.fetchall(
        "SELECT * FROM scoring_runs ORDER BY started_at DESC LIMIT ?",
        (limit,),
    )
    return [dict(r) for r in rows]


# ---------------------------------------------------------------------------
# Scores
# ---------------------------------------------------------------------------

def insert_score(db: Database, run_id: str, symbol: str, **kwargs: Any) -> int:
    """Insert a score record. Returns the score ID."""
    cols = ["run_id", "scored_at", "symbol"] + list(kwargs.keys())
    placeholders = ", ".join(["?"] * len(cols))
    col_names = ", ".join(cols)
    values = [run_id, datetime.now().isoformat(), symbol] + list(kwargs.values())
    cursor = db.execute(
        f"INSERT INTO scores ({col_names}) VALUES ({placeholders})", tuple(values)
    )
    db.conn.commit()
    return cursor.lastrowid or 0


def get_latest_scores(db: Database) -> dict[str, dict[str, Any]]:
    """Load the most recent scoring run's results as {symbol: score_data}."""
    latest_run = get_latest_scoring_run(db)
    if not latest_run:
        return {}
    rows = db.fetchall(
        "SELECT * FROM scores WHERE run_id = ?", (latest_run["run_id"],)
    )
    return {row["symbol"]: dict(row) for row in rows}


def get_score_history(
    db: Database, symbol: str, limit: int = 52
) -> list[dict[str, Any]]:
    """Load DCS history for a ticker."""
    rows = db.fetchall(
        "SELECT * FROM scores WHERE symbol = ? ORDER BY scored_at DESC LIMIT ?",
        (symbol, limit),
    )
    return [dict(r) for r in rows]


# ---------------------------------------------------------------------------
# Signals
# ---------------------------------------------------------------------------

def insert_signal(
    db: Database,
    score_id: int,
    signal_type: str,
    severity: str,
    criterion: str,
    message: str,
    metadata: dict[str, Any] | None = None,
) -> int:
    """Insert a signal. Returns signal ID."""
    meta_json = json.dumps(metadata) if metadata else None
    cursor = db.execute(
        """INSERT INTO signals (score_id, signal_type, severity, criterion, message, metadata_json)
        VALUES (?, ?, ?, ?, ?, ?)""",
        (score_id, signal_type, severity, criterion, message, meta_json),
    )
    db.conn.commit()
    return cursor.lastrowid or 0


# ---------------------------------------------------------------------------
# Drawdown Classifications
# ---------------------------------------------------------------------------

def upsert_drawdown_classification(
    db: Database,
    backtest_date: str,
    symbol: str,
    classification: str,
    **kwargs: Any,
) -> None:
    """Insert or update a drawdown classification."""
    cols = ["backtest_date", "symbol", "classification"] + list(kwargs.keys())
    placeholders = ", ".join(["?"] * len(cols))
    col_names = ", ".join(cols)
    values = [backtest_date, symbol, classification] + list(kwargs.values())
    db.execute(
        f"""INSERT INTO drawdown_classifications ({col_names}) VALUES ({placeholders})
        ON CONFLICT(backtest_date, symbol) DO UPDATE SET
            classification=excluded.classification""",
        tuple(values),
    )
    db.conn.commit()


def get_drawdown_classifications(db: Database) -> dict[str, dict[str, Any]]:
    """Get the latest drawdown classifications as {symbol: data}."""
    rows = db.fetchall(
        """SELECT * FROM drawdown_classifications
        WHERE backtest_date = (SELECT MAX(backtest_date) FROM drawdown_classifications)
        ORDER BY symbol"""
    )
    return {row["symbol"]: dict(row) for row in rows}


# ---------------------------------------------------------------------------
# Data Freshness
# ---------------------------------------------------------------------------

def update_data_freshness(
    db: Database, source: str, status: str, details: str | None = None
) -> None:
    """Update data freshness tracking."""
    db.execute(
        """INSERT INTO data_freshness (source, last_updated, last_status, details, updated_at)
        VALUES (?, datetime('now'), ?, ?, datetime('now'))
        ON CONFLICT(source) DO UPDATE SET
            last_updated=datetime('now'), last_status=excluded.last_status,
            details=excluded.details, updated_at=datetime('now')""",
        (source, status, details),
    )
    db.conn.commit()


def get_data_freshness(db: Database) -> dict[str, dict[str, Any]]:
    """Get all data freshness records."""
    rows = db.fetchall("SELECT * FROM data_freshness ORDER BY source")
    return {row["source"]: dict(row) for row in rows}


# ---------------------------------------------------------------------------
# Alden Categories
# ---------------------------------------------------------------------------

def seed_alden_categories(db: Database, categories: dict[str, Any]) -> None:
    """Seed the alden_categories table from config."""
    for name, cat in categories.items():
        db.execute(
            """INSERT INTO alden_categories (name, target_low, target_high, tsp_pct,
                is_cross_cutting, is_catchall)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(name) DO UPDATE SET
                target_low=excluded.target_low, target_high=excluded.target_high,
                tsp_pct=excluded.tsp_pct, is_cross_cutting=excluded.is_cross_cutting,
                is_catchall=excluded.is_catchall""",
            (
                name, cat.target[0], cat.target[1],
                cat.tsp_pct, int(cat.cross_cutting), int(cat.is_catchall),
            ),
        )
    db.conn.commit()
