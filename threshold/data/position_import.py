"""Position import from SA export Holdings sheets.

Reads the Holdings sheet from each account's SA Excel export and
populates the positions table in the database. Handles CASH/TOTAL
rows, duplicate tickers, and multi-account aggregation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

# Rows to always skip in Holdings sheets (aggregation rows, not positions).
# CASH is intentionally NOT here — it has real market value.
# Note: empty string "" is NOT in this set — blank symbols may be cash rows.
_HOLDINGS_SKIP = frozenset({"TOTAL", "ACCOUNT TOTAL", "NAN"})


@dataclass
class ImportResult:
    """Result of a position import run."""

    positions_imported: int = 0
    accounts_processed: int = 0
    errors: list[str] = field(default_factory=list)


def _safe_float(val: Any) -> float:
    """Convert a value to float, returning 0.0 on failure.

    SA exports use "-" for fields that don't apply (e.g. shares/cost on
    cash rows). This must not break parsing of other fields.
    """
    try:
        return float(val or 0)
    except (ValueError, TypeError):
        return 0.0


# Symbols that represent cash positions, even when the Symbol column
# is blank or contains a money-market fund name.
_CASH_SYMBOLS = frozenset({"CASH", "SPAXX", "FDRXX", "FCASH", "CORE"})


def _parse_holdings_sheet(
    holdings_df: pd.DataFrame,
) -> list[dict[str, Any]]:
    """Parse an SA export Holdings sheet into position dicts.

    Parameters
    ----------
    holdings_df : pd.DataFrame
        Holdings sheet from an SA export file.

    Returns
    -------
    list[dict]
        List of position dicts with: symbol, shares, cost_basis,
        market_value, weight.

    Notes
    -----
    Cash rows are always captured (as symbol ``CASH``) because they
    contribute real dollar value to account totals. They may appear as:
    - Symbol = "CASH" with shares = "-"
    - Symbol blank / NaN but Value > 0 (some export formats)
    - Symbol = a money-market fund (SPAXX, FDRXX, etc.)
    """
    if holdings_df is None or holdings_df.empty:
        return []

    positions: list[dict[str, Any]] = []

    for _, row in holdings_df.iterrows():
        raw_symbol = row.get("Symbol", "")
        symbol = str(raw_symbol).strip().upper() if not pd.isna(raw_symbol) else ""

        # Skip aggregation rows (TOTAL, ACCOUNT TOTAL, etc.)
        if symbol in _HOLDINGS_SKIP:
            continue

        # Detect cash rows: explicit CASH symbol, money-market fund,
        # or blank symbol with positive value.
        is_cash = symbol in _CASH_SYMBOLS
        if not symbol:
            # Row with no symbol — check if it has value (likely cash)
            test_val = _safe_float(row.get("Value", row.get("Market Value", 0)))
            if test_val > 0:
                is_cash = True
            else:
                continue  # truly empty row

        if is_cash:
            symbol = "CASH"  # normalize all cash variants

        shares = _safe_float(row.get("Shares", 0))
        cost_basis = _safe_float(row.get("Cost", row.get("Cost Basis", 0)))
        market_value = _safe_float(row.get("Value", row.get("Market Value", 0)))
        raw_weight = _safe_float(row.get("Weight", row.get("% of Portfolio", 0)))
        weight = raw_weight if raw_weight <= 1.0 else raw_weight / 100.0

        if shares <= 0 and market_value <= 0:
            continue

        positions.append({
            "symbol": symbol,
            "shares": shares,
            "cost_basis": cost_basis,
            "market_value": market_value,
            "weight": weight,
        })

    return positions


def import_positions_from_export(
    db: Any,
    account_id: str,
    export_path: str | Path,
    snapshot_date: str | None = None,
) -> int:
    """Import positions from a single SA export file for one account.

    Parameters
    ----------
    db : Database
        Open database connection.
    account_id : str
        Account ID to associate positions with.
    export_path : str | Path
        Path to the SA export .xlsx file.
    snapshot_date : str | None
        Date for this snapshot (YYYY-MM-DD). If None, uses today.

    Returns
    -------
    int
        Number of positions imported.
    """
    from threshold.data.adapters.sa_export_reader import read_sa_export

    if snapshot_date is None:
        snapshot_date = date.today().isoformat()

    sheets = read_sa_export(export_path)
    holdings = sheets.get("Holdings")
    if holdings is None:
        logger.warning("No Holdings sheet in %s", export_path)
        return 0

    positions = _parse_holdings_sheet(holdings)
    if not positions:
        logger.warning("No valid positions in Holdings sheet of %s", export_path)
        return 0

    # Ensure CASH ticker exists in the DB (FK requirement).
    # Auto-register it the first time a cash position appears.
    has_cash = any(p["symbol"] == "CASH" for p in positions)
    if has_cash:
        from threshold.storage.queries import get_ticker, upsert_ticker

        if get_ticker(db, "CASH") is None:
            upsert_ticker(
                db, symbol="CASH", name="Cash & Cash Equivalents",
                type="fund", sector="Cash", is_cash=True,
            )
            db.conn.commit()
            logger.info("Auto-registered CASH ticker")

    count = 0
    for pos in positions:
        db.execute(
            """INSERT INTO positions (account_id, symbol, shares, cost_basis,
                market_value, weight, snapshot_date, source)
            VALUES (?, ?, ?, ?, ?, ?, ?, 'sa_export')
            ON CONFLICT(account_id, symbol, snapshot_date) DO UPDATE SET
                shares=excluded.shares, cost_basis=excluded.cost_basis,
                market_value=excluded.market_value, weight=excluded.weight,
                source=excluded.source""",
            (
                account_id,
                pos["symbol"],
                pos["shares"],
                pos["cost_basis"],
                pos["market_value"],
                pos["weight"],
                snapshot_date,
            ),
        )
        count += 1

    db.conn.commit()
    logger.info("Imported %d positions for account %s", count, account_id)
    return count


def import_all_positions(
    db: Any,
    config: Any,
    snapshot_date: str | None = None,
) -> ImportResult:
    """Import positions from all accounts' SA exports.

    Parameters
    ----------
    db : Database
        Open database connection.
    config : ThresholdConfig
        Configuration with export_dir and account definitions.
    snapshot_date : str | None
        Date for this snapshot (YYYY-MM-DD). If None, uses today.

    Returns
    -------
    ImportResult
        Summary of the import.
    """
    from threshold.data.adapters.sa_export_reader import find_latest_export_per_account
    from threshold.storage.queries import list_accounts

    result = ImportResult()

    export_dir = config.data_sources.seeking_alpha.export_dir
    if not export_dir:
        result.errors.append("No SA export directory configured")
        return result

    accounts = list_accounts(db)
    if not accounts:
        result.errors.append("No accounts configured in database")
        return result

    # Build account list for finder
    account_defs = [
        {
            "id": acct["id"],
            "sa_export_prefix": acct.get("sa_export_prefix", ""),
            "sa_export_prefix_old": acct.get("sa_export_prefix_old", ""),
        }
        for acct in accounts
    ]

    exports = find_latest_export_per_account(export_dir, account_defs)

    for acct in accounts:
        acct_id = acct["id"]
        export_path = exports.get(acct_id)
        if export_path is None:
            logger.debug("No export found for account %s", acct_id)
            continue

        try:
            count = import_positions_from_export(
                db, acct_id, export_path, snapshot_date,
            )
            result.positions_imported += count
            result.accounts_processed += 1
        except Exception as e:
            logger.error("Failed to import positions for %s: %s", acct_id, e)
            result.errors.append(f"{acct_id}: {e}")

    logger.info(
        "Position import: %d positions from %d accounts",
        result.positions_imported,
        result.accounts_processed,
    )
    return result


def load_positions_from_db(
    db: Any,
    snapshot_date: str | None = None,
) -> list[dict[str, Any]]:
    """Load positions from the database for a given snapshot date.

    Parameters
    ----------
    db : Database
        Open database connection.
    snapshot_date : str | None
        Date to load. If None, loads the most recent snapshot.

    Returns
    -------
    list[dict]
        Position records from the database.
    """
    if snapshot_date is None:
        row = db.fetchone(
            "SELECT MAX(snapshot_date) as max_date FROM positions"
        )
        if row and row["max_date"]:
            snapshot_date = row["max_date"]
        else:
            return []

    rows = db.fetchall(
        "SELECT * FROM positions WHERE snapshot_date = ? ORDER BY account_id, symbol",
        (snapshot_date,),
    )
    return [dict(r) for r in rows]
