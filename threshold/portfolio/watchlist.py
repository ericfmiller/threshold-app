"""Watchlist management — tracking candidate tickers for future allocation.

Manages watchlists sourced from Seeking Alpha Z-file exports. Tickers
in a watchlist get DCS-scored alongside portfolio holdings so their
buy signals are visible in the dashboard and narrative.

Supports multiple named watchlists:
  - "energy": Energy sector candidates (Z-Energy Assets)
  - "stocks_etfs": General equities (Z-Stocks and ETFs)

Uses the ``watchlists`` table from migration 001.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class WatchlistImportResult:
    """Result from importing a Z-file watchlist."""

    name: str = ""
    added: int = 0
    skipped: int = 0
    errors: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# CRUD operations
# ---------------------------------------------------------------------------

def add_to_watchlist(
    db: Any,
    name: str,
    symbol: str,
    sa_quant: float | None = None,
    source_file: str = "",
) -> bool:
    """Add a ticker to a named watchlist.

    Parameters
    ----------
    db : Database
        Open database connection.
    name : str
        Watchlist name (e.g., 'energy', 'stocks_etfs').
    symbol : str
        Ticker symbol.
    sa_quant : float | None
        Current SA Quant score (if known).
    source_file : str
        Source file path (for tracking).

    Returns
    -------
    bool
        True if added or updated, False on error.
    """
    try:
        db.execute(
            """INSERT INTO watchlists (name, symbol, sa_quant, source_file)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(name, symbol) DO UPDATE SET
                sa_quant=excluded.sa_quant,
                source_file=excluded.source_file""",
            (name, symbol.upper(), sa_quant, source_file),
        )
        db.conn.commit()
        return True
    except Exception as e:
        logger.error("Failed to add %s to watchlist %s: %s", symbol, name, e)
        return False


def remove_from_watchlist(db: Any, name: str, symbol: str) -> bool:
    """Remove a ticker from a named watchlist.

    Returns True if a row was deleted.
    """
    cursor = db.execute(
        "DELETE FROM watchlists WHERE name = ? AND symbol = ?",
        (name, symbol.upper()),
    )
    db.conn.commit()
    return cursor.rowcount > 0


def list_watchlist(
    db: Any,
    name: str | None = None,
) -> list[dict[str, Any]]:
    """List tickers in a watchlist (or all watchlists).

    Parameters
    ----------
    db : Database
        Open database connection.
    name : str | None
        Watchlist name. If None, returns all watchlists.

    Returns
    -------
    list[dict]
        Watchlist entries with symbol, sa_quant, etc.
    """
    if name is not None:
        rows = db.fetchall(
            "SELECT * FROM watchlists WHERE name = ? ORDER BY symbol",
            (name,),
        )
    else:
        rows = db.fetchall(
            "SELECT * FROM watchlists ORDER BY name, symbol",
        )
    return [dict(r) for r in rows]


def clear_watchlist(db: Any, name: str) -> int:
    """Clear all entries from a named watchlist.

    Returns the number of rows deleted.
    """
    cursor = db.execute(
        "DELETE FROM watchlists WHERE name = ?", (name,),
    )
    db.conn.commit()
    return cursor.rowcount


def get_watchlist_symbols(db: Any, name: str | None = None) -> list[str]:
    """Get just the ticker symbols from a watchlist.

    Parameters
    ----------
    db : Database
        Open database connection.
    name : str | None
        Watchlist name. If None, returns from all watchlists.

    Returns
    -------
    list[str]
        Unique ticker symbols.
    """
    entries = list_watchlist(db, name)
    seen = set()
    symbols = []
    for entry in entries:
        sym = entry.get("symbol", "")
        if sym and sym not in seen:
            seen.add(sym)
            symbols.append(sym)
    return symbols


# ---------------------------------------------------------------------------
# Z-file import
# ---------------------------------------------------------------------------

def import_zfile_watchlist(
    db: Any,
    name: str,
    export_path: str,
) -> WatchlistImportResult:
    """Import a Z-file SA export into a named watchlist.

    Reads the Summary sheet from the Z-file, extracts ticker symbols
    and Quant scores, and populates the watchlist.

    Parameters
    ----------
    db : Database
        Open database connection.
    name : str
        Watchlist name.
    export_path : str
        Path to the Z-file xlsx export.

    Returns
    -------
    WatchlistImportResult
        Import statistics.
    """
    from threshold.data.adapters.sa_export_reader import read_sa_export

    result = WatchlistImportResult(name=name)

    try:
        sheets = read_sa_export(export_path)
    except Exception as e:
        result.errors.append(f"Failed to read {export_path}: {e}")
        return result

    summary = sheets.get("Summary")
    if summary is None or summary.empty:
        result.errors.append(f"No Summary sheet in {export_path}")
        return result

    # Extract symbols and quant scores
    skip_symbols = {"CASH", "TOTAL", "ACCOUNT TOTAL", ""}
    for _, row in summary.iterrows():
        symbol = str(row.get("Symbol", "")).strip().upper()
        if not symbol or symbol in skip_symbols:
            continue

        quant = row.get("Quant Rating")
        if quant is not None:
            try:
                quant = float(quant)
            except (ValueError, TypeError):
                quant = None

        if add_to_watchlist(db, name, symbol, quant, export_path):
            result.added += 1
        else:
            result.skipped += 1

    logger.info(
        "Imported %d tickers to watchlist '%s' from %s",
        result.added, name, export_path,
    )
    return result
