"""Decision journal — trade logging with outcome tracking.

Records trade decisions with context (DCS, VIX regime, thesis) and
tracks outcomes at 1-week, 4-week, 12-week, and 26-week intervals.
Supports behavioral analysis: was it panic or process? Was there a thesis?

Uses the ``trade_journal`` and ``trade_outcomes`` tables from migration 004.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class TradeEntry:
    """A trade journal entry."""

    id: int = 0
    ticker: str = ""
    action: str = ""  # BUY, SELL, TRIM, ADD
    account: str = ""
    shares: float = 0.0
    price: float = 0.0
    trade_date: str = ""
    thesis: str = ""
    dcs_at_decision: float | None = None
    vix_regime: str = ""
    is_panic_or_process: str = ""  # "panic" or "process"
    has_thesis: bool = True
    deployment_gates_passed: bool = True
    notes: str = ""


@dataclass
class TradeOutcome:
    """Outcome tracking at specific time windows."""

    entry_id: int = 0
    window: str = ""  # 1w, 4w, 12w, 26w
    ticker_return: float = 0.0
    spy_return: float = 0.0
    alpha: float = 0.0


@dataclass
class JournalSummary:
    """Summary statistics for a set of journal entries."""

    total_trades: int = 0
    buys: int = 0
    sells: int = 0
    avg_alpha_4w: float | None = None
    process_pct: float = 0.0  # % of trades marked "process"
    thesis_pct: float = 0.0  # % of trades with thesis


# ---------------------------------------------------------------------------
# Journal CRUD
# ---------------------------------------------------------------------------

def create_trade_entry(
    db: Any,
    ticker: str,
    action: str,
    account: str = "",
    shares: float = 0.0,
    price: float = 0.0,
    trade_date: str | None = None,
    thesis: str = "",
    dcs_at_decision: float | None = None,
    vix_regime: str = "",
    is_panic_or_process: str = "process",
    has_thesis: bool = True,
    deployment_gates_passed: bool = True,
    notes: str = "",
) -> int:
    """Record a new trade in the journal.

    Parameters
    ----------
    db : Database
        Open database connection.
    ticker : str
        Ticker symbol.
    action : str
        Trade action: BUY, SELL, TRIM, ADD.
    account : str
        Account ID (e.g., 'ind', 'roth').
    shares : float
        Number of shares traded.
    price : float
        Execution price per share.
    trade_date : str | None
        Date of trade (YYYY-MM-DD). Uses today if None.
    thesis : str
        Investment thesis or reason for trade.
    dcs_at_decision : float | None
        DCS score at the time of decision.
    vix_regime : str
        VIX regime at the time (COMPLACENT/NORMAL/FEAR/PANIC).
    is_panic_or_process : str
        Behavioral check: was this "panic" or "process"?
    has_thesis : bool
        Was there a documented thesis?
    deployment_gates_passed : bool
        Did the trade pass all deployment gates?
    notes : str
        Additional notes.

    Returns
    -------
    int
        ID of the created journal entry.
    """
    if trade_date is None:
        trade_date = date.today().isoformat()

    cursor = db.execute(
        """INSERT INTO trade_journal (
            ticker, action, account, shares, price, trade_date,
            thesis, dcs_at_decision, vix_regime,
            is_panic_or_process, has_thesis, deployment_gates_passed, notes
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            ticker.upper(), action.upper(), account, shares, price,
            trade_date, thesis, dcs_at_decision, vix_regime,
            is_panic_or_process, int(has_thesis),
            int(deployment_gates_passed), notes,
        ),
    )
    db.conn.commit()
    logger.info("Recorded trade: %s %s %.0f shares at $%.2f", action, ticker, shares, price)
    return cursor.lastrowid or 0


def list_journal_entries(
    db: Any,
    ticker: str | None = None,
    limit: int = 50,
) -> list[dict[str, Any]]:
    """List journal entries, optionally filtered by ticker.

    Parameters
    ----------
    db : Database
        Open database connection.
    ticker : str | None
        Filter by ticker symbol.
    limit : int
        Maximum entries to return.

    Returns
    -------
    list[dict]
        Journal entries ordered by trade_date descending.
    """
    if ticker is not None:
        rows = db.fetchall(
            """SELECT * FROM trade_journal
            WHERE ticker = ? ORDER BY trade_date DESC LIMIT ?""",
            (ticker.upper(), limit),
        )
    else:
        rows = db.fetchall(
            """SELECT * FROM trade_journal
            ORDER BY trade_date DESC LIMIT ?""",
            (limit,),
        )
    return [dict(r) for r in rows]


def record_outcome(
    db: Any,
    entry_id: int,
    window: str,
    ticker_return: float,
    spy_return: float,
) -> int:
    """Record a trade outcome at a specific time window.

    Parameters
    ----------
    db : Database
        Open database connection.
    entry_id : int
        ID of the journal entry.
    window : str
        Time window: "1w", "4w", "12w", "26w".
    ticker_return : float
        Ticker return over the window.
    spy_return : float
        SPY return over the window.

    Returns
    -------
    int
        ID of the outcome record.
    """
    alpha = round(ticker_return - spy_return, 4)
    cursor = db.execute(
        """INSERT INTO trade_outcomes (entry_id, window, ticker_return, spy_return, alpha)
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(entry_id, window) DO UPDATE SET
            ticker_return=excluded.ticker_return,
            spy_return=excluded.spy_return,
            alpha=excluded.alpha""",
        (entry_id, window, round(ticker_return, 4), round(spy_return, 4), alpha),
    )
    db.conn.commit()
    return cursor.lastrowid or 0


def get_journal_summary(db: Any) -> JournalSummary:
    """Compute summary statistics across all journal entries.

    Returns
    -------
    JournalSummary
        Aggregate trade statistics.
    """
    entries = list_journal_entries(db, limit=1000)
    if not entries:
        return JournalSummary()

    buys = sum(1 for e in entries if e.get("action") in ("BUY", "ADD"))
    sells = sum(1 for e in entries if e.get("action") in ("SELL", "TRIM"))
    process_count = sum(1 for e in entries if e.get("is_panic_or_process") == "process")
    thesis_count = sum(1 for e in entries if e.get("has_thesis"))

    total = len(entries)
    summary = JournalSummary(
        total_trades=total,
        buys=buys,
        sells=sells,
        process_pct=round(process_count / total, 2) if total > 0 else 0,
        thesis_pct=round(thesis_count / total, 2) if total > 0 else 0,
    )

    # Average 4-week alpha from outcomes
    all_outcomes = db.fetchall(
        "SELECT alpha FROM trade_outcomes WHERE window = '4w'"
    )
    if all_outcomes:
        alphas = [float(r["alpha"]) for r in all_outcomes]
        summary.avg_alpha_4w = round(sum(alphas) / len(alphas), 4)

    return summary
