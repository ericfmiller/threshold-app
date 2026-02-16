"""Portfolio performance tracking — returns vs SPY benchmark.

Captures weekly performance snapshots and computes time-weighted returns
across multiple timeframes (1w, 1m, 3m, 6m, YTD, 1y, inception).
Compares portfolio performance against SPY benchmark.

Uses the ``performance_snapshots`` table from migration 004.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class PerformanceSnapshot:
    """A point-in-time performance capture."""

    snapshot_date: str = ""
    total_portfolio: float = 0.0
    spy_close: float = 0.0
    btc_price: float = 0.0


@dataclass
class PerformanceReport:
    """Performance comparison report."""

    as_of_date: str = ""
    portfolio_returns: dict[str, float] = field(default_factory=dict)
    spy_returns: dict[str, float] = field(default_factory=dict)
    alpha: dict[str, float] = field(default_factory=dict)
    snapshots_count: int = 0


# ---------------------------------------------------------------------------
# Snapshot capture
# ---------------------------------------------------------------------------

def capture_performance_snapshot(
    db: Any,
    total_portfolio: float,
    spy_close: float | None = None,
    btc_price: float | None = None,
    snapshot_date: str | None = None,
) -> PerformanceSnapshot:
    """Capture a performance snapshot to the database.

    Parameters
    ----------
    db : Database
        Open database connection.
    total_portfolio : float
        Total portfolio value.
    spy_close : float | None
        SPY closing price (fetched if None).
    btc_price : float | None
        BTC-USD price (fetched if None).
    snapshot_date : str | None
        Date (YYYY-MM-DD). Uses today if None.

    Returns
    -------
    PerformanceSnapshot
        The captured snapshot.
    """
    if snapshot_date is None:
        snapshot_date = date.today().isoformat()

    if spy_close is None:
        spy_close = _fetch_spy_close()

    snap = PerformanceSnapshot(
        snapshot_date=snapshot_date,
        total_portfolio=round(total_portfolio, 2),
        spy_close=round(spy_close or 0, 2),
        btc_price=round(btc_price or 0, 2),
    )

    db.execute(
        """INSERT INTO performance_snapshots
        (snapshot_date, total_portfolio, spy_close, btc_price)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(snapshot_date) DO UPDATE SET
            total_portfolio=excluded.total_portfolio,
            spy_close=excluded.spy_close,
            btc_price=excluded.btc_price""",
        (snap.snapshot_date, snap.total_portfolio, snap.spy_close, snap.btc_price),
    )
    db.conn.commit()
    return snap


# ---------------------------------------------------------------------------
# Return computation
# ---------------------------------------------------------------------------

def compute_returns(
    snapshots: list[dict[str, Any]],
) -> dict[str, float]:
    """Compute portfolio returns over standard timeframes.

    Parameters
    ----------
    snapshots : list[dict]
        Performance snapshots sorted by date ascending.
        Each dict has: snapshot_date, total_portfolio, spy_close.

    Returns
    -------
    dict[str, float]
        Returns by timeframe: {"1w": 0.012, "1m": 0.034, ...}
    """
    if len(snapshots) < 2:
        return {}

    df = pd.DataFrame(snapshots)
    df["snapshot_date"] = pd.to_datetime(df["snapshot_date"])
    df = df.sort_values("snapshot_date")
    df = df.set_index("snapshot_date")

    latest = df.iloc[-1]
    latest_date = df.index[-1]
    latest_value = float(latest["total_portfolio"])

    if latest_value <= 0:
        return {}

    timeframes = {
        "1w": timedelta(days=7),
        "1m": timedelta(days=30),
        "3m": timedelta(days=90),
        "6m": timedelta(days=180),
        "1y": timedelta(days=365),
    }

    returns: dict[str, float] = {}

    for label, delta in timeframes.items():
        target_date = latest_date - delta
        # Find closest snapshot on or before target date
        prior = df[df.index <= target_date]
        if not prior.empty:
            prior_value = float(prior.iloc[-1]["total_portfolio"])
            if prior_value > 0:
                returns[label] = round((latest_value / prior_value) - 1, 4)

    # YTD
    year_start = pd.Timestamp(latest_date.year, 1, 1)
    ytd_prior = df[df.index <= year_start]
    if not ytd_prior.empty:
        ytd_value = float(ytd_prior.iloc[-1]["total_portfolio"])
        if ytd_value > 0:
            returns["YTD"] = round((latest_value / ytd_value) - 1, 4)

    # Inception (first snapshot)
    first_value = float(df.iloc[0]["total_portfolio"])
    if first_value > 0:
        returns["inception"] = round((latest_value / first_value) - 1, 4)

    return returns


def generate_performance_report(
    db: Any,
) -> PerformanceReport:
    """Generate a full performance report with SPY comparison.

    Parameters
    ----------
    db : Database
        Open database connection.

    Returns
    -------
    PerformanceReport
        Portfolio returns, SPY returns, and alpha for each timeframe.
    """
    rows = db.fetchall(
        """SELECT * FROM performance_snapshots
        ORDER BY snapshot_date ASC"""
    )
    snapshots = [dict(r) for r in rows]

    if len(snapshots) < 2:
        return PerformanceReport(snapshots_count=len(snapshots))

    # Portfolio returns
    portfolio_returns = compute_returns(snapshots)

    # SPY returns (build synthetic snapshots from spy_close)
    spy_snapshots = [
        {"snapshot_date": s["snapshot_date"], "total_portfolio": s.get("spy_close", 0)}
        for s in snapshots
        if s.get("spy_close")
    ]
    spy_returns = compute_returns(spy_snapshots)

    # Alpha
    alpha = {}
    for tf in portfolio_returns:
        if tf in spy_returns:
            alpha[tf] = round(portfolio_returns[tf] - spy_returns[tf], 4)

    return PerformanceReport(
        as_of_date=snapshots[-1]["snapshot_date"],
        portfolio_returns=portfolio_returns,
        spy_returns=spy_returns,
        alpha=alpha,
        snapshots_count=len(snapshots),
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _fetch_spy_close() -> float:
    """Fetch current SPY closing price via yfinance."""
    try:
        import yfinance as yf

        spy = yf.Ticker("SPY")
        hist = spy.history(period="1d")
        if hist is not None and not hist.empty:
            return float(hist["Close"].iloc[-1])
    except Exception as e:
        logger.debug("SPY price fetch failed: %s", e)
    return 0.0
