"""Portfolio snapshot generator.

Creates a point-in-time snapshot of the entire portfolio from
positions in the database, TSP values from config, and BTC
price from yfinance. Stores snapshot in the database for
performance tracking over time.
"""

from __future__ import annotations

import json
import logging
from datetime import date
from typing import Any

logger = logging.getLogger(__name__)


def _fetch_btc_price() -> float:
    """Fetch current BTC-USD price via yfinance.

    Returns 0.0 if fetch fails.
    """
    try:
        import yfinance as yf

        btc = yf.Ticker("BTC-USD")
        hist = btc.history(period="1d")
        if hist is not None and not hist.empty:
            return float(hist["Close"].iloc[-1])
    except Exception as e:
        logger.debug("BTC price fetch failed: %s", e)
    return 0.0


def generate_snapshot(
    db: Any,
    config: Any,
    snapshot_date: str | None = None,
) -> dict[str, Any]:
    """Generate a comprehensive portfolio snapshot.

    Reads positions from the database, adds TSP and BTC values,
    and computes per-account and total portfolio values.

    Parameters
    ----------
    db : Database
        Open database connection.
    config : ThresholdConfig
        Configuration with TSP, BTC, and separate holdings settings.
    snapshot_date : str | None
        Date for this snapshot (YYYY-MM-DD). If None, uses today.

    Returns
    -------
    dict
        Snapshot data including:
        - snapshot_date, total_portfolio, fidelity_total
        - tsp_value, btc_value, btc_price, btc_quantity
        - accounts: {account_id: {value, positions_count, top_holdings}}
        - positions: {symbol: {total_value, accounts, weight}}
    """
    from threshold.data.position_import import load_positions_from_db
    from threshold.portfolio.accounts import aggregate_positions

    if snapshot_date is None:
        snapshot_date = date.today().isoformat()

    # Load positions
    raw_positions = load_positions_from_db(db, snapshot_date)
    if not raw_positions:
        # Try latest available
        raw_positions = load_positions_from_db(db)

    # Compute account totals from positions
    account_totals: dict[str, float] = {}
    for pos in raw_positions:
        acct = pos.get("account_id", "")
        value = float(pos.get("market_value", 0))
        account_totals[acct] = account_totals.get(acct, 0) + value

    # Aggregate positions
    snapshot = aggregate_positions(raw_positions, account_totals)
    snapshot.snapshot_date = snapshot_date

    # Fidelity total
    fidelity_total = sum(account_totals.values())

    # TSP value from config
    tsp_value = 0.0
    if hasattr(config, "tsp") and config.tsp:
        tsp_value = float(getattr(config.tsp, "value", 0) or 0)

    # BTC value
    btc_quantity = 0.0
    btc_price = 0.0
    btc_value = 0.0
    if hasattr(config, "separate_holdings"):
        for holding in config.separate_holdings:
            if getattr(holding, "symbol", "") == "BTC-USD":
                btc_quantity = float(getattr(holding, "quantity", 0) or 0)
                break

    if btc_quantity > 0:
        btc_price = _fetch_btc_price()
        btc_value = btc_quantity * btc_price

    total_portfolio = fidelity_total + tsp_value + btc_value

    # Build snapshot data
    result: dict[str, Any] = {
        "snapshot_date": snapshot_date,
        "total_portfolio": round(total_portfolio, 2),
        "fidelity_total": round(fidelity_total, 2),
        "tsp_value": round(tsp_value, 2),
        "btc_value": round(btc_value, 2),
        "btc_price": round(btc_price, 2),
        "btc_quantity": btc_quantity,
        "accounts": {},
        "positions": {},
    }

    # Per-account breakdown
    for acct_id, total in account_totals.items():
        acct_tickers = [
            pos["symbol"] for pos in raw_positions
            if pos.get("account_id") == acct_id
        ]
        result["accounts"][acct_id] = {
            "value": round(total, 2),
            "positions_count": len(acct_tickers),
        }

    # Per-ticker aggregation
    for symbol, position in snapshot.positions.items():
        result["positions"][symbol] = {
            "total_value": round(position.total_value, 2),
            "total_shares": round(position.total_shares, 4),
            "n_accounts": position.n_accounts,
            "accounts": {k: round(v, 2) for k, v in position.account_values.items()},
            "portfolio_weight": round(
                position.total_value / total_portfolio if total_portfolio > 0 else 0, 4
            ),
        }

    return result


def save_snapshot(
    db: Any,
    snapshot_data: dict[str, Any],
) -> None:
    """Save a portfolio snapshot to the database.

    Parameters
    ----------
    db : Database
        Open database connection.
    snapshot_data : dict
        Output from generate_snapshot().
    """
    data_json = json.dumps(snapshot_data)
    db.execute(
        """INSERT INTO portfolio_snapshots (
            snapshot_date, data_json, total_portfolio,
            fidelity_total, tsp_value, btc_value
        ) VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT(snapshot_date) DO UPDATE SET
            data_json=excluded.data_json,
            total_portfolio=excluded.total_portfolio,
            fidelity_total=excluded.fidelity_total,
            tsp_value=excluded.tsp_value,
            btc_value=excluded.btc_value""",
        (
            snapshot_data["snapshot_date"],
            data_json,
            snapshot_data["total_portfolio"],
            snapshot_data["fidelity_total"],
            snapshot_data["tsp_value"],
            snapshot_data["btc_value"],
        ),
    )
    db.conn.commit()


def load_latest_snapshot(db: Any) -> dict[str, Any] | None:
    """Load the most recent portfolio snapshot from the database.

    Returns None if no snapshots exist.
    """
    row = db.fetchone(
        "SELECT * FROM portfolio_snapshots ORDER BY snapshot_date DESC LIMIT 1"
    )
    if not row:
        return None

    try:
        return json.loads(row["data_json"])
    except (json.JSONDecodeError, KeyError):
        return None
