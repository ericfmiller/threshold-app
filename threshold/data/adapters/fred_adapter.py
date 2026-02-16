"""FRED (Federal Reserve Economic Data) adapter.

Fetches macro-economic data from the FRED API for regime analysis:
yield curve, credit spreads, Fed Funds rate, CPI, Fed balance sheet.

Requires a free FRED API key (https://fred.stlouisfed.org/docs/api/api_key.html).
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Any

logger = logging.getLogger(__name__)


def fetch_fred_series(
    api_key: str,
    series_id: str,
    lookback_days: int = 365,
) -> dict[str, Any] | None:
    """Fetch a single FRED series.

    Parameters
    ----------
    api_key : str
        FRED API key.
    series_id : str
        FRED series ID (e.g., 'T10Y2Y', 'DFF').
    lookback_days : int
        Number of days of history to request.

    Returns
    -------
    dict | None
        Dict with {series_id, latest_date, latest_value, history}
        or None if fetch fails.
    """
    try:
        from fredapi import Fred
    except ImportError:
        logger.warning("fredapi not installed — pip install fredapi")
        return None

    try:
        fred = Fred(api_key=api_key)
        start = datetime.now() - timedelta(days=lookback_days)
        data = fred.get_series(series_id, observation_start=start)

        if data is None or data.empty:
            return None

        # Drop NaN values
        data = data.dropna()
        if data.empty:
            return None

        return {
            "series_id": series_id,
            "latest_date": str(data.index[-1].date()),
            "latest_value": round(float(data.iloc[-1]), 4),
            "history": data,
        }
    except Exception as e:
        logger.debug("FRED fetch failed for %s: %s", series_id, e)
        return None


def fetch_fred_macro(
    api_key: str,
    series_config: dict[str, str] | None = None,
) -> dict[str, dict[str, Any]]:
    """Fetch all configured FRED macro series.

    Parameters
    ----------
    api_key : str
        FRED API key.
    series_config : dict | None
        Mapping of {label: series_id}. If None, uses defaults.

    Returns
    -------
    dict[str, dict]
        {label: {series_id, latest_date, latest_value, history}}
        Only includes series that were successfully fetched.
    """
    from threshold.config.defaults import FRED_SERIES

    if series_config is None:
        series_config = FRED_SERIES

    results: dict[str, dict[str, Any]] = {}
    success = 0
    failed = 0

    for label, series_id in series_config.items():
        data = fetch_fred_series(api_key, series_id)
        if data:
            results[label] = data
            success += 1
        else:
            failed += 1

    logger.info(
        "FRED macro fetch: %d/%d succeeded, %d failed",
        success,
        len(series_config),
        failed,
    )
    return results


def compute_macro_indicators(
    macro_data: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Compute derived macro indicators from raw FRED data.

    Parameters
    ----------
    macro_data : dict
        Output from fetch_fred_macro().

    Returns
    -------
    dict
        Computed indicators:
        - yield_curve_10y2y: float (positive = normal, negative = inverted)
        - yield_curve_10y3m: float
        - credit_spread: float (higher = more stress)
        - fed_funds_rate: float
        - cpi_yoy: float (approximate from CPI index)
        - fed_balance_sheet: float (trillions)
        - yield_curve_inverted: bool
        - credit_stress: bool (spread > 5%)
    """
    result: dict[str, Any] = {}

    # Yield curve
    yc_10y2y = macro_data.get("T10Y2Y", {})
    if yc_10y2y:
        val = yc_10y2y.get("latest_value", 0)
        result["yield_curve_10y2y"] = val
        result["yield_curve_inverted"] = val < 0

    yc_10y3m = macro_data.get("T10Y3M", {})
    if yc_10y3m:
        result["yield_curve_10y3m"] = yc_10y3m.get("latest_value", 0)

    # Credit spread
    credit = macro_data.get("BAMLH0A0HYM2", {})
    if credit:
        val = credit.get("latest_value", 0)
        result["credit_spread"] = val
        result["credit_stress"] = val > 5.0

    # Fed Funds Rate
    dff = macro_data.get("DFF", {})
    if dff:
        result["fed_funds_rate"] = dff.get("latest_value", 0)

    # CPI
    cpi = macro_data.get("CPIAUCSL", {})
    if cpi and cpi.get("history") is not None:
        history = cpi["history"]
        if len(history) >= 13:
            # YoY change: (current / 12-months-ago - 1) * 100
            current = float(history.iloc[-1])
            year_ago = float(history.iloc[-13])
            if year_ago > 0:
                result["cpi_yoy"] = round((current / year_ago - 1) * 100, 2)

    # Fed Balance Sheet (WALCL is in millions)
    walcl = macro_data.get("WALCL", {})
    if walcl:
        val = walcl.get("latest_value", 0)
        result["fed_balance_sheet_trillions"] = round(val / 1_000_000, 2)

    return result
