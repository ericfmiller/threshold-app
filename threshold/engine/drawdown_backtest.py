"""Drawdown defense backtest — 15-year classification of ticker behavior.

Analyzes each ticker's behavior during SPY drawdown episodes (months where
SPY was below its running maximum by >5%). Computes downside capture,
win rate, and max drawdown to classify each ticker into one of five
defense classes:

  - HEDGE (DC < 0): Moves opposite to SPY in drawdowns
  - DEFENSIVE (0 ≤ DC < 0.6): Loses less than SPY
  - MODERATE (0.6 ≤ DC < 1.0): Roughly tracks SPY losses
  - CYCLICAL (1.0 ≤ DC < 1.5): Amplifies losses modestly
  - AMPLIFIER (DC ≥ 1.5): Amplifies losses significantly

DC = Downside Capture = ticker return / SPY return during drawdown months.

Uses monthly returns over a 15-year lookback period.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class BacktestResult:
    """Results from a full drawdown defense backtest run."""

    backtest_date: str = ""
    classifications: dict[str, dict[str, Any]] = field(default_factory=dict)
    tickers_processed: int = 0
    tickers_skipped: int = 0
    spy_drawdown_months: int = 0
    errors: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# SPY drawdown identification
# ---------------------------------------------------------------------------

def identify_spy_drawdowns(
    spy_monthly: pd.Series,
    threshold: float = -0.05,
) -> pd.Series:
    """Identify months where SPY is in a drawdown.

    Parameters
    ----------
    spy_monthly : pd.Series
        Monthly returns for SPY.
    threshold : float
        Cumulative drawdown threshold (default -5%).

    Returns
    -------
    pd.Series
        Boolean mask — True for drawdown months.
    """
    # Compute cumulative wealth
    wealth = (1 + spy_monthly).cumprod()
    running_max = wealth.cummax()
    drawdown = (wealth - running_max) / running_max

    return drawdown < threshold


# ---------------------------------------------------------------------------
# Per-ticker analysis
# ---------------------------------------------------------------------------

def analyze_ticker_drawdown(
    ticker_monthly: pd.Series,
    spy_monthly: pd.Series,
    dd_mask: pd.Series,
) -> dict[str, Any] | None:
    """Analyze one ticker's behavior during SPY drawdowns.

    Parameters
    ----------
    ticker_monthly : pd.Series
        Monthly returns for the ticker.
    spy_monthly : pd.Series
        Monthly returns for SPY.
    dd_mask : pd.Series
        Boolean mask from identify_spy_drawdowns().

    Returns
    -------
    dict | None
        Metrics dict with downside_capture, win_rate, max_drawdown, etc.
        Returns None if insufficient data.
    """
    # Align all series
    common_idx = ticker_monthly.index.intersection(spy_monthly.index).intersection(dd_mask.index)
    if len(common_idx) < 12:
        return None  # Need at least 12 months of overlap

    ticker_aligned = ticker_monthly.loc[common_idx]
    spy_aligned = spy_monthly.loc[common_idx]
    mask_aligned = dd_mask.loc[common_idx]

    dd_months = mask_aligned.sum()
    if dd_months < 3:
        return None  # Need at least 3 drawdown months

    # Drawdown-period returns
    ticker_dd = ticker_aligned[mask_aligned]
    spy_dd = spy_aligned[mask_aligned]

    # Downside capture
    spy_dd_mean = spy_dd.mean()
    if abs(spy_dd_mean) < 1e-8:
        return None  # SPY didn't actually drawdown

    downside_capture = float(ticker_dd.mean() / spy_dd_mean)

    # Win rate: % of drawdown months where ticker beat SPY
    wins = (ticker_dd > spy_dd).sum()
    win_rate = float(wins / len(ticker_dd))

    # Upside capture (non-drawdown months)
    ticker_up = ticker_aligned[~mask_aligned]
    spy_up = spy_aligned[~mask_aligned]
    upside_capture = None
    if len(spy_up) > 0 and abs(spy_up.mean()) > 1e-8:
        upside_capture = float(ticker_up.mean() / spy_up.mean())

    # Max drawdown of the ticker
    ticker_wealth = (1 + ticker_aligned).cumprod()
    ticker_running_max = ticker_wealth.cummax()
    ticker_dd_series = (ticker_wealth - ticker_running_max) / ticker_running_max
    max_drawdown = float(ticker_dd_series.min())

    # Capture ratio
    capture_ratio = None
    if upside_capture is not None and abs(downside_capture) > 1e-8:
        capture_ratio = round(upside_capture / downside_capture, 4)

    return {
        "downside_capture": round(downside_capture, 4),
        "upside_capture": round(upside_capture, 4) if upside_capture is not None else None,
        "capture_ratio": capture_ratio,
        "win_rate_in_dd": round(win_rate, 4),
        "max_drawdown": round(max_drawdown, 4),
        "episodes_measured": int(dd_months),
    }


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------

def classify_defense(downside_capture: float) -> str:
    """Classify a ticker's defense class from its downside capture.

    Parameters
    ----------
    downside_capture : float
        Average ticker return / SPY return during drawdown months.

    Returns
    -------
    str
        One of: HEDGE, DEFENSIVE, MODERATE, CYCLICAL, AMPLIFIER.
    """
    if downside_capture < 0:
        return "HEDGE"
    if downside_capture < 0.6:
        return "DEFENSIVE"
    if downside_capture < 1.0:
        return "MODERATE"
    if downside_capture < 1.5:
        return "CYCLICAL"
    return "AMPLIFIER"


# ---------------------------------------------------------------------------
# Full backtest orchestrator
# ---------------------------------------------------------------------------

def run_drawdown_backtest(
    price_data: dict[str, pd.DataFrame],
    spy_prices: pd.DataFrame,
    dd_overrides: dict[str, str] | None = None,
    lookback_years: int = 15,
) -> BacktestResult:
    """Run drawdown defense backtest for all tickers.

    Parameters
    ----------
    price_data : dict[str, pd.DataFrame]
        {symbol: DataFrame with 'Close' column}.
    spy_prices : pd.DataFrame
        SPY price DataFrame with 'Close' column.
    dd_overrides : dict | None
        {symbol: classification} manual overrides from tickers table.
    lookback_years : int
        Years of history to analyze (default 15).

    Returns
    -------
    BacktestResult
        Full backtest results with per-ticker classifications.
    """
    from datetime import date

    result = BacktestResult(
        backtest_date=date.today().isoformat(),
    )

    if dd_overrides is None:
        dd_overrides = {}

    # Compute SPY monthly returns
    spy_close = spy_prices["Close"].dropna()
    if len(spy_close) < 60:  # Need at least 5 years
        result.errors.append("Insufficient SPY data")
        return result

    spy_monthly = spy_close.resample("ME").last().pct_change().dropna()

    # Cutoff for lookback
    cutoff = spy_monthly.index.max() - pd.DateOffset(years=lookback_years)
    spy_monthly = spy_monthly[spy_monthly.index >= cutoff]

    # Identify drawdown months
    dd_mask = identify_spy_drawdowns(spy_monthly)
    result.spy_drawdown_months = int(dd_mask.sum())

    for symbol, df in price_data.items():
        try:
            # Check for manual override
            if symbol in dd_overrides and dd_overrides[symbol]:
                result.classifications[symbol] = {
                    "classification": dd_overrides[symbol],
                    "downside_capture": None,
                    "source": "override",
                }
                result.tickers_processed += 1
                continue

            close = df["Close"].dropna()
            if len(close) < 60:
                result.tickers_skipped += 1
                continue

            ticker_monthly = close.resample("ME").last().pct_change().dropna()
            ticker_monthly = ticker_monthly[ticker_monthly.index >= cutoff]

            metrics = analyze_ticker_drawdown(ticker_monthly, spy_monthly, dd_mask)
            if metrics is None:
                result.tickers_skipped += 1
                continue

            classification = classify_defense(metrics["downside_capture"])
            metrics["classification"] = classification
            metrics["source"] = "backtest"
            result.classifications[symbol] = metrics
            result.tickers_processed += 1

        except Exception as e:
            result.errors.append(f"{symbol}: {e}")
            result.tickers_skipped += 1

    return result
