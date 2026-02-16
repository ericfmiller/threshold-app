"""ScoringContext -- shared per-run scoring context.

Bundles all data that is constant across ticker iterations into a single
object.  ``score_ticker()`` accepts ``ctx: ScoringContext`` instead of
6+ loose keyword arguments.

Built once per scoring run, then passed to every ``score_ticker()`` call.
Ticker-specific data (``sa_data``, ``price_df``) remains positional.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd


@dataclass
class ScoringContext:
    """Shared per-run scoring context.

    Usage::

        ctx = ScoringContext(
            market_regime_score=mr_score,
            vix_regime=classify_vix(vix_current),
            spy_close=spy_close_series,
            grade_history=grade_history,
            prev_scores=prev_scores,
            yf_fundamentals=yf_fund_data,
            drawdown_classifications=dd_classifications,
        )

        for ticker in tickers:
            result = score_ticker(ticker, sa_data, price_df, ctx)
    """

    # --- Market regime (constant per run) ---
    market_regime_score: float
    """Pre-computed MR sub-score from ``calc_market_regime()`` (15% of DCS)."""

    vix_regime: str | None = None
    """VIX regime string: 'COMPLACENT', 'NORMAL', 'FEAR', or 'PANIC'."""

    # --- SPY data (constant per run) ---
    spy_close: pd.Series | None = None
    """SPY close prices (2-year window) for relative strength calculations."""

    # --- Historical context (constant per run) ---
    grade_history: list[dict[str, Any]] | None = None
    """Prior weekly score JSONs (most recent first) for revision momentum."""

    prev_scores: dict[str, Any] | None = None
    """Previous week's per-ticker DCS results."""

    # --- Fundamentals (per-ticker access via get_yf_fundamentals) ---
    yf_fundamentals: dict[str, dict[str, Any]] | None = None
    """Per-ticker yfinance fundamentals: {ticker: {fcf_yield, ...}}."""

    # --- Risk framework (constant per run) ---
    drawdown_classifications: dict[str, dict[str, Any]] | None = None
    """Drawdown defense classifications."""

    # ------------------------------------------------------------------
    # Convenience accessors (per-ticker lookups into shared dicts)
    # ------------------------------------------------------------------

    def get_yf_fundamentals(self, ticker: str) -> dict[str, Any] | None:
        """Return yfinance fundamentals for a specific ticker, or None."""
        if self.yf_fundamentals is None:
            return None
        return self.yf_fundamentals.get(ticker)

    def get_prev_sa_data(self, ticker: str) -> dict[str, Any] | None:
        """Return previous week's SA data for a specific ticker, or None."""
        if self.prev_scores is None:
            return None
        return self.prev_scores.get(ticker)
