"""Conditional Drawdown at Risk (CDaR).

CDaR is the expected drawdown in the tail of the drawdown distribution,
analogous to CVaR but for drawdowns instead of returns. It captures the
expected severity of drawdowns beyond a given percentile.

Properties:
  α → 1: CDaR → Maximum Drawdown
  α → 0: CDaR → Average Drawdown
  Recommended α = 0.95

Reference:
  Chekhlov, A., Uryasev, S. & Zabarankin, M. (2005). "Drawdown Measure
  in Portfolio Optimization." International Journal of Theoretical and
  Applied Finance, 8(1), 13-58.
"""

from __future__ import annotations

from typing import TypedDict

import numpy as np
import pandas as pd


class CDaRResult(TypedDict):
    """Result from CDaR calculation."""
    cdar: float              # CDaR (expected tail drawdown), positive number
    dar: float               # Drawdown at Risk (threshold), positive number
    alpha: float             # Confidence level used
    max_drawdown: float      # Maximum drawdown, positive number
    avg_drawdown: float      # Average drawdown, positive number
    current_drawdown: float  # Current drawdown from peak, positive number
    n_observations: int      # Number of return observations
    n_drawdown_periods: int  # Number of distinct drawdown periods


class CDaRCalculator:
    """Conditional Drawdown at Risk calculator.

    Usage::

        calc = CDaRCalculator(alpha=0.95)
        result = calc.compute(returns)

    Parameters:
        alpha: Confidence level (default 0.95). Higher α = more conservative.
    """

    def __init__(self, alpha: float = 0.95) -> None:
        if not 0.0 < alpha < 1.0:
            raise ValueError(f"alpha must be in (0, 1), got {alpha}")
        self.alpha = alpha

    def compute_drawdowns(self, returns: pd.Series | np.ndarray) -> np.ndarray:
        """Compute drawdown series from returns.

        Parameters:
            returns: Daily or periodic return series.

        Returns:
            Array of drawdowns (positive numbers = magnitude of drawdown).
        """
        arr = np.asarray(returns, dtype=float)
        arr = arr[~np.isnan(arr)]

        if len(arr) < 2:
            return np.array([0.0])

        # Cumulative wealth
        cum_returns = np.cumprod(1 + arr)
        running_max = np.maximum.accumulate(cum_returns)

        # Drawdown = (peak - current) / peak
        drawdowns = (running_max - cum_returns) / running_max

        return drawdowns

    def _count_drawdown_periods(self, drawdowns: np.ndarray) -> int:
        """Count distinct drawdown periods (contiguous runs of dd > 0)."""
        if len(drawdowns) == 0:
            return 0
        in_dd = drawdowns > 1e-8
        # Count transitions from not-in-drawdown to in-drawdown
        starts = np.diff(in_dd.astype(int))
        return int(np.sum(starts == 1)) + (1 if in_dd[0] else 0)

    def historical_cdar(self, drawdowns: np.ndarray) -> float:
        """Compute historical CDaR from drawdown series.

        CDaR = E[DD | DD >= DaR_α], expressed as positive magnitude.
        """
        if len(drawdowns) < 2:
            return 0.0

        dar_threshold = np.percentile(drawdowns, self.alpha * 100)
        tail = drawdowns[drawdowns >= dar_threshold]

        if len(tail) == 0:
            return float(dar_threshold)

        return float(np.mean(tail))

    def compute(self, returns: pd.Series | np.ndarray) -> CDaRResult:
        """Compute CDaR from a return series.

        Parameters:
            returns: Daily or periodic return series.

        Returns:
            CDaRResult with drawdown risk metrics.
        """
        arr = np.asarray(returns, dtype=float)
        arr = arr[~np.isnan(arr)]
        n = len(arr)

        if n < 10:
            return CDaRResult(
                cdar=0.0,
                dar=0.0,
                alpha=self.alpha,
                max_drawdown=0.0,
                avg_drawdown=0.0,
                current_drawdown=0.0,
                n_observations=n,
                n_drawdown_periods=0,
            )

        drawdowns = self.compute_drawdowns(arr)
        cdar = self.historical_cdar(drawdowns)
        dar = float(np.percentile(drawdowns, self.alpha * 100))
        max_dd = float(np.max(drawdowns))
        avg_dd = float(np.mean(drawdowns))
        current_dd = float(drawdowns[-1])
        n_periods = self._count_drawdown_periods(drawdowns)

        return CDaRResult(
            cdar=round(cdar, 6),
            dar=round(dar, 6),
            alpha=self.alpha,
            max_drawdown=round(max_dd, 6),
            avg_drawdown=round(avg_dd, 6),
            current_drawdown=round(current_dd, 6),
            n_observations=n,
            n_drawdown_periods=n_periods,
        )
