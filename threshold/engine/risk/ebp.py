"""Excess Bond Premium (EBP) monitor — Gilchrist & Zakrajšek (2012).

The EBP captures the component of credit spreads unrelated to default risk,
serving as a forward-looking indicator of credit market stress. Rising EBP
signals tightening financial conditions and predicts economic downturns.

Reference:
  Gilchrist, S. & Zakrajšek, E. (2012). "Credit Spreads and Business Cycle
  Fluctuations." American Economic Review, 102(4), 1692-1720.

Data source: Federal Reserve Board CSV (updated monthly).
"""

from __future__ import annotations

from typing import TypedDict

import numpy as np
import pandas as pd


class EBPSignal(TypedDict):
    """Result from EBP analysis."""
    ebp_value: float | None
    ebp_regime: str  # HIGH_RISK, ELEVATED, NORMAL, ACCOMMODATIVE, UNAVAILABLE
    ebp_percentile: float | None
    ebp_3m_change: float | None
    ebp_trend: str  # rising, falling, stable, unknown


# Default thresholds (basis points, but stored as decimal %)
DEFAULT_THRESHOLDS = {
    "high_risk": 1.00,       # >100bp
    "elevated": 0.50,        # >50bp
    "normal": 0.00,          # >0bp
    # Below 0 = ACCOMMODATIVE
}


class EBPMonitor:
    """Monitors the Gilchrist-Zakrajšek Excess Bond Premium.

    Usage::

        monitor = EBPMonitor()
        monitor.load_data(ebp_series)  # pd.Series with datetime index
        signal = monitor.get_current_signal()
    """

    def __init__(
        self,
        thresholds: dict[str, float] | None = None,
        lookback_months: int = 3,
    ) -> None:
        self.thresholds = thresholds or dict(DEFAULT_THRESHOLDS)
        self.lookback_months = lookback_months
        self._data: pd.Series | None = None

    def load_data(self, ebp_series: pd.Series) -> None:
        """Load EBP time series.

        Parameters:
            ebp_series: Monthly EBP values with datetime index.
                        Values in decimal (0.01 = 1 basis point).
        """
        if ebp_series is None or len(ebp_series) < 2:
            self._data = None
            return
        self._data = ebp_series.sort_index().dropna()

    def _classify_regime(self, value: float) -> str:
        """Classify EBP value into risk regime."""
        if value >= self.thresholds["high_risk"]:
            return "HIGH_RISK"
        if value >= self.thresholds["elevated"]:
            return "ELEVATED"
        if value >= self.thresholds["normal"]:
            return "NORMAL"
        return "ACCOMMODATIVE"

    def _compute_percentile(self, value: float) -> float:
        """Compute percentile rank of current EBP vs full history."""
        if self._data is None or len(self._data) < 10:
            return 0.5
        return float(np.searchsorted(
            np.sort(self._data.values), value
        ) / len(self._data))

    def _compute_trend(self) -> tuple[float | None, str]:
        """Compute 3-month change and directional trend."""
        if self._data is None or len(self._data) < self.lookback_months + 1:
            return None, "unknown"

        current = float(self._data.iloc[-1])
        prior = float(self._data.iloc[-(self.lookback_months + 1)])
        change = current - prior

        if abs(change) < 0.05:  # Less than 5bp move
            trend = "stable"
        elif change > 0:
            trend = "rising"
        else:
            trend = "falling"

        return round(change, 4), trend

    def get_current_signal(self) -> EBPSignal:
        """Compute the current EBP risk signal.

        Returns:
            EBPSignal with regime classification and trend.
        """
        if self._data is None or len(self._data) < 2:
            return EBPSignal(
                ebp_value=None,
                ebp_regime="UNAVAILABLE",
                ebp_percentile=None,
                ebp_3m_change=None,
                ebp_trend="unknown",
            )

        current = float(self._data.iloc[-1])
        regime = self._classify_regime(current)
        percentile = self._compute_percentile(current)
        change_3m, trend = self._compute_trend()

        return EBPSignal(
            ebp_value=round(current, 4),
            ebp_regime=regime,
            ebp_percentile=round(percentile, 4),
            ebp_3m_change=change_3m,
            ebp_trend=trend,
        )

    def get_regime_score(self) -> float:
        """Return normalized risk score in [0, 1] for aggregation.

        0.0 = ACCOMMODATIVE (low risk)
        0.33 = NORMAL
        0.67 = ELEVATED
        1.0 = HIGH_RISK
        """
        signal = self.get_current_signal()
        regime_map = {
            "ACCOMMODATIVE": 0.0,
            "NORMAL": 0.33,
            "ELEVATED": 0.67,
            "HIGH_RISK": 1.0,
            "UNAVAILABLE": 0.5,  # Default to neutral when no data
        }
        return regime_map.get(signal["ebp_regime"], 0.5)
