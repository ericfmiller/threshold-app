"""Momentum crash protection — Daniel & Moskowitz (2016).

Dynamically adjusts momentum exposure based on bear market conditions.
When cumulative 24-month market return is negative (bear indicator), the
forecasting model predicts heightened crash risk and reduces momentum weight.

Reference:
  Daniel, K. & Moskowitz, T.J. (2016). "Momentum Crashes." Journal of
  Financial Economics, 122(2), 221-247.

Integration: Optional MQ adjustment multiplier in scorer.py.
"""

from __future__ import annotations

from typing import TypedDict

import numpy as np
import pandas as pd


class MomentumCrashSignal(TypedDict):
    """Result from momentum crash analysis."""
    is_bear_market: bool
    bear_indicator: float  # 1.0 if bear, 0.0 if bull
    cumulative_24m_return: float | None
    momentum_weight: float  # Dynamic weight multiplier [0, 1]
    wml_variance: float | None  # Winners-minus-Losers variance
    crash_probability: float  # Estimated crash probability [0, 1]
    regime: str  # NORMAL, CAUTION, HIGH_RISK, UNAVAILABLE


class MomentumCrashProtection:
    """Daniel-Moskowitz conditional momentum crash protection.

    Usage::

        mcp = MomentumCrashProtection(lookback_months=24)
        signal = mcp.compute_dynamic_weight(market_returns, wml_returns)

    Parameters:
        lookback_months: Months for bear market indicator (default 24).
        crash_threshold: Variance threshold for crash regime (default 0.02).
        min_weight: Minimum momentum weight floor (default 0.25).
    """

    def __init__(
        self,
        lookback_months: int = 24,
        crash_threshold: float = 0.02,
        min_weight: float = 0.25,
    ) -> None:
        self.lookback_months = lookback_months
        self.crash_threshold = crash_threshold
        self.min_weight = min_weight

    def _compute_bear_indicator(self, market_returns: pd.Series) -> tuple[bool, float | None]:
        """Compute bear market indicator I_B.

        I_B = 1 if cumulative market return over lookback period < 0.
        """
        if market_returns is None or len(market_returns) < self.lookback_months:
            return False, None

        # Use last lookback_months of monthly returns
        recent = market_returns.iloc[-self.lookback_months:]
        cum_return = float((1 + recent).prod() - 1)

        return cum_return < 0, cum_return

    def _estimate_wml_variance(self, wml_returns: pd.Series | None) -> float | None:
        """Estimate WML (Winners-minus-Losers) return variance.

        Uses rolling 126-day (6-month) variance estimate.
        """
        if wml_returns is None or len(wml_returns) < 60:
            return None

        lookback = min(126, len(wml_returns))
        return float(wml_returns.iloc[-lookback:].var())

    def _forecast_crash_probability(
        self,
        is_bear: bool,
        wml_variance: float | None,
    ) -> float:
        """Estimate momentum crash probability.

        Based on Daniel-Moskowitz interaction model:
        Higher variance in bear markets = higher crash probability.
        """
        if not is_bear:
            return 0.05  # Low baseline in bull markets

        if wml_variance is None:
            return 0.30  # Default moderate risk in bear with no variance data

        # Scaled crash probability using variance
        # Higher variance → higher crash risk during bear markets
        base_prob = 0.20
        variance_contrib = min(wml_variance / self.crash_threshold, 1.0) * 0.60
        return min(base_prob + variance_contrib, 0.95)

    def _compute_dynamic_weight(
        self,
        is_bear: bool,
        crash_probability: float,
    ) -> float:
        """Compute dynamic momentum weight multiplier.

        In bull markets: full weight (1.0).
        In bear markets: reduced proportionally to crash probability.
        """
        if not is_bear:
            return 1.0

        # Linear scaling: high crash prob → low weight
        weight = 1.0 - crash_probability * 0.75
        return max(weight, self.min_weight)

    def compute_dynamic_weight(
        self,
        market_returns: pd.Series,
        wml_returns: pd.Series | None = None,
    ) -> MomentumCrashSignal:
        """Compute momentum crash protection signal.

        Parameters:
            market_returns: Monthly market (e.g. SPY) returns.
            wml_returns: Optional Winners-minus-Losers factor returns.
                        If unavailable, uses market variance as proxy.

        Returns:
            MomentumCrashSignal with dynamic weight and regime.
        """
        if market_returns is None or len(market_returns) < 6:
            return MomentumCrashSignal(
                is_bear_market=False,
                bear_indicator=0.0,
                cumulative_24m_return=None,
                momentum_weight=1.0,
                wml_variance=None,
                crash_probability=0.05,
                regime="UNAVAILABLE",
            )

        is_bear, cum_return = self._compute_bear_indicator(market_returns)
        bear_ind = 1.0 if is_bear else 0.0

        # Use WML variance if available, else proxy with market variance
        if wml_returns is not None and len(wml_returns) >= 60:
            wml_var = self._estimate_wml_variance(wml_returns)
        else:
            wml_var = self._estimate_wml_variance(market_returns)

        crash_prob = self._forecast_crash_probability(is_bear, wml_var)
        weight = self._compute_dynamic_weight(is_bear, crash_prob)

        # Classify regime
        if crash_prob >= 0.50:
            regime = "HIGH_RISK"
        elif crash_prob >= 0.20:
            regime = "CAUTION"
        else:
            regime = "NORMAL"

        return MomentumCrashSignal(
            is_bear_market=is_bear,
            bear_indicator=bear_ind,
            cumulative_24m_return=round(cum_return, 4) if cum_return is not None else None,
            momentum_weight=round(weight, 4),
            wml_variance=round(wml_var, 6) if wml_var is not None else None,
            crash_probability=round(crash_prob, 4),
            regime=regime,
        )

    def get_regime_score(self, signal: MomentumCrashSignal) -> float:
        """Return normalized risk score in [0, 1] for aggregation.

        0.0 = NORMAL
        0.5 = CAUTION
        1.0 = HIGH_RISK
        """
        regime_map = {
            "NORMAL": 0.0,
            "CAUTION": 0.5,
            "HIGH_RISK": 1.0,
            "UNAVAILABLE": 0.5,
        }
        return regime_map.get(signal["regime"], 0.5)
