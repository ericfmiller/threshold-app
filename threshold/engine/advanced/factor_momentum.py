"""Factor momentum signal — Ehsani & Linnainmaa (2022).

Measures the breadth and strength of positive factor returns across
non-momentum factors. When most factors are trending positive, market
conditions are broadly supportive. When factors are mixed or negative,
momentum strategies face elevated crash risk.

Reference:
  Ehsani, S. & Linnainmaa, J.T. (2022). "Factor Momentum and the
  Momentum Factor." Journal of Finance, 77(3), 1877-1919.

Integration: Informational overlay only — does NOT modify DCS.
Stored in ScoringResult["advanced_signals"]["factor_momentum"].
"""

from __future__ import annotations

from typing import TypedDict

import numpy as np
import pandas as pd


class FactorMomentumResult(TypedDict):
    """Result from factor momentum analysis."""
    breadth: float               # Fraction of factors with positive 12m return [0, 1]
    momentum_strength: float     # Mean(positive) - Mean(negative) spread
    long_factors: list[str]      # Factor names with positive 12m return
    short_factors: list[str]     # Factor names with negative 12m return
    regime: str                  # BROAD_POSITIVE, MIXED, BROAD_NEGATIVE, UNAVAILABLE
    n_factors: int               # Total number of factors analyzed


class FactorMomentumSignal:
    """Ehsani-Linnainmaa factor momentum signal.

    Usage::

        fms = FactorMomentumSignal(lookback_months=12)
        result = fms.compute_signal(factor_returns)

    Parameters:
        lookback_months: Period for computing cumulative factor returns.
        breadth_threshold_high: Breadth above which = BROAD_POSITIVE.
        breadth_threshold_low: Breadth below which = BROAD_NEGATIVE.
    """

    def __init__(
        self,
        lookback_months: int = 12,
        breadth_threshold_high: float = 0.65,
        breadth_threshold_low: float = 0.35,
    ) -> None:
        self.lookback_months = lookback_months
        self.breadth_threshold_high = breadth_threshold_high
        self.breadth_threshold_low = breadth_threshold_low

    def _compute_cumulative_returns(
        self, factor_returns: pd.DataFrame
    ) -> pd.Series:
        """Compute cumulative return for each factor over lookback period.

        Parameters:
            factor_returns: Monthly factor returns with columns as factor names.

        Returns:
            Series of cumulative returns indexed by factor name.
        """
        if len(factor_returns) < self.lookback_months:
            recent = factor_returns
        else:
            recent = factor_returns.iloc[-self.lookback_months :]

        # Cumulative return: (1 + r1)(1 + r2)...(1 + rn) - 1
        cum_returns = (1 + recent).prod() - 1
        return cum_returns

    def _classify_regime(self, breadth: float) -> str:
        """Classify factor momentum regime from breadth."""
        if breadth >= self.breadth_threshold_high:
            return "BROAD_POSITIVE"
        if breadth <= self.breadth_threshold_low:
            return "BROAD_NEGATIVE"
        return "MIXED"

    def compute_signal(
        self, factor_returns: pd.DataFrame
    ) -> FactorMomentumResult:
        """Compute factor momentum signal from multi-factor return data.

        Parameters:
            factor_returns: DataFrame with datetime index and factor columns.
                           Each column = monthly returns for one factor.
                           Example factors: SMB, HML, RMW, CMA, MOM, BAB, QMJ, etc.

        Returns:
            FactorMomentumResult with breadth, strength, and factor lists.
        """
        if factor_returns is None or factor_returns.empty or factor_returns.shape[1] < 2:
            return FactorMomentumResult(
                breadth=0.5,
                momentum_strength=0.0,
                long_factors=[],
                short_factors=[],
                regime="UNAVAILABLE",
                n_factors=0,
            )

        cum_returns = self._compute_cumulative_returns(factor_returns)
        n_factors = len(cum_returns)

        # Split into positive and negative
        positive_mask = cum_returns > 0
        negative_mask = cum_returns <= 0

        long_factors = cum_returns[positive_mask].index.tolist()
        short_factors = cum_returns[negative_mask].index.tolist()

        # Breadth: fraction of positive factors
        breadth = float(positive_mask.sum() / n_factors) if n_factors > 0 else 0.5

        # Momentum strength: mean(positive) - mean(negative)
        pos_returns = cum_returns[positive_mask]
        neg_returns = cum_returns[negative_mask]

        mean_pos = float(pos_returns.mean()) if len(pos_returns) > 0 else 0.0
        mean_neg = float(neg_returns.mean()) if len(neg_returns) > 0 else 0.0
        momentum_strength = mean_pos - mean_neg

        regime = self._classify_regime(breadth)

        return FactorMomentumResult(
            breadth=round(breadth, 4),
            momentum_strength=round(momentum_strength, 4),
            long_factors=long_factors,
            short_factors=short_factors,
            regime=regime,
            n_factors=n_factors,
        )

    @staticmethod
    def compute_proxy_factors(
        etf_returns: pd.DataFrame,
        window: int = 252,
    ) -> pd.DataFrame:
        """Compute proxy factor returns from cross-asset ETF returns.

        When Fama-French data is unavailable, approximate factor exposures
        using observable cross-asset return differences:
          - Equity risk: SPY return
          - Value: EFA - SPY (international tends to be more value-tilted)
          - Safe haven: GLD - SPY (gold vs equity)
          - Duration: BND - SPY (bonds vs equity)
          - Commodity: GSG - SPY (commodities vs equity)

        Parameters:
            etf_returns: Daily returns DataFrame with ETF columns.
            window: Rolling window for monthly resampling.

        Returns:
            Monthly proxy factor returns DataFrame.
        """
        if etf_returns is None or etf_returns.empty:
            return pd.DataFrame()

        # Resample to monthly
        monthly = (1 + etf_returns).resample("ME").prod() - 1

        factors = pd.DataFrame(index=monthly.index)

        # Build proxy factors from available columns
        if "SPY" in monthly.columns:
            factors["equity_risk"] = monthly["SPY"]
        if "EFA" in monthly.columns and "SPY" in monthly.columns:
            factors["value_proxy"] = monthly["EFA"] - monthly["SPY"]
        if "GLD" in monthly.columns and "SPY" in monthly.columns:
            factors["safe_haven"] = monthly["GLD"] - monthly["SPY"]
        if "BND" in monthly.columns and "SPY" in monthly.columns:
            factors["duration"] = monthly["BND"] - monthly["SPY"]
        if "GSG" in monthly.columns and "SPY" in monthly.columns:
            factors["commodity"] = monthly["GSG"] - monthly["SPY"]

        return factors.dropna()
