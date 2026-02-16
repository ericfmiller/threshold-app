"""Inverse volatility weighting — Kirby & Ostdiek (2012).

Allocates capital inversely proportional to each asset's volatility,
producing a portfolio where each position contributes roughly equal risk.

Reference:
  Kirby, C. & Ostdiek, B. (2012). "It's All in the Timing: Simple Active
  Portfolio Strategies that Outperform Naïve Diversification." Journal of
  Financial and Quantitative Analysis, 47(2), 437-467.

Integration: Optional portfolio weighting in pipeline.py.
When disabled, the pipeline uses equal-weight or user-specified weights.
"""

from __future__ import annotations

from typing import TypedDict

import numpy as np
import pandas as pd


class InverseVolResult(TypedDict):
    """Result from inverse volatility allocation."""
    weights: dict[str, float]    # Ticker → weight (sums to 1.0)
    volatilities: dict[str, float]  # Ticker → annualized volatility
    eta: float                   # Risk aversion parameter used
    n_assets: int                # Number of assets allocated
    method: str                  # Always "inverse_vol"


class InverseVolWeighter:
    """Kirby-Ostdiek inverse volatility portfolio weighting.

    Usage::

        ivw = InverseVolWeighter(eta=1.0, window=120)
        result = ivw.compute_weights(returns)
        # result["weights"] = {"AAPL": 0.15, "MSFT": 0.12, ...}

    Parameters:
        eta: Risk aversion parameter. Higher eta = more aggressive
            underweighting of volatile assets. eta=1.0 is standard
            inverse-variance, eta=0.5 is inverse-vol.
        window: Lookback window in trading days for volatility estimation.
        min_periods: Minimum number of observations required per asset.
        annualization_factor: Trading days per year for annualization.
    """

    def __init__(
        self,
        eta: float = 1.0,
        window: int = 120,
        min_periods: int = 60,
        annualization_factor: int = 252,
    ) -> None:
        self.eta = eta
        self.window = window
        self.min_periods = min_periods
        self.annualization_factor = annualization_factor

    def _compute_volatilities(self, returns: pd.DataFrame) -> pd.Series:
        """Compute annualized volatility for each asset.

        Parameters:
            returns: Daily returns DataFrame with columns as asset names.

        Returns:
            Series of annualized volatilities indexed by asset name.
        """
        # Use last `window` observations
        if len(returns) > self.window:
            recent = returns.iloc[-self.window:]
        else:
            recent = returns

        # Standard deviation of daily returns, annualized
        vols = recent.std(ddof=1) * np.sqrt(self.annualization_factor)
        return vols

    def compute_weights(
        self,
        returns: pd.DataFrame,
        exclude: list[str] | None = None,
    ) -> InverseVolResult:
        """Compute inverse volatility portfolio weights.

        w_i = (1/σ²_i)^η / Σ(1/σ²_j)^η

        All weights are positive and sum to 1.0.

        Parameters:
            returns: Daily returns DataFrame. Columns = asset names.
            exclude: Optional list of assets to exclude from allocation.

        Returns:
            InverseVolResult with weights, volatilities, and metadata.
        """
        if returns is None or returns.empty:
            return InverseVolResult(
                weights={},
                volatilities={},
                eta=self.eta,
                n_assets=0,
                method="inverse_vol",
            )

        # Exclude specified assets
        cols = returns.columns.tolist()
        if exclude:
            cols = [c for c in cols if c not in exclude]
        if not cols:
            return InverseVolResult(
                weights={},
                volatilities={},
                eta=self.eta,
                n_assets=0,
                method="inverse_vol",
            )

        working = returns[cols]

        # Drop assets with insufficient data
        valid_cols = [
            c for c in working.columns
            if working[c].dropna().shape[0] >= self.min_periods
        ]
        if not valid_cols:
            return InverseVolResult(
                weights={},
                volatilities={},
                eta=self.eta,
                n_assets=0,
                method="inverse_vol",
            )

        working = working[valid_cols]
        vols = self._compute_volatilities(working)

        # Replace zero/near-zero vol with small floor to avoid division by zero
        vol_floor = 1e-6
        vols = vols.clip(lower=vol_floor)

        # Inverse variance raised to eta: w_i ∝ (1/σ²_i)^η
        inv_var = (1.0 / (vols ** 2)) ** self.eta

        # Normalize to sum to 1
        total = inv_var.sum()
        if total <= 0:
            # Equal weight fallback
            n = len(valid_cols)
            weights = {c: 1.0 / n for c in valid_cols}
        else:
            raw_weights = inv_var / total
            weights = {c: round(float(w), 6) for c, w in raw_weights.items()}

        vol_dict = {c: round(float(v), 6) for c, v in vols.items()}

        return InverseVolResult(
            weights=weights,
            volatilities=vol_dict,
            eta=self.eta,
            n_assets=len(weights),
            method="inverse_vol",
        )

    @staticmethod
    def equal_vol_weights(n: int) -> dict[str, float]:
        """Generate equal volatility weights (for testing baseline).

        When all assets have identical volatility, inverse-vol
        produces equal weights.
        """
        if n <= 0:
            return {}
        w = round(1.0 / n, 6)
        return {f"asset_{i}": w for i in range(n)}
