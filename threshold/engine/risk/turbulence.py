"""Financial turbulence index — Kritzman & Li (2010).

Measures the statistical unusualness of multi-asset returns using the
Mahalanobis distance. When returns deviate from their historical correlation
structure, turbulence rises — signaling regime stress.

Reference:
  Kritzman, M. & Li, Y. (2010). "Skulls, Financial Turbulence, and Risk
  Management." Financial Analysts Journal, 66(5), 30-41.

Uses: scipy.spatial.distance.mahalanobis, numpy.linalg
"""

from __future__ import annotations

from typing import TypedDict

import numpy as np
import pandas as pd


class TurbulenceSignal(TypedDict):
    """Result from turbulence analysis."""
    turbulence_value: float | None
    turbulence_percentile: float | None
    is_turbulent: bool
    turbulence_regime: str  # CALM, ELEVATED, TURBULENT, UNAVAILABLE
    rolling_mean: float | None


class TurbulenceIndex:
    """Kritzman-Li Mahalanobis turbulence index.

    Usage::

        ti = TurbulenceIndex(window=252, threshold_pctl=0.75)
        signal = ti.compute(price_data)

    Parameters:
        window: Rolling estimation window in trading days (default 252 = 1 year).
        threshold_pctl: Percentile threshold for "turbulent" classification.
        min_assets: Minimum number of assets for valid computation.
    """

    def __init__(
        self,
        window: int = 252,
        threshold_pctl: float = 0.75,
        min_assets: int = 3,
    ) -> None:
        self.window = window
        self.threshold_pctl = threshold_pctl
        self.min_assets = min_assets

    def _compute_mahalanobis(
        self, y: np.ndarray, mu: np.ndarray, cov_inv: np.ndarray
    ) -> float:
        """Compute Mahalanobis distance: d = (y-μ)ᵀ Σ⁻¹ (y-μ)."""
        diff = y - mu
        return float(diff @ cov_inv @ diff)

    def compute(self, price_data: pd.DataFrame) -> TurbulenceSignal:
        """Compute turbulence index from multi-asset price data.

        Parameters:
            price_data: DataFrame with datetime index and asset columns.
                        Should contain price levels (not returns).

        Returns:
            TurbulenceSignal with current turbulence state.
        """
        if price_data is None or price_data.shape[1] < self.min_assets:
            return TurbulenceSignal(
                turbulence_value=None,
                turbulence_percentile=None,
                is_turbulent=False,
                turbulence_regime="UNAVAILABLE",
                rolling_mean=None,
            )

        # Compute log returns
        returns = np.log(price_data / price_data.shift(1)).dropna()

        if len(returns) < self.window + 1:
            return TurbulenceSignal(
                turbulence_value=None,
                turbulence_percentile=None,
                is_turbulent=False,
                turbulence_regime="UNAVAILABLE",
                rolling_mean=None,
            )

        # Rolling turbulence computation
        turbulence_series = []
        n_assets = returns.shape[1]

        for i in range(self.window, len(returns)):
            window_returns = returns.iloc[i - self.window : i].values
            current_return = returns.iloc[i].values

            mu = np.mean(window_returns, axis=0)
            cov = np.cov(window_returns, rowvar=False)

            # Regularize if nearly singular
            try:
                cov_inv = np.linalg.inv(
                    cov + np.eye(n_assets) * 1e-8
                )
            except np.linalg.LinAlgError:
                turbulence_series.append(0.0)
                continue

            d_t = self._compute_mahalanobis(current_return, mu, cov_inv)
            turbulence_series.append(d_t)

        if not turbulence_series:
            return TurbulenceSignal(
                turbulence_value=None,
                turbulence_percentile=None,
                is_turbulent=False,
                turbulence_regime="UNAVAILABLE",
                rolling_mean=None,
            )

        turb_array = np.array(turbulence_series)
        current_turb = turb_array[-1]

        # Percentile rank
        percentile = float(
            np.searchsorted(np.sort(turb_array), current_turb)
            / len(turb_array)
        )

        # Rolling mean (last 21 trading days)
        lookback = min(21, len(turb_array))
        rolling_mean = float(np.mean(turb_array[-lookback:]))

        # Classify regime
        is_turbulent = percentile >= self.threshold_pctl
        if percentile >= 0.90:
            regime = "TURBULENT"
        elif percentile >= self.threshold_pctl:
            regime = "ELEVATED"
        else:
            regime = "CALM"

        return TurbulenceSignal(
            turbulence_value=round(current_turb, 4),
            turbulence_percentile=round(percentile, 4),
            is_turbulent=is_turbulent,
            turbulence_regime=regime,
            rolling_mean=round(rolling_mean, 4),
        )

    def get_regime_score(self, signal: TurbulenceSignal) -> float:
        """Return normalized risk score in [0, 1] for aggregation.

        0.0 = CALM
        0.5 = ELEVATED
        1.0 = TURBULENT
        """
        regime_map = {
            "CALM": 0.0,
            "ELEVATED": 0.5,
            "TURBULENT": 1.0,
            "UNAVAILABLE": 0.5,
        }
        return regime_map.get(signal["turbulence_regime"], 0.5)
