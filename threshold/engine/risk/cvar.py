"""Conditional Value at Risk (CVaR) — Expected Shortfall.

CVaR measures the expected loss beyond the VaR threshold, providing a
coherent risk measure that captures tail risk better than VaR alone.

Two computation methods:
  - Historical: Non-parametric from observed return distribution
  - Parametric: Assumes normal distribution (faster, less accurate for fat tails)

Reference:
  Rockafellar, R.T. & Uryasev, S. (2000). "Optimization of Conditional
  Value-at-Risk." Journal of Risk, 2(3), 21-41.
"""

from __future__ import annotations

from typing import TypedDict

import numpy as np
import pandas as pd
from scipy import stats


class CVaRResult(TypedDict):
    """Result from CVaR calculation."""
    cvar: float             # CVaR (expected shortfall) as positive loss
    var: float              # Value at Risk threshold
    alpha: float            # Confidence level used
    method: str             # "historical" or "parametric"
    n_observations: int     # Number of return observations
    worst_loss: float       # Maximum observed loss
    mean_return: float      # Mean return over period
    volatility: float       # Annualized volatility


class CVaRCalculator:
    """Conditional Value at Risk calculator.

    Usage::

        calc = CVaRCalculator(alpha=0.95, method="historical")
        result = calc.compute(returns)

    Parameters:
        alpha: Confidence level (default 0.95 = 95th percentile).
        method: "historical" (default) or "parametric".
        annualize: Whether to annualize the result (default False).
    """

    def __init__(
        self,
        alpha: float = 0.95,
        method: str = "historical",
        annualize: bool = False,
    ) -> None:
        if not 0.5 < alpha < 1.0:
            raise ValueError(f"alpha must be in (0.5, 1.0), got {alpha}")
        if method not in ("historical", "parametric"):
            raise ValueError(f"method must be 'historical' or 'parametric', got {method}")
        self.alpha = alpha
        self.method = method
        self.annualize = annualize

    def historical_cvar(self, returns: pd.Series | np.ndarray) -> float:
        """Compute historical (non-parametric) CVaR.

        CVaR = -E[R | R <= VaR], expressed as positive loss.
        """
        arr = np.asarray(returns, dtype=float)
        arr = arr[~np.isnan(arr)]

        if len(arr) < 10:
            return 0.0

        var_threshold = np.percentile(arr, (1 - self.alpha) * 100)
        tail_losses = arr[arr <= var_threshold]

        if len(tail_losses) == 0:
            return -float(var_threshold)

        return -float(np.mean(tail_losses))

    def parametric_cvar(self, returns: pd.Series | np.ndarray) -> float:
        """Compute parametric CVaR assuming normal distribution.

        CVaR = -μ + σ × φ(Φ⁻¹(1-α)) / (1-α)
        where φ = standard normal PDF, Φ⁻¹ = inverse CDF.
        Expressed as a positive loss magnitude.
        """
        arr = np.asarray(returns, dtype=float)
        arr = arr[~np.isnan(arr)]

        if len(arr) < 10:
            return 0.0

        mu = np.mean(arr)
        sigma = np.std(arr, ddof=1)

        if sigma < 1e-10:
            return 0.0

        # Inverse CDF at (1-alpha) — this is a negative z-score
        z = stats.norm.ppf(1 - self.alpha)
        # PDF at that quantile
        phi_z = stats.norm.pdf(z)

        # Parametric CVaR for normal: E[-R | R <= VaR]
        # = -mu + sigma * phi(z) / (1 - alpha)
        cvar = -mu + sigma * phi_z / (1 - self.alpha)
        return max(cvar, 0.0)

    def compute(self, returns: pd.Series | np.ndarray) -> CVaRResult:
        """Compute CVaR using the configured method.

        Parameters:
            returns: Daily or periodic return series.

        Returns:
            CVaRResult with risk metrics.
        """
        arr = np.asarray(returns, dtype=float)
        arr = arr[~np.isnan(arr)]
        n = len(arr)

        if n < 10:
            return CVaRResult(
                cvar=0.0,
                var=0.0,
                alpha=self.alpha,
                method=self.method,
                n_observations=n,
                worst_loss=0.0,
                mean_return=0.0,
                volatility=0.0,
            )

        if self.method == "historical":
            cvar = self.historical_cvar(arr)
        else:
            cvar = self.parametric_cvar(arr)

        var_value = -float(np.percentile(arr, (1 - self.alpha) * 100))
        worst_loss = -float(np.min(arr))
        mean_ret = float(np.mean(arr))
        vol = float(np.std(arr, ddof=1))

        if self.annualize:
            cvar *= np.sqrt(252)
            var_value *= np.sqrt(252)
            vol *= np.sqrt(252)

        return CVaRResult(
            cvar=round(cvar, 6),
            var=round(var_value, 6),
            alpha=self.alpha,
            method=self.method,
            n_observations=n,
            worst_loss=round(worst_loss, 6),
            mean_return=round(mean_ret, 6),
            volatility=round(vol, 6),
        )
