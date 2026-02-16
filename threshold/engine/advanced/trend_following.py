"""Continuous trend following signal — Baltas & Kosowski (2020).

Generates a continuous signal in [-1, +1] by regressing price on time
and normalizing the t-statistic. Combined with Yang-Zhang volatility
scaling for consistent risk-adjusted positioning.

Reference:
  Baltas, N. & Kosowski, R. (2020). "Demystifying Time-Series Momentum
  Strategies: Volatility Estimators, Trading Rules and Pairwise
  Correlations." Journal of Financial Markets.

Integration: Optional 20% blend into MQ sub-score when enabled.
"""

from __future__ import annotations

from typing import TypedDict

import numpy as np
import pandas as pd


class TrendSignal(TypedDict):
    """Result from continuous trend signal computation."""
    signal: float          # Continuous signal in [-1, +1]
    t_stat: float          # Raw t-statistic of OLS slope
    vol_scaled: float      # Signal / Yang-Zhang vol (risk-adjusted)
    yang_zhang: float      # Yang-Zhang volatility estimate
    regime: str            # STRONG_UP, UP, FLAT, DOWN, STRONG_DOWN


class ContinuousTrendFollower:
    """Baltas-Kosowski continuous trend following signal.

    Usage::

        tf = ContinuousTrendFollower(window=252, vol_window=60)
        signal = tf.compute_signal(close)
        # signal["signal"] in [-1, +1], blended 20% into MQ

    Parameters:
        window: OLS regression lookback in trading days (default 252).
        vol_window: Yang-Zhang volatility estimation window (default 60).
    """

    def __init__(self, window: int = 252, vol_window: int = 60) -> None:
        self.window = window
        self.vol_window = vol_window

    def yang_zhang_vol(
        self,
        open_prices: pd.Series,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        window: int | None = None,
    ) -> float:
        """Yang-Zhang (2000) volatility estimator.

        Combines overnight (close-to-open), open-to-close, and
        Rogers-Satchell intraday components for a more efficient
        volatility estimate than close-to-close.

        If OHLC data is unavailable, falls back to close-to-close vol.

        Parameters:
            open_prices, high, low, close: Price series.
            window: Lookback window (defaults to self.vol_window).

        Returns:
            Annualized volatility estimate (positive float).
        """
        w = window or self.vol_window

        if (
            open_prices is None
            or len(open_prices) < w
            or len(high) < w
            or len(low) < w
            or len(close) < w
        ):
            # Fallback: close-to-close volatility
            if len(close) < w:
                return 0.0
            log_ret = np.log(close.iloc[-w:] / close.iloc[-w:].shift(1)).dropna()
            return float(np.std(log_ret, ddof=1) * np.sqrt(252))

        o = open_prices.iloc[-w:]
        h = high.iloc[-w:]
        lo = low.iloc[-w:]
        c = close.iloc[-w:]
        c_prev = close.iloc[-(w + 1) : -1]

        n = len(o)
        if n < 2:
            return 0.0

        # Overnight return: log(open / prev_close)
        log_oc = np.log(o.values / c_prev.values)
        sigma_oc_sq = float(np.var(log_oc, ddof=1))

        # Open-to-close return
        log_co = np.log(c.values / o.values)
        sigma_co_sq = float(np.var(log_co, ddof=1))

        # Rogers-Satchell component
        log_ho = np.log(h.values / o.values)
        log_hc = np.log(h.values / c.values)
        log_lo = np.log(lo.values / o.values)
        log_lc = np.log(lo.values / c.values)
        rs = log_ho * log_hc + log_lo * log_lc
        sigma_rs_sq = float(np.mean(rs))

        # Yang-Zhang combination: k * σ²_oc + (1-k) * σ²_co + σ²_RS
        k = 0.34 / (1 + (n + 1) / (n - 1))
        sigma_yz_sq = k * sigma_oc_sq + (1 - k) * sigma_co_sq + sigma_rs_sq

        # Annualize
        sigma_yz = np.sqrt(max(sigma_yz_sq, 0.0) * 252)
        return float(sigma_yz)

    def yang_zhang_vol_from_close(self, close: pd.Series, window: int | None = None) -> float:
        """Simplified Yang-Zhang using only close prices (close-to-close vol).

        Used as fallback when OHLC data is not available.
        """
        w = window or self.vol_window
        if len(close) < w + 1:
            return 0.0
        log_ret = np.log(close.iloc[-w:] / close.iloc[-w:].shift(1)).dropna()
        if len(log_ret) < 2:
            return 0.0
        return float(np.std(log_ret, ddof=1) * np.sqrt(252))

    def _classify_regime(self, signal: float) -> str:
        """Classify continuous signal into discrete regime."""
        if signal >= 0.6:
            return "STRONG_UP"
        if signal >= 0.2:
            return "UP"
        if signal <= -0.6:
            return "STRONG_DOWN"
        if signal <= -0.2:
            return "DOWN"
        return "FLAT"

    def compute_signal(self, close: pd.Series) -> TrendSignal | None:
        """Compute the continuous trend following signal.

        OLS regression: price ~ β₀ + β₁·t over `window` days.
        Signal = clamp(t_stat(β₁) / 2, -1, +1).

        Parameters:
            close: Price series (Close prices).

        Returns:
            TrendSignal dict, or None if insufficient data.
        """
        if close is None or len(close) < self.window:
            return None

        # Use last `window` bars
        y = close.iloc[-self.window :].values.astype(float)
        n = len(y)

        # Time variable: 0, 1, 2, ..., n-1
        x = np.arange(n, dtype=float)

        # OLS: β̂₁ = Cov(x, y) / Var(x)
        x_mean = x.mean()
        y_mean = y.mean()
        ss_xx = np.sum((x - x_mean) ** 2)
        ss_xy = np.sum((x - x_mean) * (y - y_mean))

        if ss_xx < 1e-12:
            return TrendSignal(
                signal=0.0, t_stat=0.0, vol_scaled=0.0,
                yang_zhang=0.0, regime="FLAT",
            )

        beta_1 = ss_xy / ss_xx
        y_hat = x_mean + beta_1 * (x - x_mean) + (y_mean - beta_1 * x_mean)
        # Correct: y_hat = beta_0 + beta_1 * x
        beta_0 = y_mean - beta_1 * x_mean
        y_hat = beta_0 + beta_1 * x
        residuals = y - y_hat
        sse = np.sum(residuals**2)

        # Standard error of β̂₁
        s2 = sse / (n - 2) if n > 2 else 1e-12
        se_beta1 = np.sqrt(max(s2 / ss_xx, 1e-20))

        # t-statistic
        t_stat = float(beta_1 / se_beta1) if se_beta1 > 1e-12 else 0.0

        # Continuous signal: clamp(t_stat / 2, -1, +1)
        signal = float(np.clip(t_stat / 2, -1.0, 1.0))

        # Yang-Zhang volatility (close-to-close fallback)
        yz_vol = self.yang_zhang_vol_from_close(close, self.vol_window)

        # Vol-scaled signal
        vol_scaled = signal / max(yz_vol, 0.05) if yz_vol > 0 else signal

        regime = self._classify_regime(signal)

        return TrendSignal(
            signal=round(signal, 4),
            t_stat=round(t_stat, 4),
            vol_scaled=round(vol_scaled, 4),
            yang_zhang=round(yz_vol, 4),
            regime=regime,
        )
