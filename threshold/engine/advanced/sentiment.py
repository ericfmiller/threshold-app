"""Aligned sentiment index — Huang, Jiang, Tu & Zhou (2015).

Uses Partial Least Squares (PLS) regression to extract a sentiment
factor from multiple proxies that is optimally aligned with future
market returns. When sentiment is overheated, reduces market regime
sub-score to reflect elevated reversal risk.

Reference:
  Huang, D., Jiang, F., Tu, J. & Zhou, G. (2015). "Investor Sentiment
  Aligned: A Powerful Predictor of Stock Returns." Review of Financial
  Studies, 28(3), 791-837.

Requires: scikit-learn (optional dependency).
Integration: When overheated, reduces MR sub-score by configurable factor.
"""

from __future__ import annotations

from typing import TypedDict

import numpy as np
import pandas as pd


class SentimentResult(TypedDict):
    """Result from aligned sentiment computation."""
    sentiment_value: float | None  # Current aligned sentiment index value
    percentile: float | None       # Percentile rank vs history
    regime: str                    # OVERHEATED, DEPRESSED, NEUTRAL, UNAVAILABLE
    mr_adjustment: float           # MR sub-score multiplier adjustment (0 = no change)


# Proxy names for documentation and data loading guidance
BAKER_WURGLER_PROXIES = [
    "cef_discount",       # Closed-end fund discount (FRED: CEFD or computed)
    "ipo_volume",         # Monthly IPO count
    "ipo_first_day_ret",  # First-day IPO returns (mean)
    "equity_share",       # Equity share in new issues: equity / (equity + debt)
    "dividend_premium",   # Market-to-book of dividend payers vs non-payers
    "vix_inverted",       # Inverted VIX: -VIX (high VIX = low sentiment)
]


def _has_sklearn() -> bool:
    """Check if scikit-learn is available."""
    try:
        from sklearn.cross_decomposition import PLSRegression  # noqa: F401
        return True
    except ImportError:
        return False


class AlignedSentimentIndex:
    """Huang et al. aligned sentiment index via PLS regression.

    Usage::

        asi = AlignedSentimentIndex(n_components=1, mr_reduction=0.15)
        result = asi.compute(proxy_data, market_returns)
        # If result["regime"] == "OVERHEATED":
        #     mr_adjusted = mr * (1 - result["mr_adjustment"])

    Parameters:
        n_components: Number of PLS components (default 1).
        mr_reduction: MR sub-score reduction when overheated (default 0.15).
        overheated_pctl: Percentile threshold for OVERHEATED (default 0.80).
        depressed_pctl: Percentile threshold for DEPRESSED (default 0.20).
        min_observations: Minimum observations for PLS estimation.
    """

    def __init__(
        self,
        n_components: int = 1,
        mr_reduction: float = 0.15,
        overheated_pctl: float = 0.80,
        depressed_pctl: float = 0.20,
        min_observations: int = 60,
    ) -> None:
        self.n_components = n_components
        self.mr_reduction = mr_reduction
        self.overheated_pctl = overheated_pctl
        self.depressed_pctl = depressed_pctl
        self.min_observations = min_observations

    def _standardize(self, df: pd.DataFrame) -> pd.DataFrame:
        """Z-score standardize each column."""
        return (df - df.mean()) / df.std().replace(0, 1)

    def _compute_pls_sentiment(
        self,
        proxies: pd.DataFrame,
        market_returns: pd.Series,
    ) -> np.ndarray | None:
        """Extract aligned sentiment via PLS regression.

        Returns:
            Array of sentiment index values, or None if sklearn unavailable.
        """
        if not _has_sklearn():
            return None

        from sklearn.cross_decomposition import PLSRegression

        # Align indices
        common_idx = proxies.index.intersection(market_returns.index)
        if len(common_idx) < self.min_observations:
            return None

        X = proxies.loc[common_idx].values
        y = market_returns.loc[common_idx].values.reshape(-1, 1)

        # Remove NaN rows
        mask = ~(np.isnan(X).any(axis=1) | np.isnan(y.ravel()))
        X = X[mask]
        y = y[mask]

        if len(X) < self.min_observations:
            return None

        # Fit PLS
        pls = PLSRegression(n_components=self.n_components)
        pls.fit(X, y)

        # Extract first latent factor (aligned sentiment)
        sentiment = pls.transform(X).ravel()
        return sentiment

    def _compute_simple_sentiment(
        self,
        proxies: pd.DataFrame,
    ) -> np.ndarray | None:
        """Fallback: simple average of standardized proxies.

        Used when scikit-learn is not available.
        """
        if len(proxies) < self.min_observations:
            return None

        standardized = self._standardize(proxies.dropna())
        if len(standardized) < self.min_observations:
            return None

        # Equal-weight average of all proxies
        sentiment = standardized.mean(axis=1).values
        return sentiment

    def _classify_regime(self, percentile: float) -> str:
        """Classify sentiment percentile into regime."""
        if percentile >= self.overheated_pctl:
            return "OVERHEATED"
        if percentile <= self.depressed_pctl:
            return "DEPRESSED"
        return "NEUTRAL"

    def compute(
        self,
        proxies: pd.DataFrame,
        market_returns: pd.Series | None = None,
    ) -> SentimentResult:
        """Compute aligned sentiment index.

        Parameters:
            proxies: DataFrame with proxy columns (e.g. Baker-Wurgler variables).
                    Index should be datetime (monthly frequency recommended).
            market_returns: Market return series for PLS alignment.
                           If None, uses simple average of standardized proxies.

        Returns:
            SentimentResult with current regime and MR adjustment.
        """
        if proxies is None or proxies.empty or len(proxies) < self.min_observations:
            return SentimentResult(
                sentiment_value=None,
                percentile=None,
                regime="UNAVAILABLE",
                mr_adjustment=0.0,
            )

        # Try PLS alignment first, fall back to simple average
        sentiment = None
        if market_returns is not None and _has_sklearn():
            sentiment = self._compute_pls_sentiment(
                self._standardize(proxies), market_returns,
            )

        if sentiment is None:
            sentiment = self._compute_simple_sentiment(proxies)

        if sentiment is None or len(sentiment) < 2:
            return SentimentResult(
                sentiment_value=None,
                percentile=None,
                regime="UNAVAILABLE",
                mr_adjustment=0.0,
            )

        # Current value and percentile
        current = float(sentiment[-1])
        percentile = float(
            np.searchsorted(np.sort(sentiment), current) / len(sentiment)
        )

        regime = self._classify_regime(percentile)

        # MR adjustment: only when OVERHEATED
        mr_adj = self.mr_reduction if regime == "OVERHEATED" else 0.0

        return SentimentResult(
            sentiment_value=round(current, 4),
            percentile=round(percentile, 4),
            regime=regime,
            mr_adjustment=round(mr_adj, 4),
        )
