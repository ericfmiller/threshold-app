"""Pairwise correlation analysis and effective bets calculation.

Computes:
  - Pairwise correlation matrix for portfolio holdings
  - High-correlation pair detection (> threshold)
  - Effective number of bets (eigenvalue-based Shannon entropy)
  - Concentration risk warnings
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class CorrelationReport:
    """Result from portfolio correlation analysis."""

    high_corr_pairs: list[tuple[str, str, float]] = field(default_factory=list)
    """List of (ticker_a, ticker_b, correlation) where corr > threshold."""
    effective_bets: float = 0.0
    """Eigenvalue-based Shannon entropy measure of diversification."""
    correlation_matrix: dict[str, dict[str, float]] = field(default_factory=dict)
    """Full correlation matrix as nested dict."""
    n_tickers: int = 0
    min_data_days: int = 0
    is_concentrated: bool = False
    """True if effective_bets < concentration threshold (default 20)."""


def compute_correlation_report(
    returns: pd.DataFrame,
    corr_threshold: float = 0.80,
    concentration_threshold: float = 20.0,
    min_common_days: int = 30,
) -> CorrelationReport:
    """Compute pairwise correlation and effective bets for portfolio holdings.

    Parameters:
        returns: DataFrame of daily returns, one column per ticker.
        corr_threshold: Threshold for flagging high-correlation pairs.
        concentration_threshold: Effective bets below this triggers warning.
        min_common_days: Minimum overlapping days needed for valid analysis.

    Returns:
        CorrelationReport with pairs, effective bets, and concentration flag.
    """
    if returns.empty or returns.shape[1] < 2:
        n = returns.shape[1] if not returns.empty else 0
        return CorrelationReport(
            effective_bets=float(n),
            n_tickers=n,
        )

    # Drop columns with all NaN
    returns = returns.dropna(axis=1, how="all")
    if returns.shape[1] < 2:
        return CorrelationReport(
            effective_bets=float(returns.shape[1]),
            n_tickers=returns.shape[1],
        )

    # Use only rows where all tickers have data
    common = returns.dropna()
    if len(common) < min_common_days:
        return CorrelationReport(
            effective_bets=float(returns.shape[1]),
            n_tickers=returns.shape[1],
            min_data_days=len(common),
        )

    # Correlation matrix
    corr_matrix = common.corr()

    # High-correlation pairs
    high_pairs: list[tuple[str, str, float]] = []
    tickers = list(corr_matrix.columns)
    for i in range(len(tickers)):
        for j in range(i + 1, len(tickers)):
            corr_val = corr_matrix.iloc[i, j]
            if abs(corr_val) > corr_threshold:
                high_pairs.append((tickers[i], tickers[j], round(float(corr_val), 4)))

    # Sort by absolute correlation descending
    high_pairs.sort(key=lambda x: abs(x[2]), reverse=True)

    # Effective bets via eigenvalue-based Shannon entropy
    effective_bets = _compute_effective_bets(corr_matrix)

    # Convert correlation matrix to nested dict
    corr_dict: dict[str, dict[str, float]] = {}
    for t in tickers:
        corr_dict[t] = {
            t2: round(float(corr_matrix.loc[t, t2]), 4) for t2 in tickers
        }

    return CorrelationReport(
        high_corr_pairs=high_pairs,
        effective_bets=round(effective_bets, 2),
        correlation_matrix=corr_dict,
        n_tickers=len(tickers),
        min_data_days=len(common),
        is_concentrated=effective_bets < concentration_threshold,
    )


def _compute_effective_bets(corr_matrix: pd.DataFrame) -> float:
    """Compute effective number of bets using eigenvalue Shannon entropy.

    This is a diversification measure:
      - N uncorrelated assets → effective_bets = N
      - N perfectly correlated assets → effective_bets = 1
      - More diversified portfolios → higher effective_bets

    Uses the formula: exp(-sum(p_i * ln(p_i))) where p_i = eigenvalue_i / sum(eigenvalues)
    """
    try:
        eigenvalues = np.linalg.eigvalsh(corr_matrix.values)
        # Keep only positive eigenvalues (numerical stability)
        eigenvalues = eigenvalues[eigenvalues > 1e-10]
        if len(eigenvalues) == 0:
            return 0.0

        # Normalize to probabilities
        total = eigenvalues.sum()
        if total <= 0:
            return 0.0
        probs = eigenvalues / total

        # Shannon entropy → effective N
        entropy = -np.sum(probs * np.log(probs))
        return float(np.exp(entropy))
    except (np.linalg.LinAlgError, ValueError):
        return float(corr_matrix.shape[0])


def check_concentration_risk(
    high_corr_pairs: list[tuple[str, str, float]],
    effective_bets: float,
    buy_tickers: set[str],
    held_tickers: set[str],
    concentration_threshold: float = 20.0,
    pair_threshold: float = 0.70,
) -> list[dict[str, Any]]:
    """Check for concentration risk in proposed buy signals.

    When effective_bets < threshold, scans buy candidates for high
    correlation with existing holdings.

    Parameters:
        high_corr_pairs: Pre-computed high correlation pairs.
        effective_bets: Current portfolio effective bets.
        buy_tickers: Tickers with active BUY/HIGH CONVICTION signals.
        held_tickers: Currently held tickers.
        concentration_threshold: Effective bets threshold (default 20).
        pair_threshold: Correlation threshold for flagging (default 0.70).

    Returns:
        List of {ticker, correlated_with, correlation} warnings.
    """
    if effective_bets >= concentration_threshold:
        return []

    warnings: list[dict[str, Any]] = []
    for ticker_a, ticker_b, corr in high_corr_pairs:
        if abs(corr) < pair_threshold:
            continue
        # Check if one is a buy candidate and the other is held
        if ticker_a in buy_tickers and ticker_b in held_tickers:
            warnings.append({
                "ticker": ticker_a,
                "correlated_with": ticker_b,
                "correlation": corr,
            })
        elif ticker_b in buy_tickers and ticker_a in held_tickers:
            warnings.append({
                "ticker": ticker_b,
                "correlated_with": ticker_a,
                "correlation": corr,
            })

    return warnings
