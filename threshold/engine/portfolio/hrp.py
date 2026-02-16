"""Hierarchical Risk Parity — López de Prado (2016).

A machine-learning approach to portfolio allocation that avoids
matrix inversion (unlike Markowitz) by using hierarchical clustering
on the correlation structure, then recursively bisecting the dendrogram
to allocate risk.

Reference:
  López de Prado, M. (2016). "Building Diversified Portfolios that
  Outperform Out-of-Sample." Journal of Portfolio Management, 42(4), 59-69.

Steps:
  1. Compute correlation distance matrix: d(i,j) = sqrt(0.5 * (1 - ρ_ij))
  2. Single-linkage hierarchical clustering on d
  3. Quasi-diagonalize the covariance matrix by reordering to dendrogram
  4. Recursive bisection: split portfolio into halves, allocate by
     inverse variance of each half

Integration: Optional portfolio weighting in pipeline.py.
"""

from __future__ import annotations

from typing import TypedDict

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import squareform


class HRPResult(TypedDict):
    """Result from HRP allocation."""
    weights: dict[str, float]       # Ticker → weight (sums to 1.0)
    cluster_order: list[str]        # Assets in dendrogram order
    n_assets: int                   # Number of assets allocated
    method: str                     # Always "hrp"


class HRPAllocator:
    """López de Prado Hierarchical Risk Parity allocator.

    Usage::

        hrp = HRPAllocator(linkage_method="single")
        result = hrp.compute_weights(returns)
        # result["weights"] = {"AAPL": 0.08, "MSFT": 0.12, ...}

    Parameters:
        linkage_method: Clustering method for scipy linkage.
            "single" (default, per paper), "complete", or "average".
        min_periods: Minimum observations required for correlation.
    """

    def __init__(
        self,
        linkage_method: str = "single",
        min_periods: int = 60,
    ) -> None:
        self.linkage_method = linkage_method
        self.min_periods = min_periods

    def _correlation_distance(self, corr: pd.DataFrame) -> np.ndarray:
        """Compute correlation distance matrix.

        d(i,j) = sqrt(0.5 * (1 - ρ_ij))

        This maps correlation [-1, +1] to distance [0, 1]:
        - ρ = +1 → d = 0 (identical)
        - ρ = 0  → d ≈ 0.707
        - ρ = -1 → d = 1 (maximally different)
        """
        dist = np.sqrt(0.5 * (1 - corr.values))
        # Ensure diagonal is exactly zero and symmetry
        np.fill_diagonal(dist, 0.0)
        dist = (dist + dist.T) / 2  # Force symmetry
        return dist

    def _quasi_diagonalize(self, link: np.ndarray, n: int) -> list[int]:
        """Reorder assets according to dendrogram leaf order.

        Parameters:
            link: Linkage matrix from scipy.
            n: Number of original assets.

        Returns:
            List of asset indices in quasi-diagonal order.
        """
        return list(leaves_list(link))

    def _get_cluster_variance(
        self, cov: pd.DataFrame, items: list[int]
    ) -> float:
        """Compute the variance of the inverse-variance portfolio
        within a cluster.

        This is used for the recursive bisection step: each branch
        of the dendrogram gets weight proportional to the inverse
        of its cluster variance.
        """
        cov_slice = cov.iloc[items, items]
        ivp = 1.0 / np.diag(cov_slice.values)
        ivp /= ivp.sum()
        # Cluster variance = w' Σ w
        cluster_var = float(ivp @ cov_slice.values @ ivp)
        return cluster_var

    def _recursive_bisection(
        self, cov: pd.DataFrame, sort_ix: list[int]
    ) -> pd.Series:
        """Recursive bisection: allocate weights top-down.

        Starting from the full sorted list, split into halves.
        Each half gets weight proportional to the inverse of its
        cluster variance. Recurse until single assets remain.
        """
        weights = pd.Series(1.0, index=sort_ix)

        # List of clusters to process
        clusters = [sort_ix]

        while clusters:
            new_clusters = []
            for cluster in clusters:
                if len(cluster) <= 1:
                    continue

                # Split into two halves
                mid = len(cluster) // 2
                left = cluster[:mid]
                right = cluster[mid:]

                # Compute cluster variances
                var_left = self._get_cluster_variance(cov, left)
                var_right = self._get_cluster_variance(cov, right)

                # Allocation factor: inverse of cluster variance
                total_var = var_left + var_right
                alpha = 0.5 if total_var < 1e-12 else 1.0 - var_left / total_var

                # Scale weights
                weights[left] *= alpha
                weights[right] *= (1.0 - alpha)

                # Add sub-clusters for further processing
                if len(left) > 1:
                    new_clusters.append(left)
                if len(right) > 1:
                    new_clusters.append(right)

            clusters = new_clusters

        return weights

    def compute_weights(
        self,
        returns: pd.DataFrame,
        exclude: list[str] | None = None,
    ) -> HRPResult:
        """Compute HRP portfolio weights.

        Parameters:
            returns: Daily returns DataFrame. Columns = asset names.
            exclude: Optional list of assets to exclude.

        Returns:
            HRPResult with weights, cluster order, and metadata.
        """
        if returns is None or returns.empty:
            return HRPResult(
                weights={},
                cluster_order=[],
                n_assets=0,
                method="hrp",
            )

        # Exclude specified assets
        cols = returns.columns.tolist()
        if exclude:
            cols = [c for c in cols if c not in exclude]
        if len(cols) < 2:
            if len(cols) == 1:
                return HRPResult(
                    weights={cols[0]: 1.0},
                    cluster_order=cols,
                    n_assets=1,
                    method="hrp",
                )
            return HRPResult(
                weights={},
                cluster_order=[],
                n_assets=0,
                method="hrp",
            )

        working = returns[cols].dropna()
        if len(working) < self.min_periods:
            # Insufficient data — fall back to equal weight
            n = len(cols)
            return HRPResult(
                weights={c: round(1.0 / n, 6) for c in cols},
                cluster_order=cols,
                n_assets=n,
                method="hrp",
            )

        # Step 1: Correlation → distance
        corr = working.corr()
        dist_matrix = self._correlation_distance(corr)

        # Convert to condensed form for scipy
        condensed = squareform(dist_matrix, checks=False)

        # Step 2: Hierarchical clustering
        link = linkage(condensed, method=self.linkage_method)

        # Step 3: Quasi-diagonalize
        sort_ix = self._quasi_diagonalize(link, len(cols))

        # Step 4: Recursive bisection
        cov = working.cov()
        weights_series = self._recursive_bisection(cov, sort_ix)

        # Normalize to sum to 1 (should already be close)
        total = weights_series.sum()
        if total > 0:
            weights_series /= total

        # Map back to asset names
        sorted_names = [cols[i] for i in sort_ix]
        weights_dict = {}
        for idx, w in weights_series.items():
            asset_name = cols[idx]
            weights_dict[asset_name] = round(float(w), 6)

        return HRPResult(
            weights=weights_dict,
            cluster_order=sorted_names,
            n_assets=len(weights_dict),
            method="hrp",
        )
