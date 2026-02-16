"""Tests for threshold.engine.portfolio — inverse vol, HRP, and tax modules."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from threshold.engine.portfolio.hrp import HRPAllocator
from threshold.engine.portfolio.inverse_vol import InverseVolWeighter
from threshold.engine.portfolio.tax import HIFOSelector, TaxLossHarvester

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def equal_vol_returns() -> pd.DataFrame:
    """3 assets with identical volatility (same seed, same distribution)."""
    np.random.seed(42)
    n = 250
    dates = pd.date_range("2023-01-01", periods=n, freq="B")
    data = {}
    for name in ["A", "B", "C"]:
        data[name] = np.random.normal(0.0005, 0.015, n)
    return pd.DataFrame(data, index=dates)


@pytest.fixture
def different_vol_returns() -> pd.DataFrame:
    """3 assets with very different volatilities."""
    np.random.seed(42)
    n = 250
    dates = pd.date_range("2023-01-01", periods=n, freq="B")
    return pd.DataFrame({
        "LOW_VOL": np.random.normal(0.0003, 0.005, n),   # Low vol
        "MED_VOL": np.random.normal(0.0005, 0.015, n),   # Medium vol
        "HIGH_VOL": np.random.normal(0.0008, 0.04, n),   # High vol
    }, index=dates)


@pytest.fixture
def correlated_returns() -> pd.DataFrame:
    """5 assets with realistic correlation structure for HRP testing."""
    np.random.seed(42)
    n = 300
    dates = pd.date_range("2023-01-01", periods=n, freq="B")

    # Common factor + idiosyncratic noise
    market = np.random.normal(0.0005, 0.01, n)
    data = {}
    for i, name in enumerate(["SPY", "QQQ", "GLD", "BND", "EFA"]):
        # SPY/QQQ correlated (~0.8), GLD/BND less correlated, EFA moderate
        beta = [1.0, 1.2, -0.2, -0.3, 0.7][i]
        idio_vol = [0.005, 0.008, 0.012, 0.003, 0.007][i]
        data[name] = beta * market + np.random.normal(0, idio_vol, n)

    return pd.DataFrame(data, index=dates)


@pytest.fixture
def sample_tax_lots() -> list[dict]:
    """Sample open tax lots for HIFO testing."""
    return [
        {
            "lot_id": 1,
            "account_id": "brokerage",
            "symbol": "AAPL",
            "shares": 50,
            "cost_basis_per_share": 120.00,
            "acquired_at": "2022-01-15",
            "lot_type": "buy",
            "is_open": True,
        },
        {
            "lot_id": 2,
            "account_id": "brokerage",
            "symbol": "AAPL",
            "shares": 30,
            "cost_basis_per_share": 180.00,  # Highest cost basis
            "acquired_at": "2023-06-01",
            "lot_type": "buy",
            "is_open": True,
        },
        {
            "lot_id": 3,
            "account_id": "brokerage",
            "symbol": "AAPL",
            "shares": 20,
            "cost_basis_per_share": 150.00,
            "acquired_at": "2023-01-20",
            "lot_type": "buy",
            "is_open": True,
        },
    ]


@pytest.fixture
def sample_positions() -> list[dict]:
    """Sample positions for tax-loss harvesting scan."""
    return [
        {
            "symbol": "AAPL",
            "account_id": "brokerage",
            "shares": 100,
            "cost_basis_per_share": 180.00,
            "acquired_at": "2023-06-01",
        },
        {
            "symbol": "MSFT",
            "account_id": "brokerage",
            "shares": 50,
            "cost_basis_per_share": 350.00,
            "acquired_at": "2024-01-15",
        },
        {
            "symbol": "GLD",
            "account_id": "roth",
            "shares": 200,
            "cost_basis_per_share": 170.00,
            "acquired_at": "2022-03-10",
        },
    ]


# ---------------------------------------------------------------------------
# Inverse Volatility Tests
# ---------------------------------------------------------------------------

class TestInverseVolWeighter:
    def test_equal_vol_equal_weights(self, equal_vol_returns):
        """With identical volatilities, weights should be approximately equal."""
        ivw = InverseVolWeighter(eta=1.0, window=120)
        result = ivw.compute_weights(equal_vol_returns)
        assert result["n_assets"] == 3
        assert result["method"] == "inverse_vol"
        # All weights should be close to 1/3
        for w in result["weights"].values():
            assert abs(w - 1 / 3) < 0.05  # Within 5% of equal

    def test_low_vol_gets_higher_weight(self, different_vol_returns):
        """Lower volatility assets should get higher weight."""
        ivw = InverseVolWeighter(eta=1.0, window=120)
        result = ivw.compute_weights(different_vol_returns)
        weights = result["weights"]
        assert weights["LOW_VOL"] > weights["MED_VOL"]
        assert weights["MED_VOL"] > weights["HIGH_VOL"]

    def test_weights_sum_to_one(self, different_vol_returns):
        """Weights must sum to 1.0."""
        ivw = InverseVolWeighter(eta=1.0, window=120)
        result = ivw.compute_weights(different_vol_returns)
        total = sum(result["weights"].values())
        assert abs(total - 1.0) < 1e-6

    def test_eta_effect(self, different_vol_returns):
        """Higher eta should more aggressively underweight volatile assets."""
        ivw_low = InverseVolWeighter(eta=0.5, window=120)
        ivw_high = InverseVolWeighter(eta=2.0, window=120)
        result_low = ivw_low.compute_weights(different_vol_returns)
        result_high = ivw_high.compute_weights(different_vol_returns)
        # Higher eta → LOW_VOL gets even more weight relative to HIGH_VOL
        ratio_low = result_low["weights"]["LOW_VOL"] / result_low["weights"]["HIGH_VOL"]
        ratio_high = result_high["weights"]["LOW_VOL"] / result_high["weights"]["HIGH_VOL"]
        assert ratio_high > ratio_low

    def test_empty_returns(self):
        """Empty DataFrame should return empty result."""
        ivw = InverseVolWeighter()
        result = ivw.compute_weights(pd.DataFrame())
        assert result["n_assets"] == 0
        assert result["weights"] == {}

    def test_exclude_assets(self, different_vol_returns):
        """Excluded assets should not appear in weights."""
        ivw = InverseVolWeighter(window=120)
        result = ivw.compute_weights(different_vol_returns, exclude=["HIGH_VOL"])
        assert "HIGH_VOL" not in result["weights"]
        assert result["n_assets"] == 2
        total = sum(result["weights"].values())
        assert abs(total - 1.0) < 1e-6


# ---------------------------------------------------------------------------
# HRP Tests
# ---------------------------------------------------------------------------

class TestHRPAllocator:
    def test_weights_sum_to_one(self, correlated_returns):
        """HRP weights must sum to 1.0."""
        hrp = HRPAllocator(min_periods=60)
        result = hrp.compute_weights(correlated_returns)
        total = sum(result["weights"].values())
        assert abs(total - 1.0) < 1e-4  # Rounding tolerance
        assert result["n_assets"] == 5
        assert result["method"] == "hrp"

    def test_all_weights_positive(self, correlated_returns):
        """All HRP weights should be positive (no short positions)."""
        hrp = HRPAllocator(min_periods=60)
        result = hrp.compute_weights(correlated_returns)
        for w in result["weights"].values():
            assert w > 0

    def test_cluster_order_populated(self, correlated_returns):
        """Cluster order should contain all assets."""
        hrp = HRPAllocator(min_periods=60)
        result = hrp.compute_weights(correlated_returns)
        assert len(result["cluster_order"]) == 5
        assert set(result["cluster_order"]) == {"SPY", "QQQ", "GLD", "BND", "EFA"}

    def test_insufficient_data_equal_weight(self):
        """With insufficient data, HRP should fall back to equal weight."""
        hrp = HRPAllocator(min_periods=60)
        short_data = pd.DataFrame({
            "A": np.random.normal(0, 0.01, 10),
            "B": np.random.normal(0, 0.01, 10),
        })
        result = hrp.compute_weights(short_data)
        # Should get equal weights as fallback
        for w in result["weights"].values():
            assert abs(w - 0.5) < 1e-6

    def test_single_asset(self):
        """Single asset should get weight = 1.0."""
        hrp = HRPAllocator()
        data = pd.DataFrame({"ONLY": np.random.normal(0, 0.01, 100)})
        result = hrp.compute_weights(data)
        assert result["weights"] == {"ONLY": 1.0}
        assert result["n_assets"] == 1

    def test_empty_returns(self):
        """Empty DataFrame should return empty result."""
        hrp = HRPAllocator()
        result = hrp.compute_weights(pd.DataFrame())
        assert result["n_assets"] == 0
        assert result["weights"] == {}

    def test_no_full_concentration(self, correlated_returns):
        """HRP should not put 100% in any single asset."""
        hrp = HRPAllocator(min_periods=60)
        result = hrp.compute_weights(correlated_returns)
        for w in result["weights"].values():
            # HRP may concentrate in low-vol assets by design,
            # but should never put all weight in one asset
            assert w < 0.95


# ---------------------------------------------------------------------------
# HIFO Tax Lot Tests
# ---------------------------------------------------------------------------

class TestHIFOSelector:
    def test_hifo_selects_highest_cost_first(self, sample_tax_lots):
        """HIFO should pick the lot with highest cost basis first."""
        selector = HIFOSelector()
        result = selector.select_lots(
            sample_tax_lots, shares_to_sell=20, current_price=160.0
        )
        # Lot 2 has highest cost basis ($180), should be selected first
        assert result["selected_lots"][0]["lot_id"] == 2
        assert result["total_shares"] == 20

    def test_hifo_multiple_lots(self, sample_tax_lots):
        """Selling more than one lot's worth should span multiple lots."""
        selector = HIFOSelector()
        result = selector.select_lots(
            sample_tax_lots, shares_to_sell=40, current_price=160.0
        )
        # Need 40 shares: 30 from lot 2 ($180) + 10 from lot 3 ($150)
        assert len(result["selected_lots"]) == 2
        assert result["selected_lots"][0]["lot_id"] == 2
        assert result["selected_lots"][0]["shares_to_sell"] == 30
        assert result["selected_lots"][1]["lot_id"] == 3
        assert result["selected_lots"][1]["shares_to_sell"] == 10
        assert result["total_shares"] == 40

    def test_hifo_gain_estimate(self, sample_tax_lots):
        """Estimated gain should be correct."""
        selector = HIFOSelector()
        result = selector.select_lots(
            sample_tax_lots, shares_to_sell=30, current_price=160.0
        )
        # 30 shares from lot 2 at cost $180, selling at $160
        # Gain = 30 * 160 - 30 * 180 = 4800 - 5400 = -600 (loss)
        assert result["estimated_gain"] == -600.0
        assert result["total_cost_basis"] == 5400.0

    def test_hifo_empty_lots(self):
        """Empty lot list should return empty result."""
        selector = HIFOSelector()
        result = selector.select_lots([], shares_to_sell=10, current_price=100.0)
        assert result["total_shares"] == 0
        assert result["selected_lots"] == []

    def test_holding_period_classification(self):
        """Should correctly classify short-term vs long-term."""
        selector = HIFOSelector(long_term_days=366)
        assert selector._holding_period("2022-01-01", "2024-01-01") == "long_term"
        assert selector._holding_period("2024-06-01", "2024-12-01") == "short_term"


# ---------------------------------------------------------------------------
# Tax-Loss Harvesting Tests
# ---------------------------------------------------------------------------

class TestTaxLossHarvester:
    def test_finds_losing_positions(self, sample_positions):
        """Should find positions with unrealized losses."""
        harvester = TaxLossHarvester(loss_threshold_pct=0.02)
        prices = {"AAPL": 150.0, "MSFT": 300.0, "GLD": 180.0}
        # AAPL: cost 180, price 150 → -16.7% loss
        # MSFT: cost 350, price 300 → -14.3% loss
        # GLD: cost 170, price 180 → +5.9% gain (no harvest)
        opps = harvester.scan_opportunities(sample_positions, prices)
        symbols = [o["symbol"] for o in opps]
        assert "AAPL" in symbols
        assert "MSFT" in symbols
        assert "GLD" not in symbols  # GLD has a gain

    def test_sorted_by_largest_loss(self, sample_positions):
        """Opportunities should be sorted by largest loss first."""
        harvester = TaxLossHarvester(loss_threshold_pct=0.02)
        prices = {"AAPL": 150.0, "MSFT": 300.0, "GLD": 180.0}
        opps = harvester.scan_opportunities(sample_positions, prices)
        # MSFT loss: 50 * (300-350) = -2500
        # AAPL loss: 100 * (150-180) = -3000
        # AAPL loss is larger, should be first
        assert opps[0]["symbol"] == "AAPL"
        assert opps[0]["unrealized_loss"] < opps[1]["unrealized_loss"]

    def test_wash_sale_detection(self):
        """Should detect wash sale from recent buy within 30 days."""
        harvester = TaxLossHarvester()
        trades = [
            {"symbol": "AAPL", "date": "2024-12-20", "action": "buy"},
        ]
        # Selling on 2025-01-10 → within 30 days of buy on 2024-12-20
        assert harvester.check_wash_sale("AAPL", trades, "2025-01-10") is True
        # Selling on 2025-02-20 → outside 30 days
        assert harvester.check_wash_sale("AAPL", trades, "2025-02-20") is False

    def test_wash_sale_different_symbol(self):
        """Wash sale should not trigger for different symbol."""
        harvester = TaxLossHarvester()
        trades = [
            {"symbol": "MSFT", "date": "2025-01-01", "action": "buy"},
        ]
        assert harvester.check_wash_sale("AAPL", trades, "2025-01-10") is False

    def test_threshold_filter(self, sample_positions):
        """Positions with losses below threshold should be excluded."""
        harvester = TaxLossHarvester(loss_threshold_pct=0.20)
        # AAPL loss is ~16.7%, MSFT loss is ~14.3% — both below 20%
        prices = {"AAPL": 150.0, "MSFT": 300.0, "GLD": 180.0}
        opps = harvester.scan_opportunities(sample_positions, prices)
        assert len(opps) == 0

    def test_empty_positions(self):
        """Empty positions list should return no opportunities."""
        harvester = TaxLossHarvester()
        assert harvester.scan_opportunities([], {}) == []


# ---------------------------------------------------------------------------
# Integration: Imports & Config
# ---------------------------------------------------------------------------

class TestPortfolioImports:
    def test_package_imports(self):
        from threshold.engine.portfolio import (
            HIFOSelector,
            HRPAllocator,
            InverseVolWeighter,
            TaxLossHarvester,
        )
        assert HRPAllocator is not None
        assert HIFOSelector is not None
        assert InverseVolWeighter is not None
        assert TaxLossHarvester is not None


class TestConfigHasPortfolio:
    def test_config_portfolio_section(self):
        from threshold.config.schema import ThresholdConfig
        config = ThresholdConfig()
        assert hasattr(config, "portfolio_construction")
        assert config.portfolio_construction.inverse_vol.enabled is False
        assert config.portfolio_construction.hrp.enabled is False
        assert config.portfolio_construction.tax.enabled is False
        # Check default values preserved
        assert config.portfolio_construction.inverse_vol.eta == 1.0
        assert config.portfolio_construction.inverse_vol.window == 120
        assert config.portfolio_construction.hrp.linkage_method == "single"
        assert config.portfolio_construction.tax.lot_method == "HIFO"
        assert config.portfolio_construction.tax.loss_threshold_pct == 0.02
        assert config.portfolio_construction.tax.wash_sale_window_days == 30
        assert config.portfolio_construction.tax.long_term_days == 366
