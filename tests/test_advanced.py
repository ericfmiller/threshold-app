"""Tests for threshold.engine.advanced — 3 advanced signal modules."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from threshold.engine.advanced.trend_following import ContinuousTrendFollower, TrendSignal
from threshold.engine.advanced.factor_momentum import FactorMomentumSignal, FactorMomentumResult


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def uptrend_close() -> pd.Series:
    """Strong uptrend: 300 bars of monotonically increasing prices."""
    np.random.seed(42)
    n = 300
    drift = np.linspace(100, 200, n)
    noise = np.random.normal(0, 0.5, n)
    return pd.Series(drift + noise, index=pd.date_range("2023-01-01", periods=n, freq="B"))


@pytest.fixture
def downtrend_close() -> pd.Series:
    """Strong downtrend: 300 bars of monotonically decreasing prices."""
    np.random.seed(42)
    n = 300
    drift = np.linspace(200, 80, n)
    noise = np.random.normal(0, 0.5, n)
    return pd.Series(drift + noise, index=pd.date_range("2023-01-01", periods=n, freq="B"))


@pytest.fixture
def flat_close() -> pd.Series:
    """Flat/sideways market: 300 bars around 100."""
    np.random.seed(42)
    n = 300
    noise = np.random.normal(0, 1.0, n)
    return pd.Series(100 + noise, index=pd.date_range("2023-01-01", periods=n, freq="B"))


@pytest.fixture
def positive_factor_returns() -> pd.DataFrame:
    """All factors with positive 12-month returns."""
    np.random.seed(42)
    n = 24  # 2 years monthly
    factors = {
        "SMB": np.random.normal(0.01, 0.01, n),
        "HML": np.random.normal(0.01, 0.01, n),
        "RMW": np.random.normal(0.01, 0.01, n),
        "CMA": np.random.normal(0.01, 0.01, n),
        "BAB": np.random.normal(0.01, 0.01, n),
    }
    return pd.DataFrame(factors, index=pd.date_range("2023-01-01", periods=n, freq="ME"))


@pytest.fixture
def mixed_factor_returns() -> pd.DataFrame:
    """Mix of positive and negative factor returns."""
    np.random.seed(42)
    n = 24
    factors = {
        "SMB": np.random.normal(0.005, 0.02, n),   # Positive
        "HML": np.random.normal(-0.005, 0.02, n),  # Negative
        "RMW": np.random.normal(0.003, 0.02, n),   # Positive
        "CMA": np.random.normal(-0.003, 0.02, n),  # Negative
        "BAB": np.random.normal(0.001, 0.02, n),   # Slightly positive
    }
    return pd.DataFrame(factors, index=pd.date_range("2023-01-01", periods=n, freq="ME"))


@pytest.fixture
def sentiment_proxies() -> pd.DataFrame:
    """Overheated sentiment: all proxies trending high."""
    np.random.seed(42)
    n = 120  # 10 years monthly
    # All proxies trend upward → high sentiment
    data = {}
    for name in ["cef_discount", "ipo_volume", "equity_share", "vix_inverted"]:
        trend = np.linspace(0, 3, n)
        noise = np.random.normal(0, 0.2, n)
        data[name] = trend + noise
    return pd.DataFrame(data, index=pd.date_range("2015-01-01", periods=n, freq="ME"))


@pytest.fixture
def depressed_proxies() -> pd.DataFrame:
    """Depressed sentiment: all proxies trending low."""
    np.random.seed(42)
    n = 120
    data = {}
    for name in ["cef_discount", "ipo_volume", "equity_share", "vix_inverted"]:
        trend = np.linspace(3, -1, n)
        noise = np.random.normal(0, 0.2, n)
        data[name] = trend + noise
    return pd.DataFrame(data, index=pd.date_range("2015-01-01", periods=n, freq="ME"))


# ---------------------------------------------------------------------------
# Trend Following Tests
# ---------------------------------------------------------------------------

class TestContinuousTrendFollower:
    def test_uptrend_positive_signal(self, uptrend_close):
        tf = ContinuousTrendFollower(window=252)
        signal = tf.compute_signal(uptrend_close)
        assert signal is not None
        assert signal["signal"] > 0
        assert signal["regime"] in ("STRONG_UP", "UP")

    def test_downtrend_negative_signal(self, downtrend_close):
        tf = ContinuousTrendFollower(window=252)
        signal = tf.compute_signal(downtrend_close)
        assert signal is not None
        assert signal["signal"] < 0
        assert signal["regime"] in ("STRONG_DOWN", "DOWN")

    def test_signal_clamped_to_range(self, uptrend_close):
        tf = ContinuousTrendFollower(window=252)
        signal = tf.compute_signal(uptrend_close)
        assert signal is not None
        assert -1.0 <= signal["signal"] <= 1.0

    def test_yang_zhang_vol_positive(self, uptrend_close):
        tf = ContinuousTrendFollower(vol_window=60)
        vol = tf.yang_zhang_vol_from_close(uptrend_close, 60)
        assert vol > 0

    def test_insufficient_data(self):
        tf = ContinuousTrendFollower(window=252)
        short_series = pd.Series([100, 101, 102], index=pd.date_range("2024-01-01", periods=3))
        result = tf.compute_signal(short_series)
        assert result is None

    def test_flat_market_near_zero(self, flat_close):
        tf = ContinuousTrendFollower(window=252)
        signal = tf.compute_signal(flat_close)
        assert signal is not None
        # Flat market should have signal near zero
        assert abs(signal["signal"]) < 0.5
        assert signal["regime"] in ("FLAT", "UP", "DOWN")  # Could be slightly off zero

    def test_regime_classification(self):
        tf = ContinuousTrendFollower()
        assert tf._classify_regime(0.8) == "STRONG_UP"
        assert tf._classify_regime(0.3) == "UP"
        assert tf._classify_regime(0.0) == "FLAT"
        assert tf._classify_regime(-0.3) == "DOWN"
        assert tf._classify_regime(-0.8) == "STRONG_DOWN"


# ---------------------------------------------------------------------------
# Factor Momentum Tests
# ---------------------------------------------------------------------------

class TestFactorMomentumSignal:
    def test_all_positive_factors(self, positive_factor_returns):
        fms = FactorMomentumSignal(lookback_months=12)
        result = fms.compute_signal(positive_factor_returns)
        assert result["regime"] == "BROAD_POSITIVE"
        assert result["breadth"] > 0.5
        assert len(result["long_factors"]) > len(result["short_factors"])
        assert result["n_factors"] == 5

    def test_mixed_factors(self, mixed_factor_returns):
        fms = FactorMomentumSignal(lookback_months=12)
        result = fms.compute_signal(mixed_factor_returns)
        assert 0 < result["breadth"] < 1
        assert result["n_factors"] == 5

    def test_empty_dataframe(self):
        fms = FactorMomentumSignal()
        result = fms.compute_signal(pd.DataFrame())
        assert result["regime"] == "UNAVAILABLE"
        assert result["n_factors"] == 0

    def test_single_factor_unavailable(self):
        """Need at least 2 factors."""
        fms = FactorMomentumSignal()
        df = pd.DataFrame({"SMB": [0.01, 0.02, 0.03]})
        result = fms.compute_signal(df)
        assert result["regime"] == "UNAVAILABLE"

    def test_lookback_respected(self, positive_factor_returns):
        # With lookback=6, only uses last 6 months
        fms = FactorMomentumSignal(lookback_months=6)
        result = fms.compute_signal(positive_factor_returns)
        assert result["n_factors"] == 5  # Still 5 factors, just shorter lookback

    def test_proxy_factors(self):
        """Test the proxy factor computation from ETF returns."""
        np.random.seed(42)
        n = 300
        dates = pd.date_range("2023-01-01", periods=n, freq="B")
        etf_data = pd.DataFrame({
            "SPY": np.random.normal(0.0003, 0.01, n),
            "EFA": np.random.normal(0.0002, 0.012, n),
            "GLD": np.random.normal(0.0001, 0.008, n),
            "BND": np.random.normal(0.00005, 0.004, n),
            "GSG": np.random.normal(0.00015, 0.015, n),
        }, index=dates)

        proxies = FactorMomentumSignal.compute_proxy_factors(etf_data)
        assert not proxies.empty
        assert "equity_risk" in proxies.columns
        assert "value_proxy" in proxies.columns
        assert "safe_haven" in proxies.columns


# ---------------------------------------------------------------------------
# Sentiment Tests
# ---------------------------------------------------------------------------

class TestAlignedSentimentIndex:
    def test_high_sentiment_overheated(self, sentiment_proxies):
        from threshold.engine.advanced.sentiment import AlignedSentimentIndex
        asi = AlignedSentimentIndex(
            overheated_pctl=0.80,
            depressed_pctl=0.20,
            min_observations=30,
        )
        result = asi.compute(sentiment_proxies)
        # With strongly upward-trending proxies, last value should be high
        assert result["regime"] == "OVERHEATED"
        assert result["mr_adjustment"] == 0.15
        assert result["percentile"] is not None
        assert result["percentile"] >= 0.80

    def test_low_sentiment_depressed(self, depressed_proxies):
        from threshold.engine.advanced.sentiment import AlignedSentimentIndex
        asi = AlignedSentimentIndex(
            overheated_pctl=0.80,
            depressed_pctl=0.20,
            min_observations=30,
        )
        result = asi.compute(depressed_proxies)
        assert result["regime"] == "DEPRESSED"
        assert result["mr_adjustment"] == 0.0  # No adjustment when depressed
        assert result["percentile"] is not None
        assert result["percentile"] <= 0.20

    def test_neutral_no_adjustment(self):
        from threshold.engine.advanced.sentiment import AlignedSentimentIndex
        np.random.seed(42)
        n = 120
        # Random walk proxies — neutral sentiment
        data = {}
        for name in ["p1", "p2", "p3", "p4"]:
            data[name] = np.random.normal(0, 1, n)
        proxies = pd.DataFrame(data, index=pd.date_range("2015-01-01", periods=n, freq="ME"))

        asi = AlignedSentimentIndex(min_observations=30)
        result = asi.compute(proxies)
        # With random noise, the last observation is unlikely to be extreme
        # The mr_adjustment should be 0 unless regime is OVERHEATED
        if result["regime"] == "NEUTRAL":
            assert result["mr_adjustment"] == 0.0

    def test_insufficient_data(self):
        from threshold.engine.advanced.sentiment import AlignedSentimentIndex
        asi = AlignedSentimentIndex(min_observations=60)
        small_df = pd.DataFrame({"p1": [1, 2, 3], "p2": [4, 5, 6]})
        result = asi.compute(small_df)
        assert result["regime"] == "UNAVAILABLE"
        assert result["mr_adjustment"] == 0.0


# ---------------------------------------------------------------------------
# Integration: Imports & Config
# ---------------------------------------------------------------------------

class TestAdvancedImports:
    def test_package_imports(self):
        from threshold.engine.advanced import (
            ContinuousTrendFollower,
            FactorMomentumSignal,
        )
        assert ContinuousTrendFollower is not None
        assert FactorMomentumSignal is not None

    def test_sentiment_importable(self):
        """Sentiment should be importable directly even if sklearn unavailable."""
        from threshold.engine.advanced.sentiment import AlignedSentimentIndex
        assert AlignedSentimentIndex is not None


class TestConfigHasAdvanced:
    def test_config_advanced_section(self):
        from threshold.config.schema import ThresholdConfig
        config = ThresholdConfig()
        assert hasattr(config, "advanced")
        assert config.advanced.trend_following.enabled is False
        assert config.advanced.factor_momentum.enabled is False
        assert config.advanced.sentiment.enabled is False
        # Check default values preserved
        assert config.advanced.trend_following.window == 252
        assert config.advanced.trend_following.mq_blend_weight == 0.20
        assert config.advanced.sentiment.mr_reduction == 0.15
        assert config.advanced.factor_momentum.lookback_months == 12
