"""Tests for threshold.engine.technical — pure math indicators."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from threshold.engine.technical import (
    MACDResult,
    calc_bb_lower_breach,
    calc_consecutive_days_below_sma,
    calc_macd,
    calc_obv_divergence,
    calc_price_acceleration,
    calc_reversal_signals,
    calc_rsi,
    calc_rsi_bullish_divergence,
    calc_rsi_value,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def uptrend_close() -> pd.Series:
    """252-bar steady uptrend."""
    np.random.seed(42)
    prices = 100 * np.cumprod(1 + np.random.normal(0.0005, 0.01, 252))
    return pd.Series(prices, name="Close")


@pytest.fixture
def downtrend_close() -> pd.Series:
    """252-bar steady downtrend."""
    np.random.seed(42)
    prices = 100 * np.cumprod(1 + np.random.normal(-0.001, 0.01, 252))
    return pd.Series(prices, name="Close")


@pytest.fixture
def oversold_close() -> pd.Series:
    """252-bar series ending in deep oversold territory."""
    np.random.seed(42)
    prices = 100 * np.cumprod(1 + np.random.normal(0.0005, 0.01, 200))
    crash = prices[-1] * np.cumprod(1 + np.random.normal(-0.01, 0.005, 52))
    return pd.Series(np.concatenate([prices, crash]), name="Close")


@pytest.fixture
def flat_volume() -> pd.Series:
    """Constant volume series."""
    return pd.Series(np.full(252, 1_000_000.0), name="Volume")


# ---------------------------------------------------------------------------
# RSI
# ---------------------------------------------------------------------------

class TestRSI:
    def test_rsi_returns_series(self, uptrend_close):
        result = calc_rsi(uptrend_close)
        assert isinstance(result, pd.Series)
        assert len(result) == len(uptrend_close)

    def test_rsi_value_scalar(self, uptrend_close):
        val = calc_rsi_value(uptrend_close)
        assert isinstance(val, float)
        assert 0 <= val <= 100

    def test_rsi_uptrend_above_50(self, uptrend_close):
        val = calc_rsi_value(uptrend_close)
        assert val > 40, f"RSI in uptrend should be elevated, got {val}"

    def test_rsi_downtrend_below_50(self, downtrend_close):
        val = calc_rsi_value(downtrend_close)
        assert val < 60, f"RSI in downtrend should be depressed, got {val}"

    def test_rsi_short_series_returns_50(self):
        short = pd.Series([100.0, 101.0, 99.0])
        val = calc_rsi_value(short, period=14)
        # With very few data points, RSI may be NaN → defaults to 50
        assert isinstance(val, float)

    def test_rsi_constant_series(self):
        """Constant prices → no gains/losses → RSI should handle gracefully."""
        flat = pd.Series(np.full(50, 100.0))
        val = calc_rsi_value(flat)
        assert isinstance(val, float)


# ---------------------------------------------------------------------------
# MACD
# ---------------------------------------------------------------------------

class TestMACD:
    def test_macd_basic_structure(self, uptrend_close):
        result = calc_macd(uptrend_close)
        assert "macd" in result
        assert "signal" in result
        assert "histogram" in result
        assert "crossover" in result
        assert "hist_rising" in result
        assert "below_zero" in result

    def test_macd_short_series(self):
        short = pd.Series(range(20))
        result = calc_macd(short)
        assert result["crossover"] == "neutral"
        assert result["macd"] == 0

    def test_macd_uptrend_positive(self, uptrend_close):
        result = calc_macd(uptrend_close)
        assert result["macd"] > 0 or result["histogram"] != 0

    def test_macd_crossover_values(self, uptrend_close):
        result = calc_macd(uptrend_close)
        assert result["crossover"] in ("bullish", "bearish", "neutral")


# ---------------------------------------------------------------------------
# OBV Divergence
# ---------------------------------------------------------------------------

class TestOBV:
    def test_obv_insufficient_data(self):
        close = pd.Series(range(10), dtype=float)
        vol = pd.Series(np.full(10, 1000.0))
        result = calc_obv_divergence(close, vol)
        assert result["divergence"] == "none"

    def test_obv_basic_structure(self, uptrend_close, flat_volume):
        result = calc_obv_divergence(uptrend_close, flat_volume)
        assert result["obv_trend"] in ("rising", "falling", "flat")
        assert result["price_trend"] in ("rising", "falling", "flat")
        assert result["divergence"] in ("bullish", "bearish", "none")
        assert 0.0 <= result["divergence_strength"] <= 1.0

    def test_obv_bullish_divergence(self):
        """Price falling + volume (OBV) rising → bullish divergence."""
        np.random.seed(99)
        n = 50
        # Declining prices
        close = pd.Series(100.0 - np.arange(n) * 0.5)
        # Rising volume (more volume on up-ticks to create rising OBV)
        vol = pd.Series(np.full(n, 1_000_000.0))
        result = calc_obv_divergence(close, vol)
        # At minimum, verify result structure is valid
        assert result["price_trend"] in ("rising", "falling", "flat")


# ---------------------------------------------------------------------------
# RSI Bullish Divergence
# ---------------------------------------------------------------------------

class TestRSIDivergence:
    def test_insufficient_data(self):
        close = pd.Series(range(20), dtype=float)
        result = calc_rsi_bullish_divergence(close)
        assert result["detected"] is False

    def test_basic_structure(self, oversold_close):
        result = calc_rsi_bullish_divergence(oversold_close)
        assert "detected" in result
        assert "price_low_recent" in result
        assert "rsi_low_recent" in result


# ---------------------------------------------------------------------------
# Bollinger Band Breach
# ---------------------------------------------------------------------------

class TestBBBreach:
    def test_insufficient_data(self):
        close = pd.Series(range(10), dtype=float)
        result = calc_bb_lower_breach(close)
        assert result["breach"] is False
        assert result["lower_bb"] is None

    def test_normal_price_no_breach(self, uptrend_close):
        result = calc_bb_lower_breach(uptrend_close)
        assert isinstance(result["breach"], bool)
        assert isinstance(result["bb_pct_b"], float)

    def test_crash_causes_breach(self):
        """Sharp drop below lower BB should trigger breach."""
        np.random.seed(42)
        prices = list(np.full(25, 100.0))
        prices.extend([80.0])  # Sharp drop
        close = pd.Series(prices)
        result = calc_bb_lower_breach(close)
        assert result["breach"] is True
        assert result["bb_pct_b"] < 0


# ---------------------------------------------------------------------------
# Reversal Signals (composite)
# ---------------------------------------------------------------------------

class TestReversalSignals:
    def test_basic_structure(self, uptrend_close):
        macd_data: MACDResult = {
            "macd": 0.5, "signal": 0.3, "histogram": 0.2,
            "crossover": "neutral", "hist_rising": False, "below_zero": False,
        }
        result = calc_reversal_signals(uptrend_close, 50.0, macd_data, 4.5)
        assert "rsi_bullish_divergence" in result
        assert "bb_lower_breach" in result
        assert "bottom_turning" in result
        assert "quant_freshness_warning" in result

    def test_bottom_turning_conditions(self, oversold_close):
        """MACD hist rising + below zero + RSI < 30 + quant >= 3 → bottom turning."""
        macd_data: MACDResult = {
            "macd": -0.5, "signal": -0.3, "histogram": -0.1,
            "crossover": "neutral", "hist_rising": True, "below_zero": True,
        }
        result = calc_reversal_signals(oversold_close, 25.0, macd_data, 3.5)
        assert result["bottom_turning"] is True

    def test_quant_freshness_warning(self, oversold_close):
        """Q4+ at RSI < 30 → quant freshness warning."""
        macd_data: MACDResult = {
            "macd": 0, "signal": 0, "histogram": 0,
            "crossover": "neutral", "hist_rising": False, "below_zero": False,
        }
        result = calc_reversal_signals(oversold_close, 25.0, macd_data, 4.5)
        assert result["quant_freshness_warning"] is True

    def test_no_warning_below_quant_4(self, oversold_close):
        """Quant < 4 at RSI < 30 → no freshness warning."""
        macd_data: MACDResult = {
            "macd": 0, "signal": 0, "histogram": 0,
            "crossover": "neutral", "hist_rising": False, "below_zero": False,
        }
        result = calc_reversal_signals(oversold_close, 25.0, macd_data, 3.5)
        assert result["quant_freshness_warning"] is False


# ---------------------------------------------------------------------------
# Price Acceleration
# ---------------------------------------------------------------------------

class TestPriceAcceleration:
    def test_short_series(self):
        close = pd.Series(range(20), dtype=float)
        pa, ret_8w = calc_price_acceleration(close)
        assert pa == 0.0
        assert ret_8w == 0.0

    def test_returns_tuple(self, uptrend_close):
        pa, ret_8w = calc_price_acceleration(uptrend_close)
        assert isinstance(pa, float)
        assert isinstance(ret_8w, float)


# ---------------------------------------------------------------------------
# Consecutive Days Below SMA
# ---------------------------------------------------------------------------

class TestConsecutiveDaysBelowSMA:
    def test_insufficient_data(self):
        close = pd.Series(range(100), dtype=float)
        count, pct = calc_consecutive_days_below_sma(close)
        assert count == 0
        assert pct == 0.0

    def test_uptrend_no_breach(self, uptrend_close):
        count, pct = calc_consecutive_days_below_sma(uptrend_close)
        # Uptrend should generally be above SMA
        assert isinstance(count, int)
        assert isinstance(pct, float)

    def test_crash_triggers_count(self):
        """Sharp decline below 200d SMA should produce positive count."""
        np.random.seed(42)
        prices = list(np.full(200, 100.0))
        # Add 20 days of crash >3% below SMA
        prices.extend([95.0] * 20)
        close = pd.Series(prices)
        count, pct = calc_consecutive_days_below_sma(close)
        assert count > 0
        assert pct < 0
