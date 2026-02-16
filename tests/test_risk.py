"""Tests for threshold.engine.risk — 5 risk framework modules."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from threshold.engine.risk.cdar import CDaRCalculator
from threshold.engine.risk.cvar import CVaRCalculator
from threshold.engine.risk.ebp import EBPMonitor
from threshold.engine.risk.momentum_crash import MomentumCrashProtection
from threshold.engine.risk.turbulence import TurbulenceIndex, TurbulenceSignal

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def ebp_high_risk() -> pd.Series:
    """EBP series ending in high-risk territory (>100bp)."""
    np.random.seed(42)
    values = np.random.normal(0.3, 0.2, 60)
    values[-3:] = [1.10, 1.20, 1.30]  # Spike into high-risk
    return pd.Series(values, index=pd.date_range("2020-01-01", periods=60, freq="MS"))


@pytest.fixture
def ebp_accommodative() -> pd.Series:
    """EBP series in accommodative territory (<0)."""
    np.random.seed(42)
    values = np.random.normal(-0.3, 0.1, 60)
    return pd.Series(values, index=pd.date_range("2020-01-01", periods=60, freq="MS"))


@pytest.fixture
def calm_prices() -> pd.DataFrame:
    """Multi-asset price data with low volatility (calm market)."""
    np.random.seed(42)
    n = 300
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    data = {}
    for name in ["SPY", "EFA", "GLD", "BND"]:
        drift = 0.0003
        vol = 0.005
        returns = np.random.normal(drift, vol, n)
        prices = 100 * np.cumprod(1 + returns)
        data[name] = prices
    return pd.DataFrame(data, index=dates)


@pytest.fixture
def stressed_prices() -> pd.DataFrame:
    """Multi-asset price data with high volatility + correlation breakdown."""
    np.random.seed(42)
    n = 300
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    data = {}
    for name in ["SPY", "EFA", "GLD", "BND"]:
        drift = -0.001
        vol = 0.025
        returns = np.random.normal(drift, vol, n)
        # Add correlated shock at end
        returns[-5:] = np.random.normal(-0.03, 0.04, 5)
        prices = 100 * np.cumprod(1 + returns)
        data[name] = prices
    return pd.DataFrame(data, index=dates)


@pytest.fixture
def bull_returns() -> pd.Series:
    """Monthly market returns in a bull market (positive 24m cumulative)."""
    np.random.seed(42)
    # Ensure strongly positive returns so cumulative 24m is reliably positive
    return pd.Series(np.random.normal(0.03, 0.02, 36))


@pytest.fixture
def bear_returns() -> pd.Series:
    """Monthly market returns in a bear market (negative 24m cumulative)."""
    np.random.seed(42)
    returns = np.random.normal(-0.02, 0.04, 36)
    return pd.Series(returns)


@pytest.fixture
def daily_returns_positive() -> pd.Series:
    """Daily returns with positive drift (mild drawdowns)."""
    np.random.seed(42)
    return pd.Series(np.random.normal(0.0005, 0.01, 500))


@pytest.fixture
def daily_returns_crash() -> pd.Series:
    """Daily returns with a crash (large drawdown)."""
    np.random.seed(42)
    returns = np.random.normal(0.0003, 0.01, 500)
    # Insert crash
    returns[200:210] = -0.05
    returns[210:215] = -0.03
    return pd.Series(returns)


# ---------------------------------------------------------------------------
# EBP Tests
# ---------------------------------------------------------------------------

class TestEBPMonitor:
    def test_no_data(self):
        monitor = EBPMonitor()
        signal = monitor.get_current_signal()
        assert signal["ebp_regime"] == "UNAVAILABLE"
        assert signal["ebp_value"] is None

    def test_high_risk_regime(self, ebp_high_risk):
        monitor = EBPMonitor()
        monitor.load_data(ebp_high_risk)
        signal = monitor.get_current_signal()
        assert signal["ebp_regime"] == "HIGH_RISK"
        assert signal["ebp_value"] is not None
        assert signal["ebp_value"] >= 1.0

    def test_accommodative_regime(self, ebp_accommodative):
        monitor = EBPMonitor()
        monitor.load_data(ebp_accommodative)
        signal = monitor.get_current_signal()
        assert signal["ebp_regime"] == "ACCOMMODATIVE"
        assert signal["ebp_value"] < 0

    def test_regime_score(self, ebp_high_risk):
        monitor = EBPMonitor()
        monitor.load_data(ebp_high_risk)
        score = monitor.get_regime_score()
        assert score == 1.0  # HIGH_RISK maps to 1.0

    def test_trend_detection(self, ebp_high_risk):
        monitor = EBPMonitor()
        monitor.load_data(ebp_high_risk)
        signal = monitor.get_current_signal()
        assert signal["ebp_trend"] == "rising"
        assert signal["ebp_3m_change"] is not None
        assert signal["ebp_3m_change"] > 0

    def test_custom_thresholds(self):
        monitor = EBPMonitor(thresholds={
            "high_risk": 0.50,
            "elevated": 0.20,
            "normal": 0.00,
        })
        data = pd.Series([0.1, 0.2, 0.3], index=pd.date_range("2024-01-01", periods=3, freq="MS"))
        monitor.load_data(data)
        signal = monitor.get_current_signal()
        assert signal["ebp_regime"] == "ELEVATED"


# ---------------------------------------------------------------------------
# Turbulence Tests
# ---------------------------------------------------------------------------

class TestTurbulenceIndex:
    def test_insufficient_data(self):
        ti = TurbulenceIndex()
        small_df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6], "C": [7, 8, 9]})
        signal = ti.compute(small_df)
        assert signal["turbulence_regime"] == "UNAVAILABLE"

    def test_calm_market(self, calm_prices):
        ti = TurbulenceIndex(window=100, threshold_pctl=0.75)
        signal = ti.compute(calm_prices)
        assert signal["turbulence_value"] is not None
        assert signal["turbulence_percentile"] is not None
        assert signal["turbulence_regime"] in ("CALM", "ELEVATED", "TURBULENT")

    def test_too_few_assets(self):
        ti = TurbulenceIndex(min_assets=3)
        df = pd.DataFrame({"A": range(300), "B": range(300)})
        signal = ti.compute(df)
        assert signal["turbulence_regime"] == "UNAVAILABLE"

    def test_regime_score(self):
        ti = TurbulenceIndex()
        calm_signal = TurbulenceSignal(
            turbulence_value=1.0, turbulence_percentile=0.3,
            is_turbulent=False, turbulence_regime="CALM", rolling_mean=1.0,
        )
        assert ti.get_regime_score(calm_signal) == 0.0

        turb_signal = TurbulenceSignal(
            turbulence_value=10.0, turbulence_percentile=0.95,
            is_turbulent=True, turbulence_regime="TURBULENT", rolling_mean=8.0,
        )
        assert ti.get_regime_score(turb_signal) == 1.0


# ---------------------------------------------------------------------------
# Momentum Crash Tests
# ---------------------------------------------------------------------------

class TestMomentumCrash:
    def test_insufficient_data(self):
        mcp = MomentumCrashProtection()
        signal = mcp.compute_dynamic_weight(pd.Series([0.01, 0.02]))
        assert signal["regime"] == "UNAVAILABLE"
        assert signal["momentum_weight"] == 1.0

    def test_bull_market(self, bull_returns):
        mcp = MomentumCrashProtection(lookback_months=24)
        signal = mcp.compute_dynamic_weight(bull_returns)
        assert signal["is_bear_market"] is False
        assert signal["bear_indicator"] == 0.0
        assert signal["momentum_weight"] == 1.0
        assert signal["regime"] == "NORMAL"

    def test_bear_market(self, bear_returns):
        mcp = MomentumCrashProtection(lookback_months=24)
        signal = mcp.compute_dynamic_weight(bear_returns)
        assert signal["is_bear_market"] is True
        assert signal["bear_indicator"] == 1.0
        assert signal["momentum_weight"] < 1.0
        assert signal["momentum_weight"] >= 0.25  # min_weight floor

    def test_regime_score(self, bear_returns):
        mcp = MomentumCrashProtection(lookback_months=24)
        signal = mcp.compute_dynamic_weight(bear_returns)
        score = mcp.get_regime_score(signal)
        assert 0.0 <= score <= 1.0

    def test_min_weight_floor(self):
        mcp = MomentumCrashProtection(lookback_months=6, min_weight=0.30)
        # Very bearish: large negative returns
        returns = pd.Series([-0.05] * 12)
        signal = mcp.compute_dynamic_weight(returns)
        assert signal["momentum_weight"] >= 0.30


# ---------------------------------------------------------------------------
# CVaR Tests
# ---------------------------------------------------------------------------

class TestCVaR:
    def test_insufficient_data(self):
        calc = CVaRCalculator()
        result = calc.compute(pd.Series([0.01, -0.01]))
        assert result["cvar"] == 0.0
        assert result["n_observations"] == 2

    def test_historical_cvar(self, daily_returns_positive):
        calc = CVaRCalculator(alpha=0.95, method="historical")
        result = calc.compute(daily_returns_positive)
        assert result["cvar"] > 0  # Positive = loss magnitude
        assert result["var"] > 0
        assert result["method"] == "historical"
        assert result["n_observations"] == 500

    def test_parametric_cvar(self, daily_returns_crash):
        calc = CVaRCalculator(alpha=0.95, method="parametric")
        result = calc.compute(daily_returns_crash)
        assert result["cvar"] > 0
        assert result["method"] == "parametric"

    def test_crash_increases_cvar(self, daily_returns_positive, daily_returns_crash):
        calc = CVaRCalculator(alpha=0.95)
        calm_result = calc.compute(daily_returns_positive)
        crash_result = calc.compute(daily_returns_crash)
        assert crash_result["cvar"] > calm_result["cvar"]

    def test_invalid_alpha(self):
        with pytest.raises(ValueError, match="alpha must be"):
            CVaRCalculator(alpha=0.3)
        with pytest.raises(ValueError, match="alpha must be"):
            CVaRCalculator(alpha=1.0)

    def test_invalid_method(self):
        with pytest.raises(ValueError, match="method must be"):
            CVaRCalculator(method="monte_carlo")

    def test_var_less_than_cvar(self, daily_returns_positive):
        """CVaR should be >= VaR (expected shortfall beyond VaR)."""
        calc = CVaRCalculator(alpha=0.95, method="historical")
        result = calc.compute(daily_returns_positive)
        assert result["cvar"] >= result["var"]


# ---------------------------------------------------------------------------
# CDaR Tests
# ---------------------------------------------------------------------------

class TestCDaR:
    def test_insufficient_data(self):
        calc = CDaRCalculator()
        result = calc.compute(pd.Series([0.01, -0.01]))
        assert result["cdar"] == 0.0
        assert result["n_observations"] == 2

    def test_basic_computation(self, daily_returns_positive):
        calc = CDaRCalculator(alpha=0.95)
        result = calc.compute(daily_returns_positive)
        assert result["cdar"] > 0
        assert result["dar"] > 0
        assert result["max_drawdown"] > 0
        assert result["avg_drawdown"] >= 0
        assert result["n_observations"] == 500

    def test_crash_increases_cdar(self, daily_returns_positive, daily_returns_crash):
        calc = CDaRCalculator(alpha=0.95)
        calm_result = calc.compute(daily_returns_positive)
        crash_result = calc.compute(daily_returns_crash)
        assert crash_result["cdar"] > calm_result["cdar"]
        assert crash_result["max_drawdown"] > calm_result["max_drawdown"]

    def test_cdar_gte_dar(self, daily_returns_positive):
        """CDaR should be >= DaR (expected tail drawdown beyond threshold)."""
        calc = CDaRCalculator(alpha=0.95)
        result = calc.compute(daily_returns_positive)
        assert result["cdar"] >= result["dar"]

    def test_max_dd_gte_cdar(self, daily_returns_positive):
        """Max drawdown should be >= CDaR."""
        calc = CDaRCalculator(alpha=0.95)
        result = calc.compute(daily_returns_positive)
        assert result["max_drawdown"] >= result["cdar"]

    def test_drawdown_series(self, daily_returns_positive):
        calc = CDaRCalculator()
        drawdowns = calc.compute_drawdowns(daily_returns_positive)
        assert len(drawdowns) == len(daily_returns_positive)
        assert np.all(drawdowns >= 0)  # Drawdowns are non-negative
        assert np.all(drawdowns <= 1)  # Can't lose more than 100%

    def test_invalid_alpha(self):
        with pytest.raises(ValueError, match="alpha must be"):
            CDaRCalculator(alpha=0.0)
        with pytest.raises(ValueError, match="alpha must be"):
            CDaRCalculator(alpha=1.0)


# ---------------------------------------------------------------------------
# Integration: Module imports
# ---------------------------------------------------------------------------

class TestRiskImports:
    def test_package_imports(self):
        from threshold.engine.risk import (
            CDaRCalculator,
            CVaRCalculator,
            EBPMonitor,
            MomentumCrashProtection,
            TurbulenceIndex,
        )
        assert all([
            CDaRCalculator, CVaRCalculator, EBPMonitor,
            MomentumCrashProtection, TurbulenceIndex,
        ])

    def test_config_has_risk(self):
        from threshold.config.schema import ThresholdConfig
        config = ThresholdConfig()
        assert hasattr(config, "risk")
        assert config.risk.ebp.enabled is False
        assert config.risk.turbulence.enabled is False
        assert config.risk.momentum_crash.enabled is False
        assert config.risk.cvar.enabled is False
        assert config.risk.cdar.enabled is False
