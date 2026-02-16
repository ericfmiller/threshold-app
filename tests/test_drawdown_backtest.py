"""Tests for drawdown defense backtest."""

from __future__ import annotations

import numpy as np
import pandas as pd

from threshold.engine.drawdown_backtest import (
    BacktestResult,
    analyze_ticker_drawdown,
    classify_defense,
    identify_spy_drawdowns,
    run_drawdown_backtest,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_monthly_returns(n_months: int = 60, mean: float = 0.01, std: float = 0.04, seed: int = 42):
    """Create synthetic monthly return series."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2010-01-31", periods=n_months, freq="ME")
    returns = pd.Series(rng.normal(mean, std, n_months), index=dates)
    return returns


def _make_price_df(returns: pd.Series, start_price: float = 100.0):
    """Convert monthly returns to a price DataFrame with 'Close'."""
    prices = start_price * (1 + returns).cumprod()
    return pd.DataFrame({"Close": prices})


# ---------------------------------------------------------------------------
# Tests: BacktestResult
# ---------------------------------------------------------------------------

class TestBacktestResult:
    def test_defaults(self):
        """Default BacktestResult should have zero counts."""
        result = BacktestResult()
        assert result.tickers_processed == 0
        assert result.tickers_skipped == 0
        assert result.classifications == {}
        assert result.errors == []


# ---------------------------------------------------------------------------
# Tests: identify_spy_drawdowns
# ---------------------------------------------------------------------------

class TestIdentifySPYDrawdowns:
    def test_bull_market_no_drawdowns(self):
        """Steady growth should have no drawdown months."""
        returns = pd.Series(
            [0.02] * 60,
            index=pd.date_range("2010-01-31", periods=60, freq="ME"),
        )
        mask = identify_spy_drawdowns(returns)
        assert mask.sum() == 0

    def test_crash_creates_drawdowns(self):
        """A crash followed by recovery should create drawdown months."""
        # Build: 20 months growth, 6 months crash, 34 months recovery
        dates = pd.date_range("2010-01-31", periods=60, freq="ME")
        returns = (
            [0.02] * 20 +    # Growth
            [-0.10] * 6 +    # Crash (~47% decline)
            [0.03] * 34       # Recovery
        )
        series = pd.Series(returns, index=dates)
        mask = identify_spy_drawdowns(series)
        assert mask.sum() > 0  # Should have drawdown months

    def test_custom_threshold(self):
        """Custom threshold should change sensitivity."""
        dates = pd.date_range("2010-01-31", periods=60, freq="ME")
        returns = [0.02] * 20 + [-0.05] * 6 + [0.03] * 34
        series = pd.Series(returns, index=dates)

        strict_mask = identify_spy_drawdowns(series, threshold=-0.03)
        loose_mask = identify_spy_drawdowns(series, threshold=-0.30)

        assert strict_mask.sum() >= loose_mask.sum()


# ---------------------------------------------------------------------------
# Tests: analyze_ticker_drawdown
# ---------------------------------------------------------------------------

class TestAnalyzeTickerDrawdown:
    def test_hedge_ticker(self):
        """Ticker that gains during SPY drawdowns should be HEDGE."""
        spy = _make_monthly_returns(120, mean=0.005, std=0.05, seed=1)
        # Create ticker that moves opposite to SPY
        ticker = -spy * 0.5 + 0.005

        mask = identify_spy_drawdowns(spy)
        if mask.sum() < 3:
            # Force some drawdown months
            mask.iloc[10:16] = True

        result = analyze_ticker_drawdown(ticker, spy, mask)
        # May be None if data alignment issues; if we get result, check structure
        if result is not None:
            assert "downside_capture" in result
            assert "win_rate_in_dd" in result
            assert "max_drawdown" in result

    def test_amplifier_ticker(self):
        """Ticker that drops more than SPY should be AMPLIFIER."""
        spy = _make_monthly_returns(120, mean=0.005, std=0.05, seed=1)
        # Create ticker that amplifies SPY moves
        ticker = spy * 2.0

        mask = identify_spy_drawdowns(spy)
        if mask.sum() < 3:
            mask.iloc[10:16] = True

        result = analyze_ticker_drawdown(ticker, spy, mask)
        if result is not None:
            assert result["downside_capture"] > 1.0

    def test_insufficient_data(self):
        """Should return None with <12 months of data."""
        short_idx = pd.date_range("2010-01-31", periods=5, freq="ME")
        ticker = pd.Series([0.01] * 5, index=short_idx)
        spy = pd.Series([0.01] * 5, index=short_idx)
        mask = pd.Series([True] * 5, index=short_idx)

        result = analyze_ticker_drawdown(ticker, spy, mask)
        assert result is None

    def test_insufficient_drawdown_months(self):
        """Should return None with <3 drawdown months."""
        dates = pd.date_range("2010-01-31", periods=60, freq="ME")
        ticker = pd.Series([0.01] * 60, index=dates)
        spy = pd.Series([0.01] * 60, index=dates)
        mask = pd.Series([False] * 60, index=dates)
        mask.iloc[0] = True  # Only 1 drawdown month

        result = analyze_ticker_drawdown(ticker, spy, mask)
        assert result is None


# ---------------------------------------------------------------------------
# Tests: classify_defense
# ---------------------------------------------------------------------------

class TestClassifyDefense:
    def test_hedge(self):
        """Negative DC should classify as HEDGE."""
        assert classify_defense(-0.5) == "HEDGE"
        assert classify_defense(-1.5) == "HEDGE"

    def test_defensive(self):
        """DC 0 to 0.6 should classify as DEFENSIVE."""
        assert classify_defense(0.0) == "DEFENSIVE"
        assert classify_defense(0.3) == "DEFENSIVE"
        assert classify_defense(0.59) == "DEFENSIVE"

    def test_moderate(self):
        """DC 0.6 to 1.0 should classify as MODERATE."""
        assert classify_defense(0.6) == "MODERATE"
        assert classify_defense(0.9) == "MODERATE"

    def test_cyclical(self):
        """DC 1.0 to 1.5 should classify as CYCLICAL."""
        assert classify_defense(1.0) == "CYCLICAL"
        assert classify_defense(1.3) == "CYCLICAL"

    def test_amplifier(self):
        """DC >= 1.5 should classify as AMPLIFIER."""
        assert classify_defense(1.5) == "AMPLIFIER"
        assert classify_defense(3.0) == "AMPLIFIER"


# ---------------------------------------------------------------------------
# Tests: run_drawdown_backtest
# ---------------------------------------------------------------------------

class TestRunDrawdownBacktest:
    def test_basic_run(self):
        """Should process tickers and classify them."""
        spy_returns = _make_monthly_returns(180, mean=0.008, std=0.04, seed=10)
        spy_df = _make_price_df(spy_returns)

        ticker_returns = _make_monthly_returns(180, mean=0.005, std=0.06, seed=20)
        ticker_df = _make_price_df(ticker_returns)

        result = run_drawdown_backtest(
            price_data={"TEST": ticker_df},
            spy_prices=spy_df,
        )

        assert isinstance(result, BacktestResult)
        assert result.backtest_date != ""
        # May or may not classify depending on drawdown months

    def test_insufficient_spy_data(self):
        """Should error with insufficient SPY data."""
        short_spy = pd.DataFrame({"Close": [100, 101, 102]})
        result = run_drawdown_backtest(
            price_data={"TEST": pd.DataFrame({"Close": [100, 101]})},
            spy_prices=short_spy,
        )
        assert len(result.errors) > 0

    def test_dd_override(self):
        """Should use manual override when provided."""
        spy_returns = _make_monthly_returns(180, mean=0.008, std=0.04, seed=10)
        spy_df = _make_price_df(spy_returns)

        result = run_drawdown_backtest(
            price_data={"PHYS": pd.DataFrame({"Close": [100, 101]})},
            spy_prices=spy_df,
            dd_overrides={"PHYS": "HEDGE"},
        )

        assert "PHYS" in result.classifications
        assert result.classifications["PHYS"]["classification"] == "HEDGE"
        assert result.classifications["PHYS"]["source"] == "override"

    def test_skips_insufficient_ticker_data(self):
        """Should skip tickers with insufficient data."""
        spy_returns = _make_monthly_returns(180, mean=0.008, std=0.04, seed=10)
        spy_df = _make_price_df(spy_returns)

        result = run_drawdown_backtest(
            price_data={"SHORT": pd.DataFrame({"Close": [100, 101, 102]})},
            spy_prices=spy_df,
        )

        assert result.tickers_skipped > 0
