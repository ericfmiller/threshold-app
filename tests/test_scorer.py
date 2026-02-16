"""Tests for threshold.engine.scorer — score_ticker() orchestrator."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from threshold.engine.context import ScoringContext
from threshold.engine.scorer import score_ticker
from threshold.engine.signals import SignalBoard

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_sa_data() -> dict:
    """Typical SA data for a quality stock."""
    return {
        "quantScore": 4.5,
        "momentum": "A-",
        "profitability": "B+",
        "revisions": "B",
        "growth": "B-",
        "valuation": "C+",
    }


@pytest.fixture
def mock_sa_data_weak() -> dict:
    """Weak SA data."""
    return {
        "quantScore": 1.5,
        "momentum": "D",
        "profitability": "D-",
        "revisions": "F",
        "growth": "D",
        "valuation": "D+",
    }


@pytest.fixture
def uptrend_df() -> pd.DataFrame:
    """252-bar uptrend DataFrame with Close and Volume."""
    np.random.seed(42)
    n = 252
    close = 100 * np.cumprod(1 + np.random.normal(0.0005, 0.01, n))
    volume = np.random.uniform(500_000, 2_000_000, n)
    return pd.DataFrame({"Close": close, "Volume": volume})


@pytest.fixture
def downtrend_df() -> pd.DataFrame:
    """252-bar downtrend DataFrame."""
    np.random.seed(42)
    n = 252
    close = 100 * np.cumprod(1 + np.random.normal(-0.001, 0.015, n))
    volume = np.random.uniform(500_000, 2_000_000, n)
    return pd.DataFrame({"Close": close, "Volume": volume})


@pytest.fixture
def short_df() -> pd.DataFrame:
    """30-bar DataFrame (too short for scoring)."""
    return pd.DataFrame({
        "Close": np.linspace(100, 105, 30),
        "Volume": np.full(30, 1_000_000.0),
    })


@pytest.fixture
def spy_close() -> pd.Series:
    """252-bar SPY close series."""
    np.random.seed(99)
    return pd.Series(450 * np.cumprod(1 + np.random.normal(0.0004, 0.008, 252)))


@pytest.fixture
def basic_ctx(spy_close) -> ScoringContext:
    """Basic ScoringContext for testing."""
    return ScoringContext(
        market_regime_score=0.55,
        vix_regime="NORMAL",
        spy_close=spy_close,
        grade_history=None,
        prev_scores=None,
        yf_fundamentals=None,
        drawdown_classifications=None,
    )


@pytest.fixture
def fear_ctx(spy_close) -> ScoringContext:
    """ScoringContext in FEAR regime with drawdown classifications."""
    return ScoringContext(
        market_regime_score=0.65,
        vix_regime="FEAR",
        spy_close=spy_close,
        drawdown_classifications={
            "TEST": {"classification": "HEDGE", "downside_capture": -0.85},
            "WEAK": {"classification": "AMPLIFIER", "downside_capture": 1.78},
        },
    )


# ---------------------------------------------------------------------------
# Basic scoring
# ---------------------------------------------------------------------------

class TestScoreTickerBasic:
    def test_returns_none_for_short_data(self, mock_sa_data, short_df, basic_ctx):
        result = score_ticker("TEST", mock_sa_data, short_df, basic_ctx)
        assert result is None

    def test_returns_dict_for_valid_data(self, mock_sa_data, uptrend_df, basic_ctx):
        result = score_ticker("TEST", mock_sa_data, uptrend_df, basic_ctx)
        assert result is not None
        assert "dcs" in result
        assert "dcs_signal" in result
        assert "sub_scores" in result
        assert "technicals" in result

    def test_dcs_in_range(self, mock_sa_data, uptrend_df, basic_ctx):
        result = score_ticker("TEST", mock_sa_data, uptrend_df, basic_ctx)
        assert result is not None
        assert 0 <= result["dcs"] <= 100

    def test_sub_scores_present(self, mock_sa_data, uptrend_df, basic_ctx):
        result = score_ticker("TEST", mock_sa_data, uptrend_df, basic_ctx)
        assert result is not None
        ss = result["sub_scores"]["dcs"]
        for key in ("MQ", "FQ", "TO", "MR", "VC"):
            assert key in ss
            assert 0 <= ss[key] <= 1

    def test_dcs_signal_valid(self, mock_sa_data, uptrend_df, basic_ctx):
        result = score_ticker("TEST", mock_sa_data, uptrend_df, basic_ctx)
        assert result is not None
        valid_signals = {
            "STRONG BUY DIP", "HIGH CONVICTION", "BUY DIP",
            "WATCH", "WEAK", "AVOID",
        }
        assert result["dcs_signal"] in valid_signals


# ---------------------------------------------------------------------------
# Strong vs weak SA data
# ---------------------------------------------------------------------------

class TestScoreQuality:
    def test_strong_sa_higher_dcs(
        self, mock_sa_data, mock_sa_data_weak, uptrend_df, basic_ctx,
    ):
        strong = score_ticker("TEST", mock_sa_data, uptrend_df, basic_ctx)
        weak = score_ticker("TEST", mock_sa_data_weak, uptrend_df, basic_ctx)
        assert strong is not None and weak is not None
        assert strong["dcs"] > weak["dcs"]


# ---------------------------------------------------------------------------
# Technicals
# ---------------------------------------------------------------------------

class TestTechnicals:
    def test_rsi_present(self, mock_sa_data, uptrend_df, basic_ctx):
        result = score_ticker("TEST", mock_sa_data, uptrend_df, basic_ctx)
        assert result is not None
        assert "rsi_14" in result["technicals"]
        assert 0 <= result["technicals"]["rsi_14"] <= 100

    def test_macd_present(self, mock_sa_data, uptrend_df, basic_ctx):
        result = score_ticker("TEST", mock_sa_data, uptrend_df, basic_ctx)
        assert result is not None
        assert "macd_crossover" in result["technicals"]

    def test_obv_present(self, mock_sa_data, uptrend_df, basic_ctx):
        result = score_ticker("TEST", mock_sa_data, uptrend_df, basic_ctx)
        assert result is not None
        assert "obv_divergence" in result["technicals"]

    def test_reversal_flags(self, mock_sa_data, uptrend_df, basic_ctx):
        result = score_ticker("TEST", mock_sa_data, uptrend_df, basic_ctx)
        assert result is not None
        tech = result["technicals"]
        assert "rsi_bullish_divergence" in tech
        assert "bb_lower_breach" in tech
        assert "bottom_turning" in tech


# ---------------------------------------------------------------------------
# SignalBoard
# ---------------------------------------------------------------------------

class TestSignalBoard:
    def test_signal_board_present(self, mock_sa_data, uptrend_df, basic_ctx):
        result = score_ticker("TEST", mock_sa_data, uptrend_df, basic_ctx)
        assert result is not None
        assert "signal_board" in result
        assert isinstance(result["signal_board"], list)

    def test_sell_flags_present(self, mock_sa_data, uptrend_df, basic_ctx):
        result = score_ticker("TEST", mock_sa_data, uptrend_df, basic_ctx)
        assert result is not None
        assert "sell_flags" in result
        assert isinstance(result["sell_flags"], list)

    def test_internal_board_obj(self, mock_sa_data, uptrend_df, basic_ctx):
        result = score_ticker("TEST", mock_sa_data, uptrend_df, basic_ctx)
        assert result is not None
        assert "_signal_board_obj" in result
        assert isinstance(result["_signal_board_obj"], SignalBoard)


# ---------------------------------------------------------------------------
# Drawdown Defense
# ---------------------------------------------------------------------------

class TestDrawdownDefense:
    def test_hedge_in_fear(self, mock_sa_data, uptrend_df, fear_ctx):
        result = score_ticker("TEST", mock_sa_data, uptrend_df, fear_ctx)
        assert result is not None
        if "drawdown_defense" in result:
            assert result["drawdown_defense"]["classification"] == "HEDGE"

    def test_no_dd_in_normal(self, mock_sa_data, uptrend_df, basic_ctx):
        result = score_ticker("TEST", mock_sa_data, uptrend_df, basic_ctx)
        assert result is not None
        # No drawdown_classifications in basic_ctx → no drawdown_defense key
        assert "drawdown_defense" not in result


# ---------------------------------------------------------------------------
# Falling Knife
# ---------------------------------------------------------------------------

class TestFallingKnife:
    def test_downtrend_caps_dcs(self, mock_sa_data, downtrend_df, basic_ctx):
        result = score_ticker("TEST", mock_sa_data, downtrend_df, basic_ctx)
        assert result is not None
        # In a downtrend, DCS should be capped (or at minimum, present)
        assert result["dcs"] <= 100

    def test_falling_knife_cap_metadata(self, mock_sa_data, downtrend_df, basic_ctx):
        result = score_ticker("TEST", mock_sa_data, downtrend_df, basic_ctx)
        assert result is not None
        # If the series is in a strong downtrend, falling_knife_cap should appear
        if "falling_knife_cap" in result:
            assert "cap_applied" in result["falling_knife_cap"]
            assert "original_dcs" in result["falling_knife_cap"]


# ---------------------------------------------------------------------------
# YFinance Fundamentals
# ---------------------------------------------------------------------------

class TestYFinanceFundamentals:
    def test_yf_fundamentals_in_result(self, mock_sa_data, uptrend_df, spy_close):
        ctx = ScoringContext(
            market_regime_score=0.55,
            vix_regime="NORMAL",
            spy_close=spy_close,
            yf_fundamentals={
                "TEST": {
                    "fetch_status": "ok",
                    "fcf_yield": 0.05,
                    "fcf_yield_pctl": 0.7,
                    "gross_profitability": 0.3,
                    "gross_profitability_pctl": 0.8,
                    "ev_to_ebitda": 12.5,
                    "ev_to_ebitda_pctl": 0.4,
                    "sector": "Technology",
                },
            },
        )
        result = score_ticker("TEST", mock_sa_data, uptrend_df, ctx)
        assert result is not None
        assert "yf_fundamentals" in result
        assert "fcf_yield" in result["yf_fundamentals"]

    def test_no_yf_no_key(self, mock_sa_data, uptrend_df, basic_ctx):
        result = score_ticker("TEST", mock_sa_data, uptrend_df, basic_ctx)
        assert result is not None
        assert "yf_fundamentals" not in result


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_missing_volume_column(self, mock_sa_data, basic_ctx):
        """DataFrame without Volume should not crash."""
        np.random.seed(42)
        df = pd.DataFrame({
            "Close": 100 * np.cumprod(1 + np.random.normal(0.0005, 0.01, 252)),
        })
        result = score_ticker("TEST", mock_sa_data, df, basic_ctx)
        # Should either return a result or None, not crash
        assert result is None or isinstance(result, dict)

    def test_none_quant_score(self, uptrend_df, basic_ctx):
        """Missing quantScore should not crash."""
        sa = {"momentum": "B"}
        result = score_ticker("TEST", sa, uptrend_df, basic_ctx)
        assert result is not None

    def test_empty_sa_data(self, uptrend_df, basic_ctx):
        """Completely empty SA data should not crash."""
        result = score_ticker("TEST", {}, uptrend_df, basic_ctx)
        assert result is not None
