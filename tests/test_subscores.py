"""Tests for threshold.engine.subscores — DCS sub-score calculators."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from threshold.engine.subscores import (
    calc_fundamental_quality,
    calc_market_regime,
    calc_momentum_quality,
    calc_revision_momentum,
    calc_technical_oversold,
    calc_valuation_context,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def uptrend_close() -> pd.Series:
    """252-bar steady uptrend above both SMAs."""
    np.random.seed(42)
    prices = 100 * np.cumprod(1 + np.random.normal(0.0005, 0.01, 252))
    return pd.Series(prices, name="Close")


@pytest.fixture
def spy_close() -> pd.Series:
    """252-bar SPY-like uptrend."""
    np.random.seed(99)
    prices = 450 * np.cumprod(1 + np.random.normal(0.0004, 0.008, 252))
    return pd.Series(prices, name="SPY")


@pytest.fixture
def sa_data_strong() -> dict:
    """Strong SA data — high quant, good grades."""
    return {
        "quantScore": 4.8,
        "momentum": "A",
        "profitability": "A-",
        "revisions": "B+",
        "growth": "B",
        "valuation": "C+",
    }


@pytest.fixture
def sa_data_weak() -> dict:
    """Weak SA data — low quant, poor grades."""
    return {
        "quantScore": 1.5,
        "momentum": "D",
        "profitability": "D-",
        "revisions": "F",
        "growth": "D",
        "valuation": "D+",
    }


# ---------------------------------------------------------------------------
# MQ: Momentum Quality
# ---------------------------------------------------------------------------

class TestMomentumQuality:
    def test_returns_tuple(self, sa_data_strong, uptrend_close):
        mq, trend, vol_adj, rs = calc_momentum_quality(sa_data_strong, uptrend_close)
        assert isinstance(mq, float)
        assert isinstance(trend, float)
        assert isinstance(vol_adj, float)

    def test_mq_range(self, sa_data_strong, uptrend_close, spy_close):
        mq, _, _, _ = calc_momentum_quality(sa_data_strong, uptrend_close, spy_close)
        assert 0 <= mq <= 1, f"MQ should be in [0, 1], got {mq}"

    def test_strong_data_higher_mq(self, sa_data_strong, sa_data_weak, uptrend_close):
        mq_strong, _, _, _ = calc_momentum_quality(sa_data_strong, uptrend_close)
        mq_weak, _, _, _ = calc_momentum_quality(sa_data_weak, uptrend_close)
        assert mq_strong > mq_weak

    def test_trend_classifier_uptrend(self, uptrend_close):
        """Uptrend should produce high trend score (0.5 or 1.0)."""
        sa = {"momentum": "B"}
        _, trend, _, _ = calc_momentum_quality(sa, uptrend_close)
        assert trend >= 0.4

    def test_rs_vs_spy_none_without_spy(self, sa_data_strong, uptrend_close):
        _, _, _, rs = calc_momentum_quality(sa_data_strong, uptrend_close, None)
        assert rs is None

    def test_rs_vs_spy_with_spy(self, sa_data_strong, uptrend_close, spy_close):
        _, _, _, rs = calc_momentum_quality(sa_data_strong, uptrend_close, spy_close)
        assert rs is not None
        assert isinstance(rs, float)

    def test_short_series(self, sa_data_strong):
        """Short series should still produce valid MQ."""
        close = pd.Series(np.linspace(100, 105, 30))
        mq, trend, _, _ = calc_momentum_quality(sa_data_strong, close)
        assert 0 <= mq <= 1


# ---------------------------------------------------------------------------
# Revision Momentum
# ---------------------------------------------------------------------------

class TestRevisionMomentum:
    def test_no_history_returns_none(self):
        score, direction, delta = calc_revision_momentum("AAPL", None)
        assert score is None

    def test_insufficient_history(self):
        history = [{"scores": {"AAPL": {"sa_revisions": "B+"}}}]
        score, direction, delta = calc_revision_momentum("AAPL", history)
        assert score is None

    def test_improving_revisions(self):
        """Improving revisions should produce positive direction."""
        history = [
            {"_metadata": {"generated_at": "2026-02-15T12:00:00"},
             "scores": {"AAPL": {"sa_revisions": "A"}}},
            {"_metadata": {"generated_at": "2026-02-08T12:00:00"},
             "scores": {"AAPL": {"sa_revisions": "A-"}}},
            {"_metadata": {"generated_at": "2026-02-01T12:00:00"},
             "scores": {"AAPL": {"sa_revisions": "B+"}}},
            {"_metadata": {"generated_at": "2026-01-25T12:00:00"},
             "scores": {"AAPL": {"sa_revisions": "B"}}},
            {"_metadata": {"generated_at": "2026-01-18T12:00:00"},
             "scores": {"AAPL": {"sa_revisions": "B-"}}},
        ]
        score, direction, delta = calc_revision_momentum("AAPL", history)
        assert score is not None
        assert direction == "positive"
        assert delta > 0

    def test_calendar_span_gate(self):
        """History spanning < 21 days should return None."""
        history = [
            {"_metadata": {"generated_at": "2026-02-15T12:00:00"},
             "scores": {"AAPL": {"sa_revisions": "A"}}},
            {"_metadata": {"generated_at": "2026-02-14T12:00:00"},
             "scores": {"AAPL": {"sa_revisions": "A-"}}},
            {"_metadata": {"generated_at": "2026-02-13T12:00:00"},
             "scores": {"AAPL": {"sa_revisions": "B+"}}},
            {"_metadata": {"generated_at": "2026-02-12T12:00:00"},
             "scores": {"AAPL": {"sa_revisions": "B"}}},
        ]
        score, _, _ = calc_revision_momentum("AAPL", history)
        assert score is None

    def test_score_in_range(self):
        """Score should be in [0, 1] when valid."""
        history = [
            {"_metadata": {"generated_at": "2026-02-15T12:00:00"},
             "scores": {"AAPL": {"sa_revisions": "B"}}},
            {"_metadata": {"generated_at": "2026-02-08T12:00:00"},
             "scores": {"AAPL": {"sa_revisions": "B"}}},
            {"_metadata": {"generated_at": "2026-02-01T12:00:00"},
             "scores": {"AAPL": {"sa_revisions": "B"}}},
            {"_metadata": {"generated_at": "2026-01-25T12:00:00"},
             "scores": {"AAPL": {"sa_revisions": "B"}}},
            {"_metadata": {"generated_at": "2026-01-18T12:00:00"},
             "scores": {"AAPL": {"sa_revisions": "B"}}},
        ]
        score, direction, delta = calc_revision_momentum("AAPL", history)
        assert score is not None
        assert 0 <= score <= 1


# ---------------------------------------------------------------------------
# FQ: Fundamental Quality
# ---------------------------------------------------------------------------

class TestFundamentalQuality:
    def test_base_path(self, sa_data_strong):
        """No yfinance, no revision momentum → base weights."""
        fq = calc_fundamental_quality(sa_data_strong)
        assert 0 <= fq <= 1

    def test_with_rev_momentum(self, sa_data_strong):
        fq_base = calc_fundamental_quality(sa_data_strong)
        fq_rm = calc_fundamental_quality(sa_data_strong, rev_momentum=0.8)
        # Different paths, scores should differ
        assert fq_base != fq_rm

    def test_with_yfinance(self, sa_data_strong):
        yf = {
            "fetch_status": "ok",
            "fcf_yield_pctl": 0.7,
            "gross_profitability_pctl": 0.8,
            "ev_to_ebitda_pctl": 0.3,
        }
        fq = calc_fundamental_quality(sa_data_strong, yf_fundamentals=yf)
        assert 0 <= fq <= 1

    def test_with_yf_and_revmom(self, sa_data_strong):
        yf = {
            "fetch_status": "ok",
            "fcf_yield_pctl": 0.7,
            "gross_profitability_pctl": 0.8,
            "ev_to_ebitda_pctl": 0.3,
        }
        fq = calc_fundamental_quality(sa_data_strong, rev_momentum=0.8, yf_fundamentals=yf)
        assert 0 <= fq <= 1

    def test_strong_vs_weak(self, sa_data_strong, sa_data_weak):
        fq_strong = calc_fundamental_quality(sa_data_strong)
        fq_weak = calc_fundamental_quality(sa_data_weak)
        assert fq_strong > fq_weak

    def test_missing_quant_uses_zero(self):
        sa = {"momentum": "B"}
        fq = calc_fundamental_quality(sa)
        assert 0 <= fq <= 1


# ---------------------------------------------------------------------------
# TO: Technical Oversold
# ---------------------------------------------------------------------------

class TestTechnicalOversold:
    def test_returns_tuple(self, uptrend_close):
        to, macd = calc_technical_oversold(uptrend_close)
        assert isinstance(to, float)
        assert isinstance(macd, dict)

    def test_to_range(self, uptrend_close):
        to, _ = calc_technical_oversold(uptrend_close)
        assert 0 <= to <= 1

    def test_oversold_higher_score(self):
        """Oversold series should produce higher TO score."""
        np.random.seed(42)
        # Normal series
        normal = pd.Series(np.linspace(100, 102, 252))
        # Oversold series (sharp decline)
        prices = list(np.linspace(100, 105, 200))
        prices.extend(np.linspace(105, 80, 52))
        oversold = pd.Series(prices)

        to_normal, _ = calc_technical_oversold(normal)
        to_oversold, _ = calc_technical_oversold(oversold)
        assert to_oversold > to_normal


# ---------------------------------------------------------------------------
# MR: Market Regime
# ---------------------------------------------------------------------------

class TestMarketRegime:
    def test_basic_range(self):
        mr = calc_market_regime(18.0, 0.5, True)
        assert 0 <= mr <= 1

    def test_high_vix_higher_score(self):
        """Higher VIX = better dip-buy opportunity → higher MR score."""
        mr_low = calc_market_regime(12.0, 0.2, True)
        mr_high = calc_market_regime(30.0, 0.9, True)
        assert mr_high > mr_low

    def test_spy_above_200d_matters(self):
        mr_above = calc_market_regime(18.0, 0.5, True)
        mr_below = calc_market_regime(18.0, 0.5, False)
        assert mr_above > mr_below

    def test_with_breadth(self):
        mr_strong = calc_market_regime(18.0, 0.5, True, breadth_pct=0.80)
        mr_weak = calc_market_regime(18.0, 0.5, True, breadth_pct=0.25)
        assert mr_strong > mr_weak

    def test_without_breadth_different_weights(self):
        """Without breadth, VIX gets 60% and trend gets 40%."""
        mr = calc_market_regime(18.0, 0.5, True, breadth_pct=None)
        assert 0 <= mr <= 1

    def test_panic_vix(self):
        mr = calc_market_regime(40.0, 0.99, False, breadth_pct=0.15)
        assert mr > 0.3  # High VIX contrarian boost despite bad trends


# ---------------------------------------------------------------------------
# VC: Valuation Context
# ---------------------------------------------------------------------------

class TestValuationContext:
    def test_sa_only(self, sa_data_strong):
        vc = calc_valuation_context(sa_data_strong)
        assert 0 <= vc <= 1

    def test_with_yfinance(self, sa_data_strong):
        yf = {
            "fetch_status": "ok",
            "ev_to_ebitda_pctl": 0.3,  # Low EV/EBITDA = cheap = high VC
        }
        vc = calc_valuation_context(sa_data_strong, yf_fundamentals=yf)
        assert 0 <= vc <= 1

    def test_strong_vs_weak(self, sa_data_strong, sa_data_weak):
        vc_strong = calc_valuation_context(sa_data_strong)
        vc_weak = calc_valuation_context(sa_data_weak)
        # C+ valuation (strong) vs D+ valuation (weak)
        assert vc_strong > vc_weak
