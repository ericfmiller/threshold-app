"""Tests for threshold.engine.composite — DCS composition and classifiers."""

from __future__ import annotations

from threshold.engine.composite import (
    apply_drawdown_modifier,
    apply_falling_knife_filter,
    apply_obv_boost,
    apply_rsi_divergence_boost,
    classify_dcs,
    classify_vix,
    compose_dcs,
)

# ---------------------------------------------------------------------------
# compose_dcs
# ---------------------------------------------------------------------------

class TestComposeDCS:
    def test_default_weights(self):
        scores = {"MQ": 0.8, "FQ": 0.7, "TO": 0.6, "MR": 0.5, "VC": 0.4}
        dcs = compose_dcs(scores)
        expected = (0.8 * 30) + (0.7 * 25) + (0.6 * 20) + (0.5 * 15) + (0.4 * 10)
        assert abs(dcs - expected) < 0.01

    def test_custom_weights(self):
        scores = {"MQ": 1.0, "FQ": 1.0, "TO": 1.0, "MR": 1.0, "VC": 1.0}
        weights = {"MQ": 50, "FQ": 20, "TO": 10, "MR": 10, "VC": 10}
        dcs = compose_dcs(scores, weights)
        assert dcs == 100.0

    def test_all_zero(self):
        scores = {"MQ": 0, "FQ": 0, "TO": 0, "MR": 0, "VC": 0}
        assert compose_dcs(scores) == 0.0

    def test_all_one(self):
        scores = {"MQ": 1.0, "FQ": 1.0, "TO": 1.0, "MR": 1.0, "VC": 1.0}
        assert compose_dcs(scores) == 100.0


# ---------------------------------------------------------------------------
# OBV Boost
# ---------------------------------------------------------------------------

class TestOBVBoost:
    def test_bullish_divergence_boosts(self):
        obv = {"divergence": "bullish", "divergence_strength": 0.5}
        result = apply_obv_boost(70.0, obv, max_boost=5)
        assert result == 72.5

    def test_no_divergence_no_boost(self):
        obv = {"divergence": "none", "divergence_strength": 0.0}
        assert apply_obv_boost(70.0, obv) == 70.0

    def test_bearish_no_boost(self):
        obv = {"divergence": "bearish", "divergence_strength": 0.8}
        assert apply_obv_boost(70.0, obv) == 70.0

    def test_cap_at_100(self):
        obv = {"divergence": "bullish", "divergence_strength": 1.0}
        assert apply_obv_boost(98.0, obv, max_boost=5) == 100


# ---------------------------------------------------------------------------
# RSI Divergence Boost
# ---------------------------------------------------------------------------

class TestRSIDivergenceBoost:
    def test_boost_when_above_min(self):
        result = apply_rsi_divergence_boost(65.0, True, boost=3, min_dcs=60)
        assert result == 68.0

    def test_no_boost_below_min(self):
        result = apply_rsi_divergence_boost(55.0, True, boost=3, min_dcs=60)
        assert result == 55.0

    def test_no_boost_no_divergence(self):
        result = apply_rsi_divergence_boost(70.0, False)
        assert result == 70.0

    def test_cap_at_100(self):
        result = apply_rsi_divergence_boost(99.0, True, boost=3, min_dcs=60)
        assert result == 100


# ---------------------------------------------------------------------------
# Falling Knife Filter
# ---------------------------------------------------------------------------

class TestFallingKnifeFilter:
    def test_freefall_default_cap(self):
        dcs, cap = apply_falling_knife_filter(80.0, 0.1)
        assert dcs == 30  # Default freefall cap without classification
        assert cap == 30

    def test_freefall_hedge_cap(self):
        dcs, cap = apply_falling_knife_filter(80.0, 0.1, "HEDGE")
        assert dcs == 50
        assert cap == 50

    def test_freefall_amplifier_cap(self):
        dcs, cap = apply_falling_knife_filter(80.0, 0.1, "AMPLIFIER")
        assert dcs == 15
        assert cap == 15

    def test_downtrend_default_cap(self):
        dcs, cap = apply_falling_knife_filter(80.0, 0.3)
        assert dcs == 50
        assert cap == 50

    def test_downtrend_hedge_cap(self):
        dcs, cap = apply_falling_knife_filter(80.0, 0.3, "HEDGE")
        assert dcs == 70
        assert cap == 70

    def test_uptrend_no_cap(self):
        dcs, cap = apply_falling_knife_filter(80.0, 1.0)
        assert dcs == 80.0
        assert cap is None

    def test_already_below_cap(self):
        dcs, cap = apply_falling_knife_filter(20.0, 0.1, "AMPLIFIER")
        assert dcs == 15  # Capped down from 20
        assert cap == 15


# ---------------------------------------------------------------------------
# Drawdown Modifier (D-5)
# ---------------------------------------------------------------------------

class TestDrawdownModifier:
    def test_hedge_in_fear(self):
        dcs, mod = apply_drawdown_modifier(60.0, "HEDGE", "FEAR")
        assert dcs == 65.0
        assert mod == 5

    def test_amplifier_in_panic(self):
        dcs, mod = apply_drawdown_modifier(60.0, "AMPLIFIER", "PANIC")
        assert dcs == 55.0
        assert mod == -5

    def test_no_modification_in_normal(self):
        dcs, mod = apply_drawdown_modifier(60.0, "HEDGE", "NORMAL")
        assert dcs == 60.0
        assert mod == 0

    def test_no_classification(self):
        dcs, mod = apply_drawdown_modifier(60.0, None, "FEAR")
        assert dcs == 60.0
        assert mod == 0

    def test_moderate_no_change(self):
        dcs, mod = apply_drawdown_modifier(60.0, "MODERATE", "PANIC")
        assert dcs == 60.0
        assert mod == 0

    def test_floor_at_zero(self):
        dcs, mod = apply_drawdown_modifier(3.0, "AMPLIFIER", "FEAR")
        assert dcs == 0
        assert mod == -5

    def test_cap_at_100(self):
        dcs, mod = apply_drawdown_modifier(98.0, "HEDGE", "FEAR")
        assert dcs == 100
        assert mod == 5


# ---------------------------------------------------------------------------
# Classifiers
# ---------------------------------------------------------------------------

class TestClassifyDCS:
    def test_strong_buy_dip(self):
        assert classify_dcs(85) == "STRONG BUY DIP"

    def test_high_conviction(self):
        assert classify_dcs(75) == "HIGH CONVICTION"

    def test_buy_dip(self):
        assert classify_dcs(67) == "BUY DIP"

    def test_watch(self):
        assert classify_dcs(55) == "WATCH"

    def test_weak(self):
        assert classify_dcs(40) == "WEAK"

    def test_avoid(self):
        assert classify_dcs(20) == "AVOID"

    def test_boundary_80(self):
        assert classify_dcs(80) == "STRONG BUY DIP"

    def test_boundary_65(self):
        assert classify_dcs(65) == "BUY DIP"

    def test_custom_thresholds(self):
        custom = {"strong_buy_dip": 90, "high_conviction": 80,
                  "buy_dip": 70, "watch": 50, "weak": 30}
        assert classify_dcs(85, custom) == "HIGH CONVICTION"
        assert classify_dcs(95, custom) == "STRONG BUY DIP"


class TestClassifyVIX:
    def test_complacent(self):
        assert classify_vix(10) == "COMPLACENT"

    def test_normal(self):
        assert classify_vix(18) == "NORMAL"

    def test_fear(self):
        assert classify_vix(25) == "FEAR"

    def test_panic(self):
        assert classify_vix(35) == "PANIC"

    def test_boundary_14(self):
        assert classify_vix(14) == "NORMAL"

    def test_boundary_20(self):
        assert classify_vix(20) == "FEAR"

    def test_boundary_28(self):
        assert classify_vix(28) == "PANIC"

    def test_extreme_vix(self):
        assert classify_vix(80) == "PANIC"
