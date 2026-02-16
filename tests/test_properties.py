"""Property-based tests using Hypothesis.

Tests universal invariants that should hold for ANY valid input:
- Sub-scores always in [0, 1]
- DCS always in [0, 100]
- Signal classification is always valid
- Technical indicators are bounded
- Composite operations are monotonic where expected
"""

from __future__ import annotations

import pandas as pd
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st

from threshold.engine.composite import (
    apply_drawdown_modifier,
    apply_falling_knife_filter,
    apply_obv_boost,
    apply_rsi_divergence_boost,
    classify_dcs,
    classify_vix,
    compose_dcs,
)
from threshold.engine.grades import sa_grade_to_norm
from threshold.engine.technical import (
    calc_rsi_value,
)

# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

# Valid SA grades
sa_grades = st.sampled_from([
    "A+", "A", "A-", "B+", "B", "B-", "C+", "C", "C-", "D+", "D", "D-", "F",
])

# Quant scores 0-5
quant_scores = st.floats(min_value=0.0, max_value=5.0, allow_nan=False)

# Sub-score values [0, 1]
sub_score_values = st.floats(min_value=0.0, max_value=1.0, allow_nan=False)

# DCS values [0, 100]
dcs_values = st.floats(min_value=0.0, max_value=100.0, allow_nan=False)

# VIX values
vix_values = st.floats(min_value=5.0, max_value=80.0, allow_nan=False)

# Price series (as lists of floats > 0)
price_lists = st.lists(
    st.floats(min_value=1.0, max_value=10000.0, allow_nan=False, allow_infinity=False),
    min_size=15,
    max_size=300,
)

# Defense classes
defense_classes = st.sampled_from(["HEDGE", "DEFENSIVE", "MODERATE", "CYCLICAL", "AMPLIFIER"])

# VIX regimes
vix_regimes = st.sampled_from(["COMPLACENT", "NORMAL", "FEAR", "PANIC"])


# ---------------------------------------------------------------------------
# Grade conversion properties
# ---------------------------------------------------------------------------

class TestGradeProperties:
    @given(grade=sa_grades)
    def test_grade_to_norm_bounded(self, grade):
        """sa_grade_to_norm always returns [0, 1]."""
        val = sa_grade_to_norm(grade)
        assert 0.0 <= val <= 1.0

    @given(grade=sa_grades)
    def test_grade_to_norm_deterministic(self, grade):
        """Same grade always gives same result."""
        assert sa_grade_to_norm(grade) == sa_grade_to_norm(grade)


# ---------------------------------------------------------------------------
# DCS composition properties
# ---------------------------------------------------------------------------

class TestComposeProperties:
    @given(
        mq=sub_score_values,
        fq=sub_score_values,
        to=sub_score_values,
        mr=sub_score_values,
        vc=sub_score_values,
    )
    def test_dcs_always_bounded(self, mq, fq, to, mr, vc):
        """DCS from compose_dcs must always be in [0, 100]."""
        sub_scores = {"MQ": mq, "FQ": fq, "TO": to, "MR": mr, "VC": vc}
        dcs = compose_dcs(sub_scores)
        assert 0 <= dcs <= 100

    @given(
        mq=sub_score_values,
        fq=sub_score_values,
        to=sub_score_values,
        mr=sub_score_values,
        vc=sub_score_values,
    )
    def test_dcs_zero_when_all_zero(self, mq, fq, to, mr, vc):
        """DCS should be 0 when all sub-scores are 0."""
        sub_scores = {"MQ": 0, "FQ": 0, "TO": 0, "MR": 0, "VC": 0}
        assert compose_dcs(sub_scores) == 0

    @given(
        mq=sub_score_values,
        fq=sub_score_values,
        to=sub_score_values,
        mr=sub_score_values,
        vc=sub_score_values,
    )
    def test_dcs_max_when_all_one(self, mq, fq, to, mr, vc):
        """DCS should be 100 when all sub-scores are 1."""
        sub_scores = {"MQ": 1, "FQ": 1, "TO": 1, "MR": 1, "VC": 1}
        assert compose_dcs(sub_scores) == 100

    @given(
        base=sub_score_values,
        boost=sub_score_values,
    )
    def test_higher_mq_higher_dcs(self, base, boost):
        """Increasing MQ should increase DCS (all else equal)."""
        assume(boost > base + 0.01)  # Meaningful difference
        low = compose_dcs({"MQ": base, "FQ": 0.5, "TO": 0.5, "MR": 0.5, "VC": 0.5})
        high = compose_dcs({"MQ": boost, "FQ": 0.5, "TO": 0.5, "MR": 0.5, "VC": 0.5})
        assert high >= low


# ---------------------------------------------------------------------------
# Modifier properties
# ---------------------------------------------------------------------------

class TestModifierProperties:
    @given(dcs=dcs_values, max_boost=st.floats(min_value=0, max_value=10, allow_nan=False))
    def test_obv_boost_bounded(self, dcs, max_boost):
        """OBV boost should not push DCS below 0 or above 100."""
        obv_result = {"divergence": "bullish", "strength": 0.8}
        boosted = apply_obv_boost(dcs, obv_result, max_boost)
        assert 0 <= boosted <= 100

    @given(dcs=dcs_values)
    def test_obv_no_boost_on_neutral(self, dcs):
        """No OBV boost when divergence is neutral."""
        obv_result = {"divergence": "neutral", "strength": 0}
        assert apply_obv_boost(dcs, obv_result, 5) == dcs

    @given(dcs=dcs_values, boost=st.floats(min_value=0, max_value=10, allow_nan=False))
    def test_rsi_div_boost_bounded(self, dcs, boost):
        """RSI divergence boost should not push DCS below 0 or above 100."""
        boosted = apply_rsi_divergence_boost(dcs, True, boost, min_dcs=60)
        assert 0 <= boosted <= 100

    @given(dcs=dcs_values)
    def test_rsi_div_no_boost_when_false(self, dcs):
        """No boost when RSI divergence is False."""
        assert apply_rsi_divergence_boost(dcs, False, 3, 60) == dcs


# ---------------------------------------------------------------------------
# Classification properties
# ---------------------------------------------------------------------------

class TestClassificationProperties:
    @given(dcs=dcs_values)
    def test_classify_dcs_always_valid(self, dcs):
        """classify_dcs always returns a valid signal string."""
        signal = classify_dcs(dcs)
        valid = {"STRONG BUY DIP", "HIGH CONVICTION", "BUY DIP", "WATCH", "WEAK", "AVOID"}
        assert signal in valid

    @given(vix=vix_values)
    def test_classify_vix_always_valid(self, vix):
        """classify_vix always returns a valid regime string."""
        regime = classify_vix(vix)
        valid = {"COMPLACENT", "NORMAL", "FEAR", "PANIC"}
        assert regime in valid

    @given(dcs=dcs_values)
    def test_classify_dcs_monotonic(self, dcs):
        """Higher DCS should never produce a weaker signal."""
        signal_order = {
            "AVOID": 0, "WEAK": 1, "WATCH": 2,
            "BUY DIP": 3, "HIGH CONVICTION": 4, "STRONG BUY DIP": 5,
        }
        s = classify_dcs(dcs)
        # If DCS >= 80, signal must be STRONG BUY DIP
        if dcs >= 80:
            assert signal_order[s] >= signal_order["STRONG BUY DIP"]
        # If DCS >= 65, signal must be at least BUY DIP
        if dcs >= 65:
            assert signal_order[s] >= signal_order["BUY DIP"]

    @given(vix=vix_values)
    def test_classify_vix_monotonic(self, vix):
        """Higher VIX should never produce a calmer regime."""
        regime_order = {"COMPLACENT": 0, "NORMAL": 1, "FEAR": 2, "PANIC": 3}
        r = classify_vix(vix)
        if vix >= 28:
            assert regime_order[r] >= regime_order["PANIC"]
        if vix >= 20:
            assert regime_order[r] >= regime_order["FEAR"]


# ---------------------------------------------------------------------------
# Drawdown modifier properties
# ---------------------------------------------------------------------------

class TestDrawdownModifierProperties:
    @given(
        dcs=dcs_values,
        dd_class=defense_classes,
        vix_regime=vix_regimes,
    )
    def test_drawdown_modifier_bounded(self, dcs, dd_class, vix_regime):
        """Drawdown modifier should keep DCS in [0, 100]."""
        result_dcs, modifier = apply_drawdown_modifier(dcs, dd_class, vix_regime)
        assert 0 <= result_dcs <= 100

    @given(dcs=dcs_values, dd_class=defense_classes)
    def test_no_modifier_in_normal(self, dcs, dd_class):
        """No D-5 modifier in NORMAL or COMPLACENT regime."""
        result_dcs, modifier = apply_drawdown_modifier(dcs, dd_class, "NORMAL")
        assert modifier == 0
        assert result_dcs == dcs

    @given(dcs=dcs_values)
    def test_hedge_boosted_in_fear(self, dcs):
        """HEDGE should get positive modifier in FEAR."""
        _, modifier = apply_drawdown_modifier(dcs, "HEDGE", "FEAR")
        assert modifier >= 0

    @given(dcs=dcs_values)
    def test_amplifier_penalized_in_fear(self, dcs):
        """AMPLIFIER should get negative modifier in FEAR."""
        _, modifier = apply_drawdown_modifier(dcs, "AMPLIFIER", "FEAR")
        assert modifier <= 0


# ---------------------------------------------------------------------------
# Falling knife properties
# ---------------------------------------------------------------------------

class TestFallingKnifeProperties:
    @given(
        dcs=dcs_values,
        dd_class=defense_classes,
    )
    def test_falling_knife_never_increases_dcs(self, dcs, dd_class):
        """Falling knife filter should only decrease or maintain DCS."""
        # trend_score 0.0 = freefall (strongest downtrend)
        result_dcs, cap = apply_falling_knife_filter(dcs, 0.0, dd_class)
        assert result_dcs <= dcs

    @given(dcs=dcs_values, dd_class=defense_classes)
    def test_falling_knife_bounded(self, dcs, dd_class):
        """Falling knife result should be in [0, 100]."""
        for trend_val in (0.0, 0.2, 0.5, 0.8):
            result_dcs, _ = apply_falling_knife_filter(dcs, trend_val, dd_class)
            assert 0 <= result_dcs <= 100

    @given(dcs=dcs_values, dd_class=defense_classes)
    def test_no_cap_in_uptrend(self, dcs, dd_class):
        """No falling knife cap in uptrend (high trend_score)."""
        result_dcs, cap = apply_falling_knife_filter(dcs, 0.8, dd_class)
        assert result_dcs == dcs
        assert cap is None


# ---------------------------------------------------------------------------
# RSI properties
# ---------------------------------------------------------------------------

class TestRSIProperties:
    @given(prices=price_lists)
    @settings(suppress_health_check=[HealthCheck.too_slow])
    def test_rsi_always_bounded(self, prices):
        """RSI must always be in [0, 100]."""
        assume(len(prices) >= 15)
        series = pd.Series(prices)
        rsi = calc_rsi_value(series, period=14)
        assert 0 <= rsi <= 100

    def test_rsi_extreme_uptrend(self):
        """Monotonically increasing prices should give RSI near 100."""
        prices = pd.Series(list(range(100, 200)))
        rsi = calc_rsi_value(prices, period=14)
        assert rsi > 80

    def test_rsi_extreme_downtrend(self):
        """Monotonically decreasing prices should give RSI near 0."""
        prices = pd.Series(list(range(200, 100, -1)))
        rsi = calc_rsi_value(prices, period=14)
        assert rsi < 20
