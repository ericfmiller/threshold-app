"""Tests for threshold.engine.aggregator — cross-module signal aggregation."""

from __future__ import annotations

import pytest

from threshold.engine.aggregator import SignalAggregator

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def default_aggregator() -> SignalAggregator:
    """Default aggregator with standard weights and thresholds."""
    return SignalAggregator()


@pytest.fixture
def calm_signals() -> tuple[dict, dict, dict]:
    """All calm/normal signals — low risk."""
    ebp = {"regime": "ACCOMMODATIVE", "ebp_value": -0.5}
    turb = {"regime": "CALM", "percentile": 0.15}
    crash = {"is_bear": False, "crash_probability": 0.05}
    return ebp, turb, crash


@pytest.fixture
def stressed_signals() -> tuple[dict, dict, dict]:
    """All stressed signals — high risk."""
    ebp = {"regime": "HIGH_RISK", "ebp_value": 1.5}
    turb = {"regime": "TURBULENT", "percentile": 0.92}
    crash = {"is_bear": True, "crash_probability": 0.85}
    return ebp, turb, crash


@pytest.fixture
def mixed_signals() -> tuple[dict, dict, dict]:
    """Mixed signals — elevated risk."""
    ebp = {"regime": "HIGH_RISK", "ebp_value": 1.2}
    turb = {"regime": "ELEVATED", "percentile": 0.50}
    crash = {"is_bear": False, "crash_probability": 0.10}
    return ebp, turb, crash


# ---------------------------------------------------------------------------
# Composite Risk Tests
# ---------------------------------------------------------------------------

class TestCompositeRisk:
    def test_all_calm_normal_regime(self, default_aggregator, calm_signals):
        """All calm signals should produce NORMAL regime."""
        ebp, turb, crash = calm_signals
        result = default_aggregator.compute_composite_risk(ebp, turb, crash)
        assert result["regime"] == "NORMAL"
        assert result["composite_score"] < 0.40
        assert result["dcs_penalty"] == 0

    def test_all_stressed_high_risk(self, default_aggregator, stressed_signals):
        """All stressed signals should produce HIGH_RISK regime."""
        ebp, turb, crash = stressed_signals
        result = default_aggregator.compute_composite_risk(ebp, turb, crash)
        assert result["regime"] == "HIGH_RISK"
        assert result["composite_score"] >= 0.70
        assert result["dcs_penalty"] == 10

    def test_mixed_elevated_regime(self, default_aggregator, mixed_signals):
        """Mixed signals with elevated EBP should produce ELEVATED regime."""
        ebp, turb, crash = mixed_signals
        result = default_aggregator.compute_composite_risk(ebp, turb, crash)
        assert result["regime"] == "ELEVATED"
        assert 0.40 <= result["composite_score"] < 0.70
        assert result["dcs_penalty"] == 5

    def test_single_high_signal(self, default_aggregator):
        """A single high signal should elevate the composite."""
        # Only EBP is HIGH_RISK, others are None
        result = default_aggregator.compute_composite_risk(
            ebp_signal={"regime": "HIGH_RISK"},
            turb_signal=None,
            crash_signal=None,
        )
        # EBP=1.0 * 0.40 = 0.40, others=0 → composite=0.40 → ELEVATED
        assert result["regime"] == "ELEVATED"
        assert result["ebp_contrib"] == 0.40
        assert result["turbulence_contrib"] == 0.0
        assert result["crash_contrib"] == 0.0

    def test_all_none_signals(self, default_aggregator):
        """All None signals should produce NORMAL regime with zero penalty."""
        result = default_aggregator.compute_composite_risk(None, None, None)
        assert result["regime"] == "NORMAL"
        assert result["composite_score"] == 0.0
        assert result["dcs_penalty"] == 0

    def test_composite_clamped_to_one(self):
        """Composite score should never exceed 1.0."""
        agg = SignalAggregator(
            ebp_weight=0.50,
            turbulence_weight=0.50,
            crash_weight=0.50,  # Weights sum > 1 intentionally
        )
        result = agg.compute_composite_risk(
            {"regime": "HIGH_RISK"},
            {"percentile": 1.0},
            {"crash_probability": 1.0},
        )
        assert result["composite_score"] <= 1.0

    def test_contributions_sum_to_composite(self, default_aggregator, stressed_signals):
        """Individual contributions should sum to composite score."""
        ebp, turb, crash = stressed_signals
        result = default_aggregator.compute_composite_risk(ebp, turb, crash)
        contrib_sum = (
            result["ebp_contrib"]
            + result["turbulence_contrib"]
            + result["crash_contrib"]
        )
        # Allow small rounding tolerance
        assert abs(contrib_sum - result["composite_score"]) < 0.01


# ---------------------------------------------------------------------------
# Risk Overlay Tests
# ---------------------------------------------------------------------------

class TestRiskOverlay:
    def test_high_risk_penalty(self, default_aggregator, stressed_signals):
        """HIGH_RISK should apply -10 penalty to DCS."""
        ebp, turb, crash = stressed_signals
        composite = default_aggregator.compute_composite_risk(ebp, turb, crash)
        adjusted = default_aggregator.apply_risk_overlay(72.0, composite)
        assert adjusted == 62.0

    def test_elevated_penalty(self, default_aggregator, mixed_signals):
        """ELEVATED should apply -5 penalty to DCS."""
        ebp, turb, crash = mixed_signals
        composite = default_aggregator.compute_composite_risk(ebp, turb, crash)
        adjusted = default_aggregator.apply_risk_overlay(72.0, composite)
        assert adjusted == 67.0

    def test_normal_no_penalty(self, default_aggregator, calm_signals):
        """NORMAL should not change DCS."""
        ebp, turb, crash = calm_signals
        composite = default_aggregator.compute_composite_risk(ebp, turb, crash)
        adjusted = default_aggregator.apply_risk_overlay(72.0, composite)
        assert adjusted == 72.0

    def test_penalty_floor_at_zero(self, default_aggregator, stressed_signals):
        """DCS should not go below 0 after penalty."""
        ebp, turb, crash = stressed_signals
        composite = default_aggregator.compute_composite_risk(ebp, turb, crash)
        adjusted = default_aggregator.apply_risk_overlay(5.0, composite)
        assert adjusted == 0.0

    def test_disabled_passthrough(self):
        """When aggregator is not invoked, DCS is unchanged."""
        # This tests the pattern used in scorer.py: simply don't call aggregator
        dcs = 75.0
        # Without calling apply_risk_overlay, DCS stays unchanged
        assert dcs == 75.0  # Trivial, but documents the disabled pattern


# ---------------------------------------------------------------------------
# Config Tests
# ---------------------------------------------------------------------------

class TestAggregatorConfig:
    def test_config_aggregator_section(self):
        from threshold.config.schema import ThresholdConfig
        config = ThresholdConfig()
        assert hasattr(config, "aggregator")
        assert config.aggregator.enabled is False
        assert config.aggregator.ebp_weight == 0.40
        assert config.aggregator.turbulence_weight == 0.30
        assert config.aggregator.crash_weight == 0.30
        assert config.aggregator.high_risk_threshold == 0.70
        assert config.aggregator.elevated_threshold == 0.40
        assert config.aggregator.high_risk_penalty == 10
        assert config.aggregator.elevated_penalty == 5

    def test_custom_weights(self):
        """Custom weights should produce different composite scores."""
        agg_default = SignalAggregator()
        agg_turb_heavy = SignalAggregator(
            ebp_weight=0.10,
            turbulence_weight=0.80,
            crash_weight=0.10,
        )
        signal = {"percentile": 0.95}
        r1 = agg_default.compute_composite_risk(turb_signal=signal)
        r2 = agg_turb_heavy.compute_composite_risk(turb_signal=signal)
        # Turbulence-heavy weighting should produce higher score
        assert r2["composite_score"] > r1["composite_score"]
