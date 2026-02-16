"""Tests for Gate 3 parabolic filter."""

from __future__ import annotations

from unittest.mock import MagicMock

from threshold.engine.gate3 import Gate3Result, check_gate3

# ---------------------------------------------------------------------------
# Tests: Gate3Result
# ---------------------------------------------------------------------------

class TestGate3Result:
    def test_default_passes(self):
        """Default Gate3Result should pass at FULL sizing."""
        result = Gate3Result()
        assert result.passes is True
        assert result.sizing == "FULL"


# ---------------------------------------------------------------------------
# Tests: check_gate3
# ---------------------------------------------------------------------------

class TestCheckGate3:
    def test_normal_passes(self):
        """Normal RSI and return should pass at FULL sizing."""
        result = check_gate3(rsi=55.0, ret_8w=0.10)
        assert result.passes is True
        assert result.sizing == "FULL"

    def test_both_triggered_fail(self):
        """RSI > 80 AND 8w > 30% should FAIL."""
        result = check_gate3(rsi=85.0, ret_8w=0.35)
        assert result.passes is False
        assert result.sizing == "FAIL"

    def test_rsi_only_wait(self):
        """RSI > 80 alone should WAIT."""
        result = check_gate3(rsi=85.0, ret_8w=0.10)
        assert result.passes is False
        assert result.sizing == "WAIT"

    def test_ret_only_wait(self):
        """8w return > 30% alone should WAIT."""
        result = check_gate3(rsi=55.0, ret_8w=0.35)
        assert result.passes is False
        assert result.sizing == "WAIT"

    def test_gold_exempt_at_high_rsi(self):
        """Gold at RSI > 80 should pass at THREE_QUARTER sizing (D-13)."""
        result = check_gate3(rsi=85.0, ret_8w=0.35, is_gold=True)
        assert result.passes is True
        assert result.sizing == "THREE_QUARTER"
        assert result.is_gold_exempt is True

    def test_gold_normal_full_sizing(self):
        """Gold at normal RSI should pass at FULL sizing."""
        result = check_gate3(rsi=55.0, ret_8w=0.10, is_gold=True)
        assert result.passes is True
        assert result.sizing == "FULL"
        assert result.is_gold_exempt is True

    def test_custom_thresholds_via_config(self):
        """Should use config thresholds when provided."""
        config = MagicMock()
        config.deployment.gate3_rsi_max = 70
        config.deployment.gate3_ret_8w_max = 0.20
        config.deployment.gold_rsi_max_sizing = 0.75

        # RSI 75 would pass default (80) but fail custom (70)
        result = check_gate3(rsi=75.0, ret_8w=0.25, config=config)
        assert result.passes is False
        assert result.sizing == "FAIL"

    def test_boundary_exact_rsi(self):
        """RSI exactly at threshold should pass (uses > not >=)."""
        result = check_gate3(rsi=80.0, ret_8w=0.30)
        assert result.passes is True
        assert result.sizing == "FULL"

    def test_boundary_just_above(self):
        """RSI just above threshold should WAIT."""
        result = check_gate3(rsi=80.1, ret_8w=0.10)
        assert result.passes is False
        assert result.sizing == "WAIT"

    def test_result_has_values(self):
        """Result should contain the input RSI and ret_8w values."""
        result = check_gate3(rsi=65.3, ret_8w=0.1234)
        assert result.rsi == 65.3
        assert result.ret_8w == 0.1234

    def test_no_config_uses_defaults(self):
        """Should use default thresholds when config is None."""
        result = check_gate3(rsi=85.0, ret_8w=0.35, config=None)
        assert result.passes is False
        assert result.sizing == "FAIL"
