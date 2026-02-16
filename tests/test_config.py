"""Tests for the configuration system."""

from __future__ import annotations

import os

import pytest
import yaml

from threshold.config.defaults import (
    DCS_WEIGHTS,
    FALLING_KNIFE_CAPS,
    GRADE_TO_NUM,
    MQ_WEIGHTS,
    SIGNAL_THRESHOLDS,
    VIX_REGIMES,
)
from threshold.config.loader import _expand_env_vars, load_config
from threshold.config.schema import ThresholdConfig


class TestDefaults:
    """Verify all calibrated defaults are present and correct."""

    def test_dcs_weights_sum_to_100(self):
        assert sum(DCS_WEIGHTS.values()) == 100

    def test_mq_weights_sum_to_1(self):
        total = sum(MQ_WEIGHTS.values())
        assert abs(total - 1.0) < 0.001

    def test_signal_thresholds_ordered(self):
        t = SIGNAL_THRESHOLDS
        assert t["strong_buy_dip"] > t["high_conviction"] > t["buy_dip"] > t["watch"] > t["weak"]

    def test_vix_regimes_contiguous(self):
        regimes = VIX_REGIMES
        assert regimes["COMPLACENT"][1] == regimes["NORMAL"][0]
        assert regimes["NORMAL"][1] == regimes["FEAR"][0]
        assert regimes["FEAR"][1] == regimes["PANIC"][0]

    def test_falling_knife_caps_consistent(self):
        freefall = FALLING_KNIFE_CAPS["freefall"]
        downtrend = FALLING_KNIFE_CAPS["downtrend"]
        for cls in ["HEDGE", "DEFENSIVE", "MODERATE", "CYCLICAL", "AMPLIFIER"]:
            assert freefall[cls] <= downtrend[cls], f"Freefall cap should be <= downtrend for {cls}"

    def test_grade_to_num_complete(self):
        expected_grades = ["A+", "A", "A-", "B+", "B", "B-", "C+", "C", "C-", "D+", "D", "D-", "F"]
        for grade in expected_grades:
            assert grade in GRADE_TO_NUM

    def test_grade_to_num_ordered(self):
        assert GRADE_TO_NUM["A+"] > GRADE_TO_NUM["A"] > GRADE_TO_NUM["B+"] > GRADE_TO_NUM["F"]


class TestConfigLoading:
    """Test config file loading and validation."""

    def test_load_defaults_no_file(self):
        config = load_config("/nonexistent/path.yaml")
        assert isinstance(config, ThresholdConfig)
        assert config.version == 1
        assert config.scoring.weights.MQ == 30

    def test_load_from_yaml(self, tmp_path):
        yaml_content = {
            "version": 1,
            "scoring": {"weights": {"MQ": 35, "FQ": 25, "TO": 20, "MR": 10, "VC": 10}},
        }
        config_file = tmp_path / "config.yaml"
        with open(config_file, "w") as f:
            yaml.dump(yaml_content, f)

        config = load_config(str(config_file))
        assert config.scoring.weights.MQ == 35

    def test_weights_must_sum_to_100(self):
        with pytest.raises(ValueError):
            ThresholdConfig(
                scoring={"weights": {"MQ": 30, "FQ": 25, "TO": 20, "MR": 15, "VC": 5}}
            )

    def test_env_var_expansion(self):
        os.environ["TEST_THRESHOLD_KEY"] = "secret123"
        try:
            result = _expand_env_vars("key=${TEST_THRESHOLD_KEY}")
            assert result == "key=secret123"
        finally:
            del os.environ["TEST_THRESHOLD_KEY"]

    def test_env_var_missing_returns_empty(self):
        result = _expand_env_vars("${NONEXISTENT_VAR_12345}")
        assert result == ""

    def test_nested_env_expansion(self):
        os.environ["TEST_VAL"] = "hello"
        try:
            result = _expand_env_vars({"key": "${TEST_VAL}", "nested": {"deep": "${TEST_VAL}"}})
            assert result == {"key": "hello", "nested": {"deep": "hello"}}
        finally:
            del os.environ["TEST_VAL"]

    def test_alden_categories_default_populated(self):
        config = ThresholdConfig()
        assert "US Large Cap" in config.alden_categories
        assert "Hard Assets" in config.alden_categories
        assert len(config.alden_categories) == 7


class TestConfigSchema:
    """Test Pydantic schema validation."""

    def test_full_default_config_valid(self):
        config = ThresholdConfig()
        assert config.scoring.weights.MQ == 30
        assert config.scoring.thresholds.buy_dip == 65
        assert config.deployment.gate3_rsi_max == 80

    def test_accounts_list(self):
        config = ThresholdConfig(
            accounts=[
                {
                    "id": "test",
                    "name": "Test Account",
                    "type": "taxable",
                }
            ]
        )
        assert len(config.accounts) == 1
        assert config.accounts[0].id == "test"

    def test_separate_holdings(self):
        config = ThresholdConfig(
            separate_holdings=[
                {"symbol": "BTC-USD", "quantity": 2.0, "description": "Bitcoin"}
            ]
        )
        assert len(config.separate_holdings) == 1
        assert config.separate_holdings[0].quantity == 2.0
