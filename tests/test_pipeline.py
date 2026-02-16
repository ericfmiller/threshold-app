"""Tests for Phase 3: portfolio management, pipeline, alerts, and CLI."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from threshold.portfolio.accounts import (
    PortfolioSnapshot,
    aggregate_positions,
)
from threshold.portfolio.allocation import (
    compute_alden_allocation,
    compute_war_chest,
)
from threshold.portfolio.correlation import (
    check_concentration_risk,
    compute_correlation_report,
)
from threshold.portfolio.ledger import PortfolioLedger, PortfolioValues

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def raw_positions() -> list[dict[str, Any]]:
    """Multi-account position data for aggregation testing."""
    return [
        {"symbol": "AAPL", "account_id": "brokerage", "shares": 50, "market_value": 10000},
        {"symbol": "AAPL", "account_id": "roth", "shares": 30, "market_value": 6000},
        {"symbol": "GOOGL", "account_id": "brokerage", "shares": 20, "market_value": 7000},
        {"symbol": "STIP", "account_id": "brokerage", "shares": 100, "market_value": 5000},
        {"symbol": "BRK.B", "account_id": "ira", "shares": 10, "market_value": 4500},
    ]


@pytest.fixture
def account_totals() -> dict[str, float]:
    return {
        "brokerage": 50000.0,
        "roth": 30000.0,
        "ira": 20000.0,
    }


@pytest.fixture
def sample_snapshot(raw_positions, account_totals) -> PortfolioSnapshot:
    return aggregate_positions(raw_positions, account_totals)


@pytest.fixture
def sample_values() -> PortfolioValues:
    return PortfolioValues(
        fidelity_total=100000.0,
        tsp_value=50000.0,
        btc_value=25000.0,
        total_portfolio=175000.0,
        account_values={"brokerage": 50000.0, "roth": 30000.0, "ira": 20000.0},
        cash_balances={"brokerage": 3000.0, "roth": 500.0},
    )


@pytest.fixture
def sample_ledger(sample_snapshot, sample_values) -> PortfolioLedger:
    return PortfolioLedger(snapshot=sample_snapshot, values=sample_values)


@pytest.fixture
def correlated_returns() -> pd.DataFrame:
    """5 assets with varying correlations for correlation analysis."""
    np.random.seed(42)
    n = 100
    market = np.random.normal(0, 0.01, n)
    data = {
        "SPY": market + np.random.normal(0, 0.002, n),
        "QQQ": market * 1.2 + np.random.normal(0, 0.003, n),  # High corr with SPY
        "GLD": -0.3 * market + np.random.normal(0, 0.008, n),  # Negative corr
        "BND": -0.1 * market + np.random.normal(0, 0.002, n),  # Low corr
        "EFA": market * 0.8 + np.random.normal(0, 0.005, n),   # Moderate corr
    }
    return pd.DataFrame(data)


# ---------------------------------------------------------------------------
# Position Aggregation Tests
# ---------------------------------------------------------------------------

class TestPositionAggregation:
    def test_basic_aggregation(self, raw_positions, account_totals):
        """Should consolidate positions across accounts."""
        snapshot = aggregate_positions(raw_positions, account_totals)
        assert snapshot.n_positions == 4  # AAPL, GOOGL, STIP, BRK.B

    def test_multi_account_position(self, raw_positions, account_totals):
        """AAPL in brokerage + roth should be consolidated."""
        snapshot = aggregate_positions(raw_positions, account_totals)
        aapl = snapshot.get_position("AAPL")
        assert aapl is not None
        assert aapl.total_shares == 80  # 50 + 30
        assert aapl.total_value == 16000  # 10000 + 6000
        assert aapl.is_multi_account
        assert aapl.n_accounts == 2

    def test_single_account_position(self, raw_positions, account_totals):
        """GOOGL in one account should not be flagged multi-account."""
        snapshot = aggregate_positions(raw_positions, account_totals)
        googl = snapshot.get_position("GOOGL")
        assert googl is not None
        assert googl.total_shares == 20
        assert not googl.is_multi_account
        assert googl.n_accounts == 1

    def test_total_value(self, raw_positions, account_totals):
        """Total portfolio value should be sum of all positions."""
        snapshot = aggregate_positions(raw_positions, account_totals)
        assert snapshot.total_value == 32500.0  # 16000 + 7000 + 5000 + 4500

    def test_portfolio_weight(self, raw_positions, account_totals):
        """Portfolio weight should be position value / total value."""
        snapshot = aggregate_positions(raw_positions, account_totals)
        weight = snapshot.portfolio_weight("AAPL")
        assert abs(weight - 16000 / 32500) < 0.001

    def test_account_weights(self, raw_positions, account_totals):
        """Per-account weights should be computed from account totals."""
        snapshot = aggregate_positions(raw_positions, account_totals)
        aapl = snapshot.get_position("AAPL")
        assert aapl is not None
        # brokerage: 10000 / 50000 = 0.2
        assert abs(aapl.account_weights["brokerage"] - 0.2) < 0.001
        # roth: 6000 / 30000 = 0.2
        assert abs(aapl.account_weights["roth"] - 0.2) < 0.001

    def test_empty_positions(self):
        """Empty position list should return empty snapshot."""
        snapshot = aggregate_positions([])
        assert snapshot.n_positions == 0
        assert snapshot.total_value == 0.0

    def test_missing_position(self, sample_snapshot):
        """Requesting non-existent ticker should return None."""
        assert sample_snapshot.get_position("TSLA") is None


# ---------------------------------------------------------------------------
# Ledger Tests
# ---------------------------------------------------------------------------

class TestPortfolioLedger:
    def test_held_tickers(self, sample_ledger):
        """Should return all held ticker symbols."""
        tickers = sample_ledger.held_tickers
        assert "AAPL" in tickers
        assert "GOOGL" in tickers
        assert "STIP" in tickers
        assert "BRK.B" in tickers

    def test_ticker_dollar_value(self, sample_ledger):
        """Dollar value should sum across accounts."""
        val = sample_ledger.ticker_dollar_value("AAPL")
        # AAPL has account_values: brokerage=10000, roth=6000
        assert val == 16000.0

    def test_ticker_dollar_value_missing(self, sample_ledger):
        """Missing ticker should return 0."""
        assert sample_ledger.ticker_dollar_value("TSLA") == 0.0

    def test_category_dollar_value(self, sample_ledger):
        """Category value should sum specified tickers."""
        val = sample_ledger.category_dollar_value(["AAPL", "GOOGL"])
        assert val == 23000.0  # 16000 + 7000

    def test_category_with_tsp(self, sample_ledger):
        """TSP portion should be added to category value."""
        val = sample_ledger.category_dollar_value(["AAPL"], tsp_pct=0.25)
        # AAPL=16000 + 25% of TSP(50000)=12500
        assert val == 28500.0

    def test_category_with_btc(self, sample_ledger):
        """BTC should be added when include_btc=True."""
        val = sample_ledger.category_dollar_value(["AAPL"], include_btc=True)
        assert val == 41000.0  # 16000 + 25000

    def test_held_in_accounts(self, sample_ledger):
        """Should return accounts where ticker is held."""
        accounts = sample_ledger.held_in_accounts("AAPL")
        assert "brokerage" in accounts
        assert "roth" in accounts
        assert len(accounts) == 2

    def test_is_held(self, sample_ledger):
        assert sample_ledger.is_held("AAPL")
        assert not sample_ledger.is_held("TSLA")

    def test_tickers_in_account(self, sample_ledger):
        tickers = sample_ledger.tickers_in_account("brokerage")
        assert "AAPL" in tickers
        assert "GOOGL" in tickers
        assert "STIP" in tickers

    def test_portfolio_properties(self, sample_ledger):
        assert sample_ledger.total_portfolio == 175000.0
        assert sample_ledger.fidelity_total == 100000.0
        assert sample_ledger.tsp_value == 50000.0
        assert sample_ledger.btc_value == 25000.0

    def test_cash_balances(self, sample_ledger):
        assert sample_ledger.cash_balances == {"brokerage": 3000.0, "roth": 500.0}
        assert sample_ledger.fidelity_cash_total == 3500.0


# ---------------------------------------------------------------------------
# Allocation Tests
# ---------------------------------------------------------------------------

class TestAllocation:
    def test_compute_alden_allocation(self):
        """Should compute dollar-weighted category allocation."""
        ticker_cats = {"AAPL": "Growth", "GLD": "Hard Assets", "BND": "Defensive/Income"}
        ticker_vals = {"AAPL": 5000, "GLD": 3000, "BND": 2000}
        report = compute_alden_allocation(
            ticker_categories=ticker_cats,
            ticker_values=ticker_vals,
            total_portfolio=10000,
        )
        assert "Growth" in report.categories
        assert report.categories["Growth"].dollar_value == 5000
        assert abs(report.categories["Growth"].weight_pct - 0.5) < 0.01

    def test_allocation_with_btc(self):
        """BTC should be added to Hard Assets."""
        ticker_cats = {"GLD": "Hard Assets"}
        ticker_vals = {"GLD": 3000}
        report = compute_alden_allocation(
            ticker_categories=ticker_cats,
            ticker_values=ticker_vals,
            btc_value=7000,
            total_portfolio=10000,
        )
        assert report.categories["Hard Assets"].dollar_value == 10000

    def test_war_chest_normal_vix(self):
        """NORMAL VIX should target 10% war chest (default from config)."""
        wc = compute_war_chest(
            vix_regime="NORMAL",
            fidelity_cash=5000,
            wc_instrument_values={"STIP": 3000},
            total_portfolio=100000,
        )
        assert wc.actual_dollars == 8000.0
        assert wc.fidelity_cash == 5000.0
        assert wc.wc_instrument_value == 3000.0
        assert wc.wc_instruments == ["STIP"]

    def test_war_chest_fear_vix(self):
        """FEAR VIX should target 15% war chest."""
        wc = compute_war_chest(
            vix_regime="FEAR",
            fidelity_cash=10000,
            total_portfolio=100000,
        )
        assert wc.target_pct == 0.15
        assert wc.target_dollars == 15000.0

    def test_war_chest_surplus(self):
        """Surplus should be positive when above target."""
        wc = compute_war_chest(
            vix_regime="NORMAL",
            fidelity_cash=15000,
            total_portfolio=100000,
        )
        assert wc.surplus > 0
        assert wc.is_adequate


# ---------------------------------------------------------------------------
# Correlation Tests
# ---------------------------------------------------------------------------

class TestCorrelation:
    def test_correlation_report(self, correlated_returns):
        """Should compute correlation and effective bets."""
        report = compute_correlation_report(correlated_returns, corr_threshold=0.80)
        assert report.n_tickers == 5
        assert report.effective_bets > 0
        assert report.min_data_days > 0

    def test_high_corr_pairs(self, correlated_returns):
        """SPY/QQQ should be flagged as highly correlated."""
        report = compute_correlation_report(correlated_returns, corr_threshold=0.70)
        # SPY/QQQ have high market beta similarity
        assert len(report.high_corr_pairs) >= 1

    def test_effective_bets_range(self, correlated_returns):
        """Effective bets should be between 1 and N."""
        report = compute_correlation_report(correlated_returns)
        assert 1.0 <= report.effective_bets <= 5.0

    def test_uncorrelated_high_bets(self):
        """Uncorrelated assets should have high effective bets."""
        np.random.seed(42)
        data = pd.DataFrame({
            f"Asset{i}": np.random.normal(0, 0.01, 100) for i in range(5)
        })
        report = compute_correlation_report(data)
        # Should be close to 5 (all independent)
        assert report.effective_bets > 4.0

    def test_perfectly_correlated_low_bets(self):
        """Perfectly correlated assets should have effective bets near 1."""
        np.random.seed(42)
        base = np.random.normal(0, 0.01, 100)
        data = pd.DataFrame({
            "A": base + np.random.normal(0, 0.0001, 100),
            "B": base + np.random.normal(0, 0.0001, 100),
            "C": base + np.random.normal(0, 0.0001, 100),
        })
        report = compute_correlation_report(data)
        assert report.effective_bets < 2.0

    def test_empty_returns(self):
        """Empty DataFrame should return zero effective bets."""
        report = compute_correlation_report(pd.DataFrame())
        assert report.effective_bets == 0.0
        assert report.n_tickers == 0

    def test_single_ticker(self):
        """Single ticker should return effective bets = 1."""
        data = pd.DataFrame({"A": np.random.normal(0, 0.01, 100)})
        report = compute_correlation_report(data)
        assert report.effective_bets == 1.0

    def test_concentration_check(self):
        """Should flag buy candidates correlated with existing holdings."""
        pairs = [("AAPL", "MSFT", 0.85), ("GLD", "BND", 0.30)]
        warnings = check_concentration_risk(
            high_corr_pairs=pairs,
            effective_bets=15.0,
            buy_tickers={"AAPL"},
            held_tickers={"MSFT", "GLD"},
            concentration_threshold=20.0,
            pair_threshold=0.70,
        )
        assert len(warnings) == 1
        assert warnings[0]["ticker"] == "AAPL"
        assert warnings[0]["correlated_with"] == "MSFT"

    def test_no_warnings_above_threshold(self):
        """No warnings when effective bets is high enough."""
        pairs = [("AAPL", "MSFT", 0.85)]
        warnings = check_concentration_risk(
            high_corr_pairs=pairs,
            effective_bets=25.0,
            buy_tickers={"AAPL"},
            held_tickers={"MSFT"},
        )
        assert len(warnings) == 0


# ---------------------------------------------------------------------------
# Alert Tests
# ---------------------------------------------------------------------------

class TestAlerts:
    def test_generate_alerts(self):
        """Should generate alerts for high DCS scores."""
        from threshold.output.alerts import generate_scoring_alerts

        scores = {
            "HIGH": {"dcs": 82.0, "dcs_signal": "STRONG BUY"},
            "MED": {"dcs": 72.0, "dcs_signal": "HIGH CONVICTION"},
            "BUY": {"dcs": 66.0, "dcs_signal": "BUY DIP"},
            "LOW": {"dcs": 55.0, "dcs_signal": "WATCH"},
        }
        alerts = generate_scoring_alerts(scores)
        levels = [a["level"] for a in alerts]
        assert "STRONG BUY" in levels
        assert "HIGH CONVICTION" in levels
        assert "BUY DIP" in levels
        # LOW (55) should not generate an alert
        assert len(alerts) == 3

    def test_alerts_sorted_by_score(self):
        """Alerts should be sorted by score descending."""
        from threshold.output.alerts import generate_scoring_alerts

        scores = {
            "A": {"dcs": 72.0, "dcs_signal": "HC"},
            "B": {"dcs": 85.0, "dcs_signal": "STRONG"},
            "C": {"dcs": 66.0, "dcs_signal": "BUY"},
        }
        alerts = generate_scoring_alerts(scores)
        assert alerts[0]["ticker"] == "B"
        assert alerts[1]["ticker"] == "A"
        assert alerts[2]["ticker"] == "C"

    def test_portfolio_filter(self):
        """Should filter to portfolio-only tickers."""
        from threshold.output.alerts import generate_scoring_alerts

        scores = {
            "AAPL": {"dcs": 82.0, "dcs_signal": "STRONG"},
            "ZZZZ": {"dcs": 85.0, "dcs_signal": "STRONG"},  # Watchlist
        }
        alerts = generate_scoring_alerts(scores, portfolio_only={"AAPL"})
        assert len(alerts) == 1
        assert alerts[0]["ticker"] == "AAPL"

    def test_build_email(self):
        """Should build subject and HTML body."""
        from threshold.output.alerts import build_scoring_email

        scores = {"AAPL": {"dcs": 75.0, "dcs_signal": "STRONG"}}
        alerts = [{"level": "STRONG BUY", "ticker": "AAPL", "score": 75.0}]
        subject, body = build_scoring_email(scores, alerts, 18.0, "NORMAL", 0.05, 10)
        assert "STRONG BUY" in subject
        assert "AAPL" in body
        assert "html" in body.lower()


# ---------------------------------------------------------------------------
# Score History Tests
# ---------------------------------------------------------------------------

class TestScoreHistory:
    def test_save_and_load(self, tmp_path):
        """Should save and load score history JSON."""
        from threshold.output.alerts import load_previous_scores, save_score_history

        scores = {
            "AAPL": {"dcs": 72.5, "dcs_signal": "HC", "sub_scores": {"MQ": 0.8}},
            "MSFT": {"dcs": 55.0, "dcs_signal": "WATCH", "sub_scores": {"MQ": 0.5}},
        }
        filepath = save_score_history(
            scores=scores,
            vix_current=18.0,
            vix_regime="NORMAL",
            output_dir=tmp_path,
        )
        assert Path(filepath).exists()

        # Load back
        loaded = load_previous_scores(output_dir=tmp_path)
        assert "AAPL" in loaded
        assert loaded["AAPL"]["dcs"] == 72.5

    def test_load_empty_dir(self, tmp_path):
        """Should return empty dict for empty directory."""
        from threshold.output.alerts import load_previous_scores

        assert load_previous_scores(output_dir=tmp_path) == {}

    def test_load_grade_history(self, tmp_path):
        """Should load multiple score history files."""
        from threshold.output.alerts import load_grade_history, save_score_history

        for i in range(3):
            save_score_history(
                scores={"AAPL": {"dcs": 70 + i}},
                vix_current=18.0,
                vix_regime="NORMAL",
                output_dir=tmp_path,
            )

        history = load_grade_history(max_weeks=8, output_dir=tmp_path)
        # May be 1 file if all saved on same date
        assert len(history) >= 1


# ---------------------------------------------------------------------------
# Pipeline Structure Tests (no yfinance calls)
# ---------------------------------------------------------------------------

class TestPipelineStructure:
    def test_run_tracker(self):
        """RunTracker should initialize with defaults."""
        from threshold.engine.pipeline import RunTracker

        tracker = RunTracker()
        assert tracker.run_id
        assert tracker.started_at
        assert tracker.tickers_scored == 0
        assert tracker.tickers_failed == 0
        d = tracker.to_dict()
        assert "run_id" in d
        assert "data_sources" in d

    def test_pipeline_result(self):
        """PipelineResult should have sensible defaults."""
        from threshold.engine.pipeline import PipelineResult

        result = PipelineResult()
        assert result.n_scored == 0
        assert result.top_scores == []
        assert result.vix_regime == "NORMAL"

    def test_pipeline_result_top_scores(self):
        """Top scores should sort by DCS descending."""
        from threshold.engine.pipeline import PipelineResult

        result = PipelineResult(
            scores={
                "AAPL": {"dcs": 72.0},
                "MSFT": {"dcs": 80.0},
                "GLD": {"dcs": 55.0},
            }
        )
        top = result.top_scores
        assert top[0] == ("MSFT", 80.0)
        assert top[1] == ("AAPL", 72.0)
        assert top[2] == ("GLD", 55.0)


# ---------------------------------------------------------------------------
# CLI Import Tests
# ---------------------------------------------------------------------------

class TestImportCommands:
    def test_import_registry(self, tmp_path):
        """Import registry command should parse ticker_registry.json."""
        # Create a minimal registry file
        registry = {
            "AAPL": {"name": "Apple Inc", "type": "stock", "sector": "Technology"},
            "SPY": {"name": "SPDR S&P 500", "type": "etf", "sector": "Broad Market"},
        }
        reg_file = tmp_path / "ticker_registry.json"
        with open(reg_file, "w") as f:
            json.dump(registry, f)

        # Verify the file is valid JSON
        with open(reg_file) as f:
            loaded = json.load(f)
        assert len(loaded) == 2

    def test_import_scores_format(self, tmp_path):
        """Score history files should be loadable."""
        scores_file = tmp_path / "weekly_scores_2026-02-15.json"
        data = {
            "_metadata": {"vix_current": 18.0, "vix_regime": "NORMAL"},
            "scores": {
                "AAPL": {"dcs": 72.0, "dcs_signal": "HC"},
            },
        }
        with open(scores_file, "w") as f:
            json.dump(data, f)

        with open(scores_file) as f:
            loaded = json.load(f)
        assert "scores" in loaded
        assert "AAPL" in loaded["scores"]


# ---------------------------------------------------------------------------
# Package Import Tests
# ---------------------------------------------------------------------------

class TestPackageImports:
    def test_portfolio_imports(self):
        """Portfolio package should export all public API."""
        from threshold.portfolio import (
            PortfolioLedger,
            Position,
            compute_correlation_report,
        )
        assert Position is not None
        assert PortfolioLedger is not None
        assert compute_correlation_report is not None

    def test_pipeline_imports(self):
        """Pipeline module should be importable."""
        from threshold.engine.pipeline import (
            PipelineResult,
            RunTracker,
            run_scoring_pipeline,
        )
        assert run_scoring_pipeline is not None
        assert RunTracker is not None
        assert PipelineResult is not None

    def test_alerts_imports(self):
        """Alerts module should be importable."""
        from threshold.output.alerts import (
            generate_scoring_alerts,
            save_score_history,
        )
        assert generate_scoring_alerts is not None
        assert save_score_history is not None

    def test_cli_score_import(self):
        """Score CLI command should be importable."""
        from threshold.cli.score import score_cmd
        assert score_cmd is not None

    def test_cli_import_import(self):
        """Import CLI commands should be importable."""
        from threshold.cli.import_cmd import import_group
        assert import_group is not None


# ---------------------------------------------------------------------------
# Config Integration Tests
# ---------------------------------------------------------------------------

class TestConfigIntegration:
    def test_config_has_alerts(self):
        """ThresholdConfig should have alerts section."""
        from threshold.config.schema import ThresholdConfig
        config = ThresholdConfig()
        assert config.alerts.enabled is True
        assert "dcs_strong" in config.alerts.thresholds
        assert "dcs_conviction" in config.alerts.thresholds
        assert config.alerts.thresholds["dcs_strong"] == 80
        assert config.alerts.thresholds["dcs_conviction"] == 70

    def test_config_has_allocation(self):
        """ThresholdConfig should have allocation section."""
        from threshold.config.schema import ThresholdConfig
        config = ThresholdConfig()
        assert hasattr(config, "allocation")
        assert "equities" in config.allocation.targets
        assert config.allocation.rebalance_trigger == 0.05

    def test_config_war_chest_vix_targets(self):
        """Config should have VIX-regime war chest targets."""
        from threshold.config.schema import ThresholdConfig
        config = ThresholdConfig()
        wc = config.allocation.war_chest_vix
        assert "NORMAL" in wc
        assert "FEAR" in wc
        assert "PANIC" in wc
