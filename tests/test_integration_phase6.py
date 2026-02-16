"""Integration tests for Phase 6E — pipeline wiring, narrative, dashboard."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from threshold.engine.exemptions import ExemptionResult
from threshold.engine.pipeline import PipelineResult, RunTracker
from threshold.output.narrative import (
    _build_exemption_section,
    _build_gate3_section,
    _build_grace_period_section,
    generate_narrative,
)
from threshold.portfolio.correlation import CorrelationReport

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_pipeline_result(**overrides):
    """Build a minimal PipelineResult for testing."""
    defaults = {
        "run_id": "test-123",
        "vix_current": 16.5,
        "vix_regime": "NORMAL",
        "spy_pct_from_200d": 0.02,
        "spy_above_200d": True,
        "market_regime_score": 0.55,
        "breadth_pct": 0.62,
        "tracker": RunTracker(run_id="test-123"),
        "correlation": CorrelationReport(),
    }
    defaults.update(overrides)
    return PipelineResult(**defaults)


# ---------------------------------------------------------------------------
# Tests: Grace Period Narrative Section
# ---------------------------------------------------------------------------

class TestGracePeriodNarrativeSection:
    def test_empty_grace_periods(self):
        """Should show 'no active' message when list is empty."""
        output = _build_grace_period_section([])
        assert "No active grace periods" in output

    def test_none_grace_periods(self):
        """Should handle None input."""
        output = _build_grace_period_section(None)
        assert "No active grace periods" in output

    def test_renders_active_grace_periods(self):
        """Should render table with active grace periods."""
        periods = [
            {
                "symbol": "AAPL",
                "tier": 180,
                "days_remaining": 120,
                "reason": "Momentum fading but thesis intact",
                "expires_at": "2026-07-15",
            },
            {
                "symbol": "MSFT",
                "tier": 90,
                "days_remaining": 45,
                "reason": "Below SMA but earnings strong",
                "expires_at": "2026-04-01",
            },
        ]
        output = _build_grace_period_section(periods)
        assert "active grace period(s)" in output
        assert "AAPL" in output
        assert "MSFT" in output
        assert "180d" in output
        assert "90d" in output
        assert "120" in output
        assert "45" in output


# ---------------------------------------------------------------------------
# Tests: Exemption Narrative Section
# ---------------------------------------------------------------------------

class TestExemptionNarrativeSection:
    def test_empty_exemptions(self):
        """Should show 'no exemptions' message."""
        output = _build_exemption_section({})
        assert "No tickers with exemptions" in output

    def test_none_exemptions(self):
        """Should handle None input."""
        output = _build_exemption_section(None)
        assert "No tickers with exemptions" in output

    def test_renders_crypto_exemptions(self):
        """Should render crypto halving exemptions."""
        exemptions = {
            "FBTC": ExemptionResult(
                is_exempt=True,
                reason="FBTC exempt — Bitcoin halving cycle hold",
                exemption_type="crypto_halving",
                expires_at="2026-11-15",
            ),
            "MSTR": ExemptionResult(
                is_exempt=True,
                reason="MSTR exempt — Bitcoin halving cycle hold",
                exemption_type="crypto_halving",
                expires_at="2026-11-15",
            ),
        }
        output = _build_exemption_section(exemptions)
        assert "Crypto Halving Cycle" in output
        assert "FBTC" in output
        assert "MSTR" in output
        assert "2026-11-15" in output

    def test_renders_cash_exemptions(self):
        """Should render cash/war chest exemptions."""
        exemptions = {
            "STIP": ExemptionResult(
                is_exempt=True,
                reason="STIP is a cash/war chest position",
                exemption_type="cash",
            ),
        }
        output = _build_exemption_section(exemptions)
        assert "Cash / War Chest" in output
        assert "STIP" in output

    def test_renders_expired_exemptions(self):
        """Should show expired exemptions."""
        exemptions = {
            "FBTC": ExemptionResult(
                is_exempt=False,
                reason="Crypto exemption expired",
                exemption_type="crypto_halving",
                expires_at="2025-01-01",
                is_expired=True,
            ),
        }
        output = _build_exemption_section(exemptions)
        assert "Expired Exemptions" in output
        assert "2025-01-01" in output

    def test_mixed_exemptions(self):
        """Should render all types together."""
        exemptions = {
            "FBTC": ExemptionResult(
                is_exempt=True,
                exemption_type="crypto_halving",
                expires_at="2026-11-15",
            ),
            "STIP": ExemptionResult(
                is_exempt=True,
                exemption_type="cash",
            ),
        }
        output = _build_exemption_section(exemptions)
        assert "Crypto Halving Cycle" in output
        assert "Cash / War Chest" in output


# ---------------------------------------------------------------------------
# Tests: Gate 3 Narrative Section
# ---------------------------------------------------------------------------

class TestGate3NarrativeSection:
    def test_no_buy_candidates(self):
        """Should show 'no buy candidates' when no DCS >= 65."""
        result = _make_pipeline_result(scores={
            "AAPL": {"dcs": 50, "dcs_signal": "WATCH", "technicals": {}, "signal_board": []},
        })
        output = _build_gate3_section(result)
        assert "No buy candidates" in output

    def test_pass_gate3(self):
        """Should show PASS for tickers without parabolic warning."""
        result = _make_pipeline_result(scores={
            "MU": {
                "dcs": 72,
                "dcs_signal": "HIGH CONVICTION",
                "technicals": {"rsi_14": 55, "ret_8w": 0.15},
                "signal_board": [],
            },
        })
        output = _build_gate3_section(result)
        assert "MU" in output
        assert "PASS" in output
        assert "72" in output

    def test_gate3_warning(self):
        """Should highlight tickers with GATE3 signal."""
        result = _make_pipeline_result(scores={
            "NVDA": {
                "dcs": 70,
                "dcs_signal": "HIGH CONVICTION",
                "technicals": {"rsi_14": 85, "ret_8w": 0.35},
                "signal_board": [
                    {
                        "signal_type": "DEPLOYMENT_GATE",
                        "legacy_prefix": "GATE3:",
                        "metadata": {"sizing": "FAIL", "rsi": 85, "ret_8w": 0.35},
                    },
                ],
            },
        })
        output = _build_gate3_section(result)
        assert "NVDA" in output
        assert "FAIL" in output
        assert "deployment restrictions" in output


# ---------------------------------------------------------------------------
# Tests: Pipeline exemption & grace period fields
# ---------------------------------------------------------------------------

class TestPipelineResultFields:
    def test_has_exempt_tickers_field(self):
        """PipelineResult should have exempt_tickers dict."""
        result = PipelineResult()
        assert result.exempt_tickers == {}

    def test_has_active_grace_periods_field(self):
        """PipelineResult should have active_grace_periods list."""
        result = PipelineResult()
        assert result.active_grace_periods == []


# ---------------------------------------------------------------------------
# Tests: Full narrative generation
# ---------------------------------------------------------------------------

class TestNarrativeGeneration:
    def test_generates_with_all_sections(self, tmp_path):
        """Should generate narrative with all sections including new ones."""
        result = _make_pipeline_result(
            scores={
                "AAPL": {
                    "dcs": 72,
                    "dcs_signal": "HIGH CONVICTION",
                    "technicals": {"rsi_14": 55, "ret_8w": 0.12, "pct_from_200d": 0.05},
                    "sell_flags": [],
                    "signal_board": [],
                    "sub_scores": {"MQ": 0.7, "FQ": 0.6},
                },
            },
            active_grace_periods=[
                {
                    "symbol": "MSFT",
                    "tier": 180,
                    "days_remaining": 90,
                    "reason": "test",
                    "expires_at": "2026-06-01",
                },
            ],
            exempt_tickers={
                "FBTC": ExemptionResult(
                    is_exempt=True,
                    exemption_type="crypto_halving",
                    expires_at="2026-11-15",
                ),
            },
        )

        filepath = generate_narrative(
            result,
            output_dir=str(tmp_path),
        )

        assert filepath.endswith(".md")

        with open(filepath) as f:
            content = f.read()

        # Check key sections are present
        assert "Macro Backdrop" in content
        assert "Dip-Buy Opportunities" in content
        assert "Gate 3" in content
        assert "Grace Periods" in content
        assert "Exemption Status" in content
        assert "Action Items" in content
        assert "AAPL" in content
        assert "MSFT" in content
        assert "FBTC" in content


# ---------------------------------------------------------------------------
# Tests: Pipeline integration
# ---------------------------------------------------------------------------

class TestPipelineIntegration:
    """Test that Phase 6E wiring is present in the pipeline.

    These tests mock the yfinance download layer to prevent the pipeline
    from going into market context computation.  We only verify that the
    Phase 6E functions (expire, exemptions, grace periods) are called
    and their results are populated on the PipelineResult.
    """

    def _pipeline_patches(self):
        """Return the common set of patches for pipeline tests."""
        import pandas as pd

        return {
            "expire": patch("threshold.engine.pipeline.expire_overdue_grace_periods", return_value=0),
            "list_tickers": patch("threshold.engine.pipeline.list_tickers", return_value=[]),
            "get_exempt": patch("threshold.engine.pipeline.get_exempt_tickers", return_value={}),
            "list_grace": patch("threshold.engine.pipeline.list_active_grace_periods", return_value=[]),
            "fetch_prices": patch("threshold.engine.pipeline._fetch_prices_yfinance", return_value=pd.DataFrame()),
        }

    def test_pipeline_expires_grace_periods(self):
        """Pipeline should call expire_overdue_grace_periods at startup."""
        patches = self._pipeline_patches()
        with patches["expire"] as mock_expire, \
             patches["list_tickers"], \
             patches["get_exempt"], \
             patches["list_grace"], \
             patches["fetch_prices"]:
            mock_db = MagicMock()
            mock_expire.return_value = 2

            from threshold.config.schema import ThresholdConfig

            config = ThresholdConfig()

            from threshold.engine.pipeline import run_scoring_pipeline

            run_scoring_pipeline(config=config, db=mock_db, dry_run=True)

            mock_expire.assert_called_once_with(mock_db)

    def test_pipeline_loads_exemptions(self):
        """Pipeline should populate exempt_tickers on result."""
        exemptions = {
            "FBTC": ExemptionResult(
                is_exempt=True,
                exemption_type="crypto_halving",
            ),
        }
        patches = self._pipeline_patches()
        with patches["expire"], \
             patch("threshold.engine.pipeline.list_tickers", return_value=[
                 {"symbol": "FBTC", "is_crypto_exempt": True, "is_cash": False},
             ]), \
             patch("threshold.engine.pipeline.get_exempt_tickers", return_value=exemptions), \
             patches["list_grace"], \
             patches["fetch_prices"]:
            mock_db = MagicMock()

            from threshold.config.schema import ThresholdConfig

            config = ThresholdConfig()

            from threshold.engine.pipeline import run_scoring_pipeline

            result = run_scoring_pipeline(config=config, db=mock_db, dry_run=True)

            assert "FBTC" in result.exempt_tickers

    def test_pipeline_loads_grace_periods(self):
        """Pipeline should populate active_grace_periods on result."""
        grace_periods = [
            {"symbol": "AAPL", "tier": 180, "days_remaining": 90},
        ]
        patches = self._pipeline_patches()
        with patches["expire"], \
             patches["list_tickers"], \
             patches["get_exempt"], \
             patch("threshold.engine.pipeline.list_active_grace_periods", return_value=grace_periods), \
             patches["fetch_prices"]:
            mock_db = MagicMock()

            from threshold.config.schema import ThresholdConfig

            config = ThresholdConfig()

            from threshold.engine.pipeline import run_scoring_pipeline

            result = run_scoring_pipeline(config=config, db=mock_db, dry_run=True)

            assert len(result.active_grace_periods) == 1
            assert result.active_grace_periods[0]["symbol"] == "AAPL"


# ---------------------------------------------------------------------------
# Tests: Sync CLI command
# ---------------------------------------------------------------------------

class TestSyncCommand:
    def test_sync_cmd_exists(self):
        """Sync command should be importable."""
        from threshold.cli.sync_cmd import sync_cmd
        assert sync_cmd is not None
        assert sync_cmd.name == "sync"

    def test_sync_registered_in_cli(self):
        """Sync command should be registered in the main CLI group."""
        from threshold.cli.main import cli
        commands = cli.list_commands(ctx=None)
        assert "sync" in commands
