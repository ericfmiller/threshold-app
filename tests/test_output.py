"""Tests for Phase 4: Output & Dashboard.

Tests charts, dashboard assembly, narrative generation, and CLI commands.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from threshold.engine.pipeline import PipelineResult
from threshold.engine.scorer import ScoringResult
from threshold.portfolio.correlation import CorrelationReport

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_scoring_result(
    dcs: float = 55.0,
    signal: str = "WATCH",
    rsi: float = 50.0,
    sell_flags: list[str] | None = None,
    reversal_confirmed: bool = False,
    bottom_turning: bool = False,
    rsi_divergence: bool = False,
    quant_warning: bool = False,
    trend_score: float = 0.5,
    is_etf: bool = False,
    sub_scores: dict | None = None,
    falling_knife: dict | None = None,
) -> ScoringResult:
    """Helper to build a ScoringResult."""
    result: ScoringResult = {
        "dcs": dcs,
        "dcs_signal": signal,
        "sub_scores": sub_scores or {"MQ": 60, "FQ": 55, "TO": 50, "MR": 45, "VC": 40},
        "is_etf": is_etf,
        "technicals": {
            "rsi_14": rsi,
            "pct_from_200d": 0.05,
            "ret_8w": 0.08,
            "macd_crossover": "neutral",
        },
        "trend_score": trend_score,
        "sell_flags": sell_flags or [],
    }
    if reversal_confirmed:
        result["reversal_confirmed"] = True
    if bottom_turning:
        result["bottom_turning"] = True
    if rsi_divergence:
        result["rsi_bullish_divergence"] = True
    if quant_warning:
        result["quant_freshness_warning"] = True
    if falling_knife:
        result["falling_knife_cap"] = falling_knife
    return result


@pytest.fixture
def sample_scores() -> dict[str, ScoringResult]:
    """A diverse set of scoring results for testing."""
    return {
        "AAPL": _make_scoring_result(dcs=82.0, signal="STRONG BUY DIP", rsi=25),
        "MSFT": _make_scoring_result(dcs=72.0, signal="HIGH CONVICTION", rsi=35),
        "GOOGL": _make_scoring_result(dcs=66.0, signal="BUY DIP", rsi=40),
        "AMZN": _make_scoring_result(dcs=58.0, signal="WATCH", rsi=55),
        "META": _make_scoring_result(dcs=45.0, signal="WEAK", rsi=65),
        "TSLA": _make_scoring_result(
            dcs=38.0, signal="AVOID", rsi=72,
            sell_flags=["QUANT_BELOW_2", "BELOW_200D"],
        ),
        "NVDA": _make_scoring_result(
            dcs=70.0, signal="HIGH CONVICTION", rsi=28,
            reversal_confirmed=True,
        ),
        "AMD": _make_scoring_result(
            dcs=55.0, signal="WATCH", rsi=32,
            bottom_turning=True,
            rsi_divergence=True,
        ),
    }


@pytest.fixture
def sample_pipeline_result(sample_scores) -> PipelineResult:
    """A PipelineResult with realistic data."""
    return PipelineResult(
        run_id="test-run",
        scores=sample_scores,
        correlation=CorrelationReport(
            high_corr_pairs=[("AAPL", "MSFT", 0.87), ("GOOGL", "META", 0.82)],
            effective_bets=6.5,
            n_tickers=8,
            is_concentrated=False,
        ),
        concentration_warnings=[
            {"ticker": "NVDA", "correlated_with": "AMD", "correlation": 0.91},
        ],
        vix_current=18.5,
        vix_regime="NORMAL",
        spy_pct_from_200d=0.03,
        spy_above_200d=True,
        market_regime_score=0.62,
        breadth_pct=0.58,
    )


@pytest.fixture
def sample_ticker_sectors() -> dict[str, str]:
    return {
        "AAPL": "Technology",
        "MSFT": "Technology",
        "GOOGL": "Communication Services",
        "AMZN": "Consumer Discretionary",
        "META": "Communication Services",
        "TSLA": "Consumer Discretionary",
        "NVDA": "Technology",
        "AMD": "Technology",
    }


@pytest.fixture
def sample_drawdown_classifications() -> dict[str, str]:
    return {
        "AAPL": "MODERATE",
        "MSFT": "DEFENSIVE",
        "GOOGL": "MODERATE",
        "AMZN": "CYCLICAL",
        "META": "CYCLICAL",
        "TSLA": "AMPLIFIER",
        "NVDA": "MODERATE",
        "AMD": "CYCLICAL",
    }


# ---------------------------------------------------------------------------
# Charts tests
# ---------------------------------------------------------------------------

class TestChartsImports:
    """Test that all chart functions are importable."""

    def test_import_charts_module(self):
        from threshold.output import charts
        assert hasattr(charts, "build_dcs_scatter")
        assert hasattr(charts, "build_war_chest_gauge")
        assert hasattr(charts, "build_drawdown_defense_bars")
        assert hasattr(charts, "build_correlation_heatmap")
        assert hasattr(charts, "build_sector_rrg")
        assert hasattr(charts, "build_sector_treemap")
        assert hasattr(charts, "build_signal_cards_html")
        assert hasattr(charts, "build_market_context_html")

    def test_colors_defined(self):
        from threshold.output.charts import COLORS, SECTOR_COLORS
        assert "bg" in COLORS
        assert "green" in COLORS
        assert "red" in COLORS
        assert "Technology" in SECTOR_COLORS


class TestDCSScatter:
    """Test the DCS vs RSI scatter plot builder."""

    def test_basic_scatter(self, sample_scores):
        from threshold.output.charts import build_dcs_scatter
        fig = build_dcs_scatter(sample_scores)
        assert fig is not None
        assert len(fig.data) > 0  # Has traces

    def test_scatter_with_sectors(self, sample_scores, sample_ticker_sectors):
        from threshold.output.charts import build_dcs_scatter
        fig = build_dcs_scatter(sample_scores, ticker_sectors=sample_ticker_sectors)
        assert fig is not None
        # Should have traces (Holdings/Watchlist split or sector-grouped)
        assert len(fig.data) >= 1

    def test_scatter_with_held_symbols(self, sample_scores, sample_ticker_sectors):
        from threshold.output.charts import build_dcs_scatter
        held = {"AAPL", "MSFT", "NVDA"}
        fig = build_dcs_scatter(sample_scores, ticker_sectors=sample_ticker_sectors, held_symbols=held)
        assert fig is not None
        trace_names = {t.name for t in fig.data if hasattr(t, "name") and t.name}
        # Should have both Holdings and Watchlist traces
        assert any("Holdings" in n for n in trace_names)
        assert any("Watchlist" in n for n in trace_names)

    def test_scatter_empty_scores(self):
        from threshold.output.charts import build_dcs_scatter
        fig = build_dcs_scatter({})
        assert fig is not None

    def test_scatter_has_threshold_lines(self, sample_scores):
        from threshold.output.charts import build_dcs_scatter
        fig = build_dcs_scatter(sample_scores)
        # Plotly stores hlines/vlines in layout shapes
        shapes = fig.layout.shapes or []
        # Should have signal zone rectangles and threshold lines
        assert len(shapes) >= 2  # At least the green and red zone rectangles


class TestWarChestGauge:
    def test_basic_gauge(self):
        from threshold.output.charts import build_war_chest_gauge
        fig = build_war_chest_gauge(actual_pct=0.08, target_pct=0.10, vix_regime="NORMAL")
        assert fig is not None
        assert len(fig.data) == 1  # One indicator

    def test_gauge_above_target(self):
        from threshold.output.charts import build_war_chest_gauge
        fig = build_war_chest_gauge(actual_pct=0.15, target_pct=0.10)
        assert fig is not None

    def test_gauge_fear_regime(self):
        from threshold.output.charts import build_war_chest_gauge
        fig = build_war_chest_gauge(actual_pct=0.12, target_pct=0.15, vix_regime="FEAR")
        assert fig is not None
        assert "FEAR" in fig.data[0].title.text


class TestDrawdownDefenseBars:
    def test_basic_bars(self, sample_drawdown_classifications):
        from threshold.output.charts import build_drawdown_defense_bars
        fig = build_drawdown_defense_bars(sample_drawdown_classifications)
        assert fig is not None
        assert len(fig.data) >= 1  # At least one bar trace (count; optionally dollar-weighted)

    def test_bars_counts(self, sample_drawdown_classifications):
        from threshold.output.charts import build_drawdown_defense_bars
        fig = build_drawdown_defense_bars(sample_drawdown_classifications)
        y_values = list(fig.data[0].y)
        # Now uses percentages — should sum to ~100%
        assert abs(sum(y_values) - 100.0) < 1.0

    def test_bars_dollar_weighted(self, sample_drawdown_classifications):
        from threshold.output.charts import build_drawdown_defense_bars
        ticker_values = {
            "AAPL": 50000, "MSFT": 30000, "GOOGL": 20000, "AMZN": 15000,
            "META": 10000, "TSLA": 5000, "NVDA": 40000, "AMD": 25000,
        }
        fig = build_drawdown_defense_bars(
            sample_drawdown_classifications, ticker_values=ticker_values,
        )
        assert fig is not None
        # Should have 2 bar traces (count + dollar-weighted)
        assert len(fig.data) == 2

    def test_empty_classifications(self):
        from threshold.output.charts import build_drawdown_defense_bars
        fig = build_drawdown_defense_bars({})
        assert fig is not None


class TestCorrelationHeatmap:
    def test_basic_heatmap(self):
        from threshold.output.charts import build_correlation_heatmap
        matrix = {
            "A": {"A": 1.0, "B": 0.8},
            "B": {"A": 0.8, "B": 1.0},
        }
        fig = build_correlation_heatmap(matrix)
        assert fig is not None
        assert len(fig.data) == 1  # One heatmap trace

    def test_empty_matrix(self):
        from threshold.output.charts import build_correlation_heatmap
        fig = build_correlation_heatmap({})
        assert fig is not None
        # Should have an annotation for "insufficient data"
        assert len(fig.layout.annotations) > 0


class TestSectorRRG:
    def test_basic_rrg(self):
        from threshold.output.charts import build_sector_rrg
        rankings = [
            {"sector": "Technology", "rs_vs_spy": 1.05, "momentum": 0.02, "quadrant": "LEADING"},
            {"sector": "Energy", "rs_vs_spy": 0.95, "momentum": -0.01, "quadrant": "LAGGING"},
        ]
        fig = build_sector_rrg(rankings)
        assert fig is not None
        assert len(fig.data) == 2

    def test_empty_rrg(self):
        from threshold.output.charts import build_sector_rrg
        fig = build_sector_rrg([])
        assert fig is not None


class TestSectorTreemap:
    def test_basic_treemap(self, sample_scores, sample_ticker_sectors):
        from threshold.output.charts import build_sector_treemap
        fig = build_sector_treemap(
            sample_scores,
            ticker_sectors=sample_ticker_sectors,
        )
        assert fig is not None
        assert len(fig.data) == 1  # One treemap trace

    def test_treemap_with_values(self, sample_scores, sample_ticker_sectors):
        from threshold.output.charts import build_sector_treemap
        values = {ticker: 1000 * (i + 1) for i, ticker in enumerate(sample_scores)}
        fig = build_sector_treemap(
            sample_scores,
            ticker_sectors=sample_ticker_sectors,
            ticker_values=values,
        )
        assert fig is not None


class TestSignalCards:
    def test_basic_cards(self, sample_scores):
        from threshold.output.charts import build_signal_cards_html
        html = build_signal_cards_html(sample_scores)
        assert isinstance(html, str)
        assert "STRONG BUY" in html
        assert "HIGH CONVICTION" in html

    def test_empty_scores(self):
        from threshold.output.charts import build_signal_cards_html
        html = build_signal_cards_html({})
        assert isinstance(html, str)


class TestMarketContextHTML:
    def test_basic_context(self):
        from threshold.output.charts import build_market_context_html
        html = build_market_context_html(
            vix_current=18.5,
            vix_regime="NORMAL",
            spy_pct=0.03,
            spy_above_200d=True,
            breadth_pct=0.58,
            effective_bets=6.5,
        )
        assert isinstance(html, str)
        assert "18.5" in html
        assert "NORMAL" in html

    def test_fear_regime(self):
        from threshold.output.charts import build_market_context_html
        html = build_market_context_html(
            vix_current=25.0,
            vix_regime="FEAR",
            spy_pct=-0.05,
            spy_above_200d=False,
            breadth_pct=0.35,
        )
        assert "FEAR" in html
        assert "BELOW" in html


# ---------------------------------------------------------------------------
# Dashboard tests
# ---------------------------------------------------------------------------

class TestDashboard:
    def test_generate_dashboard(self, sample_pipeline_result, tmp_path):
        from threshold.output.dashboard import generate_dashboard
        filepath = generate_dashboard(
            sample_pipeline_result,
            output_dir=str(tmp_path),
            auto_open=False,
        )
        assert Path(filepath).exists()
        content = Path(filepath).read_text()
        assert "<!DOCTYPE html>" in content
        assert "Threshold" in content
        assert "plotly" in content.lower()

    def test_dashboard_has_sections(self, sample_pipeline_result, tmp_path):
        from threshold.output.dashboard import generate_dashboard
        filepath = generate_dashboard(
            sample_pipeline_result,
            output_dir=str(tmp_path),
            auto_open=False,
        )
        content = Path(filepath).read_text()
        assert "Macro Regime" in content
        assert "Selection" in content

    def test_dashboard_with_sectors(self, sample_pipeline_result, sample_ticker_sectors, tmp_path):
        from threshold.output.dashboard import generate_dashboard
        filepath = generate_dashboard(
            sample_pipeline_result,
            ticker_sectors=sample_ticker_sectors,
            output_dir=str(tmp_path),
            auto_open=False,
        )
        content = Path(filepath).read_text()
        assert "Holdings by Sector" in content

    def test_dashboard_with_drawdown(
        self, sample_pipeline_result, sample_drawdown_classifications, tmp_path,
    ):
        from threshold.output.dashboard import generate_dashboard
        filepath = generate_dashboard(
            sample_pipeline_result,
            drawdown_classifications=sample_drawdown_classifications,
            output_dir=str(tmp_path),
            auto_open=False,
        )
        content = Path(filepath).read_text()
        assert "Drawdown Defense" in content

    def test_dashboard_with_sector_rrg(self, sample_pipeline_result, tmp_path):
        from threshold.output.dashboard import generate_dashboard
        rankings = [
            {"sector": "Tech", "rs_vs_spy": 1.05, "momentum": 0.02, "quadrant": "LEADING"},
        ]
        filepath = generate_dashboard(
            sample_pipeline_result,
            sector_rankings=rankings,
            output_dir=str(tmp_path),
            auto_open=False,
        )
        content = Path(filepath).read_text()
        assert "Sector Rotation" in content

    def test_dashboard_correlation(self, sample_pipeline_result, tmp_path):
        from threshold.output.dashboard import generate_dashboard
        # Add a minimal correlation matrix
        sample_pipeline_result.correlation.correlation_matrix = {
            "AAPL": {"AAPL": 1.0, "MSFT": 0.87},
            "MSFT": {"AAPL": 0.87, "MSFT": 1.0},
        }
        filepath = generate_dashboard(
            sample_pipeline_result,
            output_dir=str(tmp_path),
            auto_open=False,
        )
        content = Path(filepath).read_text()
        assert "Correlation" in content

    def test_dashboard_minimal_result(self, tmp_path):
        """Dashboard should handle a minimal PipelineResult."""
        from threshold.output.dashboard import generate_dashboard
        result = PipelineResult(
            scores={"AAPL": _make_scoring_result(dcs=60)},
        )
        filepath = generate_dashboard(
            result,
            output_dir=str(tmp_path),
            auto_open=False,
        )
        assert Path(filepath).exists()


class TestDashboardHelpers:
    def test_embed_plotly(self):
        import plotly.graph_objects as go

        from threshold.output.dashboard import _embed_plotly
        fig = go.Figure(go.Scatter(x=[1, 2], y=[3, 4]))
        html = _embed_plotly(fig, "test-div")
        assert "test-div" in html
        assert "plotly" in html.lower() or "div" in html.lower()

    def test_embed_plotly_error(self):
        from threshold.output.dashboard import _embed_plotly
        html = _embed_plotly("not a figure", "test-div")
        assert "error" in html.lower()

    def test_html_header(self):
        from threshold.output.dashboard import _html_header
        html = _html_header("Test Dashboard", "2026-02-16")
        assert "<!DOCTYPE html>" in html
        assert "Test Dashboard" in html
        assert "plotly" in html.lower()

    def test_navbar(self):
        from threshold.output.dashboard import _navbar
        html = _navbar("2026-02-16")
        assert "Threshold" in html
        assert "nav" in html.lower()


# ---------------------------------------------------------------------------
# Narrative tests
# ---------------------------------------------------------------------------

class TestNarrative:
    def test_generate_narrative(self, sample_pipeline_result, tmp_path):
        from threshold.output.narrative import generate_narrative
        filepath = generate_narrative(
            sample_pipeline_result,
            output_dir=str(tmp_path),
        )
        assert Path(filepath).exists()
        content = Path(filepath).read_text()
        assert "# Threshold Scoring Report" in content

    def test_narrative_has_all_sections(self, sample_pipeline_result, tmp_path):
        from threshold.output.narrative import generate_narrative
        filepath = generate_narrative(
            sample_pipeline_result,
            output_dir=str(tmp_path),
        )
        content = Path(filepath).read_text()
        # 23-section layout (some may be empty but headers should be present)
        assert "## 1. Macro Backdrop" in content
        assert "## 2. Dip-Buy Opportunities" in content
        assert "## 3. Falling Knife" in content
        assert "## 5. Watch Zone" in content
        assert "## 7. Reversal Signals" in content
        assert "## 12. Sell Criteria" in content
        assert "## 15. Drawdown Defense" in content
        assert "## 16. Correlation" in content
        assert "## 17. Sector Exposure" in content
        assert "## 18. War Chest" in content
        assert "## 20. Action Items" in content
        assert "## 21. Quick Reference" in content
        # New sections from Phase 5
        assert "## 8. Sub-Score Driver" in content
        assert "## 9. Relative Strength" in content
        assert "## 10. EPS Revision" in content
        assert "## 11. OBV Divergence" in content
        assert "## 19. Per-Account" in content

    def test_narrative_header_info(self, sample_pipeline_result, tmp_path):
        from threshold.output.narrative import generate_narrative
        filepath = generate_narrative(
            sample_pipeline_result,
            output_dir=str(tmp_path),
        )
        content = Path(filepath).read_text()
        assert "test-run" in content
        assert "18.5" in content  # VIX
        assert "NORMAL" in content  # VIX regime

    def test_narrative_dipbuys(self, sample_pipeline_result, tmp_path):
        from threshold.output.narrative import generate_narrative
        filepath = generate_narrative(
            sample_pipeline_result,
            output_dir=str(tmp_path),
        )
        content = Path(filepath).read_text()
        # Should contain tickers with DCS >= 65
        assert "AAPL" in content
        assert "MSFT" in content
        assert "GOOGL" in content

    def test_narrative_sell_flags(self, sample_pipeline_result, tmp_path):
        from threshold.output.narrative import generate_narrative
        # Sell alerts only show for holdings — mark TSLA as held
        sample_pipeline_result.held_symbols = {"TSLA"}
        filepath = generate_narrative(
            sample_pipeline_result,
            output_dir=str(tmp_path),
        )
        content = Path(filepath).read_text()
        assert "TSLA" in content
        assert "QUANT_BELOW_2" in content

    def test_narrative_reversals(self, sample_pipeline_result, tmp_path):
        from threshold.output.narrative import generate_narrative
        filepath = generate_narrative(
            sample_pipeline_result,
            output_dir=str(tmp_path),
        )
        content = Path(filepath).read_text()
        assert "Reversal Confirmed" in content
        assert "NVDA" in content
        assert "Bottom Turning" in content
        assert "AMD" in content

    def test_narrative_correlation(self, sample_pipeline_result, tmp_path):
        from threshold.output.narrative import generate_narrative
        filepath = generate_narrative(
            sample_pipeline_result,
            output_dir=str(tmp_path),
        )
        content = Path(filepath).read_text()
        assert "6.5" in content  # effective bets
        assert "AAPL" in content and "MSFT" in content  # high corr pair

    def test_narrative_concentration_warnings(self, sample_pipeline_result, tmp_path):
        from threshold.output.narrative import generate_narrative
        filepath = generate_narrative(
            sample_pipeline_result,
            output_dir=str(tmp_path),
        )
        content = Path(filepath).read_text()
        assert "Concentration" in content
        assert "NVDA" in content

    def test_narrative_with_drawdown(
        self, sample_pipeline_result, sample_drawdown_classifications, tmp_path,
    ):
        from threshold.output.narrative import generate_narrative
        filepath = generate_narrative(
            sample_pipeline_result,
            drawdown_classifications=sample_drawdown_classifications,
            output_dir=str(tmp_path),
        )
        content = Path(filepath).read_text()
        assert "HEDGE" in content or "DEFENSIVE" in content
        assert "AMPLIFIER" in content

    def test_narrative_with_sectors(
        self, sample_pipeline_result, sample_ticker_sectors, tmp_path,
    ):
        from threshold.output.narrative import generate_narrative
        filepath = generate_narrative(
            sample_pipeline_result,
            ticker_sectors=sample_ticker_sectors,
            output_dir=str(tmp_path),
        )
        content = Path(filepath).read_text()
        assert "Technology" in content

    def test_narrative_war_chest(self, sample_pipeline_result, tmp_path):
        from threshold.output.narrative import generate_narrative
        filepath = generate_narrative(
            sample_pipeline_result,
            war_chest_pct=0.08,
            war_chest_target=0.10,
            output_dir=str(tmp_path),
        )
        content = Path(filepath).read_text()
        assert "War Chest" in content
        assert "BELOW TARGET" in content

    def test_narrative_action_items(self, sample_pipeline_result, tmp_path):
        from threshold.output.narrative import generate_narrative
        filepath = generate_narrative(
            sample_pipeline_result,
            output_dir=str(tmp_path),
        )
        content = Path(filepath).read_text()
        assert "Action Items" in content
        assert "STRONG BUY" in content  # AAPL has DCS 82

    def test_narrative_fear_regime(self, tmp_path):
        from threshold.output.narrative import generate_narrative
        result = PipelineResult(
            run_id="fear-test",
            scores={"AAPL": _make_scoring_result(dcs=75)},
            vix_current=26.0,
            vix_regime="FEAR",
            spy_above_200d=False,
            spy_pct_from_200d=-0.08,
            breadth_pct=0.35,
        )
        filepath = generate_narrative(result, output_dir=str(tmp_path))
        content = Path(filepath).read_text()
        assert "FEAR" in content
        assert "D-5 modifiers" in content

    def test_narrative_minimal(self, tmp_path):
        """Narrative handles a minimal PipelineResult."""
        from threshold.output.narrative import generate_narrative
        result = PipelineResult(scores={})
        filepath = generate_narrative(result, output_dir=str(tmp_path))
        assert Path(filepath).exists()
        content = Path(filepath).read_text()
        assert "Threshold Scoring Report" in content


class TestNarrativeHelpers:
    def test_pct_formatting(self):
        from threshold.output.narrative import _pct
        assert _pct(0.05) == "5.0%"
        assert _pct(0.123, 2) == "12.30%"
        assert _pct(-0.05) == "-5.0%"

    def test_dcs_emoji(self):
        from threshold.output.narrative import _dcs_emoji
        assert "STRONG" in _dcs_emoji(85)
        assert "HC" in _dcs_emoji(72)
        assert "BUY" in _dcs_emoji(66)
        assert "WATCH" in _dcs_emoji(55)
        assert "WEAK" in _dcs_emoji(40)

    def test_vix_emoji(self):
        from threshold.output.narrative import _vix_emoji
        assert _vix_emoji("COMPLACENT") == "LOW"
        assert _vix_emoji("NORMAL") == "NORMAL"
        assert "FEAR" in _vix_emoji("FEAR")
        assert "PANIC" in _vix_emoji("PANIC")

    def test_format_sell_flags(self):
        from threshold.output.narrative import _format_sell_flags
        assert _format_sell_flags([]) == "-"
        assert _format_sell_flags(["A", "B"]) == "A, B"


class TestNarrativeFallingKnife:
    def test_falling_knife_section(self, tmp_path):
        from threshold.output.narrative import generate_narrative
        scores = {
            "TSLA": _make_scoring_result(
                dcs=30,
                falling_knife={
                    "cap_applied": True,
                    "original_dcs": 65,
                    "classification": "AMPLIFIER",
                },
            ),
        }
        result = PipelineResult(scores=scores)
        filepath = generate_narrative(result, output_dir=str(tmp_path))
        content = Path(filepath).read_text()
        assert "Falling Knife" in content
        assert "TSLA" in content
        assert "AMPLIFIER" in content

    def test_no_falling_knives(self, tmp_path):
        from threshold.output.narrative import generate_narrative
        result = PipelineResult(
            scores={"AAPL": _make_scoring_result(dcs=60)},
        )
        filepath = generate_narrative(result, output_dir=str(tmp_path))
        content = Path(filepath).read_text()
        assert "No falling knife" in content


# ---------------------------------------------------------------------------
# CLI command tests
# ---------------------------------------------------------------------------

class TestCLIDashboardCommand:
    def test_dashboard_cmd_importable(self):
        from threshold.cli.dashboard_cmd import dashboard_cmd
        assert dashboard_cmd is not None

    def test_narrative_cmd_importable(self):
        from threshold.cli.dashboard_cmd import narrative_cmd
        assert narrative_cmd is not None


class TestCLIRegistration:
    def test_dashboard_registered(self):
        from threshold.cli.main import cli
        commands = cli.list_commands(None)
        assert "dashboard" in commands

    def test_narrative_registered(self):
        from threshold.cli.main import cli
        commands = cli.list_commands(None)
        assert "narrative" in commands


# ---------------------------------------------------------------------------
# Package imports
# ---------------------------------------------------------------------------

class TestOutputPackageImports:
    def test_import_output_package(self):
        from threshold.output import (
            generate_dashboard,
            generate_narrative,
            generate_scoring_alerts,
        )
        assert callable(generate_scoring_alerts)
        assert callable(generate_dashboard)
        assert callable(generate_narrative)

    def test_import_charts(self):
        from threshold.output.charts import (
            build_dcs_scatter,
            build_market_context_html,
        )
        assert callable(build_dcs_scatter)
        assert callable(build_market_context_html)

    def test_import_dashboard(self):
        from threshold.output.dashboard import generate_dashboard
        assert callable(generate_dashboard)

    def test_import_narrative(self):
        from threshold.output.narrative import generate_narrative
        assert callable(generate_narrative)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_single_ticker_dashboard(self, tmp_path):
        from threshold.output.dashboard import generate_dashboard
        result = PipelineResult(
            scores={"AAPL": _make_scoring_result(dcs=75)},
        )
        filepath = generate_dashboard(result, output_dir=str(tmp_path), auto_open=False)
        assert Path(filepath).exists()

    def test_single_ticker_narrative(self, tmp_path):
        from threshold.output.narrative import generate_narrative
        result = PipelineResult(
            scores={"AAPL": _make_scoring_result(dcs=75)},
        )
        filepath = generate_narrative(result, output_dir=str(tmp_path))
        assert Path(filepath).exists()

    def test_all_strong_buy(self, tmp_path):
        from threshold.output.narrative import generate_narrative
        scores = {
            f"T{i}": _make_scoring_result(dcs=85, signal="STRONG BUY DIP")
            for i in range(5)
        }
        result = PipelineResult(scores=scores)
        filepath = generate_narrative(result, output_dir=str(tmp_path))
        content = Path(filepath).read_text()
        assert "STRONG BUY" in content

    def test_all_sell_flagged(self, tmp_path):
        from threshold.output.narrative import generate_narrative
        scores = {
            f"T{i}": _make_scoring_result(
                dcs=30,
                sell_flags=["QUANT_BELOW_2", "BELOW_200D"],
            )
            for i in range(5)
        }
        # Sell alerts only show for holdings — mark all as held
        held = {f"T{i}" for i in range(5)}
        result = PipelineResult(scores=scores, held_symbols=held)
        filepath = generate_narrative(result, output_dir=str(tmp_path))
        content = Path(filepath).read_text()
        # New format: urgent section with REVIEW REQUIRED for 2+ flags
        assert "REVIEW REQUIRED" in content
        assert "5 tickers with 2+ flags" in content

    def test_drawdown_all_hedge(self, tmp_path):
        from threshold.output.narrative import generate_narrative
        dd = {"GOLD": "HEDGE", "BND": "HEDGE", "TIP": "HEDGE"}
        result = PipelineResult(scores={})
        filepath = generate_narrative(
            result,
            drawdown_classifications=dd,
            output_dir=str(tmp_path),
        )
        content = Path(filepath).read_text()
        assert "HEDGE" in content
        assert "0%" in content  # 0% offense

    def test_complacent_regime_narrative(self, tmp_path):
        from threshold.output.narrative import generate_narrative
        result = PipelineResult(
            scores={"AAPL": _make_scoring_result(dcs=60)},
            vix_current=12.0,
            vix_regime="COMPLACENT",
        )
        filepath = generate_narrative(result, output_dir=str(tmp_path))
        content = Path(filepath).read_text()
        assert "Complacent" in content or "half-size" in content


# ---------------------------------------------------------------------------
# New section tests (Phase 5/6 enrichments)
# ---------------------------------------------------------------------------


class TestNarrativeNewSections:
    """Test the 9 new narrative sections added in Phase 5."""

    def _generate(self, tmp_path, **kwargs):
        from threshold.output.narrative import generate_narrative
        defaults = dict(output_dir=str(tmp_path))
        defaults.update(kwargs)
        result = defaults.pop("result", None)
        if result is None:
            result = PipelineResult(scores={})
        filepath = generate_narrative(result, **defaults)
        return Path(filepath).read_text()

    def test_dipbuy_holdings_watchlist_split(self, tmp_path):
        """Dip-buy section should split holdings vs watchlist when held_symbols given."""
        from threshold.output.narrative import generate_narrative
        scores = {
            "HELD1": _make_scoring_result(dcs=70, signal="HIGH CONVICTION"),
            "WL1": _make_scoring_result(dcs=68, signal="BUY DIP"),
        }
        scores["HELD1"]["is_holding"] = True
        scores["WL1"]["is_holding"] = False
        result = PipelineResult(scores=scores, held_symbols={"HELD1"})
        filepath = generate_narrative(result, output_dir=str(tmp_path))
        content = Path(filepath).read_text()
        assert "Portfolio Holdings" in content
        assert "Watchlist Candidates" in content

    def test_hedge_downtrend_section(self, tmp_path):
        """Hedges/defensives with falling knife caps should appear separately."""
        from threshold.output.narrative import generate_narrative
        scores = {
            "GOLD": _make_scoring_result(
                dcs=55,
                falling_knife={"reason": "20d_velocity", "capped_dcs": 55},
            ),
        }
        dd = {"GOLD": "HEDGE"}
        result = PipelineResult(scores=scores)
        filepath = generate_narrative(
            result, drawdown_classifications=dd, output_dir=str(tmp_path),
        )
        content = Path(filepath).read_text()
        assert "## 4. Hedges & Defensives" in content

    def test_bitcoin_crypto_section(self, tmp_path):
        """Crypto section appears when exempt tickers exist."""
        from threshold.output.narrative import generate_narrative
        scores = {
            "FBTC": _make_scoring_result(dcs=30, signal="WEAK"),
        }
        result = PipelineResult(
            scores=scores,
            exempt_tickers={"FBTC": {"type": "crypto_halving", "reason": "4-year cycle"}},
        )
        filepath = generate_narrative(result, output_dir=str(tmp_path))
        content = Path(filepath).read_text()
        assert "## 6. Bitcoin & Crypto" in content

    def test_subscore_driver_section(self, tmp_path):
        """Sub-score driver analysis should show top DCS tickers."""
        from threshold.output.narrative import generate_narrative
        scores = {
            "TOP": _make_scoring_result(
                dcs=75, signal="STRONG BUY",
                sub_scores={"MQ": 85, "FQ": 70, "TO": 65, "MR": 50, "VC": 55},
            ),
        }
        result = PipelineResult(scores=scores)
        filepath = generate_narrative(result, output_dir=str(tmp_path))
        content = Path(filepath).read_text()
        assert "## 8. Sub-Score Driver" in content
        assert "TOP" in content
        assert "MQ" in content

    def test_relative_strength_section(self, tmp_path):
        """RS vs SPY section surfaces technicals.rs_vs_spy."""
        from threshold.output.narrative import generate_narrative
        scores = {
            "OUTPERFORMER": _make_scoring_result(dcs=70),
            "LAGGARD": _make_scoring_result(dcs=40),
        }
        scores["OUTPERFORMER"]["technicals"]["rs_vs_spy"] = 1.25
        scores["LAGGARD"]["technicals"]["rs_vs_spy"] = 0.65
        result = PipelineResult(scores=scores)
        filepath = generate_narrative(result, output_dir=str(tmp_path))
        content = Path(filepath).read_text()
        assert "## 9. Relative Strength" in content

    def test_revision_momentum_section(self, tmp_path):
        """EPS revision momentum section surfaces revision_momentum data."""
        from threshold.output.narrative import generate_narrative
        scores = {
            "IMPROVING": _make_scoring_result(dcs=65),
            "DECLINING": _make_scoring_result(dcs=50),
        }
        scores["IMPROVING"]["revision_momentum"] = {"direction": "improving", "delta_4w": 0.15}
        scores["DECLINING"]["revision_momentum"] = {"direction": "declining", "delta_4w": -0.20}
        result = PipelineResult(scores=scores)
        filepath = generate_narrative(result, output_dir=str(tmp_path))
        content = Path(filepath).read_text()
        assert "## 10. EPS Revision" in content

    def test_obv_divergence_section(self, tmp_path):
        """OBV divergence section surfaces obv data from technicals."""
        from threshold.output.narrative import generate_narrative
        scores = {
            "ACCUM": _make_scoring_result(dcs=60),
        }
        scores["ACCUM"]["technicals"]["obv_divergence"] = "bullish"
        scores["ACCUM"]["technicals"]["obv_divergence_strength"] = 0.8
        result = PipelineResult(scores=scores)
        filepath = generate_narrative(result, output_dir=str(tmp_path))
        content = Path(filepath).read_text()
        assert "## 11. OBV Divergence" in content

    def test_per_account_section(self, tmp_path):
        """Per-account holdings health appears when positions provided."""
        from threshold.output.narrative import generate_narrative
        positions = [
            {"account_id": "Brokerage", "symbol": "AAPL", "market_value": 10000, "quantity": 50},
            {"account_id": "Brokerage", "symbol": "MSFT", "market_value": 8000, "quantity": 30},
            {"account_id": "Roth", "symbol": "GOOGL", "market_value": 5000, "quantity": 20},
        ]
        scores = {
            "AAPL": _make_scoring_result(dcs=70, signal="HIGH CONVICTION"),
            "MSFT": _make_scoring_result(dcs=55, signal="WATCH"),
            "GOOGL": _make_scoring_result(dcs=60, signal="LEAN BUY"),
        }
        result = PipelineResult(scores=scores)
        filepath = generate_narrative(
            result, positions=positions, output_dir=str(tmp_path),
        )
        content = Path(filepath).read_text()
        assert "## 19. Per-Account" in content
        assert "Brokerage" in content
        assert "Roth" in content

    def test_quick_reference_section(self, sample_pipeline_result, tmp_path):
        """Quick reference section appears at the end."""
        from threshold.output.narrative import generate_narrative
        filepath = generate_narrative(sample_pipeline_result, output_dir=str(tmp_path))
        content = Path(filepath).read_text()
        assert "## 21. Quick Reference" in content
        assert "VIX" in content
        assert "Top DCS" in content

    def test_war_chest_with_values(self, tmp_path):
        """War chest section shows dollar amounts when provided."""
        from threshold.output.narrative import generate_narrative
        result = PipelineResult(scores={}, vix_regime="NORMAL")
        filepath = generate_narrative(
            result,
            war_chest_pct=0.08,
            war_chest_target=0.12,
            war_chest_value=32000.0,
            total_portfolio_value=400000.0,
            output_dir=str(tmp_path),
        )
        content = Path(filepath).read_text()
        assert "## 18. War Chest" in content
        assert "$32,000" in content or "32,000" in content
        assert "SHORTFALL" in content or "below" in content.lower()

    def test_drawdown_dollar_weighted(self, tmp_path):
        """Drawdown section shows dollar-weighted columns when values provided."""
        from threshold.output.narrative import generate_narrative
        dd = {"AAPL": "MODERATE", "GOLD": "HEDGE", "TSLA": "AMPLIFIER"}
        tv = {"AAPL": 50000, "GOLD": 30000, "TSLA": 20000}
        result = PipelineResult(scores={})
        filepath = generate_narrative(
            result,
            drawdown_classifications=dd,
            ticker_values=tv,
            output_dir=str(tmp_path),
        )
        content = Path(filepath).read_text()
        assert "## 15. Drawdown Defense" in content
        assert "$ Value" in content or "Dollar" in content or "$" in content


class TestDashboardNewSections:
    """Test the 6 new dashboard sections added in Phase 6."""

    def test_dashboard_has_deployment_section(self, sample_pipeline_result, tmp_path):
        from threshold.output.dashboard import generate_dashboard
        filepath = generate_dashboard(
            sample_pipeline_result,
            output_dir=str(tmp_path),
            auto_open=False,
        )
        content = Path(filepath).read_text()
        assert "deployment" in content.lower()

    def test_dashboard_has_sell_alerts(self, tmp_path):
        from threshold.output.dashboard import generate_dashboard
        scores = {
            "BAD": _make_scoring_result(dcs=30, sell_flags=["QUANT_BELOW_2", "BELOW_200D"]),
        }
        result = PipelineResult(scores=scores)
        filepath = generate_dashboard(result, output_dir=str(tmp_path), auto_open=False)
        content = Path(filepath).read_text()
        assert "sell-alerts" in content.lower() or "alert" in content.lower()

    def test_dashboard_has_holdings_section(self, tmp_path):
        from threshold.output.dashboard import generate_dashboard
        positions = [
            {"account": "Brokerage", "symbol": "AAPL", "market_value": 10000, "quantity": 50},
        ]
        scores = {"AAPL": _make_scoring_result(dcs=70)}
        result = PipelineResult(scores=scores, held_symbols={"AAPL"})
        filepath = generate_dashboard(
            result, positions=positions, output_dir=str(tmp_path), auto_open=False,
        )
        content = Path(filepath).read_text()
        assert "holdings" in content.lower()

    def test_dashboard_has_behavioral_section(self, sample_pipeline_result, tmp_path):
        from threshold.output.dashboard import generate_dashboard
        filepath = generate_dashboard(
            sample_pipeline_result,
            output_dir=str(tmp_path),
            auto_open=False,
        )
        content = Path(filepath).read_text()
        assert "behavioral" in content.lower()
        assert "FOMO" in content or "fomo" in content.lower()

    def test_dashboard_war_chest_with_values(self, tmp_path):
        from threshold.output.dashboard import generate_dashboard
        result = PipelineResult(scores={}, vix_regime="FEAR")
        filepath = generate_dashboard(
            result,
            war_chest_pct=0.18,
            war_chest_target=0.15,
            war_chest_value=72000.0,
            total_portfolio_value=400000.0,
            output_dir=str(tmp_path),
            auto_open=False,
        )
        content = Path(filepath).read_text()
        assert "72,000" in content or "$72" in content

    def test_dashboard_navbar_has_new_links(self, sample_pipeline_result, tmp_path):
        from threshold.output.dashboard import generate_dashboard
        filepath = generate_dashboard(
            sample_pipeline_result,
            output_dir=str(tmp_path),
            auto_open=False,
        )
        content = Path(filepath).read_text()
        # New navbar should have these sections
        for section in ["macro", "allocation", "drawdown", "selection", "behavioral"]:
            assert section in content.lower()
