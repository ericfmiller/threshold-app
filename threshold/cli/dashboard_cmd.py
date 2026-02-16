"""CLI command: threshold dashboard — Generate and open the HTML dashboard."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import click

logger = logging.getLogger(__name__)


def _load_score_history_with_metadata(
    history_dir: str | Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load the most recent score history JSON and return (scores, metadata)."""
    history_dir = Path(history_dir).expanduser()
    if not history_dir.exists():
        return {}, {}

    files = sorted(history_dir.glob("weekly_scores_*.json"), reverse=True)
    if not files:
        return {}, {}

    try:
        with open(files[0]) as f:
            data = json.load(f)
        scores = data.get("scores", {})
        metadata = data.get("_metadata", {})
        return scores, metadata
    except (json.JSONDecodeError, OSError):
        return {}, {}


@click.command("dashboard")
@click.option(
    "--output-dir",
    type=click.Path(),
    default=None,
    help="Output directory for dashboard HTML (default: ~/.threshold/dashboards)",
)
@click.option(
    "--no-open",
    is_flag=True,
    help="Generate dashboard without opening in browser.",
)
@click.pass_context
def dashboard_cmd(ctx: click.Context, output_dir: str | None, no_open: bool) -> None:
    """Generate the Decision Hierarchy HTML dashboard.

    Reads the most recent scoring run from the database and generates
    an interactive Plotly dashboard organized by investment decision level.
    """
    from threshold.config.loader import load_config, resolve_path
    from threshold.engine.pipeline import PipelineResult
    from threshold.output.dashboard import generate_dashboard
    from threshold.portfolio.correlation import CorrelationReport
    from threshold.storage.database import Database
    from threshold.storage.queries import get_drawdown_classifications

    config = load_config(ctx.obj.get("config_path"))

    # Load most recent scores + metadata
    history_dir = resolve_path(config.output.score_history_dir)
    scores, metadata = _load_score_history_with_metadata(history_dir)

    if not scores:
        click.echo("No score history found. Run 'threshold score' first.")
        return

    # Build PipelineResult with market context from metadata
    result = PipelineResult(
        run_id=metadata.get("run_id", ""),
        scores=scores,
        correlation=CorrelationReport(
            n_tickers=len(scores),
            effective_bets=metadata.get("effective_bets", 0.0),
        ),
        vix_current=metadata.get("vix_current", 0.0),
        vix_regime=metadata.get("vix_regime", "NORMAL"),
        spy_pct_from_200d=metadata.get("spy_pct_from_200d", 0.0),
        spy_above_200d=metadata.get("spy_pct_from_200d", 0.0) > 0,
        market_regime_score=metadata.get("market_regime_score", 0.5),
        breadth_pct=metadata.get("breadth_pct", 0.0),
    )

    # Load drawdown classifications and sector mapping from DB
    dd_classes = {}
    ticker_sectors = {}
    ticker_values = {}
    try:
        db = Database(resolve_path(config.database.path))
        db.connect()
        dd_raw = get_drawdown_classifications(db)
        dd_classes = {sym: d.get("classification", "") for sym, d in dd_raw.items()}
        # Build sector mapping from tickers table
        rows = db.fetchall("SELECT symbol, sector FROM tickers WHERE sector IS NOT NULL")
        ticker_sectors = {r["symbol"]: r["sector"] for r in rows}
        # Build value mapping from positions
        rows = db.fetchall(
            "SELECT symbol, SUM(market_value) as total_value FROM positions "
            "GROUP BY symbol HAVING total_value > 0"
        )
        ticker_values = {r["symbol"]: r["total_value"] for r in rows}
        db.close()
    except Exception as e:
        logger.debug("Could not load supplemental data: %s", e)

    # Determine output directory
    out_dir = resolve_path(config.output.dashboard_dir) if output_dir is None else output_dir

    filepath = generate_dashboard(
        result,
        ticker_sectors=ticker_sectors,
        ticker_values=ticker_values,
        drawdown_classifications=dd_classes,
        output_dir=out_dir,
        auto_open=not no_open,
    )

    click.echo(f"Dashboard generated: {filepath}")


@click.command("narrative")
@click.option(
    "--output-dir",
    type=click.Path(),
    default=None,
    help="Output directory for narrative Markdown (default: ~/.threshold/narratives)",
)
@click.pass_context
def narrative_cmd(ctx: click.Context, output_dir: str | None) -> None:
    """Generate a Markdown narrative report.

    Reads the most recent scoring run from the database and produces
    a human-readable report organized by the Decision Hierarchy.
    """
    from threshold.config.loader import load_config, resolve_path
    from threshold.engine.pipeline import PipelineResult
    from threshold.output.narrative import generate_narrative
    from threshold.portfolio.correlation import CorrelationReport
    from threshold.storage.database import Database
    from threshold.storage.queries import get_drawdown_classifications

    config = load_config(ctx.obj.get("config_path"))

    # Load most recent scores + metadata
    history_dir = resolve_path(config.output.score_history_dir)
    scores, metadata = _load_score_history_with_metadata(history_dir)

    if not scores:
        click.echo("No score history found. Run 'threshold score' first.")
        return

    # Build PipelineResult with market context from metadata
    result = PipelineResult(
        run_id=metadata.get("run_id", ""),
        scores=scores,
        correlation=CorrelationReport(
            n_tickers=len(scores),
            effective_bets=metadata.get("effective_bets", 0.0),
        ),
        vix_current=metadata.get("vix_current", 0.0),
        vix_regime=metadata.get("vix_regime", "NORMAL"),
        spy_pct_from_200d=metadata.get("spy_pct_from_200d", 0.0),
        spy_above_200d=metadata.get("spy_pct_from_200d", 0.0) > 0,
        market_regime_score=metadata.get("market_regime_score", 0.5),
        breadth_pct=metadata.get("breadth_pct", 0.0),
    )

    # Load supplemental data from DB
    dd_classes = {}
    ticker_sectors = {}
    try:
        db = Database(resolve_path(config.database.path))
        db.connect()
        dd_raw = get_drawdown_classifications(db)
        dd_classes = {sym: d.get("classification", "") for sym, d in dd_raw.items()}
        rows = db.fetchall("SELECT symbol, sector FROM tickers WHERE sector IS NOT NULL")
        ticker_sectors = {r["symbol"]: r["sector"] for r in rows}
        db.close()
    except Exception as e:
        logger.debug("Could not load supplemental data: %s", e)

    # Determine output directory
    out_dir = resolve_path(config.output.narrative_dir) if output_dir is None else output_dir

    filepath = generate_narrative(
        result,
        ticker_sectors=ticker_sectors,
        drawdown_classifications=dd_classes,
        output_dir=out_dir,
    )

    click.echo(f"Narrative generated: {filepath}")
