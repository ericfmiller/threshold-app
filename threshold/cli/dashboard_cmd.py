"""CLI command: threshold dashboard — Generate and open the HTML dashboard."""

from __future__ import annotations

import logging

import click

logger = logging.getLogger(__name__)


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
    from threshold.output.alerts import load_previous_scores
    from threshold.output.dashboard import generate_dashboard
    from threshold.portfolio.correlation import CorrelationReport

    config = load_config(ctx.obj.get("config_path"))

    # Load most recent scores
    history_dir = resolve_path(config.output.score_history_dir)
    scores = load_previous_scores(str(history_dir))

    if not scores:
        click.echo("No score history found. Run 'threshold score' first.")
        return

    # Build a minimal PipelineResult from saved history
    # Note: In a full implementation, this would load from the database
    result = PipelineResult(
        scores=scores,
        correlation=CorrelationReport(n_tickers=len(scores)),
    )

    # Determine output directory
    out_dir = resolve_path(config.output.dashboard_dir) if output_dir is None else output_dir

    filepath = generate_dashboard(
        result,
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
    from threshold.output.alerts import load_previous_scores
    from threshold.output.narrative import generate_narrative
    from threshold.portfolio.correlation import CorrelationReport

    config = load_config(ctx.obj.get("config_path"))

    # Load most recent scores
    history_dir = resolve_path(config.output.score_history_dir)
    scores = load_previous_scores(str(history_dir))

    if not scores:
        click.echo("No score history found. Run 'threshold score' first.")
        return

    # Build a minimal PipelineResult from saved history
    result = PipelineResult(
        scores=scores,
        correlation=CorrelationReport(n_tickers=len(scores)),
    )

    # Determine output directory
    out_dir = resolve_path(config.output.narrative_dir) if output_dir is None else output_dir

    filepath = generate_narrative(
        result,
        output_dir=out_dir,
    )

    click.echo(f"Narrative generated: {filepath}")
