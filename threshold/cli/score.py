"""CLI command: threshold score — Run the scoring pipeline."""

from __future__ import annotations

import logging

import click

logger = logging.getLogger(__name__)


@click.command("score")
@click.option("--ticker", "-t", default=None, help="Score a single ticker only")
@click.option("--dry-run", is_flag=True, help="Score but don't persist to database")
@click.option("--no-email", is_flag=True, help="Skip sending email alerts")
@click.pass_context
def score_cmd(ctx: click.Context, ticker: str | None, dry_run: bool, no_email: bool) -> None:
    """Score portfolio tickers and generate DCS signals.

    Runs the full scoring pipeline: fetch prices, compute DCS for each ticker,
    generate alerts, and persist results to the database.
    """
    from threshold.config.loader import load_config, resolve_path
    from threshold.engine.pipeline import run_scoring_pipeline
    from threshold.output.alerts import (
        build_scoring_email,
        generate_scoring_alerts,
        save_score_history,
        send_email,
    )
    from threshold.storage.database import Database

    config = load_config(ctx.obj.get("config_path"))
    db_path = resolve_path(config.database.path)

    click.echo("Threshold Scoring Pipeline")
    click.echo("=" * 40)

    if ticker:
        click.echo(f"Single ticker mode: {ticker}")
    if dry_run:
        click.echo("DRY RUN — results will not be persisted")

    with Database(db_path) as db:
        # Run the pipeline
        result = run_scoring_pipeline(
            config=config,
            db=db,
            ticker_filter=ticker,
            dry_run=dry_run,
        )

        # Display results
        click.echo(f"\nScored: {result.n_scored} tickers")
        click.echo(f"VIX: {result.vix_current:.1f} ({result.vix_regime})")
        click.echo(
            f"SPY: {'above' if result.spy_above_200d else 'BELOW'} 200d "
            f"({result.spy_pct_from_200d:+.1%})"
        )

        if result.correlation.effective_bets > 0:
            click.echo(f"Effective bets: {result.correlation.effective_bets:.1f}")

        # Top scores
        if result.top_scores:
            click.echo("\nTop DCS Scores:")
            for sym, dcs in result.top_scores[:10]:
                signal = result.scores[sym].get("dcs_signal", "")
                click.echo(f"  {sym:8s} {dcs:5.1f}  {signal}")

        # Alerts
        alerts = generate_scoring_alerts(result.scores, config)
        if alerts:
            click.echo(f"\n{'='*40}")
            click.echo(f"ALERTS ({len(alerts)}):")
            for alert in alerts:
                click.echo(f"  [{alert['level']}] {alert['ticker']} DCS={alert['score']:.0f}")

        # Save score history
        if not dry_run:
            save_score_history(
                scores=result.scores,
                vix_current=result.vix_current,
                vix_regime=result.vix_regime,
                spy_pct=result.spy_pct_from_200d,
                breadth_pct=result.breadth_pct,
                effective_bets=result.correlation.effective_bets,
                market_regime_score=result.market_regime_score,
                run_metadata=result.tracker.to_dict(),
                output_dir=resolve_path(config.output.score_history_dir),
            )

        # Email alerts
        if alerts and not dry_run and not no_email and config.alerts.enabled:
            subject, body = build_scoring_email(
                scores=result.scores,
                alerts=alerts,
                vix_current=result.vix_current,
                vix_regime=result.vix_regime,
                spy_pct=result.spy_pct_from_200d,
                scored_count=result.n_scored,
            )
            if send_email(subject, body, config):
                click.echo(f"\nEmail sent: {subject}")

        # Concentration warnings
        if result.concentration_warnings:
            click.echo(f"\n⚠️  CONCENTRATION WARNINGS:")
            for warn in result.concentration_warnings:
                click.echo(
                    f"  {warn['ticker']} highly correlated ({warn['correlation']:.2f}) "
                    f"with {warn['correlated_with']}"
                )

        # Summary
        tracker = result.tracker
        if tracker.errors:
            click.echo(f"\nErrors ({len(tracker.errors)}):")
            for err in tracker.errors[:5]:
                click.echo(f"  {err}")

        click.echo(f"\nPipeline complete: {tracker.tickers_scored} scored, "
                    f"{tracker.tickers_failed} failed, "
                    f"{tracker.tickers_skipped} skipped")
