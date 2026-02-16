"""CLI command: threshold sync — Full workflow orchestration.

Runs the complete data-to-analysis workflow:
  1. Detect new SA exports (if export dir configured)
  2. Onboard new tickers from exports
  3. Import positions from Holdings sheets
  4. Generate portfolio snapshot
  5. Score all tickers
  6. Capture performance snapshot (vs SPY)

This is the primary "one-command" entry point for weekly reviews.
"""

from __future__ import annotations

import logging

import click

logger = logging.getLogger(__name__)


@click.command("sync")
@click.option("--dry-run", is_flag=True, help="Show what would happen without persisting")
@click.option("--no-score", is_flag=True, help="Skip the scoring pipeline")
@click.option("--no-email", is_flag=True, help="Skip sending email alerts")
@click.pass_context
def sync_cmd(
    ctx: click.Context,
    dry_run: bool,
    no_score: bool,
    no_email: bool,
) -> None:
    """Run full sync: detect exports, onboard, import, score, snapshot.

    Orchestrates the complete weekly workflow in a single command.
    """
    from threshold.config.loader import load_config, resolve_path
    from threshold.storage.database import Database

    config = load_config(ctx.obj.get("config_path"))
    db_path = resolve_path(config.database.path)

    click.echo("Threshold Sync")
    click.echo("=" * 40)
    if dry_run:
        click.echo("DRY RUN — no changes will be persisted\n")

    with Database(db_path) as db:
        # Step 1: Onboard new tickers from SA exports
        _step_onboard(db, config, dry_run)

        # Step 2: Import positions from Holdings sheets
        _step_import_positions(db, config, dry_run)

        # Step 3: Generate portfolio snapshot
        _step_snapshot(db, config, dry_run)

        # Step 4: Run scoring pipeline
        if not no_score:
            result = _step_score(db, config, dry_run, no_email)
        else:
            click.echo("\n[4/5] Scoring skipped (--no-score)")
            result = None

        # Step 5: Capture performance snapshot
        _step_performance(db, config, dry_run, result)

    click.echo("\n" + "=" * 40)
    click.echo("Sync complete.")


def _step_onboard(db, config, dry_run: bool) -> None:  # noqa: ANN001
    """Step 1: Detect and onboard new tickers."""
    click.echo("\n[1/5] Checking for new tickers...")

    try:
        from threshold.data.onboarding import run_onboarding

        onboard_result = run_onboarding(db, config, dry_run=dry_run)
        if onboard_result.added > 0:
            click.echo(f"  Onboarded {onboard_result.added} new ticker(s)")
        else:
            click.echo("  No new tickers found")
        if onboard_result.errors:
            for err in onboard_result.errors[:3]:
                click.echo(f"  Warning: {err}")
    except Exception as e:
        click.echo(f"  Onboarding skipped: {e}")


def _step_import_positions(db, config, dry_run: bool) -> None:  # noqa: ANN001
    """Step 2: Import positions from SA export Holdings sheets + synthetic."""
    click.echo("\n[2/5] Importing positions...")

    try:
        from threshold.data.position_import import (
            import_all_positions,
            import_synthetic_positions,
        )

        import_result = import_all_positions(db, config)
        total = import_result.positions_imported
        if total > 0:
            click.echo(f"  Imported {total} position(s) across {import_result.accounts_processed} account(s)")
        else:
            click.echo("  No SA export positions to import")

        # Import TSP + BTC synthetic positions
        synthetic = import_synthetic_positions(db, config)
        if synthetic > 0:
            click.echo(f"  Imported {synthetic} synthetic position(s) (TSP/BTC)")
    except Exception as e:
        click.echo(f"  Position import skipped: {e}")


def _step_snapshot(db, config, dry_run: bool) -> None:  # noqa: ANN001
    """Step 3: Generate portfolio snapshot."""
    click.echo("\n[3/5] Generating portfolio snapshot...")

    try:
        from threshold.data.snapshot import generate_snapshot, save_snapshot

        snapshot = generate_snapshot(db, config)
        total = snapshot.get("total_portfolio", 0)
        if total > 0 and not dry_run:
            save_snapshot(db, snapshot)
            click.echo(f"  Total portfolio: ${total:,.2f}")
        elif total > 0:
            click.echo(f"  Total portfolio: ${total:,.2f} (dry run, not saved)")
        else:
            click.echo("  No portfolio data available for snapshot")
    except Exception as e:
        click.echo(f"  Snapshot skipped: {e}")


def _step_score(db, config, dry_run: bool, no_email: bool):  # noqa: ANN001, ANN202
    """Step 4: Run the scoring pipeline."""
    click.echo("\n[4/5] Running scoring pipeline...")

    from threshold.engine.pipeline import run_scoring_pipeline

    result = run_scoring_pipeline(
        config=config,
        db=db,
        dry_run=dry_run,
    )

    click.echo(f"  Scored: {result.n_scored} tickers")
    click.echo(f"  VIX: {result.vix_current:.1f} ({result.vix_regime})")

    if result.top_scores:
        top3 = ", ".join(f"{s}={d:.0f}" for s, d in result.top_scores[:3])
        click.echo(f"  Top DCS: {top3}")

    if result.exempt_tickers:
        click.echo(f"  Exempt: {len(result.exempt_tickers)} ticker(s)")

    if result.active_grace_periods:
        click.echo(f"  Grace periods: {len(result.active_grace_periods)} active")

    # Alerts
    if not dry_run and not no_email:
        try:
            from threshold.output.alerts import (
                build_scoring_email,
                generate_scoring_alerts,
                send_email,
            )

            alerts = generate_scoring_alerts(result.scores, config)
            if alerts and config.alerts.enabled:
                subject, body = build_scoring_email(
                    scores=result.scores,
                    alerts=alerts,
                    vix_current=result.vix_current,
                    vix_regime=result.vix_regime,
                    spy_pct=result.spy_pct_from_200d,
                    scored_count=result.n_scored,
                )
                if send_email(subject, body, config):
                    click.echo(f"  Email sent: {subject}")
        except Exception as e:
            logger.debug("Alert email skipped: %s", e)

    return result


def _step_performance(db, config, dry_run: bool, result) -> None:  # noqa: ANN001
    """Step 5: Capture performance snapshot vs SPY."""
    click.echo("\n[5/5] Capturing performance snapshot...")

    try:
        # Get total portfolio from snapshot
        from threshold.data.snapshot import load_latest_snapshot
        from threshold.portfolio.performance import capture_performance_snapshot

        snapshot = load_latest_snapshot(db)
        total = snapshot.get("total_portfolio", 0) if snapshot else 0

        if total > 0:
            spy_close = None
            if result and result.scores:
                # Try to get SPY close from pipeline context
                pass  # SPY close not directly accessible; let the module fetch it

            perf = capture_performance_snapshot(
                db,
                total_portfolio=total,
                spy_close=spy_close,
            )
            if not dry_run:
                click.echo(
                    f"  Snapshot captured: ${perf.total_portfolio:,.2f} "
                    f"(SPY={perf.spy_close:.2f})"
                )
            else:
                click.echo(f"  Would capture: ${total:,.2f} (dry run)")
        else:
            click.echo("  No portfolio value available — skipping")
    except Exception as e:
        click.echo(f"  Performance snapshot skipped: {e}")
