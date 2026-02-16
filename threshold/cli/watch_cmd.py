"""CLI command: threshold watch — Watch for new SA exports."""

from __future__ import annotations

import logging

import click

logger = logging.getLogger(__name__)


@click.command("watch")
@click.option("--once", is_flag=True, help="Run one check and exit (cron-friendly)")
@click.option("--interval", default=600, help="Poll interval in seconds (default 600)")
@click.option("--status", "show_status", is_flag=True, help="Show watcher state and exit")
@click.pass_context
def watch_cmd(
    ctx: click.Context, once: bool, interval: int, show_status: bool
) -> None:
    """Watch for new SA export files and auto-onboard tickers.

    By default runs as a polling daemon. Use --once for a single check
    (suitable for cron jobs).
    """
    from threshold.config.loader import load_config, resolve_path
    from threshold.storage.database import Database

    config = load_config(ctx.obj.get("config_path"))
    db_path = resolve_path(config.database.path)
    sa_config = config.data_sources.seeking_alpha

    export_dir = sa_config.export_dir
    if not export_dir:
        click.echo("Error: No export_dir configured in data_sources.seeking_alpha")
        raise SystemExit(1)

    z_file_dirs = [sa_config.z_file_dir] if sa_config.z_file_dir else None

    if show_status:
        _show_status(db_path, export_dir, z_file_dirs)
        return

    with Database(db_path) as db:
        if once:
            from threshold.data.watcher import run_watch_cycle

            result = run_watch_cycle(db, export_dir, z_file_dirs)
            if result.new_files:
                click.echo(
                    f"Processed {len(result.new_files)} files: "
                    f"{result.new_tickers} new tickers, "
                    f"{result.review_needed} need review"
                )
            else:
                click.echo("No new export files found")
        else:
            from threshold.data.watcher import run_daemon

            run_daemon(
                db,
                export_dir,
                z_file_dirs=z_file_dirs,
                interval=interval,
            )


def _show_status(
    db_path: str, export_dir: str, z_file_dirs: list[str] | None
) -> None:
    """Show the current watcher state."""
    from pathlib import Path

    from threshold.data.watcher import find_new_exports, get_last_processed_mtime
    from threshold.storage.database import Database

    with Database(db_path) as db:
        last_mtime = get_last_processed_mtime(db)

    dirs = [Path(export_dir)]
    if z_file_dirs:
        dirs.extend(Path(d) for d in z_file_dirs)

    pending = find_new_exports(dirs, last_mtime)

    click.echo("Watcher status:")
    click.echo(f"  Last processed mtime: {last_mtime:.0f}")
    click.echo(f"  Export dir: {export_dir}")
    if z_file_dirs:
        click.echo(f"  Z-file dirs: {', '.join(z_file_dirs)}")
    click.echo(f"  Pending new files: {len(pending)}")
    for f in pending[:10]:
        click.echo(f"    - {f.name}")
    if len(pending) > 10:
        click.echo(f"    ... and {len(pending) - 10} more")
