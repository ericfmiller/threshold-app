"""Top-level CLI entry point for Threshold."""

from __future__ import annotations

import logging
from pathlib import Path

import click

from threshold import __version__


@click.group()
@click.version_option(version=__version__, prog_name="threshold")
@click.option(
    "--config",
    type=click.Path(),
    default=None,
    envvar="THRESHOLD_CONFIG",
    help="Path to config.yaml",
)
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose logging")
@click.pass_context
def cli(ctx: click.Context, config: str | None, verbose: bool) -> None:
    """Threshold -- Quantitative Investment Analysis System."""
    ctx.ensure_object(dict)
    ctx.obj["config_path"] = config

    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


# Register sub-commands
from threshold.cli.config_cmd import config_group  # noqa: E402
from threshold.cli.dashboard_cmd import dashboard_cmd, narrative_cmd  # noqa: E402
from threshold.cli.import_cmd import import_group  # noqa: E402
from threshold.cli.score import score_cmd  # noqa: E402
from threshold.cli.ticker import ticker_group  # noqa: E402

cli.add_command(config_group, "config")
cli.add_command(dashboard_cmd, "dashboard")
cli.add_command(import_group, "import")
cli.add_command(narrative_cmd, "narrative")
cli.add_command(score_cmd, "score")
cli.add_command(ticker_group, "ticker")


@cli.command()
@click.pass_context
def init(ctx: click.Context) -> None:
    """Initialize Threshold: create directories, database, and example config."""
    from threshold.config.loader import load_config, resolve_path
    from threshold.storage.database import Database
    from threshold.storage.migrations import ensure_schema
    from threshold.storage.queries import seed_alden_categories, upsert_account

    config = load_config(ctx.obj.get("config_path"))

    # Create directories
    threshold_dir = Path("~/.threshold").expanduser()
    threshold_dir.mkdir(parents=True, exist_ok=True)

    for dir_attr in ["score_history_dir", "dashboard_dir", "narrative_dir"]:
        dir_path = resolve_path(getattr(config.output, dir_attr))
        dir_path.mkdir(parents=True, exist_ok=True)
        click.echo(f"  Created {dir_path}")

    # Create/migrate database
    db_path = resolve_path(config.database.path)
    click.echo(f"  Database: {db_path}")

    with Database(db_path) as db:
        version = ensure_schema(db)
        click.echo(f"  Schema version: {version}")

        # Seed alden categories from config
        seed_alden_categories(db, config.alden_categories)

        # Seed accounts from config
        for acct in config.accounts:
            upsert_account(
                db,
                id=acct.id,
                name=acct.name,
                type=acct.type,
                institution=acct.institution,
                tax_treatment=acct.tax_treatment,
                sa_export_prefix=acct.sa_export_prefix,
                sa_export_prefix_old=acct.sa_export_prefix_old,
            )
        if config.accounts:
            click.echo(f"  Seeded {len(config.accounts)} accounts")

    # Copy example config if none exists
    user_config = threshold_dir / "config.yaml"
    if not user_config.exists():
        example = Path(__file__).parent.parent.parent / "config.yaml.example"
        if example.exists():
            import shutil
            shutil.copy2(example, user_config)
            click.echo(f"  Copied example config to {user_config}")

    click.echo("\nThreshold initialized successfully.")
    click.echo("Next steps:")
    click.echo(f"  1. Edit {user_config} to add your API keys and accounts")
    click.echo("  2. Run: threshold ticker add AAPL  (to register tickers)")
    click.echo("  3. Run: threshold score            (to score your portfolio)")
