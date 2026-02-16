"""Config CLI commands: show, validate."""

from __future__ import annotations

import click


@click.group("config")
def config_group() -> None:
    """Manage configuration."""
    pass


@config_group.command("show")
@click.pass_context
def config_show(ctx: click.Context) -> None:
    """Print the resolved configuration."""
    from threshold.config.loader import load_config

    config = load_config(ctx.obj.get("config_path"))
    # Pretty-print as YAML-like output
    import json

    click.echo(json.dumps(config.model_dump(), indent=2, default=str))


@config_group.command("validate")
@click.pass_context
def config_validate(ctx: click.Context) -> None:
    """Validate config.yaml against the schema."""
    from threshold.config.loader import load_config

    try:
        config = load_config(ctx.obj.get("config_path"))
        click.echo("Config is valid.")
        click.echo(f"  Version: {config.version}")
        click.echo(f"  Accounts: {len(config.accounts)}")
        click.echo(f"  Data sources: yfinance={config.data_sources.yfinance.enabled}, "
                    f"tiingo={config.data_sources.tiingo.enabled}, "
                    f"sa={config.data_sources.seeking_alpha.enabled}, "
                    f"fred={config.data_sources.fred.enabled}")
        click.echo(f"  Database: {config.database.path}")
    except Exception as e:
        click.echo(f"Config validation failed: {e}", err=True)
        raise SystemExit(1) from None
