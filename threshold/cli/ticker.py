"""Ticker management CLI commands: add, list, info, remove, review."""

from __future__ import annotations

import click


@click.group("ticker")
def ticker_group() -> None:
    """Manage ticker registry."""
    pass


@ticker_group.command("add")
@click.argument("symbol")
@click.option("--dry-run", is_flag=True, help="Show what would be added without writing")
@click.pass_context
def ticker_add(ctx: click.Context, symbol: str, dry_run: bool) -> None:
    """Register a new ticker with auto-enrichment via yfinance."""
    from threshold.config.loader import load_config, resolve_path
    from threshold.data.adapters.yfinance_adapter import enrich_ticker
    from threshold.storage.database import Database
    from threshold.storage.migrations import ensure_schema
    from threshold.storage.queries import get_ticker, upsert_ticker

    config = load_config(ctx.obj.get("config_path"))
    symbol = symbol.upper().strip()

    click.echo(f"Enriching {symbol} via yfinance...")
    info = enrich_ticker(symbol)

    if info is None:
        click.echo(f"Could not find {symbol} on yfinance. Check the symbol.", err=True)
        raise SystemExit(1)

    click.echo(f"  Name: {info['name']}")
    click.echo(f"  Type: {info['type']}")
    click.echo(f"  Sector: {info['sector']}")
    click.echo(f"  Sector Detail: {info['sector_detail']}")
    click.echo(f"  Needs Review: {info['needs_review']}")

    if dry_run:
        click.echo("\n[Dry run] No changes written.")
        return

    db_path = resolve_path(config.database.path)
    with Database(db_path) as db:
        ensure_schema(db)
        existing = get_ticker(db, symbol)
        upsert_ticker(db, symbol, **info)

        if existing:
            click.echo(f"\nUpdated {symbol} in database.")
        else:
            click.echo(f"\nRegistered {symbol} in database.")


@ticker_group.command("list")
@click.option("--review", is_flag=True, help="Show only tickers needing review")
@click.pass_context
def ticker_list(ctx: click.Context, review: bool) -> None:
    """List all registered tickers."""
    from threshold.config.loader import load_config, resolve_path
    from threshold.storage.database import Database
    from threshold.storage.migrations import ensure_schema
    from threshold.storage.queries import list_tickers

    config = load_config(ctx.obj.get("config_path"))
    db_path = resolve_path(config.database.path)

    with Database(db_path) as db:
        ensure_schema(db)
        tickers = list_tickers(db, needs_review=True if review else None)

    if not tickers:
        click.echo("No tickers registered." if not review else "No tickers need review.")
        return

    # Table header
    click.echo(f"{'Symbol':<10} {'Name':<35} {'Type':<8} {'Sector':<20} {'Category':<18} {'Review'}")
    click.echo("-" * 100)

    for t in tickers:
        name = (t["name"] or "")[:34]
        review_flag = "*" if t["needs_review"] else ""
        click.echo(
            f"{t['symbol']:<10} {name:<35} {(t['type'] or ''):<8} "
            f"{(t['sector'] or ''):<20} {(t['alden_category'] or ''):<18} {review_flag}"
        )

    click.echo(f"\nTotal: {len(tickers)} tickers")


@ticker_group.command("info")
@click.argument("symbol")
@click.pass_context
def ticker_info(ctx: click.Context, symbol: str) -> None:
    """Show full metadata for a ticker."""
    from threshold.config.loader import load_config, resolve_path
    from threshold.storage.database import Database
    from threshold.storage.migrations import ensure_schema
    from threshold.storage.queries import get_ticker

    config = load_config(ctx.obj.get("config_path"))
    db_path = resolve_path(config.database.path)
    symbol = symbol.upper().strip()

    with Database(db_path) as db:
        ensure_schema(db)
        t = get_ticker(db, symbol)

    if not t:
        click.echo(f"Ticker {symbol} not found.", err=True)
        raise SystemExit(1)

    click.echo(f"Ticker: {t['symbol']}")
    click.echo(f"  Name:            {t['name']}")
    click.echo(f"  Type:            {t['type']}")
    click.echo(f"  Sector:          {t['sector']}")
    click.echo(f"  Sector Detail:   {t['sector_detail']}")
    click.echo(f"  YF Symbol:       {t['yf_symbol']}")
    click.echo(f"  Alden Category:  {t['alden_category']}")
    click.echo(f"  Flags:")
    for flag in [
        "is_gold", "is_hard_money", "is_crypto", "is_crypto_exempt",
        "is_cash", "is_war_chest", "is_international",
        "is_amplifier_trim", "is_defensive_add",
    ]:
        if t[flag]:
            click.echo(f"    {flag}: True")
    if t["dd_override"]:
        click.echo(f"  DD Override:     {t['dd_override']}")
    click.echo(f"  Verified:        {t['verified_at'] or 'Not verified'}")
    click.echo(f"  Needs Review:    {'Yes' if t['needs_review'] else 'No'}")
    if t["notes"]:
        click.echo(f"  Notes:           {t['notes']}")


@ticker_group.command("remove")
@click.argument("symbol")
@click.confirmation_option(prompt="Are you sure you want to remove this ticker?")
@click.pass_context
def ticker_remove(ctx: click.Context, symbol: str) -> None:
    """Remove a ticker from the registry."""
    from threshold.config.loader import load_config, resolve_path
    from threshold.storage.database import Database
    from threshold.storage.migrations import ensure_schema
    from threshold.storage.queries import delete_ticker

    config = load_config(ctx.obj.get("config_path"))
    db_path = resolve_path(config.database.path)
    symbol = symbol.upper().strip()

    with Database(db_path) as db:
        ensure_schema(db)
        removed = delete_ticker(db, symbol)

    if removed:
        click.echo(f"Removed {symbol}.")
    else:
        click.echo(f"Ticker {symbol} not found.", err=True)
