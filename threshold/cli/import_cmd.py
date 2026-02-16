"""CLI command: threshold import — Import legacy data into the database."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import click

logger = logging.getLogger(__name__)


@click.group("import")
@click.pass_context
def import_group(ctx: click.Context) -> None:
    """Import legacy data into the Threshold database."""
    pass


@import_group.command("registry")
@click.argument("path", type=click.Path(exists=True))
@click.pass_context
def import_registry(ctx: click.Context, path: str) -> None:
    """Import ticker_registry.json into the tickers table.

    PATH: path to ticker_registry.json file.
    """
    from threshold.config.loader import load_config, resolve_path
    from threshold.storage.database import Database
    from threshold.storage.queries import upsert_ticker

    config = load_config(ctx.obj.get("config_path"))
    db_path = resolve_path(config.database.path)

    with open(path) as f:
        registry = json.load(f)

    count = 0
    with Database(db_path) as db:
        for symbol, data in registry.items():
            if symbol.startswith("_"):
                continue  # Skip metadata keys
            upsert_ticker(
                db,
                symbol=symbol,
                name=data.get("name"),
                type=data.get("type"),
                sector=data.get("sector"),
                sector_detail=data.get("sector_detail"),
                yf_symbol=data.get("yf_symbol"),
                alden_category=data.get("alden_category", "Other"),
                is_gold=data.get("is_gold", False),
                is_hard_money=data.get("is_hard_money", False),
                is_crypto=data.get("is_crypto", False),
                is_crypto_exempt=data.get("is_crypto_exempt", False),
                is_cash=data.get("is_cash", False),
                is_war_chest=data.get("is_war_chest", False),
                is_international=data.get("is_international", False),
                is_amplifier_trim=data.get("is_amplifier_trim", False),
                is_defensive_add=data.get("is_defensive_add", False),
                dd_override=data.get("dd_override"),
                needs_review=data.get("needs_review", False),
                notes=data.get("notes"),
            )
            count += 1

    click.echo(f"Imported {count} tickers from {path}")


@import_group.command("scores")
@click.argument("dir_path", type=click.Path(exists=True))
@click.option("--limit", default=52, help="Max files to import (most recent first)")
@click.pass_context
def import_scores(ctx: click.Context, dir_path: str, limit: int) -> None:
    """Import weekly_scores_*.json files into the scores table.

    DIR_PATH: directory containing weekly_scores_*.json files.
    """
    from threshold.config.loader import load_config, resolve_path
    from threshold.storage.database import Database
    from threshold.storage.queries import insert_score, insert_scoring_run

    config = load_config(ctx.obj.get("config_path"))
    db_path = resolve_path(config.database.path)

    score_dir = Path(dir_path)
    files = sorted(score_dir.glob("weekly_scores_*.json"), reverse=True)[:limit]

    if not files:
        click.echo(f"No weekly_scores_*.json files found in {dir_path}")
        return

    imported_runs = 0
    imported_scores = 0

    with Database(db_path) as db:
        for filepath in reversed(files):  # Import oldest first
            try:
                with open(filepath) as f:
                    data = json.load(f)
            except (json.JSONDecodeError, OSError) as e:
                click.echo(f"  Skipping {filepath.name}: {e}")
                continue

            meta = data.get("_metadata", {})
            scores_dict = data.get("scores", {})
            if not scores_dict:
                continue

            # Generate a run_id from the filename date
            date_part = filepath.stem.replace("weekly_scores_", "")
            run_id = f"import-{date_part}"

            # Insert scoring run
            insert_scoring_run(
                db,
                run_id=run_id,
                vix_current=meta.get("vix_current", 0),
                vix_regime=meta.get("vix_regime", ""),
                tickers_scored=len(scores_dict),
                status="imported",
            )
            imported_runs += 1

            # Insert scores
            for symbol, score_data in scores_dict.items():
                sub_scores = score_data.get("sub_scores", {})
                insert_score(
                    db,
                    run_id=run_id,
                    symbol=symbol,
                    dcs=score_data.get("dcs", 0),
                    dcs_signal=score_data.get("dcs_signal", ""),
                    mq=sub_scores.get("MQ", 0),
                    fq=sub_scores.get("FQ", 0),
                    to=sub_scores.get("TO", 0),
                    mr=sub_scores.get("MR", 0),
                    vc=sub_scores.get("VC", 0),
                    is_etf=int(score_data.get("is_etf", False)),
                )
                imported_scores += 1

    click.echo(
        f"Imported {imported_runs} scoring runs, {imported_scores} ticker scores"
    )


@import_group.command("drawdown")
@click.argument("path", type=click.Path(exists=True))
@click.pass_context
def import_drawdown(ctx: click.Context, path: str) -> None:
    """Import drawdown_defense_*.json into drawdown_classifications.

    PATH: path to drawdown_defense JSON file.
    """
    from threshold.config.loader import load_config, resolve_path
    from threshold.storage.database import Database
    from threshold.storage.queries import upsert_drawdown_classification

    config = load_config(ctx.obj.get("config_path"))
    db_path = resolve_path(config.database.path)

    with open(path) as f:
        data = json.load(f)

    # Extract date from filename or metadata
    filepath = Path(path)
    date_part = filepath.stem.replace("drawdown_defense_", "")
    backtest_date = data.get("_metadata", {}).get("backtest_date", date_part)

    classifications = data.get("classifications", data)
    if isinstance(classifications, dict) and "_metadata" in classifications:
        # Top-level dict with metadata — remove it
        classifications = {k: v for k, v in classifications.items() if not k.startswith("_")}

    count = 0
    with Database(db_path) as db:
        for symbol, dd_data in classifications.items():
            if symbol.startswith("_"):
                continue
            if isinstance(dd_data, dict):
                classification = dd_data.get("classification", dd_data.get("class", "MODERATE"))
                upsert_drawdown_classification(
                    db,
                    backtest_date=backtest_date,
                    symbol=symbol,
                    classification=classification,
                    downside_capture=dd_data.get("downside_capture", 0),
                    win_rate=dd_data.get("win_rate", 0),
                )
                count += 1

    click.echo(f"Imported {count} drawdown classifications from {path}")
