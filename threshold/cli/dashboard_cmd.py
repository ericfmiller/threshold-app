"""CLI command: threshold dashboard — Generate and open the HTML dashboard."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import click

logger = logging.getLogger(__name__)

# War chest instrument symbols (cash and near-cash holdings)
WAR_CHEST_SYMBOLS = frozenset({
    "STIP", "CASH", "SPAXX", "FDRXX", "FCASH", "CORE", "FMPXX",
    "VMFXX", "SWVXX", "SPRXX", "TTTXX",
})

# VIX-regime war chest targets
WAR_CHEST_TARGETS = {
    "COMPLACENT": 0.10,
    "NORMAL": 0.12,
    "FEAR": 0.15,
    "PANIC": 0.20,
}


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


def _load_supplemental_data(
    db: Any,
    config: Any = None,
) -> dict[str, Any]:
    """Load all supplemental data from the database.

    Returns dict with: dd_classes, ticker_sectors, ticker_values,
    positions, war_chest_pct, total_portfolio_value, held_symbols,
    exempt_tickers, active_grace_periods.
    """
    from threshold.storage.queries import (
        get_drawdown_classifications,
        get_latest_positions,
        list_tickers,
    )

    data: dict[str, Any] = {
        "dd_classes": {},
        "ticker_sectors": {},
        "ticker_values": {},
        "positions": [],
        "war_chest_value": 0.0,
        "total_portfolio_value": 0.0,
        "war_chest_pct": 0.0,
        "held_symbols": set(),
        "exempt_tickers": {},
        "active_grace_periods": [],
    }

    try:
        # Drawdown classifications
        dd_raw = get_drawdown_classifications(db)
        data["dd_classes"] = {sym: d.get("classification", "") for sym, d in dd_raw.items()}

        # Sector mapping from tickers table
        rows = db.fetchall("SELECT symbol, sector FROM tickers WHERE sector IS NOT NULL")
        data["ticker_sectors"] = {r["symbol"]: r["sector"] for r in rows}

        # Positions (for per-account reporting and war chest)
        positions = get_latest_positions(db)
        data["positions"] = positions

        # Build value mapping and war chest from positions
        total_value = 0.0
        war_chest_value = 0.0
        symbol_values: dict[str, float] = {}
        held_symbols: set[str] = set()

        for pos in positions:
            symbol = pos.get("symbol", "")
            value = float(pos.get("market_value", 0))
            if value > 0:
                symbol_values[symbol] = symbol_values.get(symbol, 0) + value
                total_value += value
                held_symbols.add(symbol)
                if symbol.upper() in WAR_CHEST_SYMBOLS:
                    war_chest_value += value

        data["ticker_values"] = symbol_values
        data["total_portfolio_value"] = total_value
        data["war_chest_value"] = war_chest_value
        data["war_chest_pct"] = war_chest_value / total_value if total_value > 0 else 0.0
        data["held_symbols"] = held_symbols

        # Exempt tickers (crypto, cash — for narrative crypto section)
        try:
            from threshold.engine.exemptions import get_exempt_tickers
            if config is not None:
                all_tickers = list_tickers(db)
                data["exempt_tickers"] = get_exempt_tickers(all_tickers, config)
        except Exception:
            pass

        # Active grace periods
        try:
            from threshold.engine.grace_period import list_active_grace_periods
            data["active_grace_periods"] = list_active_grace_periods(db)
        except Exception:
            pass

    except Exception as e:
        logger.debug("Could not load supplemental data: %s", e)

    return data


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
    db = Database(resolve_path(config.database.path))
    db.connect()
    supp = _load_supplemental_data(db, config=config)
    db.close()

    result.held_symbols = supp["held_symbols"]
    result.exempt_tickers = supp["exempt_tickers"]
    result.active_grace_periods = supp["active_grace_periods"]

    # Compute war chest target based on VIX regime
    war_chest_target = WAR_CHEST_TARGETS.get(result.vix_regime, 0.12)

    # Determine output directory
    out_dir = resolve_path(config.output.dashboard_dir) if output_dir is None else output_dir

    filepath = generate_dashboard(
        result,
        ticker_sectors=supp["ticker_sectors"],
        ticker_values=supp["ticker_values"],
        drawdown_classifications=supp["dd_classes"],
        positions=supp["positions"],
        war_chest_pct=supp["war_chest_pct"],
        war_chest_target=war_chest_target,
        war_chest_value=supp["war_chest_value"],
        total_portfolio_value=supp["total_portfolio_value"],
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
    db = Database(resolve_path(config.database.path))
    db.connect()
    supp = _load_supplemental_data(db, config=config)
    db.close()

    result.held_symbols = supp["held_symbols"]
    result.exempt_tickers = supp["exempt_tickers"]
    result.active_grace_periods = supp["active_grace_periods"]

    # Compute war chest target based on VIX regime
    war_chest_target = WAR_CHEST_TARGETS.get(result.vix_regime, 0.12)

    # Determine output directory
    out_dir = resolve_path(config.output.narrative_dir) if output_dir is None else output_dir

    filepath = generate_narrative(
        result,
        ticker_sectors=supp["ticker_sectors"],
        ticker_values=supp["ticker_values"],
        drawdown_classifications=supp["dd_classes"],
        positions=supp["positions"],
        war_chest_pct=supp["war_chest_pct"],
        war_chest_target=war_chest_target,
        war_chest_value=supp["war_chest_value"],
        total_portfolio_value=supp["total_portfolio_value"],
        output_dir=out_dir,
    )

    click.echo(f"Narrative generated: {filepath}")
