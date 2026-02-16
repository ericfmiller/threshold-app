"""Full portfolio scoring pipeline orchestrator.

Coordinates the end-to-end scoring run:
  1. Load configuration and open database
  2. Build ticker universe (portfolio + watchlists)
  3. Fetch price data (yfinance primary, Tiingo fallback)
  4. Fetch macro data (FRED, VIX regime, breadth)
  5. Build ScoringContext
  6. Score each ticker via scorer.score_ticker()
  7. Run correlation / concentration analysis
  8. Persist results to SQLite
  9. Generate alerts

This module is the core of ``threshold score``.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

from threshold.config.schema import ThresholdConfig
from threshold.engine.composite import classify_vix
from threshold.engine.context import ScoringContext
from threshold.engine.scorer import ScoringResult, score_ticker
from threshold.engine.technical import calc_rsi_value
from threshold.portfolio.accounts import PortfolioSnapshot, aggregate_positions
from threshold.portfolio.correlation import (
    CorrelationReport,
    check_concentration_risk,
    compute_correlation_report,
)
from threshold.storage.database import Database
from threshold.storage.queries import (
    get_drawdown_classifications,
    get_latest_scores,
    insert_score,
    insert_scoring_run,
    insert_signal,
    list_tickers,
    update_data_freshness,
    update_scoring_run,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Run tracker
# ---------------------------------------------------------------------------

@dataclass
class RunTracker:
    """Tracks the state and metrics of a scoring run."""

    run_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    started_at: str = field(default_factory=lambda: datetime.now().isoformat())
    data_sources: dict[str, str] = field(default_factory=dict)
    """source_name → status ('ok', 'failed', 'skipped', 'stale')."""
    tickers_scored: int = 0
    tickers_failed: int = 0
    tickers_skipped: int = 0
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "started_at": self.started_at,
            "data_sources": self.data_sources,
            "tickers_scored": self.tickers_scored,
            "tickers_failed": self.tickers_failed,
            "tickers_skipped": self.tickers_skipped,
            "errors": self.errors,
            "warnings": self.warnings,
        }


# ---------------------------------------------------------------------------
# Pipeline result
# ---------------------------------------------------------------------------

@dataclass
class PipelineResult:
    """Result from a full scoring pipeline run."""

    run_id: str = ""
    scores: dict[str, ScoringResult] = field(default_factory=dict)
    """symbol → ScoringResult for each scored ticker."""
    correlation: CorrelationReport = field(default_factory=CorrelationReport)
    concentration_warnings: list[dict[str, Any]] = field(default_factory=list)
    vix_current: float = 0.0
    vix_regime: str = "NORMAL"
    spy_pct_from_200d: float = 0.0
    spy_above_200d: bool = True
    market_regime_score: float = 0.5
    breadth_pct: float = 0.0
    tracker: RunTracker = field(default_factory=RunTracker)
    alerts: list[dict[str, Any]] = field(default_factory=list)

    @property
    def n_scored(self) -> int:
        return len(self.scores)

    @property
    def top_scores(self) -> list[tuple[str, float]]:
        """Top 10 tickers by DCS, descending."""
        scored = [
            (sym, res["dcs"])
            for sym, res in self.scores.items()
            if "dcs" in res
        ]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:10]


# ---------------------------------------------------------------------------
# Price fetching
# ---------------------------------------------------------------------------

def _fetch_prices_yfinance(
    tickers: list[str],
    period: str = "2y",
) -> pd.DataFrame:
    """Fetch price data via yfinance.

    Returns:
        DataFrame with MultiIndex columns (Price, Ticker) for Close prices,
        or a simpler DataFrame depending on yfinance version.
    """
    try:
        import yfinance as yf
        data = yf.download(
            tickers,
            period=period,
            progress=False,
            auto_adjust=True,
            threads=True,
        )
        return data
    except Exception as e:
        logger.error("yfinance batch download failed: %s", e)
        return pd.DataFrame()


def _extract_close(
    batch_data: pd.DataFrame,
    ticker: str,
) -> pd.Series | None:
    """Extract close price series for a single ticker from batch download."""
    if batch_data.empty:
        return None
    try:
        # yfinance returns MultiIndex columns: (Price, Ticker)
        if isinstance(batch_data.columns, pd.MultiIndex):
            if "Close" in batch_data.columns.get_level_values(0):
                series = batch_data["Close"][ticker].dropna()
                if len(series) > 0:
                    return series
        else:
            # Single ticker download — columns are price types
            if "Close" in batch_data.columns:
                return batch_data["Close"].dropna()
    except (KeyError, TypeError):
        pass
    return None


# ---------------------------------------------------------------------------
# Market context
# ---------------------------------------------------------------------------

def _compute_market_context(
    batch_data: pd.DataFrame,
    config: ThresholdConfig,
) -> dict[str, Any]:
    """Compute market-level context from batch price data.

    Returns dict with:
        spy_close, spy_above_200d, spy_pct_from_200d,
        vix_current, vix_percentile, vix_regime,
        breadth_pct, breadth_above, breadth_total,
        market_regime_score
    """
    from threshold.engine.subscores import calc_market_regime

    result: dict[str, Any] = {
        "spy_close": None,
        "spy_above_200d": True,
        "spy_pct_from_200d": 0.0,
        "vix_current": 15.0,
        "vix_percentile": 0.5,
        "vix_regime": "NORMAL",
        "breadth_pct": 0.5,
        "breadth_above": 0,
        "breadth_total": 0,
        "market_regime_score": 0.5,
    }

    # SPY
    spy_close = _extract_close(batch_data, "SPY")
    if spy_close is not None and len(spy_close) >= 200:
        result["spy_close"] = spy_close
        sma_200 = spy_close.rolling(200).mean().iloc[-1]
        current_price = spy_close.iloc[-1]
        result["spy_above_200d"] = current_price > sma_200
        result["spy_pct_from_200d"] = (
            (current_price - sma_200) / sma_200 if sma_200 > 0 else 0.0
        )

    # VIX
    vix_close = _extract_close(batch_data, "^VIX")
    if vix_close is not None and len(vix_close) > 0:
        result["vix_current"] = float(vix_close.iloc[-1])
        # Percentile over available history
        result["vix_percentile"] = float(
            (vix_close < vix_close.iloc[-1]).mean()
        )
        result["vix_regime"] = classify_vix(
            result["vix_current"], config.scoring.vix_regimes
        )

    # Breadth: % of tickers above their 200d SMA
    above_count = 0
    total_count = 0
    if isinstance(batch_data.columns, pd.MultiIndex):
        for ticker in batch_data["Close"].columns:
            if ticker in ("SPY", "^VIX") or ticker.startswith("^"):
                continue
            series = batch_data["Close"][ticker].dropna()
            if len(series) >= 200:
                sma = series.rolling(200).mean().iloc[-1]
                if series.iloc[-1] > sma:
                    above_count += 1
                total_count += 1

    result["breadth_above"] = above_count
    result["breadth_total"] = total_count
    result["breadth_pct"] = (
        above_count / total_count if total_count > 0 else 0.5
    )

    # Market regime score
    result["market_regime_score"] = calc_market_regime(
        vix_current=result["vix_current"],
        vix_percentile=result["vix_percentile"],
        spy_above_200d=result["spy_above_200d"],
        breadth_pct=result["breadth_pct"],
        config=config,
    )

    return result


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_scoring_pipeline(
    config: ThresholdConfig,
    db: Database,
    *,
    sa_data: dict[str, dict[str, Any]] | None = None,
    ticker_filter: str | None = None,
    dry_run: bool = False,
) -> PipelineResult:
    """Execute the full scoring pipeline.

    Parameters:
        config: Resolved ThresholdConfig.
        db: Open Database connection.
        sa_data: Pre-loaded SA ratings {symbol: sa_dict}. If None, attempts
            to load from database.
        ticker_filter: If provided, only score this single ticker.
        dry_run: If True, compute scores but don't persist to database.

    Returns:
        PipelineResult with all scores, correlation, and alerts.
    """
    tracker = RunTracker()
    result = PipelineResult(run_id=tracker.run_id, tracker=tracker)

    # ------------------------------------------------------------------
    # Step 1: Build ticker universe
    # ------------------------------------------------------------------
    logger.info("[1/6] Building ticker universe...")

    all_tickers_db = list_tickers(db)
    ticker_symbols = [t["symbol"] for t in all_tickers_db]
    ticker_meta = {t["symbol"]: t for t in all_tickers_db}

    if ticker_filter:
        ticker_symbols = [t for t in ticker_symbols if t == ticker_filter]
        if not ticker_symbols:
            logger.warning("Ticker %s not found in database", ticker_filter)
            return result

    # Build SA data dict
    sa_ratings = sa_data or {}

    # Exempt tickers (crypto, war chest, etc.)
    exempt_tickers = {
        t["symbol"] for t in all_tickers_db
        if t.get("is_crypto_exempt") or t.get("is_cash")
    }
    tickers_to_score = [
        t for t in ticker_symbols if t not in exempt_tickers
    ]

    logger.info(
        "  %d tickers (%d to score, %d exempt)",
        len(ticker_symbols),
        len(tickers_to_score),
        len(exempt_tickers),
    )

    # ------------------------------------------------------------------
    # Step 2: Fetch price data
    # ------------------------------------------------------------------
    logger.info("[2/6] Fetching price data...")

    # Build download list: scored tickers + SPY + VIX
    download_tickers = list(set(tickers_to_score + ["SPY", "^VIX"]))
    period = config.data_sources.yfinance.price_period

    batch_data = _fetch_prices_yfinance(download_tickers, period=period)

    if batch_data.empty:
        tracker.data_sources["yfinance"] = "failed"
        tracker.errors.append("yfinance batch download returned no data")
        logger.error("  yfinance failed — cannot score without price data")
        return result

    tracker.data_sources["yfinance"] = "ok"
    logger.info("  Downloaded %d days of data", len(batch_data))

    # ------------------------------------------------------------------
    # Step 3: Compute market context
    # ------------------------------------------------------------------
    logger.info("[3/6] Computing market context...")

    market_ctx = _compute_market_context(batch_data, config)
    result.vix_current = market_ctx["vix_current"]
    result.vix_regime = market_ctx["vix_regime"]
    result.spy_above_200d = market_ctx["spy_above_200d"]
    result.spy_pct_from_200d = market_ctx["spy_pct_from_200d"]
    result.market_regime_score = market_ctx["market_regime_score"]
    result.breadth_pct = market_ctx["breadth_pct"]

    logger.info(
        "  VIX=%.1f (%s), SPY %s 200d (%.1f%%), Breadth=%.0f%%",
        result.vix_current,
        result.vix_regime,
        "above" if result.spy_above_200d else "BELOW",
        result.spy_pct_from_200d * 100,
        result.breadth_pct * 100,
    )

    # ------------------------------------------------------------------
    # Step 4: Build scoring context
    # ------------------------------------------------------------------
    logger.info("[4/6] Scoring tickers...")

    # Load previous scores and drawdown classifications from DB
    prev_scores = get_latest_scores(db)
    dd_classifications = get_drawdown_classifications(db)

    ctx = ScoringContext(
        market_regime_score=result.market_regime_score,
        vix_regime=result.vix_regime,
        spy_close=market_ctx.get("spy_close"),
        prev_scores=prev_scores,
        drawdown_classifications=dd_classifications,
    )

    # ------------------------------------------------------------------
    # Step 5: Score each ticker
    # ------------------------------------------------------------------
    scored_results: dict[str, ScoringResult] = {}
    errors: list[str] = []

    for ticker in tickers_to_score:
        try:
            sa = sa_ratings.get(ticker, {})
            close = _extract_close(batch_data, ticker)

            if close is None or len(close) < 50:
                tracker.tickers_skipped += 1
                logger.debug("  %s: insufficient data (%d bars)", ticker, len(close) if close is not None else 0)
                continue

            # Build price DataFrame (score_ticker expects DataFrame with Close column)
            price_df = pd.DataFrame({"Close": close})

            scoring_result = score_ticker(
                ticker=ticker,
                sa_data=sa,
                price_df=price_df,
                ctx=ctx,
                config=config,
            )

            if scoring_result is not None:
                scored_results[ticker] = scoring_result
                tracker.tickers_scored += 1
            else:
                tracker.tickers_skipped += 1

        except Exception as e:
            tracker.tickers_failed += 1
            errors.append(f"{ticker}: {e}")
            logger.error("  %s: scoring error: %s", ticker, e)

    result.scores = scored_results
    logger.info(
        "  Scored %d, skipped %d, failed %d",
        tracker.tickers_scored,
        tracker.tickers_skipped,
        tracker.tickers_failed,
    )

    if tracker.tickers_scored > 0:
        top = result.top_scores[:3]
        top_str = ", ".join(f"{s}={d:.0f}" for s, d in top)
        logger.info("  Top DCS: %s", top_str)

    # ------------------------------------------------------------------
    # Step 5b: Correlation & concentration analysis
    # ------------------------------------------------------------------
    logger.info("[5/6] Correlation analysis...")

    # Build returns DataFrame for held tickers
    held_tickers = [
        t for t in scored_results
        if ticker_meta.get(t, {}).get("type") != "watchlist"
    ]

    if len(held_tickers) >= 3:
        returns_data: dict[str, pd.Series] = {}
        for ticker in held_tickers:
            close = _extract_close(batch_data, ticker)
            if close is not None and len(close) >= 90:
                # Use last 90 days of daily returns
                returns_data[ticker] = close.pct_change().dropna().tail(90)

        if returns_data:
            returns_df = pd.DataFrame(returns_data)
            result.correlation = compute_correlation_report(returns_df)

            logger.info(
                "  Effective bets: %.1f, High-corr pairs: %d",
                result.correlation.effective_bets,
                len(result.correlation.high_corr_pairs),
            )

            # Check concentration risk
            buy_tickers = {
                t for t, r in scored_results.items()
                if r.get("dcs", 0) >= config.scoring.thresholds.buy_dip
            }
            result.concentration_warnings = check_concentration_risk(
                high_corr_pairs=result.correlation.high_corr_pairs,
                effective_bets=result.correlation.effective_bets,
                buy_tickers=buy_tickers,
                held_tickers=set(held_tickers),
            )
            if result.concentration_warnings:
                logger.warning(
                    "  CONCENTRATION: %d warnings",
                    len(result.concentration_warnings),
                )

    # ------------------------------------------------------------------
    # Step 6: Persist to database
    # ------------------------------------------------------------------
    if not dry_run:
        logger.info("[6/6] Persisting results...")
        _persist_results(db, result, tracker)
    else:
        logger.info("[6/6] Dry run — skipping persistence")

    tracker.data_sources["scoring"] = "ok"
    return result


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def _persist_results(
    db: Database,
    result: PipelineResult,
    tracker: RunTracker,
) -> None:
    """Persist scoring results to the database."""
    try:
        # Insert scoring run
        insert_scoring_run(
            db,
            run_id=tracker.run_id,
            vix_current=result.vix_current,
            vix_regime=result.vix_regime,
            spy_above_200d=int(result.spy_above_200d),
            spy_pct_from_200d=round(result.spy_pct_from_200d, 4),
            breadth_pct=round(result.breadth_pct, 4),
            market_regime_score=round(result.market_regime_score, 4),
            tickers_scored=tracker.tickers_scored,
            tickers_failed=tracker.tickers_failed,
            effective_bets=round(result.correlation.effective_bets, 2),
        )

        # Insert individual scores
        for symbol, score_data in result.scores.items():
            sub_scores = score_data.get("sub_scores", {})
            score_id = insert_score(
                db,
                run_id=tracker.run_id,
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

            # Insert signals from signal board
            for sig in score_data.get("signal_board", []):
                insert_signal(
                    db,
                    score_id=score_id,
                    signal_type=sig.get("signal_type", "UNKNOWN"),
                    severity=sig.get("severity", "INFO"),
                    criterion=sig.get("criterion", ""),
                    message=sig.get("message", ""),
                    metadata=sig.get("metadata"),
                )

        # Update data freshness
        update_data_freshness(db, "scoring", "ok", f"Run {tracker.run_id}")

        # Update scoring run with completion
        update_scoring_run(
            db,
            tracker.run_id,
            completed_at=datetime.now().isoformat(),
            status="completed",
        )

        logger.info(
            "  Persisted %d scores under run %s",
            len(result.scores),
            tracker.run_id,
        )

    except Exception as e:
        logger.error("  Persistence error: %s", e)
        tracker.errors.append(f"Persistence: {e}")
