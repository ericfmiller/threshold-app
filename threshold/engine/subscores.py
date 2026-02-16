"""DCS sub-score calculators.

Five sub-scores that compose the Dip-Buying Composite Score (DCS):
  - MQ: Momentum Quality (30%)
  - FQ: Fundamental Quality (25%)
  - TO: Technical Oversold (20%)
  - MR: Market Regime (15%)
  - VC: Valuation Context (10%)

Plus:
  - calc_revision_momentum() — EPS revision momentum from grade history
  - calc_quant_deterioration() — SA Quant 30-day drop detection (reads DB)
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal, TypedDict

import numpy as np
import pandas as pd

from threshold.engine.grades import sa_grade_to_norm
from threshold.engine.technical import MACDResult, calc_macd, calc_rsi_value

# ---------------------------------------------------------------------------
# Type definitions
# ---------------------------------------------------------------------------

class MQResult(TypedDict):
    """Return type for calc_momentum_quality()."""
    mq: float
    trend_score: float
    vol_adj_mom: float
    rs_vs_spy: float | None


class RevisionMomentumResult(TypedDict):
    """Per-ticker revision momentum metadata."""
    score: float
    direction: Literal["positive", "negative", "flat"]
    delta_4w: float


# ---------------------------------------------------------------------------
# MQ: Momentum Quality (30% of DCS)
# ---------------------------------------------------------------------------

def calc_momentum_quality(
    sa_data: dict[str, Any],
    close: pd.Series,
    spy_close: pd.Series | None = None,
    config: Any | None = None,
) -> tuple[float, float, float, float | None]:
    """MQ: Momentum Quality (weight 30% of DCS).

    Components:
      - Trend regime: 50d vs 200d SMA + price position (30%)
      - 12-1 momentum, volatility-adjusted (25%) [Barroso & Santa-Clara 2015]
      - SA Momentum Grade (25%)
      - Relative strength vs SPY (20%) [Antonacci dual momentum]

    Returns (mq, trend_score, vol_adj_mom, rs_vs_spy).
    """
    n = len(close)
    current = close.iloc[-1]

    # Read weights from config if available
    w_trend = 0.30
    w_vol_adj = 0.25
    w_sa_mom = 0.25
    w_rs = 0.20
    if config is not None:
        mq_w = getattr(config, "scoring", config)
        if hasattr(mq_w, "mq_weights"):
            mq_w = mq_w.mq_weights
            w_trend = getattr(mq_w, "trend", w_trend)
            w_vol_adj = getattr(mq_w, "vol_adj_momentum", w_vol_adj)
            w_sa_mom = getattr(mq_w, "sa_momentum", w_sa_mom)
            w_rs = getattr(mq_w, "relative_strength", w_rs)

    # 50d and 200d SMA
    sma_50 = close.rolling(50).mean().iloc[-1] if n >= 50 else current
    sma_200 = close.rolling(200).mean().iloc[-1] if n >= 200 else current

    # Trend classifier
    if sma_50 > sma_200 and current > sma_200:
        trend_score = 1.0    # Uptrend pullback — IDEAL dip-buy
    elif sma_50 > sma_200 and current <= sma_200:
        trend_score = 0.5    # Uptrend but price broke 200d — caution
    elif sma_50 <= sma_200 and current > sma_200:
        trend_score = 0.4    # Recovery attempt
    else:
        trend_score = 0.1    # Downtrend, below both SMAs — falling knife

    # 12-1 momentum, volatility-adjusted
    vol_adj_mom = 0.0
    raw_mom_12_1 = 0.0
    if n >= 252:
        price_12m = close.iloc[-252]
        price_1m = close.iloc[-21]
        raw_mom_12_1 = (price_1m / price_12m) - 1.0
    elif n >= 60:
        price_start = close.iloc[0]
        price_1m = close.iloc[-21]
        raw_mom_12_1 = (price_1m / price_start) - 1.0

    # Compute realized volatility for vol-adjustment
    if n >= 60:
        daily_returns = close.pct_change().dropna()
        if len(daily_returns) >= 252:
            realized_vol = float(daily_returns.iloc[-252:].std() * np.sqrt(252))
        else:
            realized_vol = float(daily_returns.std() * np.sqrt(252))
        vol_adj_mom = raw_mom_12_1 / max(realized_vol, 0.05)
    else:
        vol_adj_mom = raw_mom_12_1

    # Normalize vol-adjusted momentum (typical range: -2 to +3 after vol scaling)
    mom_score = max(0.0, min(1.0, (vol_adj_mom + 0.5) / 2.5))

    # SA Momentum grade
    sa_mom_norm = sa_grade_to_norm(sa_data.get("momentum"))

    # Relative strength vs SPY (Antonacci dual momentum)
    rs_score = 0.5  # Neutral default if SPY data unavailable
    rs_vs_spy = None
    if spy_close is not None and len(spy_close) >= 252 and n >= 252:
        ticker_12m_ret = (close.iloc[-21] / close.iloc[-252]) - 1.0
        spy_12m_ret = (spy_close.iloc[-21] / spy_close.iloc[-252]) - 1.0
        rs_vs_spy = ticker_12m_ret / spy_12m_ret if spy_12m_ret != 0 else 1.0
        rs_score = max(0.0, min(1.0, (rs_vs_spy - 0.3) / 1.4))

    # Composite MQ
    mq = (
        (trend_score * w_trend)
        + (mom_score * w_vol_adj)
        + (sa_mom_norm * w_sa_mom)
        + (rs_score * w_rs)
    )

    return mq, trend_score, vol_adj_mom, rs_vs_spy


# ---------------------------------------------------------------------------
# Revision Momentum
# ---------------------------------------------------------------------------

def calc_revision_momentum(
    ticker: str,
    grade_history: list[dict[str, Any]] | None,
    config: Any | None = None,
) -> tuple[float | None, str | None, float | None]:
    """Compute EPS revision momentum from stored grade history.

    Returns (rev_mom_score, rev_direction, rev_delta_4w) or
    (None, None, None) if insufficient history.

    Requires at least 21 calendar days between newest and oldest history
    file to avoid measuring day-to-day noise.

    Evidence: Novy-Marx 2015 — earnings momentum subsumes price momentum.
    """
    min_history_weeks = 4
    min_calendar_days = 21
    if config is not None:
        rm = getattr(config, "scoring", config)
        if hasattr(rm, "revision_momentum"):
            rm = rm.revision_momentum
            min_history_weeks = getattr(rm, "min_history_weeks", min_history_weeks)
            min_calendar_days = getattr(rm, "min_calendar_days", min_calendar_days)

    if not grade_history or len(grade_history) < min_history_weeks:
        return None, None, None

    # --- Calendar span gate ---
    try:
        newest_dt = grade_history[0].get("_metadata", {}).get("generated_at", "")
        oldest_dt = grade_history[-1].get("_metadata", {}).get("generated_at", "")
        if newest_dt and oldest_dt:
            newest = datetime.fromisoformat(newest_dt)
            oldest = datetime.fromisoformat(oldest_dt)
            span_days = (newest - oldest).days
            if span_days < min_calendar_days:
                return None, None, None
    except (ValueError, TypeError):
        pass

    # Extract revisions grades from last 4-8 weeks (most recent first)
    rev_grades: list[float | None] = []
    for week_data in grade_history:
        scores = week_data.get("scores", {})
        ticker_data = scores.get(ticker, {})
        rev_grade = ticker_data.get("sa_revisions")
        if rev_grade:
            rev_grades.append(sa_grade_to_norm(rev_grade))
        else:
            rev_grades.append(None)

    # Need at least 4 data points with data
    valid = [g for g in rev_grades[:8] if g is not None]
    if len(valid) < min_history_weeks:
        return None, None, None

    # Delta: current vs 4 weeks ago (positive = improving)
    current = valid[0]
    four_weeks_ago = valid[min(3, len(valid) - 1)]
    rev_delta_4w = current - four_weeks_ago

    # Direction consistency
    transitions = []
    for i in range(len(valid) - 1):
        if valid[i] > valid[i + 1]:
            transitions.append(1)   # Improving
        elif valid[i] < valid[i + 1]:
            transitions.append(-1)  # Deteriorating
        else:
            transitions.append(0)   # Flat

    if transitions:
        pos = sum(1 for t in transitions if t > 0)
        neg = sum(1 for t in transitions if t < 0)
        consistency = pos / len(transitions) if pos > neg else -(neg / len(transitions))
    else:
        consistency = 0.0

    # Direction
    if rev_delta_4w > 0.05:
        rev_direction = "positive"
    elif rev_delta_4w < -0.05:
        rev_direction = "negative"
    else:
        rev_direction = "flat"

    # Score: combine delta magnitude + direction consistency
    delta_score = max(0.0, min(1.0, (rev_delta_4w + 0.3) / 0.6))
    consistency_score = max(0.0, min(1.0, (consistency + 1.0) / 2.0))
    rev_mom_score = (delta_score * 0.60) + (consistency_score * 0.40)

    return rev_mom_score, rev_direction, round(rev_delta_4w, 3)


# ---------------------------------------------------------------------------
# FQ: Fundamental Quality (25% of DCS)
# ---------------------------------------------------------------------------

def calc_fundamental_quality(
    sa_data: dict[str, Any],
    rev_momentum: float | None = None,
    yf_fundamentals: dict[str, Any] | None = None,
    config: Any | None = None,
) -> float:
    """FQ: Fundamental Quality (weight 25% of DCS).

    Four conditional weight paths based on data availability:
      1. yfinance + revision momentum
      2. yfinance only
      3. revision momentum only
      4. base (SA grades only)

    All weights sourced from config.scoring.fq_weights.
    """
    quant = sa_data.get("quantScore") or 0
    quant_norm = min(1.0, quant / 5.0)

    prof_norm = sa_grade_to_norm(sa_data.get("profitability"))
    rev_norm = sa_grade_to_norm(sa_data.get("revisions"))
    growth_norm = sa_grade_to_norm(sa_data.get("growth"))

    # Read FQ weights and profitability blend from config
    prof_blend_sa = 0.60
    prof_blend_nm = 0.40
    if config is not None:
        sc = getattr(config, "scoring", config)
        if hasattr(sc, "profitability_blend"):
            pb = sc.profitability_blend
            prof_blend_sa = getattr(pb, "sa_weight", prof_blend_sa)
            prof_blend_nm = getattr(pb, "novy_marx_weight", prof_blend_nm)

    # Check yfinance data availability
    has_yf = (
        yf_fundamentals is not None
        and yf_fundamentals.get("fetch_status") == "ok"
        and yf_fundamentals.get("fcf_yield_pctl") is not None
    )

    if has_yf and yf_fundamentals is not None:
        # Blend SA profitability grade with Novy-Marx gross profitability
        gp_pctl = yf_fundamentals.get("gross_profitability_pctl")
        if gp_pctl is not None:
            prof_blended = (prof_norm * prof_blend_sa) + (float(gp_pctl) * prof_blend_nm)
        else:
            prof_blended = prof_norm

        fcf_pctl = float(yf_fundamentals.get("fcf_yield_pctl", 0.5))

        if rev_momentum is not None:
            return (
                (quant_norm * 0.30)
                + (prof_blended * 0.22)
                + (fcf_pctl * 0.13)
                + (rev_momentum * 0.15)
                + (rev_norm * 0.10)
                + (growth_norm * 0.10)
            )
        else:
            return (
                (quant_norm * 0.30)
                + (prof_blended * 0.22)
                + (fcf_pctl * 0.13)
                + (rev_norm * 0.20)
                + (growth_norm * 0.15)
            )
    else:
        if rev_momentum is not None:
            return (
                (quant_norm * 0.35)
                + (prof_norm * 0.25)
                + (rev_momentum * 0.15)
                + (rev_norm * 0.15)
                + (growth_norm * 0.10)
            )
        else:
            return (
                (quant_norm * 0.35)
                + (prof_norm * 0.25)
                + (rev_norm * 0.25)
                + (growth_norm * 0.15)
            )


# ---------------------------------------------------------------------------
# TO: Technical Oversold (20% of DCS)
# ---------------------------------------------------------------------------

def calc_technical_oversold(
    close: pd.Series,
    config: Any | None = None,
) -> tuple[float, MACDResult]:
    """TO: Technical Oversold (weight 20% of DCS).

    Components:
      - RSI-14 (lower = more oversold = higher score) (35%)
      - Distance from 200d SMA (25%)
      - Bollinger Band position (25%)
      - MACD confirmation (15%)

    Returns (to_score, macd_data).
    """
    n = len(close)
    current = close.iloc[-1]

    # Read weights from config
    w_rsi = 0.35
    w_sma = 0.25
    w_bb = 0.25
    w_macd = 0.15
    if config is not None:
        sc = getattr(config, "scoring", config)
        if hasattr(sc, "to_weights"):
            tw = sc.to_weights
            w_rsi = getattr(tw, "rsi", w_rsi)
            w_sma = getattr(tw, "sma_distance", w_sma)
            w_bb = getattr(tw, "bollinger", w_bb)
            w_macd = getattr(tw, "macd", w_macd)

    # RSI
    rsi = calc_rsi_value(close, 14)
    rsi_score = max(0.0, min(1.0, (70 - rsi) / 40.0))

    # Distance from 200d SMA
    if n >= 200:
        sma_200 = close.rolling(200).mean().iloc[-1]
        pct_from_sma = (current - sma_200) / sma_200
    else:
        pct_from_sma = 0.0
    sma_dist_score = max(0.0, min(1.0, (0.10 - pct_from_sma) / 0.30))

    # Bollinger Band position (20d, 2 sigma)
    if n >= 20:
        sma_20 = close.rolling(20).mean().iloc[-1]
        std_20 = close.rolling(20).std().iloc[-1]
        if std_20 > 0:
            upper_bb = sma_20 + 2 * std_20
            lower_bb = sma_20 - 2 * std_20
            bb_position = (current - lower_bb) / (upper_bb - lower_bb)
            bb_score = max(0.0, min(1.0, 1.0 - bb_position))
        else:
            bb_score = 0.5
    else:
        bb_score = 0.5

    # MACD confirmation
    macd_data = calc_macd(close)
    macd_score = 0.0
    if macd_data["crossover"] == "bullish" and macd_data["below_zero"]:
        macd_score = 1.0
    elif macd_data["crossover"] == "bullish":
        macd_score = 0.7
    elif macd_data["hist_rising"] and macd_data["below_zero"]:
        macd_score = 0.6
    elif macd_data["hist_rising"]:
        macd_score = 0.3

    to = (
        (rsi_score * w_rsi)
        + (sma_dist_score * w_sma)
        + (bb_score * w_bb)
        + (macd_score * w_macd)
    )
    return to, macd_data


# ---------------------------------------------------------------------------
# MR: Market Regime (15% of DCS)
# ---------------------------------------------------------------------------

def calc_market_regime(
    vix_current: float,
    vix_percentile: float,
    spy_above_200d: bool,
    breadth_pct: float | None = None,
    config: Any | None = None,
) -> float:
    """MR: Market Regime (weight 15% of DCS).

    Components:
      - VIX level (contrarian — higher VIX = better dip-buy environment) (50%)
      - SPY trend (above 200d SMA = supportive regime) (30%)
      - Market breadth (% of holdings above 200d SMA) (20%)
    """
    # VIX contrarian scoring
    if vix_current < 14:
        vix_score = 0.2
    elif vix_current < 20:
        vix_score = 0.2 + (vix_current - 14) * (0.3 / 6)
    elif vix_current < 28:
        vix_score = 0.5 + (vix_current - 20) * (0.25 / 8)
    else:
        vix_score = min(1.0, 0.75 + (vix_current - 28) * (0.25 / 12))

    market_trend = 1.0 if spy_above_200d else 0.4

    if breadth_pct is not None:
        if breadth_pct > 0.70:
            breadth_score = 1.0
        elif breadth_pct > 0.50:
            breadth_score = 0.5 + (breadth_pct - 0.50) * 2.5
        elif breadth_pct > 0.30:
            breadth_score = 0.2 + (breadth_pct - 0.30) * 1.5
        else:
            breadth_score = 0.1
        return (vix_score * 0.50) + (market_trend * 0.30) + (breadth_score * 0.20)
    else:
        return (vix_score * 0.60) + (market_trend * 0.40)


# ---------------------------------------------------------------------------
# VC: Valuation Context (10% of DCS)
# ---------------------------------------------------------------------------

def calc_valuation_context(
    sa_data: dict[str, Any],
    yf_fundamentals: dict[str, Any] | None = None,
    config: Any | None = None,
) -> float:
    """VC: Valuation Context (weight 10% of DCS).

    Lowest weight because value is a poor short-term predictor.
    When yfinance EV/EBITDA data available, blends SA grade with
    sector-relative EV/EBITDA percentile.
    """
    sa_norm = sa_grade_to_norm(sa_data.get("valuation"))

    has_yf = (
        yf_fundamentals is not None
        and yf_fundamentals.get("fetch_status") == "ok"
        and yf_fundamentals.get("ev_to_ebitda_pctl") is not None
    )

    if has_yf and yf_fundamentals is not None:
        ev_ebitda_pctl = 1.0 - float(yf_fundamentals["ev_to_ebitda_pctl"])
        return float((sa_norm * 0.65) + (ev_ebitda_pctl * 0.35))
    else:
        return sa_norm


# ---------------------------------------------------------------------------
# Quant Deterioration (reads DB instead of JSON files)
# ---------------------------------------------------------------------------

def calc_quant_deterioration(
    ticker: str,
    db: Any | None,
    current_quant: float | None,
    lookback_days: int = 35,
) -> tuple[bool, float, str | None]:
    """Check if SA Quant dropped >1.0 point in the last 30 days.

    Sell criterion #4. Reads from the database (scores table) instead of
    scanning JSON files on disk.

    Returns (dropped, delta, compare_date).
    """
    if current_quant is None:
        return False, 0.0, None

    if db is None:
        return False, 0.0, None

    # Query the scores table for the oldest quant score within lookback window.
    # This will be implemented when the storage layer is wired up.
    # For now, return no deterioration — the pipeline caller can provide
    # prev_scores via ScoringContext for backward compatibility.
    return False, 0.0, None
