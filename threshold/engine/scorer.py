"""score_ticker() — Central orchestrator for DCS scoring of one ticker.

Computes the Dip-Buying Composite Score (DCS) for a single ticker by:
  1. Computing 5 sub-scores (MQ, FQ, TO, MR, VC)
  2. Composing raw DCS from weighted sub-scores
  3. Applying post-composition modifiers (OBV boost, RSI divergence boost)
  4. Applying falling knife filter (defense-aware)
  5. Applying drawdown defense modifier (D-5 rule)
  6. Evaluating sell criteria and building SignalBoard
  7. Returning full ScoringResult dict
"""

from __future__ import annotations

from typing import Any, TypedDict

import pandas as pd

from threshold.engine.composite import (
    apply_drawdown_modifier,
    apply_falling_knife_filter,
    apply_obv_boost,
    apply_rsi_divergence_boost,
    classify_dcs,
    compose_dcs,
)
from threshold.engine.context import ScoringContext
from threshold.engine.signals import (
    SignalBoard,
    make_amplifier_warning,
    make_bottom_turning,
    make_defensive_hold,
    make_eps_rev_sell,
    make_eps_rev_warning,
    make_quant_drop_sell,
    make_quant_freshness_warning,
    make_reversal_confirmed,
    make_sma_breach_sell,
    make_sma_breach_warning,
)
from threshold.engine.subscores import (
    calc_fundamental_quality,
    calc_momentum_quality,
    calc_quant_deterioration,
    calc_revision_momentum,
    calc_technical_oversold,
    calc_valuation_context,
)
from threshold.engine.technical import (
    calc_consecutive_days_below_sma,
    calc_obv_divergence,
    calc_price_acceleration,
    calc_reversal_signals,
    calc_rsi_value,
)


# ---------------------------------------------------------------------------
# Type definitions
# ---------------------------------------------------------------------------

class Technicals(TypedDict, total=False):
    """Technicals dict returned inside ScoringResult."""
    rsi_14: float
    pct_from_200d: float
    ret_8w: float
    macd_crossover: str
    macd_histogram: float
    obv_divergence: str
    obv_divergence_strength: float
    rsi_bullish_divergence: bool
    bb_lower_breach: bool
    bb_pct_b: float
    bottom_turning: bool
    quant_freshness_warning: bool
    reversal_confirmed: bool
    vol_adj_mom: float
    rs_vs_spy: float


class ScoringResult(TypedDict, total=False):
    """Return type for score_ticker() — None when insufficient data."""
    dcs: float
    dcs_signal: str
    sub_scores: dict[str, Any]
    is_etf: bool
    technicals: Technicals
    trend_score: float
    days_below_sma_3pct: int
    sell_flags: list[str]
    signal_board: list[dict[str, Any]]
    quant_deterioration: dict[str, Any]
    revision_momentum: dict[str, Any]
    reversal_confirmed: bool
    bottom_turning: bool
    rsi_bullish_divergence: bool
    quant_freshness_warning: bool
    yf_fundamentals: dict[str, Any]
    drawdown_defense: dict[str, Any]
    falling_knife_cap: dict[str, Any]


# ---------------------------------------------------------------------------
# Main scoring function
# ---------------------------------------------------------------------------

def score_ticker(
    ticker: str,
    sa_data: dict[str, Any],
    price_df: pd.DataFrame,
    ctx: ScoringContext,
    config: Any | None = None,
) -> ScoringResult | None:
    """Calculate DCS for a single ticker.

    Parameters:
        ticker: Ticker symbol (e.g. "AAPL").
        sa_data: SA ratings dict with keys: quantScore, momentum,
                 profitability, revisions, growth, valuation.
        price_df: DataFrame with 'Close' and 'Volume' columns.
        ctx: ScoringContext with market regime, SPY data, history, etc.
        config: ThresholdConfig (optional). When None, uses calibrated defaults.

    Returns ScoringResult dict, or None if insufficient data (<50 bars).
    """
    close = price_df["Close"].dropna()
    volume = price_df["Volume"].dropna() if "Volume" in price_df.columns else pd.Series(dtype=float)

    if len(close) < 50:
        return None  # Insufficient data

    # --- OBV divergence ---
    obv_data = calc_obv_divergence(close, volume)

    # --- Sell criterion: consecutive days below 200d SMA ---
    days_below_sma, _ = calc_consecutive_days_below_sma(close)

    # --- Quant deterioration (from DB or prev_scores) ---
    current_quant = sa_data.get("quantScore")
    quant_dropped = False
    quant_delta = 0.0
    quant_compare_date: str | None = None

    # Try DB-based detection first, then fall back to prev_scores
    prev_sa = ctx.get_prev_sa_data(ticker)
    if prev_sa is not None:
        prev_quant = prev_sa.get("quantScore")
        if prev_quant is not None and current_quant is not None:
            quant_delta = round(current_quant - prev_quant, 2)
            quant_dropped = quant_delta < -1.0
            quant_compare_date = prev_sa.get("_date")

    # --- Revision Momentum ---
    rev_mom_score, rev_direction, rev_delta_4w = calc_revision_momentum(
        ticker, ctx.grade_history, config,
    )

    # --- Retrieve yfinance fundamentals for this ticker ---
    yf_fundamentals = ctx.get_yf_fundamentals(ticker)

    # --- Sub-scores ---
    mq, trend_score, vol_adj_mom, rs_vs_spy = calc_momentum_quality(
        sa_data, close, ctx.spy_close, config,
    )
    fq = calc_fundamental_quality(
        sa_data, rev_mom_score, yf_fundamentals=yf_fundamentals, config=config,
    )
    to, macd_data = calc_technical_oversold(close, config)
    mr = ctx.market_regime_score
    vc = calc_valuation_context(sa_data, yf_fundamentals=yf_fundamentals, config=config)

    # --- Advanced signal overlays (Phase 2C, all disabled by default) ---
    advanced_signals: dict[str, Any] = {}
    if config is not None and hasattr(config, "advanced"):
        adv = config.advanced

        # Trend Following: blend into MQ sub-score
        if hasattr(adv, "trend_following") and adv.trend_following.enabled:
            from threshold.engine.advanced.trend_following import ContinuousTrendFollower
            tf = ContinuousTrendFollower(
                window=adv.trend_following.window,
                vol_window=adv.trend_following.vol_window,
            )
            trend_sig = tf.compute_signal(close)
            if trend_sig is not None:
                blend_w = adv.trend_following.mq_blend_weight
                trend_norm = (trend_sig["signal"] + 1) / 2  # [-1,1] → [0,1]
                mq = (1 - blend_w) * mq + blend_w * trend_norm
                advanced_signals["trend_following"] = dict(trend_sig)

        # Sentiment: reduce MR when overheated
        if hasattr(adv, "sentiment") and adv.sentiment.enabled:
            try:
                from threshold.engine.advanced.sentiment import AlignedSentimentIndex
                asi = AlignedSentimentIndex(
                    n_components=adv.sentiment.n_components,
                    mr_reduction=adv.sentiment.mr_reduction,
                    overheated_pctl=adv.sentiment.overheated_pctl,
                    depressed_pctl=adv.sentiment.depressed_pctl,
                    min_observations=adv.sentiment.min_observations,
                )
                # Sentiment requires proxy data from context
                proxy_data = getattr(ctx, "sentiment_proxies", None)
                market_rets = getattr(ctx, "market_returns", None)
                if proxy_data is not None:
                    sent_result = asi.compute(proxy_data, market_rets)
                    if sent_result["mr_adjustment"] > 0:
                        mr = mr * (1 - sent_result["mr_adjustment"])
                    advanced_signals["sentiment"] = dict(sent_result)
            except ImportError:
                pass  # scikit-learn not installed, skip

    # --- Compose raw DCS ---
    sub_score_dict = {"MQ": mq, "FQ": fq, "TO": to, "MR": mr, "VC": vc}

    # Read weights from config
    weights: dict[str, int] | None = None
    if config is not None:
        sc = getattr(config, "scoring", config)
        if hasattr(sc, "weights"):
            w = sc.weights
            weights = {
                "MQ": getattr(w, "MQ", 30),
                "FQ": getattr(w, "FQ", 25),
                "TO": getattr(w, "TO", 20),
                "MR": getattr(w, "MR", 15),
                "VC": getattr(w, "VC", 10),
            }

    dcs_raw = compose_dcs(sub_score_dict, weights)

    # --- Post-composition modifiers ---
    # Read modifier settings from config
    obv_max = 5
    rsi_div_boost = 3
    rsi_div_min_dcs = 60
    if config is not None:
        sc = getattr(config, "scoring", config)
        if hasattr(sc, "modifiers"):
            mod = sc.modifiers
            obv_max = getattr(mod, "obv_bullish_max", obv_max)
            rsi_div_boost = getattr(mod, "rsi_divergence_boost", rsi_div_boost)
            rsi_div_min_dcs = getattr(mod, "rsi_divergence_min_dcs", rsi_div_min_dcs)

    dcs_raw = apply_obv_boost(dcs_raw, obv_data, obv_max)

    # --- Technical data (ret_8w needed by Gate 3) ---
    _, ret_8w = calc_price_acceleration(close)

    # Technicals for display
    rsi = calc_rsi_value(close, 14)
    sma_200 = (
        close.rolling(200).mean().iloc[-1] if len(close) >= 200 else close.mean()
    )
    pct_from_200d = (close.iloc[-1] - sma_200) / sma_200

    # --- Reversal Signal Detection (Phase 2 backtest-validated) ---
    sa_quant = sa_data.get("quantScore")
    reversal = calc_reversal_signals(close, rsi, macd_data, sa_quant)

    # RSI Bullish Divergence boosts DCS
    dcs_raw = apply_rsi_divergence_boost(
        dcs_raw, reversal["rsi_bullish_divergence"], rsi_div_boost, rsi_div_min_dcs,
    )

    # --- Drawdown Defense classification lookup ---
    dd_classification: str | None = None
    dd_downside_capture: float | None = None
    if ctx.drawdown_classifications:
        dd_info = ctx.drawdown_classifications.get(
            ticker
        ) or ctx.drawdown_classifications.get(ticker.replace("-", "."))
        if dd_info:
            dd_classification = dd_info.get("classification")
            dd_downside_capture = dd_info.get("downside_capture")

    # --- Falling knife filter (defense-aware) ---
    dcs, fk_cap_applied = apply_falling_knife_filter(
        dcs_raw, trend_score, dd_classification, config,
    )

    # --- Drawdown Defense DCS Modifier (Rule D-5) ---
    dcs, dd_modifier = apply_drawdown_modifier(
        dcs, dd_classification, ctx.vix_regime, config,
    )

    # Derive reversal tags
    reversal_confirmed = dcs >= 65 and reversal["bb_lower_breach"]

    technicals: dict[str, Any] = {
        "rsi_14": round(rsi, 1),
        "pct_from_200d": round(pct_from_200d, 4),
        "ret_8w": round(ret_8w, 4),
        "macd_crossover": macd_data["crossover"],
        "macd_histogram": macd_data["histogram"],
        "obv_divergence": obv_data["divergence"],
        "obv_divergence_strength": obv_data["divergence_strength"],
        "rsi_bullish_divergence": reversal["rsi_bullish_divergence"],
        "bb_lower_breach": reversal["bb_lower_breach"],
        "bb_pct_b": reversal["bb_pct_b"],
        "bottom_turning": reversal["bottom_turning"],
        "quant_freshness_warning": reversal["quant_freshness_warning"],
        "reversal_confirmed": reversal_confirmed,
    }
    if vol_adj_mom is not None:
        technicals["vol_adj_mom"] = round(vol_adj_mom, 3)
    if rs_vs_spy is not None:
        technicals["rs_vs_spy"] = round(rs_vs_spy, 3)

    # --- Sell criterion flags (SignalBoard taxonomy) ---
    board = SignalBoard()

    # Read sell criteria thresholds from config
    sma_sell_days = 10
    sma_warn_days = 7
    eps_sell_subgrades = 3.0
    eps_warn_subgrades = 2.0
    if config is not None:
        sc = getattr(config, "scoring", config)
        if hasattr(sc, "revision_momentum"):
            rm = sc.revision_momentum
            eps_sell_subgrades = float(
                getattr(rm, "sell_threshold_subgrades", eps_sell_subgrades)
            )
            eps_warn_subgrades = float(
                getattr(rm, "warning_threshold_subgrades", eps_warn_subgrades)
            )
        sell_c = getattr(config, "sell_criteria", None)
        if sell_c is not None:
            sma_sell_days = getattr(sell_c, "sma_breach_days", sma_sell_days)
            sma_warn_days = getattr(sell_c, "sma_breach_warning_days", sma_warn_days)

    # Sell #1: SMA breach
    if days_below_sma >= sma_sell_days:
        board.add(make_sma_breach_sell(days_below_sma))
    elif days_below_sma >= sma_warn_days:
        board.add(make_sma_breach_warning(days_below_sma))

    # Sell #2: Quant drop
    if quant_dropped and quant_compare_date is not None:
        board.add(make_quant_drop_sell(quant_delta, quant_compare_date))

    # Sell #3: EPS Revision Momentum
    if rev_delta_4w is not None:
        sub_grade_steps = abs(rev_delta_4w) / (1.0 / 13.0)
        if rev_delta_4w <= -eps_sell_subgrades / 13.0:
            board.add(make_eps_rev_sell(sub_grade_steps, rev_delta_4w))
        elif rev_delta_4w <= -eps_warn_subgrades / 13.0:
            board.add(make_eps_rev_warning(sub_grade_steps, rev_delta_4w))

    # Quant Freshness Warning
    if reversal["quant_freshness_warning"]:
        board.add(make_quant_freshness_warning())

    # Drawdown Defense sell/hold flags (Rules D-7, D-8)
    if dd_classification and ctx.vix_regime in ("FEAR", "PANIC"):
        sell_count = len(board.sells)
        if dd_classification in ("HEDGE", "DEFENSIVE") and sell_count == 1:
            if dd_downside_capture is not None:
                board.add(make_defensive_hold(dd_classification, dd_downside_capture))
        elif dd_classification == "AMPLIFIER" and sell_count >= 1:
            if dd_downside_capture is not None:
                board.add(make_amplifier_warning(dd_downside_capture))

    # Reversal buy signals
    if reversal_confirmed:
        board.add(make_reversal_confirmed())
    if reversal["bottom_turning"]:
        board.add(make_bottom_turning())

    # Derive legacy sell_flags from SignalBoard
    sell_flags = board.to_legacy_flags()

    # --- Classify DCS ---
    thresholds = None
    if config is not None:
        sc = getattr(config, "scoring", config)
        if hasattr(sc, "thresholds"):
            t = sc.thresholds
            thresholds = {
                "strong_buy_dip": getattr(t, "strong_buy_dip", 80),
                "high_conviction": getattr(t, "high_conviction", 70),
                "buy_dip": getattr(t, "buy_dip", 65),
                "watch": getattr(t, "watch", 50),
                "weak": getattr(t, "weak", 35),
            }

    dcs_signal = classify_dcs(dcs, thresholds)

    # --- Build result ---
    result: dict[str, Any] = {
        "dcs": round(dcs, 1),
        "dcs_signal": dcs_signal,
        "sub_scores": {
            "dcs": {
                "MQ": round(mq, 3),
                "FQ": round(fq, 3),
                "TO": round(to, 3),
                "MR": round(mr, 3),
                "VC": round(vc, 3),
            },
        },
        "technicals": technicals,
        "trend_score": trend_score,
        "days_below_sma_3pct": days_below_sma,
        "sell_flags": sell_flags,
        "signal_board": board.to_dict(),
        "_signal_board_obj": board,
    }

    if dd_classification:
        result["drawdown_defense"] = {
            "classification": dd_classification,
            "downside_capture": dd_downside_capture,
            "dd_modifier_applied": dd_modifier,
        }

    if fk_cap_applied is not None:
        result["falling_knife_cap"] = {
            "classification": dd_classification,
            "cap_applied": fk_cap_applied,
            "original_dcs": round(dcs_raw, 1),
        }

    if quant_dropped:
        result["quant_deterioration"] = {
            "delta": quant_delta,
            "since": quant_compare_date,
        }

    if rev_mom_score is not None:
        result["revision_momentum"] = {
            "score": round(rev_mom_score, 3),
            "direction": rev_direction,
            "delta_4w": rev_delta_4w,
        }

    # Reversal signal metadata
    if reversal_confirmed:
        result["reversal_confirmed"] = True
    if reversal["bottom_turning"]:
        result["bottom_turning"] = True
    if reversal["rsi_bullish_divergence"]:
        result["rsi_bullish_divergence"] = True
    if reversal["quant_freshness_warning"]:
        result["quant_freshness_warning"] = True

    # yfinance fundamentals metadata
    if yf_fundamentals and yf_fundamentals.get("fetch_status") == "ok":
        yf_meta: dict[str, Any] = {}
        for key in (
            "fcf_yield", "gross_profitability", "ev_to_ebitda",
            "gross_margin", "sector", "fcf_yield_pctl",
            "gross_profitability_pctl", "ev_to_ebitda_pctl",
        ):
            val = yf_fundamentals.get(key)
            if val is not None:
                yf_meta[key] = round(val, 4) if isinstance(val, float) else val
        if yf_meta:
            result["yf_fundamentals"] = yf_meta

    # Advanced signal metadata (Phase 2C)
    if advanced_signals:
        result["advanced_signals"] = advanced_signals

    return result  # type: ignore[return-value]
