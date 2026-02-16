"""DCS composition, post-composition modifiers, and classifiers.

Functions:
  compose_dcs               — Raw DCS = weighted sum of 5 sub-scores
  apply_obv_boost           — +up to 5 pts for OBV bullish divergence
  apply_rsi_divergence_boost — +3 pts for RSI divergence when DCS >= 60
  apply_falling_knife_filter — Cap DCS in downtrends (defense-aware)
  apply_drawdown_modifier    — D-5 rule: adjust DCS by defense class in FEAR/PANIC
  classify_dcs              — DCS -> signal classification string
  classify_vix              — VIX -> regime classification string
"""

from __future__ import annotations

from typing import Any

# ---------------------------------------------------------------------------
# DCS Composition
# ---------------------------------------------------------------------------

def compose_dcs(
    sub_scores: dict[str, float],
    weights: dict[str, int] | None = None,
) -> float:
    """Compute raw DCS from weighted sub-scores.

    sub_scores: {"MQ": 0.72, "FQ": 0.65, "TO": 0.80, "MR": 0.55, "VC": 0.40}
    weights: {"MQ": 30, "FQ": 25, "TO": 20, "MR": 15, "VC": 10} (must sum to 100)

    Returns DCS in [0, 100].
    """
    if weights is None:
        weights = {"MQ": 30, "FQ": 25, "TO": 20, "MR": 15, "VC": 10}

    return sum(sub_scores.get(k, 0.0) * weights.get(k, 0) for k in weights)


# ---------------------------------------------------------------------------
# Post-Composition Modifiers
# ---------------------------------------------------------------------------

def apply_obv_boost(
    dcs: float,
    obv_result: dict[str, Any],
    max_boost: int = 5,
) -> float:
    """Boost DCS by up to max_boost points for OBV bullish divergence.

    Granville OBV divergence: volume precedes price by 2-6 weeks.
    """
    if obv_result.get("divergence") == "bullish":
        strength = obv_result.get("divergence_strength", 0.0)
        return min(100, dcs + max_boost * strength)
    return dcs


def apply_rsi_divergence_boost(
    dcs: float,
    has_divergence: bool,
    boost: int = 3,
    min_dcs: int = 60,
) -> float:
    """Boost DCS by `boost` points for RSI bullish divergence.

    Only applied when DCS >= min_dcs (default 60).
    Phase 2 validated: +2.2pp edge, walk-forward stable.
    """
    if has_divergence and dcs >= min_dcs:
        return min(100, dcs + boost)
    return dcs


def apply_falling_knife_filter(
    dcs_raw: float,
    trend_score: float,
    dd_classification: str | None = None,
    config: Any | None = None,
) -> tuple[float, int | None]:
    """Cap DCS if trend context is bearish — defense-aware.

    Hedges/defensives get softer caps (counter-cyclical value).
    Amplifiers/cyclicals get harsher caps (magnify losses).

    Returns (capped_dcs, cap_applied) — cap_applied is None if no cap.
    """
    # Default caps
    freefall_caps = {
        "HEDGE": 50, "DEFENSIVE": 45, "MODERATE": 30,
        "CYCLICAL": 20, "AMPLIFIER": 15,
    }
    downtrend_caps = {
        "HEDGE": 70, "DEFENSIVE": 60, "MODERATE": 50,
        "CYCLICAL": 40, "AMPLIFIER": 30,
    }

    # Override from config if available
    if config is not None:
        sc = getattr(config, "scoring", config)
        if hasattr(sc, "falling_knife"):
            fk = sc.falling_knife
            if hasattr(fk, "freefall"):
                freefall_caps = dict(fk.freefall)
            if hasattr(fk, "downtrend"):
                downtrend_caps = dict(fk.downtrend)

    if trend_score <= 0.1:
        cap = freefall_caps.get(dd_classification, 30) if dd_classification else 30
        return min(dcs_raw, cap), cap
    if trend_score <= 0.4:
        cap = downtrend_caps.get(dd_classification, 50) if dd_classification else 50
        return min(dcs_raw, cap), cap
    return dcs_raw, None


def apply_drawdown_modifier(
    dcs: float,
    dd_classification: str | None,
    vix_regime: str | None,
    config: Any | None = None,
) -> tuple[float, int]:
    """Apply D-5 drawdown defense modifier in FEAR/PANIC regimes.

    HEDGE +5, DEFENSIVE +3, MODERATE 0, CYCLICAL -3, AMPLIFIER -5.
    Only active when VIX regime is FEAR or PANIC.

    Returns (modified_dcs, modifier_applied).
    """
    if vix_regime not in ("FEAR", "PANIC") or dd_classification is None:
        return dcs, 0

    # Default modifiers
    modifiers = {
        "HEDGE": 5, "DEFENSIVE": 3, "MODERATE": 0,
        "CYCLICAL": -3, "AMPLIFIER": -5,
    }

    # Override from config
    if config is not None:
        sc = getattr(config, "scoring", config)
        if hasattr(sc, "drawdown_modifiers"):
            dm = sc.drawdown_modifiers
            modifiers = {
                "HEDGE": getattr(dm, "HEDGE", 5),
                "DEFENSIVE": getattr(dm, "DEFENSIVE", 3),
                "MODERATE": getattr(dm, "MODERATE", 0),
                "CYCLICAL": getattr(dm, "CYCLICAL", -3),
                "AMPLIFIER": getattr(dm, "AMPLIFIER", -5),
            }

    modifier = modifiers.get(dd_classification, 0)
    return max(0, min(100, dcs + modifier)), modifier


# ---------------------------------------------------------------------------
# Classifiers
# ---------------------------------------------------------------------------

def classify_dcs(dcs: float, thresholds: dict[str, int] | None = None) -> str:
    """Classify DCS into a signal string.

    Default thresholds (Phase 2 backtest-calibrated):
      >= 80: STRONG BUY DIP
      >= 70: HIGH CONVICTION
      >= 65: BUY DIP
      >= 50: WATCH
      >= 35: WEAK
      <  35: AVOID
    """
    if thresholds is None:
        thresholds = {
            "strong_buy_dip": 80,
            "high_conviction": 70,
            "buy_dip": 65,
            "watch": 50,
            "weak": 35,
        }

    if dcs >= thresholds.get("strong_buy_dip", 80):
        return "STRONG BUY DIP"
    if dcs >= thresholds.get("high_conviction", 70):
        return "HIGH CONVICTION"
    if dcs >= thresholds.get("buy_dip", 65):
        return "BUY DIP"
    if dcs >= thresholds.get("watch", 50):
        return "WATCH"
    if dcs >= thresholds.get("weak", 35):
        return "WEAK"
    return "AVOID"


def classify_vix(vix: float, boundaries: dict[str, list[int]] | None = None) -> str:
    """Classify VIX into a regime string.

    Default boundaries: COMPLACENT <14, NORMAL 14-20, FEAR 20-28, PANIC 28+.
    """
    if boundaries is None:
        boundaries = {
            "COMPLACENT": [0, 14],
            "NORMAL": [14, 20],
            "FEAR": [20, 28],
            "PANIC": [28, 999],
        }

    # Support both plain dicts and Pydantic model objects
    if hasattr(boundaries, "model_dump"):
        boundaries = boundaries.model_dump()

    for regime, (lo, hi) in boundaries.items():
        if lo <= vix < hi:
            return regime
    return "PANIC"  # Fallback for extreme values
