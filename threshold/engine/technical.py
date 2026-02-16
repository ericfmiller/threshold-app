"""Technical indicator calculations — pure math, zero I/O.

Functions:
  calc_rsi          — Standard RSI(14) series
  calc_rsi_value    — Most recent RSI scalar
  calc_macd         — MACD line, signal, histogram, crossover state
  calc_obv_divergence — On-Balance Volume divergence detection
  calc_rsi_bullish_divergence — RSI bullish divergence (Phase 2 validated)
  calc_bb_lower_breach — Bollinger Band lower breach
  calc_reversal_signals — Composite reversal signal dispatcher
  calc_price_acceleration — 8-week return + weekly acceleration
  calc_consecutive_days_below_sma — Sell criterion #3 helper
"""

from __future__ import annotations

from typing import Any, TypedDict

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Type definitions
# ---------------------------------------------------------------------------

class MACDResult(TypedDict):
    """Return type for calc_macd()."""
    macd: float
    signal: float
    histogram: float
    crossover: str          # 'bullish', 'bearish', or 'neutral'
    hist_rising: bool
    below_zero: bool


class OBVResult(TypedDict):
    """Return type for calc_obv_divergence()."""
    obv_trend: str          # 'rising', 'falling', 'flat'
    price_trend: str        # 'rising', 'falling', 'flat'
    divergence: str         # 'bullish', 'bearish', 'none'
    divergence_strength: float


class RSIDivergenceResult(TypedDict):
    """Return type for calc_rsi_bullish_divergence()."""
    detected: bool
    price_low_recent: float | None
    rsi_low_recent: float | None


class BBBreachResult(TypedDict):
    """Return type for calc_bb_lower_breach()."""
    breach: bool
    bb_pct_b: float
    lower_bb: float | None


class ReversalSignals(TypedDict):
    """Return type for calc_reversal_signals()."""
    rsi_bullish_divergence: bool
    bb_lower_breach: bool
    bb_pct_b: float
    bottom_turning: bool
    quant_freshness_warning: bool


# ---------------------------------------------------------------------------
# Core Indicators
# ---------------------------------------------------------------------------

def calc_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Standard RSI calculation using Wilder's EMA method."""
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta.where(delta < 0, 0.0))
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def calc_rsi_value(series: pd.Series, period: int = 14) -> float:
    """Return single most recent RSI value."""
    rsi_series = calc_rsi(series, period)
    val = rsi_series.iloc[-1]
    return float(val) if not np.isnan(val) else 50.0


def calc_macd(
    close: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> MACDResult:
    """Compute MACD line, signal line, and histogram.

    Returns dict with current values and signal state.
    """
    if len(close) < slow + signal:
        return {
            "macd": 0, "signal": 0, "histogram": 0,
            "crossover": "neutral", "hist_rising": False, "below_zero": False,
        }

    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line

    macd_now = float(macd_line.iloc[-1])
    signal_now = float(signal_line.iloc[-1])
    hist_now = float(histogram.iloc[-1])
    hist_prev = float(histogram.iloc[-2]) if len(histogram) >= 2 else 0.0

    # Crossover detection (last 3 bars)
    crossover = "neutral"
    if len(macd_line) >= 3:
        for i in range(-3, 0):
            prev_m = float(macd_line.iloc[i - 1])
            prev_s = float(signal_line.iloc[i - 1])
            curr_m = float(macd_line.iloc[i])
            curr_s = float(signal_line.iloc[i])
            if prev_m <= prev_s and curr_m > curr_s:
                crossover = "bullish"
            elif prev_m >= prev_s and curr_m < curr_s:
                crossover = "bearish"

    return {
        "macd": round(macd_now, 4),
        "signal": round(signal_now, 4),
        "histogram": round(hist_now, 4),
        "crossover": crossover,
        "hist_rising": hist_now > hist_prev,
        "below_zero": macd_now < 0,
    }


def calc_obv_divergence(
    close: pd.Series,
    volume: pd.Series,
    lookback: int = 20,
) -> OBVResult:
    """Compute On-Balance Volume and detect price/OBV divergences.

    Granville (1963): OBV divergences lead price by 2-6 weeks.

    Returns dict:
      - obv_trend: 'rising', 'falling', or 'flat' (20-day OBV regression slope)
      - price_trend: 'rising', 'falling', or 'flat' (20-day price regression slope)
      - divergence: 'bullish' (price falling, OBV rising),
                    'bearish' (price rising, OBV falling), or 'none'
      - divergence_strength: 0.0-1.0 (how strong the divergence is)
    """
    n = min(len(close), len(volume))
    if n < lookback + 5:
        return {
            "obv_trend": "flat", "price_trend": "flat",
            "divergence": "none", "divergence_strength": 0.0,
        }

    # Compute OBV
    obv = pd.Series(0.0, index=close.index[:n])
    for i in range(1, n):
        if close.iloc[i] > close.iloc[i - 1]:
            obv.iloc[i] = obv.iloc[i - 1] + volume.iloc[i]
        elif close.iloc[i] < close.iloc[i - 1]:
            obv.iloc[i] = obv.iloc[i - 1] - volume.iloc[i]
        else:
            obv.iloc[i] = obv.iloc[i - 1]

    # Linear regression slopes over lookback period (normalized)
    recent_close = close.iloc[-lookback:].values
    recent_obv = obv.iloc[-lookback:].values
    x = np.arange(lookback)

    # Price trend
    price_slope = np.polyfit(x, recent_close, 1)[0]
    price_norm = price_slope / (np.mean(recent_close) + 1e-10)

    # OBV trend
    obv_slope = np.polyfit(x, recent_obv, 1)[0]
    obv_norm = obv_slope / (np.abs(np.mean(recent_obv)) + 1e-10)

    def classify_trend(norm_slope: float) -> str:
        if norm_slope > 0.001:
            return "rising"
        elif norm_slope < -0.001:
            return "falling"
        return "flat"

    price_trend = classify_trend(price_norm)
    obv_trend = classify_trend(obv_norm)

    # Divergence detection
    divergence = "none"
    strength = 0.0
    if price_trend == "falling" and obv_trend == "rising":
        divergence = "bullish"
        strength = min(1.0, abs(obv_norm) * 100)
    elif price_trend == "rising" and obv_trend == "falling":
        divergence = "bearish"
        strength = min(1.0, abs(obv_norm) * 100)

    return {
        "obv_trend": obv_trend,
        "price_trend": price_trend,
        "divergence": divergence,
        "divergence_strength": round(strength, 3),
    }


# ---------------------------------------------------------------------------
# Reversal Signal Detection (Phase 2 backtest-validated)
# ---------------------------------------------------------------------------

def calc_rsi_bullish_divergence(
    close: pd.Series,
    rsi_period: int = 14,
    lookback: int = 40,
) -> RSIDivergenceResult:
    """Detect RSI bullish divergence: price makes a lower low over two windows
    while RSI makes a higher low.

    Walk-forward stable: Cal=58.6%, Val=57.1%, +2.2pp edge,
    11,259 signals in Phase 2 backtest.
    """
    n = len(close)
    if n < lookback:
        return {"detected": False, "price_low_recent": None, "rsi_low_recent": None}

    rsi_series = calc_rsi(close, rsi_period)
    rsi_clean = rsi_series.dropna()
    if len(rsi_clean) < lookback:
        return {"detected": False, "price_low_recent": None, "rsi_low_recent": None}

    # Compare two windows: days -40 to -20 vs days -20 to now
    half = lookback // 2
    price_w1 = close.iloc[-lookback:-half]
    price_w2 = close.iloc[-half:]
    rsi_w1 = rsi_series.iloc[-lookback:-half].dropna()
    rsi_w2 = rsi_series.iloc[-half:].dropna()

    if len(rsi_w1) == 0 or len(rsi_w2) == 0:
        return {"detected": False, "price_low_recent": None, "rsi_low_recent": None}

    price_low1 = float(price_w1.min())
    price_low2 = float(price_w2.min())
    rsi_low1 = float(rsi_w1.min())
    rsi_low2 = float(rsi_w2.min())

    # Price made lower low, RSI made higher low
    detected = price_low2 < price_low1 and rsi_low2 > rsi_low1

    return {
        "detected": detected,
        "price_low_recent": round(price_low2, 2),
        "rsi_low_recent": round(rsi_low2, 1),
    }


def calc_bb_lower_breach(close: pd.Series) -> BBBreachResult:
    """Detect Bollinger Band lower breach: price below 20d SMA - 2*std.

    DCS >= 65 + BB breach: 60.4% win rate, +4.6pp edge,
    Cal=64.5%, Val=57.5%.
    """
    n = len(close)
    if n < 20:
        return {"breach": False, "bb_pct_b": 0.5, "lower_bb": None}

    current = float(close.iloc[-1])
    sma_20 = float(close.rolling(20).mean().iloc[-1])
    std_20 = float(close.rolling(20).std().iloc[-1])

    if std_20 <= 0:
        return {"breach": False, "bb_pct_b": 0.5, "lower_bb": None}

    upper_bb = sma_20 + 2 * std_20
    lower_bb = sma_20 - 2 * std_20
    bb_pct_b = (current - lower_bb) / (upper_bb - lower_bb)

    return {
        "breach": current < lower_bb,
        "bb_pct_b": round(bb_pct_b, 3),
        "lower_bb": round(lower_bb, 2),
    }


def calc_reversal_signals(
    close: pd.Series,
    rsi_value: float,
    macd_data: MACDResult,
    sa_quant: float | None,
) -> ReversalSignals:
    """Compute all backtest-validated reversal signals for a ticker.

    Called from score_ticker(). Returns dict of signal flags and metadata.

    Signals:
      1. RSI Bullish Divergence (+2.2pp edge, walk-forward stable)
      2. BB Lower Breach (used with DCS for "Reversal Confirmed" tag)
      3. MACD Hist Rising below zero + RSI < 30 + Quant >= 3 (Bottom-Turning)
      4. Q4+ RSI < 30 quant freshness warning (41% chance quant drops below 4)
    """
    result: ReversalSignals = {
        "rsi_bullish_divergence": False,
        "bb_lower_breach": False,
        "bb_pct_b": 0.5,
        "bottom_turning": False,
        "quant_freshness_warning": False,
    }

    # 1. RSI Bullish Divergence
    div_data = calc_rsi_bullish_divergence(close)
    result["rsi_bullish_divergence"] = div_data["detected"]

    # 2. BB Lower Breach
    bb_data = calc_bb_lower_breach(close)
    result["bb_lower_breach"] = bb_data["breach"]
    result["bb_pct_b"] = bb_data["bb_pct_b"]

    # 3. Bottom-Turning: MACD hist rising (below zero) + RSI < 30 + Quant >= 3
    #    Walk-forward stable: Cal=58.5%, Val=63.3%, +4.4pp edge, 544 signals
    hist_rising_below_zero = (
        macd_data.get("hist_rising", False) and macd_data.get("below_zero", False)
    )
    quant = sa_quant if sa_quant is not None else 0
    if hist_rising_below_zero and rsi_value < 30 and quant >= 3.0:
        result["bottom_turning"] = True

    # 4. Quant Freshness Warning: Q4+ stock at RSI < 30
    #    Phase 2 backtest: 41.2% of Q4+ stocks that hit RSI < 30 drop below
    #    quant 4 at next observation
    if quant >= 4.0 and rsi_value < 30:
        result["quant_freshness_warning"] = True

    return result


# ---------------------------------------------------------------------------
# Technical Helpers (used by Gate 3 and DCS)
# ---------------------------------------------------------------------------

def calc_price_acceleration(close: pd.Series) -> tuple[float, float]:
    """Calculate price acceleration and 8-week return.

    Returns (pa_score, ret_8w). ret_8w is used by Gate 3 deployment checks.
    """
    n = len(close)
    if n < 40:
        return 0.0, 0.0

    # 8-week return
    ret_8w = (close.iloc[-1] / close.iloc[-40]) - 1.0

    # Weekly returns for last 8 weeks
    weekly_returns = []
    for i in range(8):
        end_idx = -(i * 5) - 1 if i > 0 else -1
        start_idx = -((i + 1) * 5) - 1
        if abs(start_idx) < n:
            w_ret = (close.iloc[end_idx] / close.iloc[start_idx]) - 1.0
            weekly_returns.append(w_ret)
    weekly_returns.reverse()

    # Acceleration: later weeks' returns larger than earlier
    acceleration = 0.0
    if len(weekly_returns) >= 4:
        first_half = np.mean(weekly_returns[:4])
        second_half = (
            np.mean(weekly_returns[4:]) if len(weekly_returns) > 4 else first_half
        )
        acceleration = second_half - first_half

    # 8-week return scoring
    if ret_8w < 0.15:
        ret_score = 0.0
    elif ret_8w < 0.30:
        ret_score = (ret_8w - 0.15) / 0.15 * 0.5
    elif ret_8w < 0.50:
        ret_score = 0.5 + (ret_8w - 0.30) / 0.20 * 0.3
    else:
        ret_score = min(1.0, 0.8 + (ret_8w - 0.50) / 0.30 * 0.2)

    # Acceleration scoring
    accel_score = max(0.0, min(1.0, acceleration / 0.03))

    return (ret_score * 0.60) + (accel_score * 0.40), ret_8w


def calc_consecutive_days_below_sma(
    close: pd.Series,
    threshold: float = -0.03,
) -> tuple[int, float]:
    """Count consecutive trading days where price is >3% below 200d SMA.

    Sell criterion #3 requires 10+ consecutive days.
    Returns (count, pct_from_sma_current).
    """
    n = len(close)
    if n < 200:
        return 0, 0.0

    sma_200 = close.rolling(200).mean()
    pct_from_sma = (close - sma_200) / sma_200

    # Count consecutive days from the most recent bar going backwards
    count = 0
    for i in range(len(pct_from_sma) - 1, -1, -1):
        val = pct_from_sma.iloc[i]
        if not np.isnan(val) and val < threshold:
            count += 1
        else:
            break

    current_pct = (
        float(pct_from_sma.iloc[-1])
        if not np.isnan(pct_from_sma.iloc[-1])
        else 0.0
    )
    return count, current_pct
