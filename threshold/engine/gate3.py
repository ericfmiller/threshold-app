"""Gate 3 parabolic filter — deployment discipline for buy signals.

Gate 3 checks whether a ticker is in a parabolic move before allowing
full deployment of capital. A parabolic move is defined as:

  - RSI > config.deployment.gate3_rsi_max (default 80), AND
  - 8-week return > config.deployment.gate3_ret_8w_max (default 30%)

When Gate 3 fires:
  - Standard tickers: deployment is blocked (WAIT/FAIL)
  - Gold tickers: deployment at 0.75x size (Rule D-13)
  - Gold tickers exempt from Gate 3 entirely (regime-driven moves)

Sizing levels returned:
  - FULL: No parabolic signal, deploy at full position size
  - THREE_QUARTER: Gold at RSI > 80, deploy at 0.75x
  - HALF: Near thresholds but not fully parabolic
  - WAIT: One criterion triggered but not both
  - FAIL: Both criteria triggered — do not deploy
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class Gate3Result:
    """Result of a Gate 3 parabolic filter check."""

    passes: bool = True
    sizing: str = "FULL"  # FULL, THREE_QUARTER, HALF, WAIT, FAIL
    reason: str = ""
    rsi: float = 0.0
    ret_8w: float = 0.0
    is_gold_exempt: bool = False


# ---------------------------------------------------------------------------
# Core function
# ---------------------------------------------------------------------------

def check_gate3(
    rsi: float,
    ret_8w: float,
    config: Any | None = None,
    is_gold: bool = False,
) -> Gate3Result:
    """Check Gate 3 parabolic filter for deployment sizing.

    Parameters
    ----------
    rsi : float
        Current 14-day RSI value.
    ret_8w : float
        8-week return (0.30 = 30%).
    config : ThresholdConfig | None
        Optional config for thresholds. Uses defaults if None.
    is_gold : bool
        Whether the ticker is a gold position (D-13 exemption).

    Returns
    -------
    Gate3Result
        Whether the ticker passes Gate 3 and at what deployment size.
    """
    # Read thresholds from config
    rsi_max = 80
    ret_8w_max = 0.30
    gold_sizing = 0.75
    if config is not None:
        deploy = getattr(config, "deployment", None)
        if deploy is not None:
            rsi_max = getattr(deploy, "gate3_rsi_max", rsi_max)
            ret_8w_max = getattr(deploy, "gate3_ret_8w_max", ret_8w_max)
            gold_sizing = getattr(deploy, "gold_rsi_max_sizing", gold_sizing)

    result = Gate3Result(rsi=round(rsi, 1), ret_8w=round(ret_8w, 4))

    rsi_triggered = rsi > rsi_max
    ret_triggered = ret_8w > ret_8w_max

    # Gold exemption (Rule D-13)
    if is_gold:
        result.is_gold_exempt = True
        if rsi_triggered:
            # Gold at RSI > 80: 0.75x sizing, not blocked
            result.passes = True
            result.sizing = "THREE_QUARTER"
            result.reason = (
                f"Gold RSI {rsi:.0f} > {rsi_max} — deploy at "
                f"{gold_sizing:.0%} size (D-13: gold exempt from Gate 3)"
            )
        else:
            result.passes = True
            result.sizing = "FULL"
            result.reason = "Gold exempt from Gate 3 parabolic filter (D-13)"
        return result

    # Both triggered → FAIL
    if rsi_triggered and ret_triggered:
        result.passes = False
        result.sizing = "FAIL"
        result.reason = (
            f"PARABOLIC: RSI {rsi:.0f} > {rsi_max} AND "
            f"8w return {ret_8w:.1%} > {ret_8w_max:.0%} — do NOT deploy"
        )
        return result

    # One triggered → WAIT
    if rsi_triggered:
        result.passes = False
        result.sizing = "WAIT"
        result.reason = (
            f"RSI {rsi:.0f} > {rsi_max} — wait for RSI pullback "
            f"before deploying"
        )
        return result

    if ret_triggered:
        result.passes = False
        result.sizing = "WAIT"
        result.reason = (
            f"8w return {ret_8w:.1%} > {ret_8w_max:.0%} — "
            f"wait for consolidation before deploying"
        )
        return result

    # Neither triggered → FULL
    result.passes = True
    result.sizing = "FULL"
    result.reason = "Gate 3 passed — deploy at full size"
    return result
