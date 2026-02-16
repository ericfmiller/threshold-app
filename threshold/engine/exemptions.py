"""Exemption logic for the scoring pipeline.

Certain tickers are exempt from standard sell criteria:

  - **Crypto halving cycle**: FBTC, FETH, MSTR, TSWCF — exempt from
    sell signals because the Bitcoin 4-year halving cycle overrides
    momentum signals. Still scored (DCS computed for tracking).
    Exemption has a configurable expiry date.

  - **Cash/war chest**: STIP and cash-like positions — exempt from
    sell scoring entirely (always hold).

Tickers are flagged in the ``tickers`` table with ``is_crypto_exempt``
and ``is_cash`` boolean columns. The exemption module checks these
flags and config-level expiry dates to determine exemption status.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class ExemptionResult:
    """Result of an exemption check for one ticker."""

    is_exempt: bool = False
    reason: str = ""
    exemption_type: str = ""  # "crypto_halving", "cash", "none"
    expires_at: str = ""  # ISO date or empty
    is_expired: bool = False


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------

def is_exempt_from_sell(
    ticker_meta: dict[str, Any],
    config: Any | None = None,
) -> ExemptionResult:
    """Check whether a ticker is exempt from sell signal generation.

    Parameters
    ----------
    ticker_meta : dict
        Ticker metadata from the ``tickers`` table. Expected keys:
        ``is_crypto_exempt``, ``is_cash``, ``symbol``.
    config : ThresholdConfig | None
        Optional config for crypto exemption expiry date.

    Returns
    -------
    ExemptionResult
        Exemption status with reason and type.
    """
    symbol = ticker_meta.get("symbol", "")

    # Check cash exemption first (no expiry)
    if ticker_meta.get("is_cash"):
        return ExemptionResult(
            is_exempt=True,
            reason=f"{symbol} is a cash/war chest position",
            exemption_type="cash",
        )

    # Check crypto halving cycle exemption
    if ticker_meta.get("is_crypto_exempt"):
        expiry = _get_crypto_expiry(config)

        if expiry:
            today = date.today().isoformat()
            if today > expiry:
                logger.info(
                    "Crypto exemption for %s expired on %s", symbol, expiry,
                )
                return ExemptionResult(
                    is_exempt=False,
                    reason=f"Crypto exemption expired on {expiry}",
                    exemption_type="crypto_halving",
                    expires_at=expiry,
                    is_expired=True,
                )

        return ExemptionResult(
            is_exempt=True,
            reason=f"{symbol} exempt — Bitcoin halving cycle hold",
            exemption_type="crypto_halving",
            expires_at=expiry or "",
        )

    return ExemptionResult(
        exemption_type="none",
    )


def get_exempt_tickers(
    all_tickers: list[dict[str, Any]],
    config: Any | None = None,
) -> dict[str, ExemptionResult]:
    """Check exemptions for all tickers in the universe.

    Parameters
    ----------
    all_tickers : list[dict]
        List of ticker metadata dicts from the ``tickers`` table.
    config : ThresholdConfig | None
        Optional config for crypto exemption expiry.

    Returns
    -------
    dict[str, ExemptionResult]
        Mapping of symbol → ExemptionResult for all tickers
        that have ANY exemption (active or expired).
    """
    result: dict[str, ExemptionResult] = {}
    for ticker in all_tickers:
        exemption = is_exempt_from_sell(ticker, config)
        if exemption.exemption_type != "none":
            result[ticker.get("symbol", "")] = exemption
    return result


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_crypto_expiry(config: Any | None) -> str:
    """Extract crypto exemption expiry date from config.

    Returns empty string if not configured.
    """
    if config is None:
        return ""

    # Try scoring.crypto_exempt_expiry
    scoring = getattr(config, "scoring", None)
    if scoring is not None:
        expiry = getattr(scoring, "crypto_exempt_expiry", None)
        if expiry:
            return str(expiry)

    return ""
