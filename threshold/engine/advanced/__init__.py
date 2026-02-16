"""Advanced signal modules — toggle-able overlays for enhanced scoring.

Each module is behind ``config.advanced.X.enabled`` (default False).
When disabled, DCS scoring is unchanged.

Modules:
  trend_following   — Baltas-Kosowski continuous trend signal (blends into MQ)
  factor_momentum   — Ehsani-Linnainmaa factor momentum (informational only)
  sentiment         — Huang et al. aligned sentiment index (adjusts MR)
"""

from threshold.engine.advanced.factor_momentum import FactorMomentumSignal
from threshold.engine.advanced.trend_following import ContinuousTrendFollower

# Sentiment requires optional dep (scikit-learn) — lazy import
try:
    from threshold.engine.advanced.sentiment import AlignedSentimentIndex
except ImportError:
    AlignedSentimentIndex = None  # type: ignore[assignment, misc]

__all__ = [
    "ContinuousTrendFollower",
    "FactorMomentumSignal",
    "AlignedSentimentIndex",
]
