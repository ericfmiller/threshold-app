"""Scoring engine organized by the Decision Hierarchy.

Public API:
  score_ticker   — Score a single ticker -> ScoringResult
  ScoringResult  — TypedDict for scoring output
  ScoringContext — Shared per-run context (market regime, SPY, history)
  SignalBoard    — Typed container for scoring signals
"""

from threshold.engine.context import ScoringContext
from threshold.engine.scorer import ScoringResult, score_ticker
from threshold.engine.signals import SignalBoard

__all__ = [
    "ScoringContext",
    "ScoringResult",
    "SignalBoard",
    "score_ticker",
]
