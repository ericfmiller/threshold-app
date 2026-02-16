"""SignalBoard — typed signal taxonomy for the scoring pipeline.

Replaces the flat ``sell_flags: list[str]`` with structured ``Signal``
objects grouped by a ``SignalBoard`` container.  Provides:

  - ``SignalType`` enum: SELL_HARD, EARLY_WARNING, BUY_CONFIRMED, etc.
  - ``Severity`` enum: CRITICAL, HIGH, MEDIUM, LOW, INFO
  - ``Signal`` frozen dataclass: one signal event with metadata
  - ``SignalBoard`` container: typed access, net_action resolution,
    and ``to_legacy_flags()`` for backward compatibility
  - 11 factory functions: one per signal origin in the scoring pipeline

Backward compatibility contract
-------------------------------
``board.to_legacy_flags()`` produces the **exact same** ``list[str]``
that the old ``sell_flags.append(...)`` calls produced.  Consumers
(narrative, dashboard, alerts) continue reading ``sell_flags`` unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class SignalType(Enum):
    """Semantic category of a scoring signal."""
    SELL_HARD = "SELL_HARD"             # 200d SMA breach 10d+, quant drop, EPS rev 3+
    EARLY_WARNING = "EARLY_WARNING"     # SMA 7-9d, EPS rev 2 sub-grades
    BUY_CONFIRMED = "BUY_CONFIRMED"    # REVERSAL CONFIRMED
    BUY_WATCHLIST = "BUY_WATCHLIST"     # BOTTOM TURNING
    HOLD_OVERRIDE = "HOLD_OVERRIDE"    # DEFENSIVE_HOLD (drawdown insurance)
    TRIM_PRIORITY = "TRIM_PRIORITY"    # AMPLIFIER_WARNING
    DEPLOYMENT_GATE = "DEPLOYMENT_GATE"  # CONCENTRATION
    VERIFY = "VERIFY"                   # QUANT_CHECK


class Severity(Enum):
    """Signal severity for downstream prioritization."""
    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    INFO = "INFO"


# ---------------------------------------------------------------------------
# Signal dataclass
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Signal:
    """One scoring signal with typed metadata.

    Attributes:
        signal_type: Semantic category (SELL_HARD, BUY_CONFIRMED, etc.).
        severity: Urgency level.
        message: Human-readable description (the part AFTER the prefix).
        legacy_prefix: The colon-terminated prefix used in legacy sell_flags
            strings (e.g. ``"SELL:"``).
        metadata: Structured data for programmatic consumers (thresholds,
            criterion names, etc.).
    """
    signal_type: SignalType
    severity: Severity
    message: str
    legacy_prefix: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_legacy_flag(self) -> str:
        """Reconstruct the exact legacy sell_flags string."""
        return f"{self.legacy_prefix} {self.message}"

    def to_dict(self) -> dict[str, Any]:
        """Serialize for JSON persistence (score_history)."""
        return {
            "signal_type": self.signal_type.value,
            "severity": self.severity.value,
            "message": self.message,
            "legacy_prefix": self.legacy_prefix,
            "legacy_flag": self.to_legacy_flag(),
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Signal:
        """Deserialize from JSON."""
        return cls(
            signal_type=SignalType(d["signal_type"]),
            severity=Severity(d["severity"]),
            message=d["message"],
            legacy_prefix=d["legacy_prefix"],
            metadata=d.get("metadata", {}),
        )


# ---------------------------------------------------------------------------
# SignalBoard container
# ---------------------------------------------------------------------------

class SignalBoard:
    """Typed container for all signals produced during ``score_ticker()``.

    Provides filtered views (``sells``, ``warnings``, ``buy_signals``),
    conflict resolution (``net_action``), and backward-compatible
    legacy string conversion (``to_legacy_flags()``).
    """

    def __init__(self) -> None:
        self._signals: list[Signal] = []

    # -- mutation --

    def add(self, signal: Signal) -> None:
        """Append a signal to the board."""
        self._signals.append(signal)

    # -- read-only views --

    @property
    def signals(self) -> list[Signal]:
        """All signals in insertion order."""
        return list(self._signals)

    @property
    def sells(self) -> list[Signal]:
        """SELL_HARD signals only."""
        return [s for s in self._signals if s.signal_type is SignalType.SELL_HARD]

    @property
    def warnings(self) -> list[Signal]:
        """EARLY_WARNING signals only."""
        return [s for s in self._signals if s.signal_type is SignalType.EARLY_WARNING]

    @property
    def buy_signals(self) -> list[Signal]:
        """BUY_CONFIRMED + BUY_WATCHLIST signals."""
        return [
            s for s in self._signals
            if s.signal_type in (SignalType.BUY_CONFIRMED, SignalType.BUY_WATCHLIST)
        ]

    @property
    def hold_overrides(self) -> list[Signal]:
        """HOLD_OVERRIDE signals (DEFENSIVE_HOLD)."""
        return [s for s in self._signals if s.signal_type is SignalType.HOLD_OVERRIDE]

    @property
    def trim_signals(self) -> list[Signal]:
        """TRIM_PRIORITY signals (AMPLIFIER_WARNING)."""
        return [s for s in self._signals if s.signal_type is SignalType.TRIM_PRIORITY]

    @property
    def deployment_gates(self) -> list[Signal]:
        """DEPLOYMENT_GATE signals (CONCENTRATION)."""
        return [s for s in self._signals if s.signal_type is SignalType.DEPLOYMENT_GATE]

    @property
    def verify_signals(self) -> list[Signal]:
        """VERIFY signals (QUANT_CHECK)."""
        return [s for s in self._signals if s.signal_type is SignalType.VERIFY]

    @property
    def has_sell_review(self) -> bool:
        """True if 2+ SELL_HARD signals -> review required (investment_rules.md)."""
        return len(self.sells) >= 2

    # -- conflict resolution --

    @property
    def net_action(self) -> str:
        """Compute net recommended action from all signals.

        Priority (highest first):
          1. 2+ SELL_HARD -> "REVIEW"
          2. 1 SELL_HARD + HOLD_OVERRIDE -> "HOLD"
          3. 1 SELL_HARD alone -> "WATCH"
          4. TRIM_PRIORITY (no sells) -> "TRIM"
          5. BUY_CONFIRMED (no sells) -> "BUY"
          6. BUY_WATCHLIST (no sells) -> "WATCHLIST"
          7. Warnings only -> "WATCH"
          8. Nothing -> "NONE"
        """
        n_sells = len(self.sells)

        if n_sells >= 2:
            return "REVIEW"
        if n_sells == 1:
            if self.hold_overrides:
                return "HOLD"
            return "WATCH"

        # No hard sells below this point
        if self.trim_signals:
            return "TRIM"
        if any(s.signal_type is SignalType.BUY_CONFIRMED for s in self._signals):
            return "BUY"
        if any(s.signal_type is SignalType.BUY_WATCHLIST for s in self._signals):
            return "WATCHLIST"
        if self.warnings:
            return "WATCH"

        return "NONE"

    # -- legacy conversion --

    def to_legacy_flags(self) -> list[str]:
        """Convert all signals to the legacy ``sell_flags`` string format.

        Order matches insertion order, which matches the original
        ``sell_flags.append()`` call sequence in ``score_ticker()``.
        """
        return [s.to_legacy_flag() for s in self._signals]

    # -- serialization --

    def to_dict(self) -> list[dict[str, Any]]:
        """Serialize the board to a JSON-safe list of signal dicts."""
        return [s.to_dict() for s in self._signals]

    @classmethod
    def from_dict(cls, data: list[dict[str, Any]]) -> SignalBoard:
        """Deserialize from a list of signal dicts."""
        board = cls()
        for d in data:
            board.add(Signal.from_dict(d))
        return board

    def __len__(self) -> int:
        return len(self._signals)

    def __bool__(self) -> bool:
        return bool(self._signals)

    def __repr__(self) -> str:
        return f"SignalBoard({len(self._signals)} signals, net_action={self.net_action!r})"


# ---------------------------------------------------------------------------
# Factory functions — one per signal origin
# ---------------------------------------------------------------------------

def make_sma_breach_sell(days_below: int) -> Signal:
    """Sell #1: 10+ consecutive days >3% below 200d SMA."""
    return Signal(
        signal_type=SignalType.SELL_HARD,
        severity=Severity.HIGH,
        message=f"{days_below} consecutive days >3% below 200d SMA",
        legacy_prefix="SELL:",
        metadata={"criterion": "sma_breach", "days_below": days_below},
    )


def make_sma_breach_warning(days_below: int) -> Signal:
    """Early warning: 7-9 consecutive days >3% below 200d SMA."""
    return Signal(
        signal_type=SignalType.EARLY_WARNING,
        severity=Severity.MEDIUM,
        message=f"{days_below} consecutive days >3% below 200d SMA (trigger at 10)",
        legacy_prefix="WARNING:",
        metadata={"criterion": "sma_breach", "days_below": days_below},
    )


def make_quant_drop_sell(quant_delta: float, compare_date: str) -> Signal:
    """Sell #2: SA Quant dropped significantly (>1.0) in 30 days."""
    return Signal(
        signal_type=SignalType.SELL_HARD,
        severity=Severity.HIGH,
        message=f"SA Quant dropped {quant_delta:+.2f} since {compare_date}",
        legacy_prefix="SELL:",
        metadata={
            "criterion": "quant_drop",
            "delta": quant_delta,
            "compare_date": compare_date,
        },
    )


def make_eps_rev_sell(sub_grade_steps: float, delta_4w: float) -> Signal:
    """Sell #3: EPS Revisions dropped 3+ sub-grades in 4 weeks."""
    return Signal(
        signal_type=SignalType.SELL_HARD,
        severity=Severity.HIGH,
        message=(
            f"EPS Revisions dropped {sub_grade_steps:.0f} sub-grades in 4 weeks "
            f"(delta {delta_4w:+.3f})"
        ),
        legacy_prefix="SELL:",
        metadata={
            "criterion": "eps_revision",
            "sub_grade_steps": sub_grade_steps,
            "delta_4w": delta_4w,
        },
    )


def make_eps_rev_warning(sub_grade_steps: float, delta_4w: float) -> Signal:
    """Early warning: EPS Revisions declined 2 sub-grades in 4 weeks."""
    return Signal(
        signal_type=SignalType.EARLY_WARNING,
        severity=Severity.MEDIUM,
        message=(
            f"EPS Revisions declined {sub_grade_steps:.0f} sub-grades in 4 weeks "
            f"(delta {delta_4w:+.3f}, trigger at 3)"
        ),
        legacy_prefix="WARNING:",
        metadata={
            "criterion": "eps_revision",
            "sub_grade_steps": sub_grade_steps,
            "delta_4w": delta_4w,
        },
    )


def make_quant_freshness_warning() -> Signal:
    """Verify: RSI < 30 on Q4+ stock — quant may be stale."""
    return Signal(
        signal_type=SignalType.VERIFY,
        severity=Severity.INFO,
        message=(
            "RSI < 30 on Q4+ stock — verify quant score is current "
            "(41% of Q4+ stocks at RSI<30 drop below quant 4 at next observation)"
        ),
        legacy_prefix="QUANT_CHECK:",
        metadata={"criterion": "quant_freshness"},
    )


def make_defensive_hold(classification: str, downside_capture: float) -> Signal:
    """Hold override: HEDGE/DEFENSIVE asset provides drawdown insurance (D-7)."""
    return Signal(
        signal_type=SignalType.HOLD_OVERRIDE,
        severity=Severity.MEDIUM,
        message=(
            f"{classification} asset (DC={downside_capture:.2f}) provides "
            f"drawdown insurance — consider extended grace (270d)"
        ),
        legacy_prefix="DEFENSIVE_HOLD:",
        metadata={
            "criterion": "defensive_hold",
            "classification": classification,
            "downside_capture": downside_capture,
        },
    )


def make_amplifier_warning(downside_capture: float) -> Signal:
    """Trim priority: AMPLIFIER asset amplifies losses in drawdowns (D-8)."""
    return Signal(
        signal_type=SignalType.TRIM_PRIORITY,
        severity=Severity.HIGH,
        message=(
            f"DC={downside_capture:.2f} — amplifies losses "
            f"in drawdowns. Consider priority trim."
        ),
        legacy_prefix="AMPLIFIER_WARNING:",
        metadata={
            "criterion": "amplifier_warning",
            "downside_capture": downside_capture,
        },
    )


def make_reversal_confirmed() -> Signal:
    """Buy confirmed: DCS >= 65 + BB lower breach — higher conviction dip-buy."""
    return Signal(
        signal_type=SignalType.BUY_CONFIRMED,
        severity=Severity.LOW,
        message=(
            "DCS >= 65 + BB lower breach — "
            "higher-conviction dip-buy (+4.6pp edge, walk-forward stable)"
        ),
        legacy_prefix="REVERSAL CONFIRMED:",
        metadata={"criterion": "reversal_confirmed"},
    )


def make_bottom_turning() -> Signal:
    """Buy watchlist: MACD hist rising + RSI < 30 + Q3+ — watchlist alert."""
    return Signal(
        signal_type=SignalType.BUY_WATCHLIST,
        severity=Severity.LOW,
        message=(
            "MACD hist rising from below zero + RSI < 30 + Q3+ — "
            "watchlist alert (+4.4pp edge, walk-forward stable)"
        ),
        legacy_prefix="BOTTOM TURNING:",
        metadata={"criterion": "bottom_turning"},
    )


def make_concentration_warning(
    correlated_with: list[str],
    effective_bets: float,
) -> Signal:
    """Deployment gate: high correlation with existing holdings."""
    return Signal(
        signal_type=SignalType.DEPLOYMENT_GATE,
        severity=Severity.MEDIUM,
        message=(
            f"High corr with {', '.join(correlated_with[:3])} "
            f"(eff. bets: {effective_bets:.0f})"
        ),
        legacy_prefix="CONCENTRATION:",
        metadata={
            "criterion": "concentration",
            "correlated_with": correlated_with,
            "effective_bets": effective_bets,
        },
    )
