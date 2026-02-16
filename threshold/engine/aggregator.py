"""Cross-module signal aggregation — composite risk overlay.

Combines outputs from the risk framework (EBP, turbulence, momentum crash)
into a single composite risk score, then optionally applies a DCS penalty
based on the aggregated risk level.

Integration: Optional post-DCS modifier in scorer.py.
When disabled (default), DCS scoring is unchanged.
When enabled, applies a configurable penalty:
  - HIGH_RISK (≥0.7): DCS -10
  - ELEVATED (≥0.4): DCS -5
  - NORMAL (<0.4): no change
"""

from __future__ import annotations

from typing import TypedDict


class CompositeRiskResult(TypedDict):
    """Result from composite risk aggregation."""
    composite_score: float     # Aggregated risk score [0, 1]
    regime: str                # HIGH_RISK, ELEVATED, NORMAL
    ebp_contrib: float         # EBP contribution to composite
    turbulence_contrib: float  # Turbulence contribution to composite
    crash_contrib: float       # Momentum crash contribution to composite
    dcs_penalty: int           # Points to subtract from DCS (non-negative)


class SignalAggregator:
    """Aggregates risk module signals into a composite risk overlay.

    Usage::

        agg = SignalAggregator(
            ebp_weight=0.40,
            turbulence_weight=0.30,
            crash_weight=0.30,
        )
        result = agg.compute_composite_risk(ebp_signal, turb_signal, crash_signal)
        adjusted_dcs = agg.apply_risk_overlay(dcs=72, composite=result)

    Parameters:
        ebp_weight: Weight of EBP signal in composite (default 0.40).
        turbulence_weight: Weight of turbulence signal (default 0.30).
        crash_weight: Weight of momentum crash signal (default 0.30).
        high_risk_threshold: Composite score ≥ this = HIGH_RISK (default 0.70).
        elevated_threshold: Composite score ≥ this = ELEVATED (default 0.40).
        high_risk_penalty: DCS penalty for HIGH_RISK regime (default 10).
        elevated_penalty: DCS penalty for ELEVATED regime (default 5).
    """

    def __init__(
        self,
        ebp_weight: float = 0.40,
        turbulence_weight: float = 0.30,
        crash_weight: float = 0.30,
        high_risk_threshold: float = 0.70,
        elevated_threshold: float = 0.40,
        high_risk_penalty: int = 10,
        elevated_penalty: int = 5,
    ) -> None:
        self.ebp_weight = ebp_weight
        self.turbulence_weight = turbulence_weight
        self.crash_weight = crash_weight
        self.high_risk_threshold = high_risk_threshold
        self.elevated_threshold = elevated_threshold
        self.high_risk_penalty = high_risk_penalty
        self.elevated_penalty = elevated_penalty

    def _normalize_ebp(self, ebp_signal: dict | None) -> float:
        """Normalize EBP signal to [0, 1] risk score.

        EBP regimes map to:
          HIGH_RISK → 1.0
          ELEVATED → 0.6
          NORMAL → 0.3
          ACCOMMODATIVE → 0.0
          Unknown/None → 0.0

        Parameters:
            ebp_signal: EBP result dict with "regime" key.

        Returns:
            Normalized risk score [0, 1].
        """
        if ebp_signal is None:
            return 0.0
        regime = ebp_signal.get("regime", "UNKNOWN")
        mapping = {
            "HIGH_RISK": 1.0,
            "ELEVATED": 0.6,
            "NORMAL": 0.3,
            "ACCOMMODATIVE": 0.0,
        }
        return mapping.get(regime, 0.0)

    def _normalize_turbulence(self, turb_signal: dict | None) -> float:
        """Normalize turbulence signal to [0, 1] risk score.

        Uses the percentile directly if available, otherwise maps regime:
          TURBULENT → 0.9
          ELEVATED → 0.6
          CALM → 0.1

        Parameters:
            turb_signal: Turbulence result dict.

        Returns:
            Normalized risk score [0, 1].
        """
        if turb_signal is None:
            return 0.0

        # Prefer percentile if available
        pctl = turb_signal.get("percentile")
        if pctl is not None:
            return min(max(float(pctl), 0.0), 1.0)

        regime = turb_signal.get("regime", "CALM")
        mapping = {
            "TURBULENT": 0.9,
            "ELEVATED": 0.6,
            "CALM": 0.1,
        }
        return mapping.get(regime, 0.0)

    def _normalize_crash(self, crash_signal: dict | None) -> float:
        """Normalize momentum crash signal to [0, 1] risk score.

        Uses the crash probability if available, otherwise maps bear state:
          is_bear=True → 0.8
          is_bear=False → 0.1

        Parameters:
            crash_signal: Momentum crash result dict.

        Returns:
            Normalized risk score [0, 1].
        """
        if crash_signal is None:
            return 0.0

        prob = crash_signal.get("crash_probability")
        if prob is not None:
            return min(max(float(prob), 0.0), 1.0)

        is_bear = crash_signal.get("is_bear", False)
        return 0.8 if is_bear else 0.1

    def _classify_regime(self, composite: float) -> str:
        """Classify composite risk score into regime."""
        if composite >= self.high_risk_threshold:
            return "HIGH_RISK"
        if composite >= self.elevated_threshold:
            return "ELEVATED"
        return "NORMAL"

    def _get_penalty(self, regime: str) -> int:
        """Get DCS penalty for given risk regime."""
        if regime == "HIGH_RISK":
            return self.high_risk_penalty
        if regime == "ELEVATED":
            return self.elevated_penalty
        return 0

    def compute_composite_risk(
        self,
        ebp_signal: dict | None = None,
        turb_signal: dict | None = None,
        crash_signal: dict | None = None,
    ) -> CompositeRiskResult:
        """Compute composite risk score from individual risk signals.

        composite = w_ebp * norm_ebp + w_turb * norm_turb + w_crash * norm_crash

        Parameters:
            ebp_signal: EBP result dict (or None if not computed).
            turb_signal: Turbulence result dict (or None).
            crash_signal: Momentum crash result dict (or None).

        Returns:
            CompositeRiskResult with score, regime, contributions, and penalty.
        """
        ebp_norm = self._normalize_ebp(ebp_signal)
        turb_norm = self._normalize_turbulence(turb_signal)
        crash_norm = self._normalize_crash(crash_signal)

        # Weighted composite
        ebp_contrib = self.ebp_weight * ebp_norm
        turb_contrib = self.turbulence_weight * turb_norm
        crash_contrib = self.crash_weight * crash_norm

        composite = ebp_contrib + turb_contrib + crash_contrib
        composite = min(max(composite, 0.0), 1.0)  # Clamp to [0, 1]

        regime = self._classify_regime(composite)
        penalty = self._get_penalty(regime)

        return CompositeRiskResult(
            composite_score=round(composite, 4),
            regime=regime,
            ebp_contrib=round(ebp_contrib, 4),
            turbulence_contrib=round(turb_contrib, 4),
            crash_contrib=round(crash_contrib, 4),
            dcs_penalty=penalty,
        )

    def apply_risk_overlay(
        self,
        dcs: float,
        composite: CompositeRiskResult,
    ) -> float:
        """Apply risk overlay penalty to DCS score.

        Parameters:
            dcs: Raw DCS score (0-100).
            composite: Result from compute_composite_risk.

        Returns:
            Adjusted DCS score, floored at 0 and capped at 100.
        """
        adjusted = dcs - composite["dcs_penalty"]
        return min(max(adjusted, 0.0), 100.0)
