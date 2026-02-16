"""Risk framework — toggle-able overlays for advanced risk monitoring.

Each module is behind ``config.risk.X.enabled`` (default False).
When disabled, DCS scoring is unchanged.

Modules:
  ebp             — Gilchrist-Zakrajšek Excess Bond Premium
  turbulence      — Kritzman-Li Mahalanobis turbulence index
  momentum_crash  — Daniel-Moskowitz conditional momentum crash protection
  cvar            — Conditional Value at Risk
  cdar            — Conditional Drawdown at Risk
"""

from threshold.engine.risk.cdar import CDaRCalculator
from threshold.engine.risk.cvar import CVaRCalculator
from threshold.engine.risk.ebp import EBPMonitor
from threshold.engine.risk.momentum_crash import MomentumCrashProtection
from threshold.engine.risk.turbulence import TurbulenceIndex

__all__ = [
    "CDaRCalculator",
    "CVaRCalculator",
    "EBPMonitor",
    "MomentumCrashProtection",
    "TurbulenceIndex",
]
