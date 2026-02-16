"""Portfolio construction modules — position sizing, allocation, and tax optimization.

Each module is behind ``config.portfolio.X.enabled`` (default False).
When disabled, portfolio construction uses equal-weight or user-specified weights.

Modules:
  inverse_vol   — Kirby-Ostdiek inverse volatility weighting
  hrp           — López de Prado Hierarchical Risk Parity
  tax           — HIFO tax-lot accounting + tax-loss harvesting
"""

from threshold.engine.portfolio.hrp import HRPAllocator
from threshold.engine.portfolio.inverse_vol import InverseVolWeighter
from threshold.engine.portfolio.tax import HIFOSelector, TaxLossHarvester

__all__ = [
    "HRPAllocator",
    "HIFOSelector",
    "InverseVolWeighter",
    "TaxLossHarvester",
]
