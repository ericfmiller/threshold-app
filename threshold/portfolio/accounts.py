"""Account definitions and position aggregation.

Consolidates multi-account holdings into a unified view with
total shares, dollar-weighted positions, and cross-account tracking.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Position:
    """A consolidated position across one or more accounts."""
    symbol: str
    total_shares: float = 0.0
    total_value: float = 0.0
    account_weights: dict[str, float] = field(default_factory=dict)
    """account_id → weight in that account (0-1)."""
    account_shares: dict[str, float] = field(default_factory=dict)
    """account_id → share count in that account."""
    account_values: dict[str, float] = field(default_factory=dict)
    """account_id → dollar value in that account."""
    n_accounts: int = 0

    @property
    def is_multi_account(self) -> bool:
        return self.n_accounts > 1


@dataclass
class PortfolioSnapshot:
    """Point-in-time snapshot of the entire portfolio."""
    positions: dict[str, Position] = field(default_factory=dict)
    """symbol → Position."""
    account_totals: dict[str, float] = field(default_factory=dict)
    """account_id → total account value."""
    total_value: float = 0.0
    snapshot_date: str = ""

    @property
    def n_positions(self) -> int:
        return len(self.positions)

    def get_position(self, symbol: str) -> Position | None:
        return self.positions.get(symbol)

    def portfolio_weight(self, symbol: str) -> float:
        """Get the portfolio-level weight for a ticker."""
        pos = self.positions.get(symbol)
        if pos is None or self.total_value <= 0:
            return 0.0
        return pos.total_value / self.total_value


def aggregate_positions(
    raw_positions: list[dict[str, Any]],
    account_totals: dict[str, float] | None = None,
) -> PortfolioSnapshot:
    """Aggregate raw position data into a consolidated PortfolioSnapshot.

    Parameters:
        raw_positions: List of position dicts with at minimum:
            - symbol: str
            - account_id: str
            - shares: float
            - market_value: float (or weight + account total)
        account_totals: Optional {account_id: total_value} for weight calc.

    Returns:
        PortfolioSnapshot with consolidated positions.
    """
    positions: dict[str, Position] = {}
    acct_totals = account_totals or {}

    for pos in raw_positions:
        symbol = pos.get("symbol", "")
        account_id = pos.get("account_id", "")
        shares = float(pos.get("shares", 0))
        value = float(pos.get("market_value", 0))
        weight = float(pos.get("weight", 0))

        if not symbol:
            continue

        # Compute value from weight if not provided directly
        if value <= 0 and weight > 0 and account_id in acct_totals:
            value = weight * acct_totals[account_id]

        if symbol not in positions:
            positions[symbol] = Position(symbol=symbol)

        p = positions[symbol]
        p.total_shares += shares
        p.total_value += value
        p.account_shares[account_id] = shares
        p.account_values[account_id] = value

        # Track per-account weight
        acct_total = acct_totals.get(account_id, 0)
        if acct_total > 0:
            p.account_weights[account_id] = value / acct_total

    # Finalize
    total_portfolio = sum(p.total_value for p in positions.values())

    for p in positions.values():
        p.n_accounts = len(p.account_shares)

    snapshot = PortfolioSnapshot(
        positions=positions,
        account_totals=acct_totals,
        total_value=round(total_portfolio, 2),
    )
    return snapshot
