"""HIFO tax-lot accounting and tax-loss harvesting.

HIFO (Highest-In, First-Out) selects lots with the highest cost basis
first when selling, minimizing taxable gains (or maximizing deductible losses).

Tax-loss harvesting scans positions for unrealized losses that can be
realized to offset gains, respecting the IRS 30-day wash sale rule.

Integration: Used by pipeline.py for sell-order lot selection and
periodic tax optimization scans. Data stored in the tax_lots table.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import TypedDict


class TaxLot(TypedDict):
    """A single tax lot representing a purchase of shares."""
    lot_id: int                    # Database primary key
    account_id: str                # Which account holds this lot
    symbol: str                    # Ticker symbol
    shares: float                  # Number of shares in this lot
    cost_basis_per_share: float    # Purchase price per share
    acquired_at: str               # Date acquired (ISO format YYYY-MM-DD)
    lot_type: str                  # "buy", "transfer_in", etc.
    is_open: bool                  # True if lot still held


class LotSelection(TypedDict):
    """Result of HIFO lot selection for a sell order."""
    selected_lots: list[dict]      # Lots chosen, each with lot_id, shares_to_sell
    total_shares: float            # Total shares selected
    total_cost_basis: float        # Total cost basis of selected shares
    estimated_gain: float          # Estimated realized gain at given price
    holding_periods: list[str]     # "short_term" or "long_term" per lot


class HarvestOpportunity(TypedDict):
    """A tax-loss harvesting opportunity."""
    symbol: str                    # Ticker with unrealized loss
    account_id: str                # Account holding the position
    unrealized_loss: float         # Total unrealized loss (negative value)
    loss_pct: float                # Loss as percentage of cost basis
    shares: float                  # Total shares held
    cost_basis: float              # Total cost basis
    current_value: float           # Current market value
    wash_sale_blocked: bool        # True if wash sale rule applies
    holding_period: str            # "short_term" or "long_term" (majority)


@dataclass
class HIFOSelector:
    """Highest-In, First-Out tax lot selector.

    Selects lots with the highest cost basis first when selling,
    which minimizes taxable gains (or maximizes deductible losses).

    Usage::

        selector = HIFOSelector()
        result = selector.select_lots(lots, shares_to_sell=50, current_price=150.0)
        # result["selected_lots"] = [{"lot_id": 3, "shares_to_sell": 30}, ...]

    Parameters:
        long_term_days: Days required for long-term capital gains treatment.
            Default is 366 (more than 1 year per IRS rules).
    """

    long_term_days: int = 366

    def _holding_period(self, acquired_at: str, sell_date: str | None = None) -> str:
        """Classify holding period as short-term or long-term.

        Parameters:
            acquired_at: Acquisition date in ISO format.
            sell_date: Sell date in ISO format. Defaults to today.

        Returns:
            "long_term" if held > long_term_days, else "short_term".
        """
        try:
            acq = date.fromisoformat(acquired_at)
        except (ValueError, TypeError):
            return "short_term"  # Unknown → assume short

        if sell_date:
            try:
                sell = date.fromisoformat(sell_date)
            except (ValueError, TypeError):
                sell = date.today()
        else:
            sell = date.today()

        days_held = (sell - acq).days
        return "long_term" if days_held >= self.long_term_days else "short_term"

    def select_lots(
        self,
        lots: list[TaxLot],
        shares_to_sell: float,
        current_price: float,
        sell_date: str | None = None,
    ) -> LotSelection:
        """Select lots using HIFO ordering.

        Parameters:
            lots: Available open lots for a given symbol/account.
            shares_to_sell: Number of shares to sell.
            current_price: Current market price per share.
            sell_date: Date of sale (ISO format). Defaults to today.

        Returns:
            LotSelection with selected lots and tax impact estimate.
        """
        if not lots or shares_to_sell <= 0:
            return LotSelection(
                selected_lots=[],
                total_shares=0.0,
                total_cost_basis=0.0,
                estimated_gain=0.0,
                holding_periods=[],
            )

        # Filter to open lots only
        open_lots = [lot for lot in lots if lot.get("is_open", True)]
        if not open_lots:
            return LotSelection(
                selected_lots=[],
                total_shares=0.0,
                total_cost_basis=0.0,
                estimated_gain=0.0,
                holding_periods=[],
            )

        # Sort by cost basis per share DESCENDING (highest first = HIFO)
        sorted_lots = sorted(
            open_lots,
            key=lambda x: x["cost_basis_per_share"],
            reverse=True,
        )

        selected = []
        remaining = shares_to_sell
        total_cost_basis = 0.0
        holding_periods = []

        for lot in sorted_lots:
            if remaining <= 0:
                break

            available = lot["shares"]
            take = min(available, remaining)

            selected.append({
                "lot_id": lot["lot_id"],
                "shares_to_sell": round(take, 6),
                "cost_basis_per_share": lot["cost_basis_per_share"],
            })

            total_cost_basis += take * lot["cost_basis_per_share"]
            holding_periods.append(
                self._holding_period(lot["acquired_at"], sell_date)
            )
            remaining -= take

        total_shares = shares_to_sell - max(remaining, 0)
        estimated_gain = total_shares * current_price - total_cost_basis

        return LotSelection(
            selected_lots=selected,
            total_shares=round(total_shares, 6),
            total_cost_basis=round(total_cost_basis, 2),
            estimated_gain=round(estimated_gain, 2),
            holding_periods=holding_periods,
        )


@dataclass
class TaxLossHarvester:
    """Tax-loss harvesting scanner.

    Scans positions for unrealized losses that exceed a threshold,
    checking the IRS 30-day wash sale rule before recommending harvest.

    Usage::

        harvester = TaxLossHarvester(loss_threshold_pct=0.02)
        opportunities = harvester.scan_opportunities(
            positions, current_prices, recent_trades
        )

    Parameters:
        loss_threshold_pct: Minimum loss percentage to trigger harvest.
            Default 0.02 (2% loss).
        wash_sale_window_days: IRS wash sale window (30 days).
    """

    loss_threshold_pct: float = 0.02
    wash_sale_window_days: int = 30

    def check_wash_sale(
        self,
        symbol: str,
        recent_trades: list[dict],
        reference_date: str | None = None,
    ) -> bool:
        """Check if selling this symbol would trigger a wash sale.

        A wash sale occurs if you buy "substantially identical" securities
        within 30 days before or after the sale.

        Parameters:
            symbol: Ticker being considered for sale.
            recent_trades: List of recent trades across ALL accounts.
                Each dict: {"symbol": str, "date": str, "action": str}
            reference_date: Date of proposed sale (ISO). Defaults to today.

        Returns:
            True if wash sale rule would be triggered.
        """
        if reference_date:
            try:
                ref = date.fromisoformat(reference_date)
            except (ValueError, TypeError):
                ref = date.today()
        else:
            ref = date.today()

        window_start = ref - timedelta(days=self.wash_sale_window_days)
        window_end = ref + timedelta(days=self.wash_sale_window_days)

        for trade in recent_trades:
            if trade.get("symbol") != symbol:
                continue
            if trade.get("action") not in ("buy", "reinvest", "transfer_in"):
                continue
            try:
                trade_date = date.fromisoformat(trade["date"])
            except (ValueError, TypeError):
                continue

            if window_start <= trade_date <= window_end:
                return True

        return False

    def scan_opportunities(
        self,
        positions: list[dict],
        current_prices: dict[str, float],
        recent_trades: list[dict] | None = None,
        reference_date: str | None = None,
    ) -> list[HarvestOpportunity]:
        """Scan positions for tax-loss harvesting opportunities.

        Parameters:
            positions: List of position dicts with:
                - symbol: str
                - account_id: str
                - shares: float
                - cost_basis_per_share: float
                - acquired_at: str (ISO date)
            current_prices: Ticker → current market price.
            recent_trades: Recent trades for wash sale check.
            reference_date: Date for calculations (ISO). Defaults to today.

        Returns:
            List of HarvestOpportunity sorted by largest loss first.
        """
        if not positions:
            return []

        trades = recent_trades or []
        opportunities: list[HarvestOpportunity] = []

        for pos in positions:
            symbol = pos.get("symbol", "")
            price = current_prices.get(symbol)
            if price is None or price <= 0:
                continue

            shares = pos.get("shares", 0)
            cost_per_share = pos.get("cost_basis_per_share", 0)
            if shares <= 0 or cost_per_share <= 0:
                continue

            cost_basis = shares * cost_per_share
            current_value = shares * price
            unrealized_loss = current_value - cost_basis

            # Only consider losses
            if unrealized_loss >= 0:
                continue

            loss_pct = abs(unrealized_loss) / cost_basis

            # Check threshold
            if loss_pct < self.loss_threshold_pct:
                continue

            # Check wash sale
            blocked = self.check_wash_sale(symbol, trades, reference_date)

            # Determine majority holding period
            acquired = pos.get("acquired_at", "")
            try:
                acq = date.fromisoformat(acquired)
                ref = date.fromisoformat(reference_date) if reference_date else date.today()
                days_held = (ref - acq).days
                period = "long_term" if days_held >= 366 else "short_term"
            except (ValueError, TypeError):
                period = "short_term"

            opportunities.append(HarvestOpportunity(
                symbol=symbol,
                account_id=pos.get("account_id", ""),
                unrealized_loss=round(unrealized_loss, 2),
                loss_pct=round(loss_pct, 4),
                shares=shares,
                cost_basis=round(cost_basis, 2),
                current_value=round(current_value, 2),
                wash_sale_blocked=blocked,
                holding_period=period,
            ))

        # Sort by largest loss (most negative first)
        opportunities.sort(key=lambda x: x["unrealized_loss"])

        return opportunities
