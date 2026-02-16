"""Unified portfolio position and value accessor.

Provides a single-source-of-truth interface for portfolio data:
  - Account weights per ticker (what % of each account is in each ticker)
  - Dollar values per ticker across all accounts
  - Cash balances (money market funds)
  - Total portfolio value breakdown (Fidelity, TSP, BTC, separate holdings)

Designed to work with SQLite-backed positions from ``storage.queries``
and config-driven account definitions.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from threshold.portfolio.accounts import PortfolioSnapshot


@dataclass
class PortfolioValues:
    """Breakdown of total portfolio values across all sources."""

    fidelity_total: float = 0.0
    tsp_value: float = 0.0
    btc_value: float = 0.0
    separate_holdings_value: float = 0.0
    total_portfolio: float = 0.0
    account_values: dict[str, float] = field(default_factory=dict)
    """account_id → dollar value for each Fidelity account."""
    cash_balances: dict[str, float] = field(default_factory=dict)
    """account_id → money-market (SPAXX/FDRXX) balance."""

    @property
    def fidelity_cash_total(self) -> float:
        """Total cash across all Fidelity accounts."""
        return sum(self.cash_balances.values())


class PortfolioLedger:
    """Unified portfolio data accessor.

    Wraps a :class:`PortfolioSnapshot` (positions) and a
    :class:`PortfolioValues` (dollar values) into a single convenient
    interface used throughout the scoring pipeline.

    Usage::

        ledger = PortfolioLedger(snapshot=snapshot, values=values)
        # Dollar value of AAPL across all accounts
        aapl_value = ledger.ticker_dollar_value("AAPL")
        # All held tickers
        for ticker in ledger.held_tickers:
            ...
    """

    def __init__(
        self,
        snapshot: PortfolioSnapshot | None = None,
        values: PortfolioValues | None = None,
    ) -> None:
        self._snapshot = snapshot or PortfolioSnapshot()
        self._values = values or PortfolioValues()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def snapshot(self) -> PortfolioSnapshot:
        return self._snapshot

    @property
    def values(self) -> PortfolioValues:
        return self._values

    @property
    def held_tickers(self) -> set[str]:
        """Set of all tickers held across all accounts."""
        return set(self._snapshot.positions.keys())

    @property
    def total_portfolio(self) -> float:
        return self._values.total_portfolio

    @property
    def fidelity_total(self) -> float:
        return self._values.fidelity_total

    @property
    def tsp_value(self) -> float:
        return self._values.tsp_value

    @property
    def btc_value(self) -> float:
        return self._values.btc_value

    @property
    def account_values(self) -> dict[str, float]:
        return self._values.account_values

    @property
    def cash_balances(self) -> dict[str, float]:
        return self._values.cash_balances

    @property
    def fidelity_cash_total(self) -> float:
        return self._values.fidelity_cash_total

    # ------------------------------------------------------------------
    # Computed accessors
    # ------------------------------------------------------------------

    def ticker_dollar_value(self, ticker: str) -> float:
        """Total dollar value of *ticker* across all accounts.

        Computes as: sum(weight_in_account × account_value) for each
        account that holds the ticker.
        """
        pos = self._snapshot.get_position(ticker)
        if pos is None:
            return 0.0
        # If we have direct account values from the snapshot, use them
        if pos.account_values:
            return sum(pos.account_values.values())
        # Otherwise compute from weights × account totals
        total = 0.0
        for acct_id, weight in pos.account_weights.items():
            acct_val = self._values.account_values.get(acct_id, 0.0)
            total += weight * acct_val
        return total

    def category_dollar_value(
        self,
        tickers: set[str] | list[str],
        tsp_pct: float = 0.0,
        include_btc: bool = False,
    ) -> float:
        """Total dollar value for a group of tickers.

        Parameters:
            tickers: Tickers to sum.
            tsp_pct: Fraction of TSP to include (0-1).
            include_btc: Whether to include BTC value.

        Returns:
            Dollar value for the category.
        """
        total = sum(self.ticker_dollar_value(t) for t in tickers)
        if tsp_pct > 0:
            total += self._values.tsp_value * tsp_pct
        if include_btc:
            total += self._values.btc_value
        return total

    def held_in_accounts(self, ticker: str) -> list[str]:
        """Account IDs where *ticker* is held."""
        pos = self._snapshot.get_position(ticker)
        if pos is None:
            return []
        return list(pos.account_shares.keys())

    def is_held(self, ticker: str) -> bool:
        """Check if *ticker* is held in any account."""
        return ticker in self._snapshot.positions

    def tickers_in_account(self, account_id: str) -> set[str]:
        """All tickers held in a specific account."""
        result: set[str] = set()
        for symbol, pos in self._snapshot.positions.items():
            if account_id in pos.account_shares:
                result.add(symbol)
        return result

    def portfolio_weight(self, ticker: str) -> float:
        """Portfolio-level weight for a ticker (0-1)."""
        return self._snapshot.portfolio_weight(ticker)
