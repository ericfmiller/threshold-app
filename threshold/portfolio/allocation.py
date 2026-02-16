"""Alden 3-pillar allocation analysis and war chest tracking.

Computes:
  - 6-category Alden allocation (Growth, Value, Dividend, Defensive, Cyclical, Hard Assets)
  - Dollar-weighted and count-based allocation breakdowns
  - War chest status (VIX-regime-targeted cash reserves)
  - Gap analysis vs. model portfolio targets
"""

from __future__ import annotations

from dataclasses import dataclass, field

from threshold.config.schema import ThresholdConfig

# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass
class AldenAllocation:
    """Allocation result for one Alden category."""

    name: str
    tickers: list[str] = field(default_factory=list)
    dollar_value: float = 0.0
    weight_pct: float = 0.0
    target_low: float = 0.0
    target_high: float = 0.0
    gap: float = 0.0
    """Positive = over-allocated, negative = under-allocated vs midpoint."""

    @property
    def on_target(self) -> bool:
        return self.target_low <= self.weight_pct <= self.target_high


@dataclass
class WarChestStatus:
    """War chest (cash + near-cash) status relative to VIX-regime target."""

    vix_regime: str = "NORMAL"
    target_pct: float = 0.10
    target_dollars: float = 0.0
    actual_dollars: float = 0.0
    actual_pct: float = 0.0
    fidelity_cash: float = 0.0
    cash_balances: dict[str, float] = field(default_factory=dict)
    wc_instruments: list[str] = field(default_factory=list)
    wc_instrument_value: float = 0.0
    total_portfolio: float = 0.0

    @property
    def surplus(self) -> float:
        """Positive = above target, negative = below."""
        return self.actual_dollars - self.target_dollars

    @property
    def is_adequate(self) -> bool:
        return self.actual_dollars >= self.target_dollars


@dataclass
class AllocationReport:
    """Full allocation analysis result."""

    categories: dict[str, AldenAllocation] = field(default_factory=dict)
    """category_name → AldenAllocation."""
    war_chest: WarChestStatus = field(default_factory=WarChestStatus)
    total_portfolio: float = 0.0
    equities_pct: float = 0.0
    hard_money_pct: float = 0.0
    cash_pct: float = 0.0


# ---------------------------------------------------------------------------
# Allocation calculator
# ---------------------------------------------------------------------------

def compute_alden_allocation(
    ticker_categories: dict[str, str],
    ticker_values: dict[str, float],
    tsp_value: float = 0.0,
    btc_value: float = 0.0,
    total_portfolio: float = 0.0,
    config: ThresholdConfig | None = None,
) -> AllocationReport:
    """Compute Alden 6-category allocation breakdown.

    Parameters:
        ticker_categories: {ticker: alden_category_name} mapping.
        ticker_values: {ticker: dollar_value} for each held ticker.
        tsp_value: Total TSP value (allocated across categories via config).
        btc_value: Total BTC value (added to Hard Assets).
        total_portfolio: Total portfolio value for weight calculations.
        config: ThresholdConfig for category targets and TSP allocation.

    Returns:
        AllocationReport with per-category breakdown and gap analysis.
    """
    if config is None:
        from threshold.config.schema import ThresholdConfig
        config = ThresholdConfig()

    categories: dict[str, AldenAllocation] = {}

    # Initialize categories from config
    for cat_name, cat_cfg in config.alden_categories.items():
        categories[cat_name] = AldenAllocation(
            name=cat_name,
            target_low=cat_cfg.target[0],
            target_high=cat_cfg.target[1],
        )

    # Assign tickers to categories
    for ticker, cat_name in ticker_categories.items():
        value = ticker_values.get(ticker, 0.0)
        if cat_name not in categories:
            # Create a catch-all if category not in config
            categories[cat_name] = AldenAllocation(
                name=cat_name,
                target_low=0.0,
                target_high=1.0,
            )
        categories[cat_name].tickers.append(ticker)
        categories[cat_name].dollar_value += value

    # Add TSP allocation per category
    if tsp_value > 0:
        for cat_name, cat_cfg in config.alden_categories.items():
            if cat_cfg.tsp_pct > 0 and cat_name in categories:
                categories[cat_name].dollar_value += tsp_value * cat_cfg.tsp_pct

    # Add BTC to Hard Assets
    if btc_value > 0 and "Hard Assets" in categories:
        categories["Hard Assets"].dollar_value += btc_value

    # Compute weights and gaps
    portfolio_total = total_portfolio if total_portfolio > 0 else sum(
        c.dollar_value for c in categories.values()
    )

    for cat in categories.values():
        if portfolio_total > 0:
            cat.weight_pct = cat.dollar_value / portfolio_total
        midpoint = (cat.target_low + cat.target_high) / 2
        cat.gap = cat.weight_pct - midpoint

    # Compute 3-pillar summary
    equities_value = sum(
        c.dollar_value for name, c in categories.items()
        if name not in ("Hard Assets", "Defensive/Income")
    )
    hard_money_value = categories.get("Hard Assets", AldenAllocation(name="")).dollar_value
    cash_value = categories.get("Defensive/Income", AldenAllocation(name="")).dollar_value

    report = AllocationReport(
        categories=categories,
        total_portfolio=portfolio_total,
        equities_pct=equities_value / portfolio_total if portfolio_total > 0 else 0.0,
        hard_money_pct=hard_money_value / portfolio_total if portfolio_total > 0 else 0.0,
        cash_pct=cash_value / portfolio_total if portfolio_total > 0 else 0.0,
    )
    return report


# ---------------------------------------------------------------------------
# War chest calculator
# ---------------------------------------------------------------------------

def compute_war_chest(
    vix_regime: str,
    fidelity_cash: float = 0.0,
    cash_balances: dict[str, float] | None = None,
    wc_instrument_values: dict[str, float] | None = None,
    total_portfolio: float = 0.0,
    config: ThresholdConfig | None = None,
) -> WarChestStatus:
    """Compute war chest status vs VIX-regime target.

    Parameters:
        vix_regime: Current VIX regime (COMPLACENT/NORMAL/FEAR/PANIC).
        fidelity_cash: Total cash across Fidelity accounts.
        cash_balances: Per-account cash balances.
        wc_instrument_values: {ticker: dollar_value} for war chest instruments.
        total_portfolio: Total portfolio value.
        config: ThresholdConfig for VIX targets.

    Returns:
        WarChestStatus with target vs actual breakdown.
    """
    if config is None:
        from threshold.config.schema import ThresholdConfig
        config = ThresholdConfig()

    # Get VIX-regime target
    vix_targets = config.allocation.war_chest_vix
    target_pct = vix_targets.get(vix_regime, 0.10)

    # Compute actual war chest
    wc_values = wc_instrument_values or {}
    wc_total = sum(wc_values.values())

    actual = fidelity_cash + wc_total
    target_dollars = total_portfolio * target_pct if total_portfolio > 0 else 0.0

    return WarChestStatus(
        vix_regime=vix_regime,
        target_pct=target_pct,
        target_dollars=round(target_dollars, 2),
        actual_dollars=round(actual, 2),
        actual_pct=actual / total_portfolio if total_portfolio > 0 else 0.0,
        fidelity_cash=round(fidelity_cash, 2),
        cash_balances=cash_balances or {},
        wc_instruments=list(wc_values.keys()),
        wc_instrument_value=round(wc_total, 2),
        total_portfolio=round(total_portfolio, 2),
    )
