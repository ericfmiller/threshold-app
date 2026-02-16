"""All backtest-calibrated default values for the scoring engine.

These values were calibrated through Phase 2 backtesting:
- 124,073 observations, 625 tickers, Oct 2007 - Dec 2025
- Walk-forward validation: calibration 2007-2019, validation 2020-2025

Do not change these without presenting evidence that the change improves outcomes.
"""

# ---------------------------------------------------------------------------
# DCS Sub-Score Weights (must sum to 100)
# ---------------------------------------------------------------------------
DCS_WEIGHTS = {
    "MQ": 30,  # Momentum Quality
    "FQ": 25,  # Fundamental Quality
    "TO": 20,  # Technical Oversold
    "MR": 15,  # Market Regime
    "VC": 10,  # Valuation Context
}

# ---------------------------------------------------------------------------
# MQ Internal Weights (must sum to 1.0)
# ---------------------------------------------------------------------------
MQ_WEIGHTS = {
    "trend": 0.30,           # 50d/200d SMA trend classifier
    "vol_adj_momentum": 0.25,  # Barroso & Santa-Clara 2015
    "sa_momentum": 0.25,     # SA Momentum grade
    "relative_strength": 0.20,  # Antonacci dual momentum
}

# ---------------------------------------------------------------------------
# FQ Internal Weights (4 paths based on data availability)
# ---------------------------------------------------------------------------
FQ_WEIGHTS = {
    "with_yf_and_revmom": {
        "quant": 0.30,
        "prof_blended": 0.22,
        "fcf_yield": 0.13,
        "rev_momentum": 0.15,
        "revisions": 0.10,
        "growth": 0.10,
    },
    "with_yf_only": {
        "quant": 0.30,
        "prof_blended": 0.22,
        "fcf_yield": 0.13,
        "revisions": 0.20,
        "growth": 0.15,
    },
    "with_revmom_only": {
        "quant": 0.35,
        "profitability": 0.25,
        "rev_momentum": 0.15,
        "revisions": 0.15,
        "growth": 0.10,
    },
    "base": {
        "quant": 0.35,
        "profitability": 0.25,
        "revisions": 0.25,
        "growth": 0.15,
    },
}

# ---------------------------------------------------------------------------
# TO Internal Weights
# ---------------------------------------------------------------------------
TO_WEIGHTS = {
    "rsi": 0.35,
    "sma_distance": 0.25,
    "bollinger": 0.25,
    "macd": 0.15,
}

# ---------------------------------------------------------------------------
# MR Internal Weights
# ---------------------------------------------------------------------------
MR_WEIGHTS = {
    "vix_contrarian": 0.50,
    "spy_trend": 0.30,
    "breadth": 0.20,
}

# ---------------------------------------------------------------------------
# VC Internal Weights
# ---------------------------------------------------------------------------
VC_WEIGHTS = {
    "sa_value": 0.65,
    "ev_ebitda_sector": 0.35,
}

# ---------------------------------------------------------------------------
# Profitability Blending (SA grade vs Novy-Marx gross profitability)
# ---------------------------------------------------------------------------
PROFITABILITY_BLEND = {
    "sa_weight": 0.60,
    "novy_marx_weight": 0.40,
}

# ---------------------------------------------------------------------------
# DCS Signal Thresholds (Phase 2 backtest-calibrated)
# ---------------------------------------------------------------------------
SIGNAL_THRESHOLDS = {
    "strong_buy_dip": 80,   # 53.8% win rate (13 signals)
    "high_conviction": 70,  # 60.0% win rate
    "buy_dip": 65,          # 58.9% win rate, CI [57.7%-60.1%], walk-forward stable
    "watch": 50,            # 57.5% win rate
    "weak": 35,
}

# ---------------------------------------------------------------------------
# Post-Composition Modifiers
# ---------------------------------------------------------------------------
MODIFIERS = {
    "obv_bullish_max": 5,        # OBV bullish divergence boost cap
    "rsi_divergence_boost": 3,   # RSI divergence DCS boost
    "rsi_divergence_min_dcs": 60,  # Only boost when DCS >= this
}

# ---------------------------------------------------------------------------
# VIX Regime Boundaries
# ---------------------------------------------------------------------------
VIX_REGIMES = {
    "COMPLACENT": [0, 14],
    "NORMAL": [14, 20],
    "FEAR": [20, 28],
    "PANIC": [28, 999],
}

# ---------------------------------------------------------------------------
# Trend Score Classifier
# ---------------------------------------------------------------------------
TREND_SCORES = {
    "uptrend_pullback": 1.0,  # sma50 > sma200, price > sma200
    "uptrend_break": 0.5,     # sma50 > sma200, price <= sma200
    "recovery": 0.4,          # sma50 <= sma200, price > sma200
    "downtrend": 0.1,         # both SMAs above price
}

# ---------------------------------------------------------------------------
# Falling Knife DCS Caps (defense-aware)
# ---------------------------------------------------------------------------
FALLING_KNIFE_CAPS = {
    "freefall": {  # trend_score <= 0.1
        "HEDGE": 50,
        "DEFENSIVE": 45,
        "MODERATE": 30,
        "CYCLICAL": 20,
        "AMPLIFIER": 15,
    },
    "downtrend": {  # 0.1 < trend_score <= 0.4
        "HEDGE": 70,
        "DEFENSIVE": 60,
        "MODERATE": 50,
        "CYCLICAL": 40,
        "AMPLIFIER": 30,
    },
}

# ---------------------------------------------------------------------------
# Drawdown Defense DCS Modifiers (Rule D-5, FEAR/PANIC only)
# ---------------------------------------------------------------------------
DRAWDOWN_DCS_MODIFIERS = {
    "HEDGE": 5,       # DC < 0: gains when SPY falls
    "DEFENSIVE": 3,   # DC 0-0.60: loses much less than SPY
    "MODERATE": 0,    # DC 0.60-1.0: roughly tracks SPY
    "CYCLICAL": -3,   # DC 1.0-1.5: amplifies modestly
    "AMPLIFIER": -5,  # DC > 1.5: amplifies severely
}

# ---------------------------------------------------------------------------
# Allocation Framework
# ---------------------------------------------------------------------------
ALLOCATION_TARGETS = {
    "equities": 0.70,
    "hard_money": 0.20,
    "cash": 0.10,
}

INTERNATIONAL_RANGE = [0.21, 0.35]
SECTOR_CONCENTRATION_LIMIT = 0.30

DEFENSIVE_TARGET_BY_BUBBLE = {
    "0-2": 0.05,   # Lean aggressive
    "3-4": 0.10,   # Normal posture
    "5-7": 0.20,   # Maximum defense
}

DEFENSIVE_FLOOR_BY_REGIME = {
    "Risk-On Growth": 0.05,
    "Mixed/Transitional": 0.08,
    "Inflationary": 0.10,
    "Debasement": 0.10,
    "Deflationary/Risk-Off": 0.15,
}

WAR_CHEST_VIX_TARGETS = {
    "COMPLACENT": 0.10,
    "NORMAL": 0.12,
    "FEAR": 0.15,
    "PANIC": 0.20,
}

# ---------------------------------------------------------------------------
# Sell Criteria
# ---------------------------------------------------------------------------
SELL_CRITERIA = {
    "sma_breach_days": 10,
    "sma_breach_warning_days": 7,
    "sma_breach_threshold": -0.03,
    "quant_drop_threshold": -1.0,
    "quant_drop_lookback_days": 35,
}

# ---------------------------------------------------------------------------
# Revision Momentum
# ---------------------------------------------------------------------------
REVISION_MOMENTUM = {
    "min_history_weeks": 4,
    "min_calendar_days": 21,
    "sell_threshold_subgrades": 3,
    "warning_threshold_subgrades": 2,
}

# ---------------------------------------------------------------------------
# Gate 3 Deployment
# ---------------------------------------------------------------------------
DEPLOYMENT = {
    "gate3_rsi_max": 80,
    "gate3_ret_8w_max": 0.30,
    "gold_rsi_max_sizing": 0.75,
}

# ---------------------------------------------------------------------------
# Alert Thresholds
# ---------------------------------------------------------------------------
ALERT_THRESHOLDS = {
    "dcs_conviction": 70,
    "dcs_strong": 80,
}

# ---------------------------------------------------------------------------
# Data Validation
# ---------------------------------------------------------------------------
DATA_VALIDATION = {
    "min_data_points": 50,
    "preferred_data_points": 200,
    "stale_gap_days": 5,
    "extreme_move_threshold": 0.50,
}

# ---------------------------------------------------------------------------
# Grade Mappings
# ---------------------------------------------------------------------------
GRADE_TO_NUM = {
    "A+": 13, "A": 12, "A-": 11,
    "B+": 10, "B": 9, "B-": 8,
    "C+": 7, "C": 6, "C-": 5,
    "D+": 4, "D": 3, "D-": 2,
    "F": 1,
}

VALIDATION_GRADE_TO_NUM = {
    "A+": 5.0, "A": 4.5, "A-": 4.0,
    "B+": 3.5, "B": 3.0, "B-": 2.5,
    "C+": 2.0, "C": 1.5, "C-": 1.0,
    "D+": 0.5, "D": 0.0, "D-": -0.5,
    "F": -1.0,
}

# ---------------------------------------------------------------------------
# ETF-Specific Grade Column Names
# ---------------------------------------------------------------------------
ETF_GRADE_COLS = {
    "etf_momentum": "ETF Momentum",
    "etf_expenses": "ETF Expenses",
    "etf_dividends": "ETF Dividends",
    "etf_risk": "ETF Risk",
    "etf_liquidity": "ETF Liquidity",
}

# ---------------------------------------------------------------------------
# Sector ETFs (for rotation analysis)
# ---------------------------------------------------------------------------
SECTOR_ETFS = {
    "Technology": "XLK",
    "Energy": "XLE",
    "Financials": "XLF",
    "Healthcare": "XLV",
    "Industrials": "XLI",
    "Consumer Staples": "XLP",
    "Utilities": "XLU",
    "Materials": "XLB",
    "Real Estate": "XLRE",
    "Communication Services": "XLC",
    "Consumer Discretionary": "XLY",
}

CROSS_ASSET_ETFS = {
    "US Equities": "SPY",
    "Intl Equities": "EFA",
    "Gold": "GLD",
    "Bonds": "BND",
    "Commodities": "GSG",
    "Bitcoin": "FBTC",
}

# ---------------------------------------------------------------------------
# Sector-Specific DCS Thresholds (Phase 2 backtest)
# ---------------------------------------------------------------------------
SECTOR_DCS_THRESHOLDS = {
    "Energy": 70,
    "Real Estate": 70,
    "Consumer Staples": 70,
    "Communication Services": 70,
    "Information Technology": 75,
    "Industrials": 75,
    "Consumer Discretionary": 65,
    "Financials": 55,
    "Health Care": 70,
    "Materials": 55,
    "Utilities": 60,
}

# ---------------------------------------------------------------------------
# FRED Series for Macro Regime Monitoring
# ---------------------------------------------------------------------------
FRED_SERIES = {
    "T10Y2Y": "Yield Curve (10Y-2Y spread)",
    "T10Y3M": "Yield Curve (10Y-3M spread)",
    "BAMLH0A0HYM2": "HY Credit Spread (ICE BofA)",
    "DFF": "Fed Funds Rate",
    "WALCL": "Fed Balance Sheet (Total Assets)",
    "CPIAUCSL": "CPI (All Urban)",
}

# ---------------------------------------------------------------------------
# Alden Category Allocation Targets
# ---------------------------------------------------------------------------
ALDEN_CATEGORIES = {
    "US Large Cap": {"target": [0.20, 0.35], "tsp_pct": 0.24},
    "US Small/Mid": {"target": [0.05, 0.15], "tsp_pct": 0.25},
    "Intl Developed": {"target": [0.10, 0.20], "tsp_pct": 0.51},
    "Emerging Markets": {"target": [0.05, 0.15]},
    "Hard Assets": {"target": [0.15, 0.25]},
    "Defensive/Income": {"target": [0.05, 0.15], "cross_cutting": True},
    "Other": {"target": [0.00, 0.05], "is_catchall": True},
}

# ---------------------------------------------------------------------------
# Risk Framework Defaults (Phase 2B — all disabled by default)
# ---------------------------------------------------------------------------
RISK_EBP = {
    "enabled": False,
    "high_risk_threshold": 1.00,   # >100bp
    "elevated_threshold": 0.50,    # >50bp
    "normal_threshold": 0.00,      # >0bp
    "lookback_months": 3,
}

RISK_TURBULENCE = {
    "enabled": False,
    "window": 252,                 # 1-year rolling window
    "threshold_pctl": 0.75,        # 75th percentile = turbulent
    "min_assets": 3,
}

RISK_MOMENTUM_CRASH = {
    "enabled": False,
    "lookback_months": 24,         # Daniel-Moskowitz bear indicator period
    "crash_threshold": 0.02,       # Variance threshold for crash regime
    "min_weight": 0.25,            # Floor for momentum weight multiplier
}

RISK_CVAR = {
    "enabled": False,
    "alpha": 0.95,                 # 95% confidence level
    "method": "historical",        # "historical" or "parametric"
}

RISK_CDAR = {
    "enabled": False,
    "alpha": 0.95,                 # α→1 = MaxDD, α→0 = AvgDD
}

# ---------------------------------------------------------------------------
# Data Source Defaults
# ---------------------------------------------------------------------------
YFINANCE_DEFAULTS = {
    "price_period": "2y",
    "fundamentals_delay": 0.3,
}

TIINGO_DEFAULTS = {
    "base_url": "https://api.tiingo.com/tiingo/daily",
    "rate_delay": 1.5,
}

SA_DEFAULTS = {
    "api_url": "https://seekingalpha.com/api/v3/symbols/{symbol}/ratings",
    "stale_threshold_days": 2,
}
