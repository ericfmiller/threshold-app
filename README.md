# Threshold

[![CI](https://github.com/ericfmiller/threshold-app/actions/workflows/ci.yml/badge.svg)](https://github.com/ericfmiller/threshold-app/actions/workflows/ci.yml)

Quantitative investment analysis system with an empirically calibrated scoring engine.

Threshold enforces disciplined decision-making through a structured hierarchy: every investment decision passes through empirically calibrated thresholds — buy thresholds, sell thresholds, regime thresholds, deployment gates — before action is taken.

## Quick Start

```bash
# Install
pip install -e .

# Initialize (creates ~/.threshold/, database, example config)
threshold init

# Register tickers
threshold ticker add AAPL
threshold ticker add COPJ
threshold ticker list

# Score your portfolio
threshold score

# Generate dashboard and narrative report
threshold dashboard
threshold narrative
```

## Architecture

Threshold is organized around the **Decision Hierarchy** — the order in which investment decisions should be made:

1. **Macro Regime** — What environment are we in? (VIX classification, SPY trend, breadth)
2. **Sector Rotation** — Where should capital flow? (Relative Rotation Graphs)
3. **Allocation & War Chest** — Are we properly positioned? (Alden 3-pillar, VIX-regime cash targets)
4. **Drawdown Defense** — How exposed are we? (15-year backtest classifications, D-5 regime modifier)
5. **Deployment Discipline** — Three gates before deploying cash
6. **Selection** — DCS scoring, buy/sell criteria, reversal signals
7. **Behavioral** — Process over emotion (Housel framework)

### Module Structure

```
threshold/
    config/          # Pydantic schema, YAML loader, calibrated defaults
    data/adapters/   # yfinance (+ future Tiingo, SA, FRED)
    storage/         # SQLite (WAL mode), migrations, query functions
    engine/          # DCS scoring engine
        technical.py     # RSI, MACD, OBV, Bollinger, reversals
        subscores.py     # MQ, FQ, TO, MR, VC calculators
        composite.py     # DCS composition, falling knife, modifiers
        signals.py       # SignalBoard with 11 signal factories
        scorer.py        # score_ticker() orchestrator
        pipeline.py      # Full scoring pipeline
        risk/            # EBP, turbulence, momentum crash, CVaR, CDaR
        advanced/        # Trend following, factor momentum, sentiment
        portfolio/       # Inverse vol, HRP, HIFO tax lots
        aggregator.py    # Cross-module risk overlay
    portfolio/       # Account management, allocation, correlation
    output/          # Dashboard, narrative, alerts, charts
    cli/             # Click commands
    migrations/      # SQL schema migrations
```

## DCS Scoring Engine

The Dip-Buying Composite Score (DCS) is a 0-100 composite calibrated across 124,000+ observations and 625 tickers (2007-2025):

| Sub-Score | Weight | What It Measures |
|-----------|--------|-----------------|
| MQ (Momentum Quality) | 30% | Trend regime, vol-adjusted momentum, relative strength |
| FQ (Fundamental Quality) | 25% | Quant rating, profitability, FCF yield, revision momentum |
| TO (Technical Oversold) | 20% | RSI, SMA distance, Bollinger Band position, MACD |
| MR (Market Regime) | 15% | VIX contrarian signal, SPY trend, market breadth |
| VC (Valuation Context) | 10% | Value grade, sector-relative EV/EBITDA |

### Signal Thresholds

| DCS | Signal | Win Rate | Action |
|-----|--------|----------|--------|
| >= 80 | STRONG BUY DIP | 53.8% | Full size + lean in |
| >= 70 | HIGH CONVICTION | 60.0% | Full size deployment |
| >= 65 | BUY DIP | 58.9% | Standard deployment |
| 50-64 | WATCH | 57.5% | Monitor |
| < 50 | WEAK/AVOID | - | Stay away |

### Post-Composition Modifiers

- **OBV Divergence Boost**: +2 DCS when bullish OBV divergence detected
- **RSI Bullish Divergence**: +3 DCS when price lower low + RSI higher low (walk-forward stable)
- **Falling Knife Filter**: Defense-aware caps for tickers in steep downtrends
- **D-5 Regime Modifier**: In FEAR/PANIC, HEDGE +5 / DEFENSIVE +3 / CYCLICAL -3 / AMPLIFIER -5

### Advanced Modules (opt-in, disabled by default)

- **Risk Framework**: EBP, turbulence index, momentum crash protection, CVaR, CDaR
- **Trend Following**: Baltas-Kosowski continuous trend signal (blends into MQ)
- **Factor Momentum**: Ehsani-Linnainmaa cross-factor momentum (informational overlay)
- **Aligned Sentiment**: Huang et al. PLS sentiment (adjusts MR when overheated)
- **Portfolio Construction**: Inverse volatility, HRP, HIFO tax-lot selection
- **Signal Aggregation**: Composite risk overlay from multiple risk modules

## CLI Commands

```
threshold init              # Create dirs, database, example config
threshold config show       # Print resolved configuration
threshold config validate   # Validate config against schema

threshold ticker add AAPL   # Register ticker (auto-enriches via yfinance)
threshold ticker list       # List all registered tickers
threshold ticker info AAPL  # Show full metadata
threshold ticker remove XYZ # Soft-delete ticker

threshold score             # Run full scoring pipeline
threshold score --ticker AAPL  # Score single ticker
threshold score --dry-run   # Score without persisting

threshold dashboard         # Generate HTML dashboard and open in browser
threshold dashboard --no-open  # Generate without opening
threshold narrative         # Generate Markdown narrative report

threshold import registry <path>     # Import ticker_registry.json
threshold import scores <dir>        # Import weekly_scores JSON files
threshold import drawdown <path>     # Import drawdown classifications
```

## Configuration

All settings live in `~/.threshold/config.yaml`. See `config.yaml.example` for the complete reference with inline documentation.

Key configuration sections:
- **scoring**: DCS weights, thresholds, signal boundaries
- **allocation**: Alden 3-pillar targets, rebalance triggers
- **war_chest**: VIX-regime cash targets
- **alerts**: Email notifications, DCS alert thresholds
- **risk/advanced**: Toggle-able research-backed modules
- **output**: Directory paths for dashboards, narratives, score history

## Data Sources

- **yfinance** (required) — Price data, technicals, fundamentals, ticker enrichment
- **Seeking Alpha** (recommended) — Quant ratings, factor grades
- **Tiingo** (optional) — Fallback price data
- **FRED** (optional) — Macroeconomic data (VIX, yield curves, breadth)

## Development

```bash
# Install with dev dependencies
pip install -e ".[dev]"

# Run tests (405+ tests, <5 seconds)
pytest

# Run with verbose output
pytest -v

# Type checking
mypy threshold/

# Linting
ruff check threshold/
```

### Test Organization

| File | Tests | What |
|------|-------|------|
| test_technical.py | 27 | RSI, MACD, OBV, Bollinger, reversals |
| test_subscores.py | 30 | MQ, FQ, TO, MR, VC calculators |
| test_composite.py | 43 | DCS composition, modifiers, classifiers |
| test_signals.py | 30 | SignalBoard, 11 factories, net_action |
| test_scorer.py | 22 | score_ticker() integration |
| test_risk.py | 31 | EBP, turbulence, crash, CVaR, CDaR |
| test_advanced.py | 20 | Trend following, sentiment, factor momentum |
| test_aggregator.py | 14 | Composite risk overlay |
| test_portfolio_construction.py | 26 | Inverse vol, HRP, HIFO |
| test_pipeline.py | 53 | Portfolio, allocation, correlation, alerts |
| test_output.py | 67 | Charts, dashboard, narrative |
| test_regression.py | 25+ | End-to-end DCS golden ranges |
| test_data_adapters.py | 40+ | yfinance adapter, classification |
| test_properties.py | 30+ | Hypothesis property-based tests |
| test_config.py | 17 | Config loading, validation |
| test_storage.py | 25 | Database CRUD, migrations |

## License

MIT
