# Threshold

[![CI](https://github.com/ericfmiller/threshold-app/actions/workflows/ci.yml/badge.svg)](https://github.com/ericfmiller/threshold-app/actions/workflows/ci.yml)

Quantitative investment analysis system with an empirically calibrated scoring engine, interactive HTML dashboard, and actionable Markdown narrative reports.

Threshold enforces disciplined decision-making through a structured hierarchy: every investment decision passes through empirically calibrated thresholds before action is taken. The name reflects this philosophy -- buy thresholds, sell thresholds, regime thresholds, deployment gates.

## Quick Start

```bash
# Clone and install
git clone https://github.com/ericfmiller/threshold-app.git
cd threshold-app
python3 -m venv .venv && source .venv/bin/activate
pip install -e .

# Initialize (creates ~/.threshold/, database, example config)
threshold init

# Edit your config
cp config.yaml.example ~/.threshold/config.yaml
# Then edit ~/.threshold/config.yaml with your accounts, API keys, etc.

# Register tickers
threshold ticker add AAPL
threshold ticker add COPJ

# Score your portfolio
threshold score

# Generate outputs
threshold dashboard    # Interactive HTML dashboard
threshold narrative    # Actionable Markdown report

# Or run the full workflow in one command
threshold sync
```

## What It Does

Threshold takes your portfolio holdings (from Seeking Alpha Excel exports), watchlist tickers, and market data to produce:

1. **DCS Scores** -- A 0-100 Dip-Buying Composite Score for every ticker, calibrated on 124K+ observations across 625 tickers (2007-2025)
2. **Buy/Sell Signals** -- Statistically validated signal thresholds with walk-forward tested win rates
3. **Interactive Dashboard** -- 12-section HTML dashboard following the Decision Hierarchy
4. **Narrative Report** -- 21-section Markdown report with actionable analysis

### Dashboard Sections

| Section | What It Shows |
|---------|--------------|
| Macro Regime | VIX classification, SPY trend, market breadth KPIs |
| Allocation & War Chest | Asset allocation gauge, VIX-regime cash targets |
| Drawdown Defense | Count + dollar-weighted defensive composition bars |
| Deployment Discipline | 3-gate system: buy criteria, VIX sizing, parabolic filter |
| Selection | DCS vs RSI scatter (holdings/watchlist split), signal cards |
| Sell Alerts | Red/yellow severity cards for flagged holdings |
| Holdings Health | Per-account tabbed view: ticker, weight, DCS, signal, flags |
| Watchlist | Ranked watchlist with "ready to deploy" highlights |
| Sector Exposure | Treemap with 25% concentration warnings |
| Correlation | Heatmap with high-correlation pair alerts |
| Behavioral | Housel-framework checklists (pre-buy, pre-sell, "never enough") |

### Narrative Sections

The Markdown narrative includes 21 sections: macro backdrop, dip-buy opportunities (split by holdings vs watchlist), deployment gates, falling knives, hedges/defensives in downtrend, watch zone, crypto/BTC analysis, reversal signals, sub-score driver analysis, relative strength vs SPY, EPS revision momentum, OBV divergence analysis, sell criteria, grace periods, exemptions, drawdown defense (count + dollar-weighted), correlation, sector exposure, war chest status, per-account holdings health, and a quick reference summary.

## Architecture

Organized around the **Decision Hierarchy** -- the order in which investment decisions should be made:

1. **Macro Regime** -- What environment are we in? (VIX, SPY trend, breadth)
2. **Allocation & War Chest** -- Are we properly positioned? (Alden 3-pillar, VIX-regime cash targets)
3. **Drawdown Defense** -- How exposed are we? (15-year backtest classifications, D-5 regime modifier)
4. **Deployment Discipline** -- Three gates before deploying cash
5. **Selection** -- DCS scoring, buy/sell criteria, reversal signals
6. **Behavioral** -- Process over emotion (Housel framework)

### Module Structure

```
threshold/
    config/          # Pydantic schema, YAML loader, calibrated defaults
    data/
        adapters/    # SA export reader, yfinance, Tiingo, FRED
        onboarding.py    # Auto-detect and register new tickers
        position_import.py   # Import positions from SA exports + synthetic (TSP/BTC)
        snapshot.py      # Portfolio snapshot generation
        watcher.py       # File system watcher for new exports
    storage/         # SQLite (WAL mode), migrations, query functions
    engine/          # DCS scoring engine
        technical.py     # RSI, MACD, OBV, Bollinger, reversals
        subscores.py     # MQ, FQ, TO, MR, VC calculators
        composite.py     # DCS composition, falling knife, modifiers
        signals.py       # SignalBoard with 11 signal factories
        scorer.py        # score_ticker() orchestrator
        pipeline.py      # Full scoring pipeline with correlation analysis
        exemptions.py    # Crypto halving cycle, cash exemptions
        grace_period.py  # 180d/270d hold windows for weakening positions
        gate3.py         # Parabolic deployment filter
        drawdown_backtest.py  # 15-year downside capture classification
        risk/            # EBP, turbulence, momentum crash, CVaR, CDaR
        advanced/        # Trend following, factor momentum, sentiment
        portfolio/       # Inverse vol, HRP, HIFO tax lots
    portfolio/       # Account management, allocation, correlation, watchlists
    output/
        dashboard.py     # Full HTML dashboard generator
        narrative.py     # 21-section Markdown narrative
        charts.py        # Plotly charts (DCS scatter, drawdown bars, etc.)
        alerts.py        # Email alerts, score history persistence
    cli/             # Click commands (score, dashboard, narrative, sync, etc.)
    migrations/      # SQL schema migrations (4 versions)
```

## DCS Scoring Engine

The Dip-Buying Composite Score (DCS) is a 0-100 composite calibrated across 124,000+ observations and 625 tickers (2007-2025):

| Sub-Score | Weight | What It Measures |
|-----------|--------|-----------------|
| MQ (Momentum Quality) | 30% | Trend regime, vol-adjusted momentum, SA momentum grade, relative strength vs SPY |
| FQ (Fundamental Quality) | 25% | SA Quant rating, profitability (Novy-Marx + SA blend), FCF yield, revision momentum, growth |
| TO (Technical Oversold) | 20% | RSI-14, 200d SMA distance, Bollinger Band position, MACD histogram |
| MR (Market Regime) | 15% | VIX contrarian signal, SPY trend, market breadth |
| VC (Valuation Context) | 10% | SA valuation grade, sector-relative EV/EBITDA |

### Signal Thresholds (Walk-Forward Validated)

| DCS | Signal | Win Rate | Bootstrap CI | Action |
|-----|--------|----------|-------------|--------|
| >= 80 | STRONG BUY DIP | 53.8% | Small sample | Full size + lean in |
| >= 70 | HIGH CONVICTION | 60.0% | Marginal | Full size deployment |
| >= 65 | BUY DIP | 58.9% | [57.7%-60.1%] | Standard deployment |
| 50-64 | WATCH | 57.5% | Stable | Monitor |
| < 50 | WEAK/AVOID | -- | -- | Stay away |

### Post-Composition Modifiers

- **OBV Divergence Boost**: +2 DCS when bullish OBV divergence detected (Granville 1963)
- **RSI Bullish Divergence**: +3 DCS when price lower low + RSI higher low (+2.2pp edge, walk-forward stable)
- **Falling Knife Filter**: Defense-aware caps for tickers in steep downtrends (Moskowitz trend-following)
- **D-5 Regime Modifier**: In FEAR/PANIC VIX, HEDGE +5 / DEFENSIVE +3 / CYCLICAL -3 / AMPLIFIER -5

### Reversal Signals (Phase 2 Backtest-Validated)

| Signal | Trigger | Edge | Walk-Forward |
|--------|---------|------|-------------|
| RSI Bullish Divergence | Price lower low + RSI higher low | +2.2pp | Stable |
| Reversal Confirmed | DCS >= 65 + Bollinger lower breach | +4.6pp | Marginal |
| Bottom Turning | MACD hist rising below zero + RSI < 30 + Q3+ | +4.4pp | Stable |

### Sell Criteria (6-criterion framework)

Any 2 triggered = REVIEW REQUIRED:
1. SA Quant dropped to <= 2
2. Momentum turned negative (< 3)
3. Below 200-day SMA by > 3% for 10+ consecutive days
4. Fundamental thesis broken
5. Relative strength vs SPY < 0.7 for 4+ weeks
6. EPS Revision momentum declining (2+ sub-grade drop in 4 weeks)

### Advanced Modules (opt-in, disabled by default)

- **Risk Framework**: Effective bets, turbulence index, momentum crash protection, CVaR, CDaR
- **Trend Following**: Baltas-Kosowski continuous trend signal
- **Factor Momentum**: Ehsani-Linnainmaa cross-factor momentum
- **Aligned Sentiment**: Huang et al. PLS sentiment index
- **Portfolio Construction**: Inverse volatility, HRP, HIFO tax-lot selection

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

threshold sync              # Full workflow: onboard, import, snapshot, score, perf
threshold sync --dry-run    # Preview without persisting
threshold sync --no-score   # Skip scoring step
threshold sync --no-email   # Skip email alerts

threshold import positions  # Import positions from SA exports
threshold import registry <path>     # Import ticker_registry.json
threshold import scores <dir>        # Import weekly_scores JSON files
threshold import drawdown <path>     # Import drawdown classifications

threshold watch             # Watch for new SA export files (auto-onboard)
```

## Configuration

All settings live in `~/.threshold/config.yaml`. See `config.yaml.example` for the complete reference.

Key sections:
- **data_sources**: yfinance, Tiingo, Seeking Alpha, FRED configuration
- **scoring**: DCS weights, signal thresholds (backtest-calibrated defaults)
- **sell_criteria**: SMA breach, quant drop thresholds
- **allocation**: Equity/hard money/cash targets, rebalance triggers
- **deployment**: Gate 3 parabolic filter (RSI max, 8-week return max)
- **accounts**: Brokerage account definitions with SA export prefix matching
- **tsp**: Thrift Savings Plan total value and fund allocations
- **separate_holdings**: Assets outside main brokerage (e.g., crypto)
- **alerts**: Email notification settings
- **output**: Directory paths for dashboards, narratives, score history

### Environment Variables

Sensitive values are referenced via `${VAR_NAME}` syntax:

```bash
export TIINGO_API_KEY="your-key"     # Free at https://api.tiingo.com
export FRED_API_KEY="your-key"       # Free at https://fred.stlouisfed.org
export EMAIL_TO="you@example.com"
export EMAIL_FROM="alerts@example.com"
export GMAIL_APP_PASSWORD="xxxx-xxxx-xxxx-xxxx"
```

## Data Sources

| Source | Required? | What It Provides | Cost |
|--------|-----------|-----------------|------|
| yfinance | Yes | Price data, technicals, fundamentals | Free |
| Seeking Alpha exports | Recommended | Quant ratings, factor grades, momentum grades | SA subscription |
| Tiingo | Optional | Fallback price data | Free tier available |
| FRED | Optional | VIX, yield curves, credit spreads, breadth | Free |

### Seeking Alpha Export Setup

1. Link your brokerage to Seeking Alpha's portfolio tracker
2. Export each account as Excel from SA's portfolio page
3. Place exports in the directory specified by `data_sources.seeking_alpha.export_dir`
4. Place watchlist Z-files in `data_sources.seeking_alpha.z_file_dir`
5. Run `threshold score` -- SA grades are auto-extracted from exports

## Development

```bash
# Install with dev dependencies
pip install -e ".[dev]"

# Run tests (720+ tests, <5 seconds)
pytest

# Linting
ruff check threshold/ tests/

# macOS note: after pip install -e ., run this if the CLI stops working:
chflags -R nohidden .venv
```

### Test Coverage

720+ tests organized across 27 test files:

| Category | Files | Tests | Coverage |
|----------|-------|-------|----------|
| Scoring engine | test_technical, test_subscores, test_composite, test_signals, test_scorer | ~150 | RSI, MACD, OBV, all sub-scores, DCS composition, signal board |
| Risk & advanced | test_risk, test_advanced, test_aggregator, test_portfolio_construction | ~90 | EBP, CVaR, trend following, HRP, inverse vol |
| Pipeline & integration | test_pipeline, test_regression, test_integration_phase6 | ~80 | End-to-end scoring, golden ranges, full pipeline |
| Data adapters | test_data_adapters, test_sa_export_reader, test_sa_api_fetcher, test_fred_adapter | ~60 | yfinance, SA parsing, FRED adapter |
| Output | test_output | ~70 | Dashboard HTML, narrative Markdown, charts |
| Storage & config | test_storage, test_config, test_properties | ~70 | Database CRUD, config validation, property-based |
| Portfolio features | test_snapshot, test_performance, test_position_import, test_onboarding, test_watcher | ~50 | Snapshots, imports, auto-onboarding |
| Scoring features | test_exemptions, test_grace_period, test_gate3, test_drawdown_backtest, test_watchlist, test_journal | ~50 | Exemptions, grace periods, deployment gates |

## Research Basis

The scoring engine and decision framework are grounded in peer-reviewed research:

| Component | Research |
|-----------|----------|
| Vol-adjusted momentum | Barroso & Santa-Clara 2015 |
| Dual momentum | Antonacci (cross-asset momentum) |
| Gross profitability | Novy-Marx 2013 |
| Revision momentum | Novy-Marx 2015, Fed WP 2024-049 |
| OBV divergence | Granville 1963 (leads price 2-6 weeks) |
| Trend following | Moskowitz, Ooi & Pedersen 2012 |
| Factor momentum | Ehsani & Linnainmaa 2022 |
| Sentiment index | Huang, Jiang, Tu & Zhou 2015 |
| HRP construction | Lopez de Prado 2016 |
| Behavioral framework | Housel, "Psychology of Money" |

## License

MIT
