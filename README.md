# Threshold

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

# Generate dashboard
threshold dashboard
```

## Architecture

Threshold is organized around the **Decision Hierarchy** — the order in which investment decisions should be made:

1. **Macro Regime** — What environment are we in?
2. **Sector Rotation** — Where should capital flow?
3. **Deployment Discipline** — Three gates before deploying cash
4. **Selection** — DCS scoring, buy/sell criteria, reversal signals
5. **Behavioral** — Process over emotion
6. **Tax** — Account placement optimization

## DCS Scoring Engine

The Dip-Buying Composite Score (DCS) is a 0-100 composite calibrated across 124,000+ observations and 625 tickers (2007-2025):

| Sub-Score | Weight | What It Measures |
|-----------|--------|-----------------|
| MQ (Momentum Quality) | 30% | Trend regime, vol-adjusted momentum, relative strength |
| FQ (Fundamental Quality) | 25% | Quant rating, profitability, FCF yield, revision momentum |
| TO (Technical Oversold) | 20% | RSI, SMA distance, Bollinger Band position, MACD |
| MR (Market Regime) | 15% | VIX contrarian signal, SPY trend, market breadth |
| VC (Valuation Context) | 10% | Value grade, sector-relative EV/EBITDA |

Signal thresholds: **BUY DIP** (DCS >= 65, 58.9% win rate), **HIGH CONVICTION** (>= 70), **STRONG BUY** (>= 80).

## Configuration

All settings live in `~/.threshold/config.yaml`. See `config.yaml.example` for the complete reference with inline documentation.

## Data Sources

- **yfinance** (required) — Price data, technicals, fundamentals
- **Seeking Alpha** (recommended) — Quant ratings, factor grades
- **Tiingo** (optional) — Fallback price data
- **FRED** (optional) — Macroeconomic data

## Development

```bash
pip install -e ".[dev]"
pytest
```

## License

MIT
