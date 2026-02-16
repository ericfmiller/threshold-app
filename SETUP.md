# Threshold Setup Guide

Step-by-step setup for running the Threshold investment analysis system.

## Prerequisites

- **Python 3.10+** (3.12 recommended)
- **pip** (modern version)
- **Git**

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/ericfmiller/threshold-app.git
cd threshold-app
```

### 2. Create Virtual Environment

```bash
python3.12 -m venv .venv
source .venv/bin/activate   # macOS/Linux
# .venv\Scripts\activate    # Windows
```

### 3. Install Threshold

```bash
# Production install
pip install -e .

# Development install (includes pytest, hypothesis, mypy, ruff)
pip install -e ".[dev]"

# With advanced modules (scikit-learn for sentiment analysis)
pip install -e ".[dev,advanced]"
```

### 4. Initialize

```bash
threshold init
```

This creates:
- `~/.threshold/` directory
- `~/.threshold/threshold.db` (SQLite database)
- `~/.threshold/config.yaml` (from example template)
- `~/.threshold/history/`, `dashboards/`, `narratives/` directories

### 5. Configure

Edit `~/.threshold/config.yaml`:

```yaml
# Required: API keys (if using external data sources)
api_keys:
  tiingo: "YOUR_TIINGO_KEY"     # Optional, for fallback price data
  fred: "YOUR_FRED_KEY"         # Optional, for macro data

# Configure your accounts
accounts:
  - id: "brokerage"
    name: "Individual Brokerage"
    type: "taxable"
    institution: "Fidelity"
    tax_treatment: "capital_gains"

# Email alerts (optional)
alerts:
  enabled: true
  email:
    to: "you@example.com"
    from_addr: "alerts@example.com"
    app_password: "YOUR_APP_PASSWORD"
```

### 6. Register Tickers

```bash
# Add individual tickers (auto-enriches via yfinance)
threshold ticker add AAPL
threshold ticker add MSFT
threshold ticker add COPJ

# Import from existing registry file
threshold import registry /path/to/ticker_registry.json

# View registered tickers
threshold ticker list
```

### 7. Run Scoring

```bash
# Full scoring pipeline
threshold score

# Score a single ticker
threshold score --ticker AAPL

# Dry run (no database persistence)
threshold score --dry-run
```

### 8. View Results

```bash
# Generate and open HTML dashboard
threshold dashboard

# Generate Markdown narrative report
threshold narrative

# Dashboard without auto-opening browser
threshold dashboard --no-open
```

## Data Sources Setup

### yfinance (Required)

yfinance is installed automatically. No API key needed.

### Seeking Alpha Exports (Recommended)

1. Log into Seeking Alpha and navigate to your portfolio
2. Export as Excel (one file per account)
3. Place exports in your configured SA exports directory
4. Files should be named: `{Account Name} YYYY-MM-DD.xlsx`

### Tiingo (Optional)

1. Create free account at [tiingo.com](https://api.tiingo.com/)
2. Get your API token from the account page
3. Add to config: `api_keys.tiingo: "YOUR_TOKEN"`

### FRED (Optional)

1. Request free API key at [fred.stlouisfed.org](https://fred.stlouisfed.org/docs/api/api_key.html)
2. Add to config: `api_keys.fred: "YOUR_KEY"`

## Running Tests

```bash
# All tests (~500 tests, <5 seconds)
pytest

# Verbose output
pytest -v

# Specific test file
pytest tests/test_scorer.py

# Specific test
pytest tests/test_regression.py::TestGoldenDCSRanges::test_strong_sa_uptrend_normal

# With coverage
pytest --tb=short
```

## Troubleshooting

### "No module named 'threshold'"

Make sure you installed with `pip install -e .` from the repo root.

### yfinance 403 errors

yfinance may be rate-limited or blocked in some environments. The scoring engine will log warnings and continue with available data.

### Database locked errors

Threshold uses WAL mode for concurrent reads. If you see lock errors, ensure only one write process is running at a time.

### Config validation errors

Run `threshold config validate` to see detailed error messages. All config values have sensible defaults — you only need to override what you want to change.
