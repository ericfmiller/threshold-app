-- Threshold Database Schema v1
-- Applied by: threshold init / migrations.py

PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;

-- Schema version tracking
CREATE TABLE IF NOT EXISTS _schema_version (
    version     INTEGER PRIMARY KEY,
    applied_at  TEXT NOT NULL DEFAULT (datetime('now')),
    description TEXT
);

-- Tickers: Single source of truth for ticker metadata
CREATE TABLE IF NOT EXISTS tickers (
    symbol              TEXT PRIMARY KEY,
    name                TEXT,
    type                TEXT CHECK(type IN ('stock', 'etf', 'adr', 'preferred', 'crypto', 'fund')),
    sector              TEXT,
    sector_detail       TEXT,
    yf_symbol           TEXT,
    alden_category      TEXT DEFAULT 'Other',
    is_gold             INTEGER DEFAULT 0,
    is_hard_money       INTEGER DEFAULT 0,
    is_crypto           INTEGER DEFAULT 0,
    is_crypto_exempt    INTEGER DEFAULT 0,
    is_cash             INTEGER DEFAULT 0,
    is_war_chest        INTEGER DEFAULT 0,
    is_international    INTEGER DEFAULT 0,
    is_amplifier_trim   INTEGER DEFAULT 0,
    is_defensive_add    INTEGER DEFAULT 0,
    dd_override         TEXT,
    verified_at         TEXT,
    needs_review        INTEGER DEFAULT 0,
    notes               TEXT,
    created_at          TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at          TEXT NOT NULL DEFAULT (datetime('now'))
);

-- Accounts: Brokerage accounts
CREATE TABLE IF NOT EXISTS accounts (
    id                  TEXT PRIMARY KEY,
    name                TEXT NOT NULL,
    type                TEXT NOT NULL,
    institution         TEXT DEFAULT 'Fidelity',
    tax_treatment       TEXT,
    sa_export_prefix    TEXT,
    sa_export_prefix_old TEXT,
    is_active           INTEGER DEFAULT 1,
    created_at          TEXT NOT NULL DEFAULT (datetime('now'))
);

-- Positions: Current holdings per account
CREATE TABLE IF NOT EXISTS positions (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    account_id          TEXT NOT NULL REFERENCES accounts(id),
    symbol              TEXT NOT NULL REFERENCES tickers(symbol),
    shares              REAL,
    cost_basis          REAL,
    market_value        REAL,
    weight              REAL,
    snapshot_date       TEXT NOT NULL,
    source              TEXT DEFAULT 'sa_export',
    created_at          TEXT NOT NULL DEFAULT (datetime('now')),
    UNIQUE(account_id, symbol, snapshot_date)
);

CREATE INDEX IF NOT EXISTS idx_positions_symbol ON positions(symbol);
CREATE INDEX IF NOT EXISTS idx_positions_date ON positions(snapshot_date DESC);

-- Scoring runs: Metadata for each scoring execution
CREATE TABLE IF NOT EXISTS scoring_runs (
    run_id              TEXT PRIMARY KEY,
    started_at          TEXT NOT NULL,
    finished_at         TEXT,
    duration_seconds    REAL,
    status              TEXT DEFAULT 'running',
    tickers_scored      INTEGER DEFAULT 0,
    tickers_failed      INTEGER DEFAULT 0,
    vix_current         REAL,
    vix_regime          TEXT,
    vix_percentile      REAL,
    spy_price           REAL,
    spy_pct_from_200d   REAL,
    spy_above_200d      INTEGER,
    breadth_pct         REAL,
    breadth_above       INTEGER,
    breadth_total       INTEGER,
    mr_score            REAL,
    ds_sa_json          TEXT,
    ds_sa_exports       TEXT,
    ds_yfinance         TEXT,
    ds_tiingo           TEXT,
    ds_fred             TEXT,
    portfolio_total     REAL,
    fidelity_total      REAL,
    tsp_value           REAL,
    btc_value           REAL,
    effective_bets      REAL,
    yield_curve_spread  REAL,
    credit_risk         TEXT,
    fed_funds_rate      REAL,
    created_at          TEXT NOT NULL DEFAULT (datetime('now'))
);

-- Scores: Per-ticker DCS results
CREATE TABLE IF NOT EXISTS scores (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id              TEXT NOT NULL REFERENCES scoring_runs(run_id),
    scored_at           TEXT NOT NULL,
    symbol              TEXT NOT NULL REFERENCES tickers(symbol),
    dcs                 REAL NOT NULL,
    dcs_signal          TEXT NOT NULL,
    mq                  REAL,
    fq                  REAL,
    tov                 REAL,
    mr                  REAL,
    vc                  REAL,
    rsi_14              REAL,
    pct_from_200d       REAL,
    ret_8w              REAL,
    trend_score         REAL,
    vol_adj_mom         REAL,
    rs_vs_spy           REAL,
    macd_crossover      TEXT,
    macd_histogram      REAL,
    obv_divergence      TEXT,
    obv_divergence_str  REAL,
    bb_pct_b            REAL,
    rsi_bullish_div     INTEGER DEFAULT 0,
    bb_lower_breach     INTEGER DEFAULT 0,
    reversal_confirmed  INTEGER DEFAULT 0,
    bottom_turning      INTEGER DEFAULT 0,
    quant_freshness_warn INTEGER DEFAULT 0,
    dd_classification   TEXT,
    dd_downside_capture REAL,
    dd_modifier         INTEGER DEFAULT 0,
    fk_cap_applied      INTEGER,
    fk_original_dcs     REAL,
    sa_quant            REAL,
    sa_momentum         TEXT,
    sa_valuation        TEXT,
    sa_growth           TEXT,
    sa_profitability    TEXT,
    sa_revisions        TEXT,
    yf_fcf_yield        REAL,
    yf_gross_prof       REAL,
    yf_ev_ebitda        REAL,
    yf_sector           TEXT,
    rev_mom_score       REAL,
    rev_mom_direction   TEXT,
    rev_mom_delta_4w    REAL,
    quant_delta_30d     REAL,
    quant_compare_date  TEXT,
    price_data_source   TEXT DEFAULT 'yfinance',
    days_below_sma_3pct INTEGER DEFAULT 0,
    is_etf              INTEGER DEFAULT 0,
    created_at          TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_scores_symbol ON scores(symbol);
CREATE INDEX IF NOT EXISTS idx_scores_run ON scores(run_id);
CREATE INDEX IF NOT EXISTS idx_scores_date ON scores(scored_at DESC);

-- Signals: Typed signal events per score
CREATE TABLE IF NOT EXISTS signals (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    score_id            INTEGER NOT NULL REFERENCES scores(id),
    signal_type         TEXT NOT NULL,
    severity            TEXT NOT NULL,
    criterion           TEXT NOT NULL,
    message             TEXT NOT NULL,
    metadata_json       TEXT,
    created_at          TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_signals_score ON signals(score_id);

-- Drawdown classifications: Per-ticker backtest results
CREATE TABLE IF NOT EXISTS drawdown_classifications (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    backtest_date       TEXT NOT NULL,
    symbol              TEXT NOT NULL,
    classification      TEXT NOT NULL,
    downside_capture    REAL,
    upside_capture      REAL,
    capture_ratio       REAL,
    dd_beta             REAL,
    win_rate_in_dd      REAL,
    max_drawdown        REAL,
    episodes_measured   INTEGER,
    is_portfolio        INTEGER DEFAULT 0,
    source_data         TEXT DEFAULT 'yfinance',
    UNIQUE(backtest_date, symbol)
);

CREATE INDEX IF NOT EXISTS idx_dd_cls_symbol ON drawdown_classifications(symbol);
CREATE INDEX IF NOT EXISTS idx_dd_cls_date ON drawdown_classifications(backtest_date DESC);

-- Watchlists: Candidate tickers
CREATE TABLE IF NOT EXISTS watchlists (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    name                TEXT NOT NULL,
    symbol              TEXT NOT NULL,
    sa_quant            REAL,
    added_at            TEXT NOT NULL DEFAULT (datetime('now')),
    source_file         TEXT,
    UNIQUE(name, symbol)
);

-- Data freshness tracking
CREATE TABLE IF NOT EXISTS data_freshness (
    source              TEXT PRIMARY KEY,
    last_updated        TEXT NOT NULL,
    last_status         TEXT,
    details             TEXT,
    updated_at          TEXT NOT NULL DEFAULT (datetime('now'))
);

-- Grace periods
CREATE TABLE IF NOT EXISTS grace_periods (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol              TEXT NOT NULL,
    reason              TEXT NOT NULL,
    started_at          TEXT NOT NULL,
    expires_at          TEXT NOT NULL,
    duration_days       INTEGER NOT NULL DEFAULT 180,
    is_active           INTEGER DEFAULT 1,
    resolved_at         TEXT,
    resolution          TEXT,
    UNIQUE(symbol, started_at)
);

-- Alden category allocation targets
CREATE TABLE IF NOT EXISTS alden_categories (
    name                TEXT PRIMARY KEY,
    target_low          REAL NOT NULL,
    target_high         REAL NOT NULL,
    tsp_pct             REAL DEFAULT 0.0,
    is_cross_cutting    INTEGER DEFAULT 0,
    is_catchall         INTEGER DEFAULT 0
);

-- Alerts history
CREATE TABLE IF NOT EXISTS alerts (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id              TEXT REFERENCES scoring_runs(run_id),
    alert_type          TEXT NOT NULL,
    symbol              TEXT,
    score               REAL,
    message             TEXT NOT NULL,
    sent_at             TEXT,
    created_at          TEXT NOT NULL DEFAULT (datetime('now'))
);

-- Record this migration
INSERT INTO _schema_version (version, description)
VALUES (1, 'Initial schema: tickers, accounts, positions, scores, signals, drawdown, watchlists, alerts');
