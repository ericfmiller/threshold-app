-- Threshold Database Schema v2
-- Advanced modules: risk framework, advanced signals, portfolio construction
-- Applied by: migrations.py

-- Risk framework columns on scoring_runs
ALTER TABLE scoring_runs ADD COLUMN ebp_value REAL;
ALTER TABLE scoring_runs ADD COLUMN ebp_regime TEXT;
ALTER TABLE scoring_runs ADD COLUMN turbulence_value REAL;
ALTER TABLE scoring_runs ADD COLUMN turbulence_pctl REAL;
ALTER TABLE scoring_runs ADD COLUMN is_turbulent INTEGER DEFAULT 0;
ALTER TABLE scoring_runs ADD COLUMN momentum_crash_weight REAL;
ALTER TABLE scoring_runs ADD COLUMN is_bear_market INTEGER DEFAULT 0;
ALTER TABLE scoring_runs ADD COLUMN portfolio_cvar REAL;
ALTER TABLE scoring_runs ADD COLUMN portfolio_cdar REAL;

-- Advanced module columns on scores
ALTER TABLE scores ADD COLUMN cvar_95 REAL;
ALTER TABLE scores ADD COLUMN cdar_95 REAL;
ALTER TABLE scores ADD COLUMN trend_signal REAL;
ALTER TABLE scores ADD COLUMN inv_vol_weight REAL;
ALTER TABLE scores ADD COLUMN hrp_weight REAL;
ALTER TABLE scores ADD COLUMN sentiment_score REAL;
ALTER TABLE scores ADD COLUMN composite_risk REAL;

-- Risk snapshots: Per-run risk module outputs
CREATE TABLE IF NOT EXISTS risk_snapshots (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id              TEXT NOT NULL REFERENCES scoring_runs(run_id),
    module              TEXT NOT NULL,
    signal_json         TEXT NOT NULL,
    regime              TEXT,
    regime_score        REAL,
    created_at          TEXT NOT NULL DEFAULT (datetime('now')),
    UNIQUE(run_id, module)
);

CREATE INDEX IF NOT EXISTS idx_risk_snap_run ON risk_snapshots(run_id);

-- Tax lots: For HIFO tax-lot accounting (Phase 2D)
CREATE TABLE IF NOT EXISTS tax_lots (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    account_id          TEXT NOT NULL REFERENCES accounts(id),
    symbol              TEXT NOT NULL REFERENCES tickers(symbol),
    shares              REAL NOT NULL,
    cost_basis_per_share REAL NOT NULL,
    acquired_at         TEXT NOT NULL,
    lot_type            TEXT DEFAULT 'buy',
    is_open             INTEGER DEFAULT 1,
    closed_at           TEXT,
    close_price         REAL,
    realized_gain       REAL,
    holding_period      TEXT,
    wash_sale_disallowed REAL DEFAULT 0.0,
    created_at          TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_tax_lots_symbol ON tax_lots(symbol);
CREATE INDEX IF NOT EXISTS idx_tax_lots_account ON tax_lots(account_id);
CREATE INDEX IF NOT EXISTS idx_tax_lots_open ON tax_lots(is_open) WHERE is_open = 1;

-- Record this migration
INSERT INTO _schema_version (version, description)
VALUES (2, 'Advanced modules: risk framework columns, risk_snapshots, tax_lots');
