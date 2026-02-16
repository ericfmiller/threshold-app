-- Migration 004: Trade journal and performance tracking
-- Adds tables for decision journaling with outcome tracking
-- and portfolio performance snapshots for SPY comparison.

-- Performance snapshots — weekly portfolio value vs SPY
CREATE TABLE IF NOT EXISTS performance_snapshots (
    snapshot_date       TEXT PRIMARY KEY,
    total_portfolio     REAL NOT NULL DEFAULT 0,
    spy_close           REAL,
    btc_price           REAL,
    created_at          TEXT NOT NULL DEFAULT (datetime('now'))
);

-- Trade journal — decision logging with behavioral context
CREATE TABLE IF NOT EXISTS trade_journal (
    id                      INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker                  TEXT NOT NULL,
    action                  TEXT NOT NULL CHECK(action IN ('BUY', 'SELL', 'TRIM', 'ADD')),
    account                 TEXT DEFAULT '',
    shares                  REAL DEFAULT 0,
    price                   REAL DEFAULT 0,
    trade_date              TEXT NOT NULL,
    thesis                  TEXT DEFAULT '',
    dcs_at_decision         REAL,
    vix_regime              TEXT DEFAULT '',
    is_panic_or_process     TEXT DEFAULT 'process',
    has_thesis              INTEGER DEFAULT 1,
    deployment_gates_passed INTEGER DEFAULT 1,
    notes                   TEXT DEFAULT '',
    created_at              TEXT NOT NULL DEFAULT (datetime('now'))
);

-- Trade outcomes — return tracking at fixed windows
CREATE TABLE IF NOT EXISTS trade_outcomes (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    entry_id        INTEGER NOT NULL REFERENCES trade_journal(id),
    window          TEXT NOT NULL CHECK(window IN ('1w', '4w', '12w', '26w')),
    ticker_return   REAL NOT NULL DEFAULT 0,
    spy_return      REAL NOT NULL DEFAULT 0,
    alpha           REAL NOT NULL DEFAULT 0,
    recorded_at     TEXT NOT NULL DEFAULT (datetime('now')),
    UNIQUE(entry_id, window)
);

INSERT OR IGNORE INTO _schema_version (version, description)
VALUES (4, 'Trade journal, trade outcomes, performance snapshots');
