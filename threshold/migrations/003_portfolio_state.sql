-- Migration 003: Portfolio state management
-- Adds portfolio snapshots table for tracking values over time.

CREATE TABLE IF NOT EXISTS portfolio_snapshots (
    snapshot_date TEXT NOT NULL PRIMARY KEY,
    data_json TEXT NOT NULL,
    total_portfolio REAL DEFAULT 0,
    fidelity_total REAL DEFAULT 0,
    tsp_value REAL DEFAULT 0,
    btc_value REAL DEFAULT 0,
    created_at TEXT DEFAULT (datetime('now'))
);
