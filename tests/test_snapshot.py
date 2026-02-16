"""Tests for portfolio snapshot generation."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

from threshold.data.snapshot import (
    generate_snapshot,
    load_latest_snapshot,
    save_snapshot,
)

# ---------------------------------------------------------------------------
# Tests: generate_snapshot
# ---------------------------------------------------------------------------

class TestGenerateSnapshot:
    def _make_mock_db_and_config(self, positions=None):
        """Create mock db and config for snapshot tests."""
        mock_db = MagicMock()
        mock_config = MagicMock()
        mock_config.separate_holdings = []
        mock_config.tsp = MagicMock()
        mock_config.tsp.value = 0

        # Default empty positions
        if positions is None:
            positions = []

        return mock_db, mock_config, positions

    def test_empty_positions(self):
        """Should handle empty positions gracefully."""
        db, config, positions = self._make_mock_db_and_config()

        with patch("threshold.data.position_import.load_positions_from_db", return_value=[]):
            result = generate_snapshot(db, config, "2026-02-15")

        assert result["snapshot_date"] == "2026-02-15"
        assert result["total_portfolio"] == 0.0
        assert result["fidelity_total"] == 0.0

    def test_with_positions(self):
        """Should aggregate positions across accounts."""
        positions = [
            {"account_id": "ind", "symbol": "AAPL", "shares": 100, "market_value": 17000, "weight": 0.5},
            {"account_id": "ind", "symbol": "MSFT", "shares": 50, "market_value": 17000, "weight": 0.5},
            {"account_id": "roth", "symbol": "AAPL", "shares": 50, "market_value": 8500, "weight": 1.0},
        ]
        db, config, _ = self._make_mock_db_and_config()

        with patch("threshold.data.position_import.load_positions_from_db", return_value=positions):
            result = generate_snapshot(db, config, "2026-02-15")

        assert result["fidelity_total"] == 42500.0
        assert result["total_portfolio"] == 42500.0
        assert "AAPL" in result["positions"]
        assert result["positions"]["AAPL"]["total_value"] == 25500.0
        assert result["positions"]["AAPL"]["n_accounts"] == 2

    def test_with_tsp(self):
        """Should include TSP value in total."""
        db, config, _ = self._make_mock_db_and_config()
        config.tsp.value = 226000

        with patch("threshold.data.position_import.load_positions_from_db", return_value=[]):
            result = generate_snapshot(db, config, "2026-02-15")

        assert result["tsp_value"] == 226000.0
        assert result["total_portfolio"] == 226000.0

    def test_with_btc(self):
        """Should include BTC value when configured."""
        db, config, _ = self._make_mock_db_and_config()
        btc_holding = MagicMock()
        btc_holding.symbol = "BTC-USD"
        btc_holding.quantity = 3.5
        config.separate_holdings = [btc_holding]

        with (
            patch("threshold.data.position_import.load_positions_from_db", return_value=[]),
            patch("threshold.data.snapshot._fetch_btc_price", return_value=65000.0),
        ):
            result = generate_snapshot(db, config, "2026-02-15")

        assert result["btc_quantity"] == 3.5
        assert result["btc_price"] == 65000.0
        assert result["btc_value"] == 227500.0
        assert result["total_portfolio"] == 227500.0


# ---------------------------------------------------------------------------
# Tests: save_snapshot / load_latest_snapshot
# ---------------------------------------------------------------------------

class TestSnapshotPersistence:
    def test_save_calls_execute(self):
        """save_snapshot should insert into portfolio_snapshots."""
        mock_db = MagicMock()
        mock_db.conn = MagicMock()

        snapshot = {
            "snapshot_date": "2026-02-15",
            "total_portfolio": 857000.0,
            "fidelity_total": 405000.0,
            "tsp_value": 226000.0,
            "btc_value": 226000.0,
        }

        save_snapshot(mock_db, snapshot)
        mock_db.execute.assert_called_once()
        mock_db.conn.commit.assert_called_once()

    def test_load_returns_none_when_empty(self):
        """Should return None when no snapshots exist."""
        mock_db = MagicMock()
        mock_db.fetchone.return_value = None

        result = load_latest_snapshot(mock_db)
        assert result is None

    def test_load_parses_json(self):
        """Should parse stored JSON data."""
        snapshot_data = {"snapshot_date": "2026-02-15", "total_portfolio": 857000.0}
        mock_row = {"data_json": json.dumps(snapshot_data)}
        mock_db = MagicMock()
        mock_db.fetchone.return_value = mock_row

        result = load_latest_snapshot(mock_db)
        assert result is not None
        assert result["total_portfolio"] == 857000.0
