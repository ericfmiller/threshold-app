"""Tests for position import from SA export Holdings sheets."""

from __future__ import annotations

import pandas as pd

from threshold.data.position_import import (
    ImportResult,
    _parse_holdings_sheet,
)

# ---------------------------------------------------------------------------
# Tests: _parse_holdings_sheet
# ---------------------------------------------------------------------------

class TestParseHoldingsSheet:
    def test_parses_normal_holdings(self):
        """Should extract positions from a normal Holdings sheet."""
        df = pd.DataFrame({
            "Symbol": ["AAPL", "MSFT", "GOOGL"],
            "Shares": [100, 50, 25],
            "Cost Basis": [15000, 12000, 6000],
            "Market Value": [17000, 14000, 7500],
            "% of Portfolio": [40.0, 33.0, 27.0],
        })
        result = _parse_holdings_sheet(df)
        assert len(result) == 3
        assert result[0]["symbol"] == "AAPL"
        assert result[0]["shares"] == 100.0
        assert result[0]["market_value"] == 17000.0
        assert result[0]["weight"] == 0.40

    def test_keeps_cash_skips_total(self):
        """Should keep CASH (real value) but skip TOTAL / ACCOUNT TOTAL."""
        df = pd.DataFrame({
            "Symbol": ["AAPL", "CASH", "TOTAL", "ACCOUNT TOTAL"],
            "Shares": [100, 0, 0, 0],
            "Cost Basis": [15000, 5000, 0, 0],
            "Market Value": [17000, 5000, 22000, 22000],
            "% of Portfolio": [50.0, 50.0, 100.0, 100.0],
        })
        result = _parse_holdings_sheet(df)
        assert len(result) == 2
        symbols = {p["symbol"] for p in result}
        assert "AAPL" in symbols
        assert "CASH" in symbols
        assert "TOTAL" not in symbols

    def test_cash_with_dash_shares(self):
        """CASH rows often have '-' for shares — should still import."""
        df = pd.DataFrame({
            "Symbol": ["AAPL", "CASH"],
            "Shares": [100, "-"],
            "Cost": [15000, "-"],
            "Value": [17000, 5000],
            "Weight": [0.77, 0.23],
        })
        result = _parse_holdings_sheet(df)
        assert len(result) == 2
        cash = [p for p in result if p["symbol"] == "CASH"][0]
        assert cash["market_value"] == 5000.0
        assert cash["shares"] == 0.0

    def test_blank_symbol_with_value_treated_as_cash(self):
        """Rows with no symbol but positive value should be captured as CASH."""
        df = pd.DataFrame({
            "Symbol": ["AAPL", "", None],
            "Shares": [100, 0, 0],
            "Value": [17000, 3000, 0],
            "Weight": [0.85, 0.15, 0],
        })
        result = _parse_holdings_sheet(df)
        symbols = [p["symbol"] for p in result]
        assert "AAPL" in symbols
        assert "CASH" in symbols
        assert len(result) == 2  # blank with value=0 is skipped

    def test_skips_zero_value_positions(self):
        """Should skip positions with zero shares and zero value."""
        df = pd.DataFrame({
            "Symbol": ["AAPL", "DEAD"],
            "Shares": [100, 0],
            "Cost Basis": [15000, 0],
            "Market Value": [17000, 0],
            "% of Portfolio": [100.0, 0.0],
        })
        result = _parse_holdings_sheet(df)
        assert len(result) == 1

    def test_handles_empty_df(self):
        """Should return empty list for empty DataFrame."""
        result = _parse_holdings_sheet(pd.DataFrame())
        assert result == []

    def test_handles_none(self):
        """Should return empty list for None."""
        result = _parse_holdings_sheet(None)
        assert result == []

    def test_handles_missing_columns(self):
        """Should handle DataFrames with missing columns gracefully."""
        df = pd.DataFrame({
            "Symbol": ["AAPL"],
            "Shares": [100],
            # Missing: Cost Basis, Market Value, % of Portfolio
        })
        result = _parse_holdings_sheet(df)
        # Should get a record with defaults for missing fields
        assert len(result) == 1
        assert result[0]["symbol"] == "AAPL"
        assert result[0]["shares"] == 100.0

    def test_uppercases_symbols(self):
        """Should uppercase symbol names."""
        df = pd.DataFrame({
            "Symbol": ["aapl", "Msft"],
            "Shares": [100, 50],
            "Market Value": [17000, 14000],
        })
        result = _parse_holdings_sheet(df)
        assert result[0]["symbol"] == "AAPL"
        assert result[1]["symbol"] == "MSFT"


# ---------------------------------------------------------------------------
# Tests: ImportResult
# ---------------------------------------------------------------------------

class TestImportResult:
    def test_default_values(self):
        """ImportResult should have sensible defaults."""
        result = ImportResult()
        assert result.positions_imported == 0
        assert result.accounts_processed == 0
        assert result.errors == []
