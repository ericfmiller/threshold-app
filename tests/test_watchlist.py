"""Tests for watchlist management."""

from __future__ import annotations

from unittest.mock import MagicMock

from threshold.portfolio.watchlist import (
    WatchlistImportResult,
    add_to_watchlist,
    clear_watchlist,
    get_watchlist_symbols,
    list_watchlist,
    remove_from_watchlist,
)

# ---------------------------------------------------------------------------
# Tests: WatchlistImportResult
# ---------------------------------------------------------------------------

class TestWatchlistImportResult:
    def test_default_values(self):
        """Default WatchlistImportResult should have zero counts."""
        result = WatchlistImportResult()
        assert result.added == 0
        assert result.skipped == 0
        assert result.errors == []


# ---------------------------------------------------------------------------
# Tests: add_to_watchlist
# ---------------------------------------------------------------------------

class TestAddToWatchlist:
    def test_adds_ticker(self):
        """Should insert a ticker into the watchlist."""
        mock_db = MagicMock()
        result = add_to_watchlist(mock_db, "energy", "XLE", sa_quant=4.38)
        assert result is True
        mock_db.execute.assert_called_once()
        mock_db.conn.commit.assert_called_once()

    def test_uppercases_symbol(self):
        """Should uppercase the symbol."""
        mock_db = MagicMock()
        add_to_watchlist(mock_db, "energy", "xle")
        call_args = mock_db.execute.call_args[0][1]
        assert call_args[1] == "XLE"  # symbol is uppercased

    def test_handles_error(self):
        """Should return False on error."""
        mock_db = MagicMock()
        mock_db.execute.side_effect = Exception("DB error")
        result = add_to_watchlist(mock_db, "energy", "XLE")
        assert result is False


# ---------------------------------------------------------------------------
# Tests: remove_from_watchlist
# ---------------------------------------------------------------------------

class TestRemoveFromWatchlist:
    def test_removes_existing(self):
        """Should remove an existing ticker."""
        mock_db = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.rowcount = 1
        mock_db.execute.return_value = mock_cursor

        result = remove_from_watchlist(mock_db, "energy", "XLE")
        assert result is True

    def test_returns_false_when_missing(self):
        """Should return False when ticker not in watchlist."""
        mock_db = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.rowcount = 0
        mock_db.execute.return_value = mock_cursor

        result = remove_from_watchlist(mock_db, "energy", "MISSING")
        assert result is False


# ---------------------------------------------------------------------------
# Tests: list_watchlist
# ---------------------------------------------------------------------------

class TestListWatchlist:
    def test_lists_by_name(self):
        """Should list tickers for a specific watchlist."""
        mock_db = MagicMock()
        mock_db.fetchall.return_value = [
            {"symbol": "XLE", "name": "energy", "sa_quant": 4.38},
        ]
        result = list_watchlist(mock_db, "energy")
        assert len(result) == 1
        assert result[0]["symbol"] == "XLE"

    def test_lists_all(self):
        """Should list all watchlists when name is None."""
        mock_db = MagicMock()
        mock_db.fetchall.return_value = [
            {"symbol": "XLE", "name": "energy"},
            {"symbol": "AAPL", "name": "stocks_etfs"},
        ]
        result = list_watchlist(mock_db, None)
        assert len(result) == 2


# ---------------------------------------------------------------------------
# Tests: clear_watchlist
# ---------------------------------------------------------------------------

class TestClearWatchlist:
    def test_clears_watchlist(self):
        """Should clear all entries from a watchlist."""
        mock_db = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.rowcount = 5
        mock_db.execute.return_value = mock_cursor

        count = clear_watchlist(mock_db, "energy")
        assert count == 5


# ---------------------------------------------------------------------------
# Tests: get_watchlist_symbols
# ---------------------------------------------------------------------------

class TestGetWatchlistSymbols:
    def test_returns_unique_symbols(self):
        """Should return unique ticker symbols."""
        mock_db = MagicMock()
        mock_db.fetchall.return_value = [
            {"symbol": "XLE", "name": "energy"},
            {"symbol": "AAPL", "name": "stocks_etfs"},
            {"symbol": "XLE", "name": "stocks_etfs"},  # duplicate
        ]
        result = get_watchlist_symbols(mock_db)
        assert result == ["XLE", "AAPL"]

    def test_empty_list(self):
        """Should return empty list when no watchlist entries."""
        mock_db = MagicMock()
        mock_db.fetchall.return_value = []
        result = get_watchlist_symbols(mock_db)
        assert result == []
