"""Tests for decision journal — trade logging and outcome tracking."""

from __future__ import annotations

from unittest.mock import MagicMock

from threshold.portfolio.journal import (
    JournalSummary,
    TradeEntry,
    create_trade_entry,
    get_journal_summary,
    list_journal_entries,
    record_outcome,
)

# ---------------------------------------------------------------------------
# Tests: TradeEntry
# ---------------------------------------------------------------------------

class TestTradeEntry:
    def test_default_values(self):
        """Default TradeEntry should have sensible defaults."""
        entry = TradeEntry()
        assert entry.ticker == ""
        assert entry.action == ""
        assert entry.has_thesis is True
        assert entry.deployment_gates_passed is True


# ---------------------------------------------------------------------------
# Tests: create_trade_entry
# ---------------------------------------------------------------------------

class TestCreateTradeEntry:
    def test_creates_entry(self):
        """Should insert a trade journal entry."""
        mock_db = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.lastrowid = 1
        mock_db.execute.return_value = mock_cursor

        entry_id = create_trade_entry(
            mock_db,
            ticker="AAPL",
            action="BUY",
            account="ind",
            shares=100,
            price=175.50,
            thesis="Strong fundamentals, DCS >= 70",
            dcs_at_decision=72.5,
            vix_regime="NORMAL",
        )
        assert entry_id == 1
        mock_db.execute.assert_called_once()
        mock_db.conn.commit.assert_called_once()

    def test_uppercases_ticker_and_action(self):
        """Should uppercase ticker and action."""
        mock_db = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.lastrowid = 2
        mock_db.execute.return_value = mock_cursor

        create_trade_entry(mock_db, ticker="aapl", action="buy")
        call_args = mock_db.execute.call_args[0][1]
        assert call_args[0] == "AAPL"  # ticker
        assert call_args[1] == "BUY"  # action

    def test_uses_today_when_no_date(self):
        """Should use today's date when trade_date not provided."""
        mock_db = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.lastrowid = 3
        mock_db.execute.return_value = mock_cursor

        create_trade_entry(mock_db, ticker="AAPL", action="BUY")
        call_args = mock_db.execute.call_args[0][1]
        # trade_date is 6th positional arg (index 5)
        assert call_args[5] != ""  # Should have a date


# ---------------------------------------------------------------------------
# Tests: list_journal_entries
# ---------------------------------------------------------------------------

class TestListJournalEntries:
    def test_lists_all(self):
        """Should list all journal entries."""
        mock_db = MagicMock()
        mock_db.fetchall.return_value = [
            {"id": 1, "ticker": "AAPL", "action": "BUY", "trade_date": "2026-02-15"},
        ]
        result = list_journal_entries(mock_db)
        assert len(result) == 1
        assert result[0]["ticker"] == "AAPL"

    def test_filters_by_ticker(self):
        """Should filter by ticker when provided."""
        mock_db = MagicMock()
        mock_db.fetchall.return_value = [
            {"id": 1, "ticker": "AAPL", "action": "BUY"},
        ]
        result = list_journal_entries(mock_db, ticker="AAPL")
        assert len(result) == 1
        # Verify the ticker was uppercased in the query
        call_args = mock_db.fetchall.call_args[0][1]
        assert call_args[0] == "AAPL"


# ---------------------------------------------------------------------------
# Tests: record_outcome
# ---------------------------------------------------------------------------

class TestRecordOutcome:
    def test_records_outcome(self):
        """Should insert an outcome record with alpha."""
        mock_db = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.lastrowid = 10
        mock_db.execute.return_value = mock_cursor

        outcome_id = record_outcome(
            mock_db,
            entry_id=1,
            window="4w",
            ticker_return=0.08,
            spy_return=0.03,
        )
        assert outcome_id == 10
        # Verify alpha was computed
        call_args = mock_db.execute.call_args[0][1]
        assert call_args[4] == 0.05  # alpha = 0.08 - 0.03


# ---------------------------------------------------------------------------
# Tests: get_journal_summary
# ---------------------------------------------------------------------------

class TestGetJournalSummary:
    def test_empty_journal(self):
        """Should return zero summary for empty journal."""
        mock_db = MagicMock()
        mock_db.fetchall.return_value = []

        summary = get_journal_summary(mock_db)
        assert summary.total_trades == 0

    def test_computes_summary(self):
        """Should compute summary statistics."""
        mock_db = MagicMock()
        # First call returns journal entries, second returns outcomes
        mock_db.fetchall.side_effect = [
            # Journal entries
            [
                {"action": "BUY", "is_panic_or_process": "process", "has_thesis": 1},
                {"action": "SELL", "is_panic_or_process": "panic", "has_thesis": 1},
                {"action": "BUY", "is_panic_or_process": "process", "has_thesis": 0},
            ],
            # Outcomes for 4w window
            [
                {"alpha": 0.05},
                {"alpha": -0.02},
            ],
        ]

        summary = get_journal_summary(mock_db)
        assert summary.total_trades == 3
        assert summary.buys == 2
        assert summary.sells == 1
        assert summary.process_pct == round(2 / 3, 2)
        assert summary.thesis_pct == round(2 / 3, 2)
        assert summary.avg_alpha_4w == round((0.05 + -0.02) / 2, 4)


# ---------------------------------------------------------------------------
# Tests: JournalSummary
# ---------------------------------------------------------------------------

class TestJournalSummary:
    def test_default_values(self):
        """Default JournalSummary should have zero values."""
        summary = JournalSummary()
        assert summary.total_trades == 0
        assert summary.avg_alpha_4w is None
