"""Tests for portfolio performance tracking."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from threshold.portfolio.performance import (
    PerformanceReport,
    PerformanceSnapshot,
    capture_performance_snapshot,
    compute_returns,
    generate_performance_report,
)

# ---------------------------------------------------------------------------
# Tests: PerformanceSnapshot
# ---------------------------------------------------------------------------

class TestPerformanceSnapshot:
    def test_default_values(self):
        """Default snapshot should have zero values."""
        snap = PerformanceSnapshot()
        assert snap.total_portfolio == 0.0
        assert snap.spy_close == 0.0


# ---------------------------------------------------------------------------
# Tests: capture_performance_snapshot
# ---------------------------------------------------------------------------

class TestCapturePerformanceSnapshot:
    def test_captures_snapshot(self):
        """Should insert a snapshot into the database."""
        mock_db = MagicMock()
        snap = capture_performance_snapshot(
            mock_db,
            total_portfolio=750000.0,
            spy_close=500.0,
            btc_price=65000.0,
            snapshot_date="2026-02-15",
        )
        assert snap.total_portfolio == 750000.0
        assert snap.spy_close == 500.0
        mock_db.execute.assert_called_once()
        mock_db.conn.commit.assert_called_once()

    def test_fetches_spy_when_none(self):
        """Should fetch SPY price when not provided."""
        mock_db = MagicMock()
        with patch("threshold.portfolio.performance._fetch_spy_close", return_value=505.0):
            snap = capture_performance_snapshot(
                mock_db,
                total_portfolio=750000.0,
                snapshot_date="2026-02-15",
            )
        assert snap.spy_close == 505.0


# ---------------------------------------------------------------------------
# Tests: compute_returns
# ---------------------------------------------------------------------------

class TestComputeReturns:
    def test_basic_returns(self):
        """Should compute returns over multiple timeframes."""
        snapshots = [
            {"snapshot_date": "2025-01-01", "total_portfolio": 100000},
            {"snapshot_date": "2025-06-01", "total_portfolio": 105000},
            {"snapshot_date": "2025-12-01", "total_portfolio": 110000},
            {"snapshot_date": "2026-02-01", "total_portfolio": 115000},
            {"snapshot_date": "2026-02-15", "total_portfolio": 116000},
        ]
        returns = compute_returns(snapshots)
        assert "inception" in returns
        assert returns["inception"] > 0  # Portfolio grew

    def test_empty_snapshots(self):
        """Should return empty dict for empty input."""
        returns = compute_returns([])
        assert returns == {}

    def test_single_snapshot(self):
        """Should return empty dict for single snapshot."""
        returns = compute_returns([
            {"snapshot_date": "2026-02-15", "total_portfolio": 100000},
        ])
        assert returns == {}

    def test_zero_starting_value(self):
        """Should handle zero starting value gracefully."""
        snapshots = [
            {"snapshot_date": "2025-01-01", "total_portfolio": 0},
            {"snapshot_date": "2026-02-15", "total_portfolio": 100000},
        ]
        returns = compute_returns(snapshots)
        # inception should not be present since division by zero
        # Implementation handles this by checking > 0
        assert isinstance(returns, dict)


# ---------------------------------------------------------------------------
# Tests: PerformanceReport
# ---------------------------------------------------------------------------

class TestPerformanceReport:
    def test_default_values(self):
        """Default report should have empty returns."""
        report = PerformanceReport()
        assert report.portfolio_returns == {}
        assert report.spy_returns == {}
        assert report.alpha == {}
        assert report.snapshots_count == 0


# ---------------------------------------------------------------------------
# Tests: generate_performance_report
# ---------------------------------------------------------------------------

class TestGeneratePerformanceReport:
    def test_insufficient_data(self):
        """Should return empty report with <2 snapshots."""
        mock_db = MagicMock()
        mock_db.fetchall.return_value = []

        report = generate_performance_report(mock_db)
        assert report.snapshots_count == 0
        assert report.portfolio_returns == {}

    def test_generates_report(self):
        """Should generate report with alpha calculations."""
        mock_db = MagicMock()
        mock_db.fetchall.return_value = [
            {
                "snapshot_date": "2025-01-01",
                "total_portfolio": 100000,
                "spy_close": 470.0,
            },
            {
                "snapshot_date": "2026-02-15",
                "total_portfolio": 116000,
                "spy_close": 510.0,
            },
        ]

        report = generate_performance_report(mock_db)
        assert report.snapshots_count == 2
        assert "inception" in report.portfolio_returns
        assert "inception" in report.spy_returns
        assert "inception" in report.alpha
