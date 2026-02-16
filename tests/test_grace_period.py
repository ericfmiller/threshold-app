"""Tests for grace period management."""

from __future__ import annotations

from datetime import date, timedelta
from unittest.mock import MagicMock

from threshold.engine.grace_period import (
    GracePeriodStatus,
    check_grace_period,
    create_grace_period,
    expire_overdue_grace_periods,
    list_active_grace_periods,
    resolve_grace_period,
)

# ---------------------------------------------------------------------------
# Tests: GracePeriodStatus
# ---------------------------------------------------------------------------

class TestGracePeriodStatus:
    def test_default_inactive(self):
        """Default status should be inactive."""
        status = GracePeriodStatus()
        assert status.is_active is False
        assert status.tier is None
        assert status.days_remaining == 0


# ---------------------------------------------------------------------------
# Tests: check_grace_period
# ---------------------------------------------------------------------------

class TestCheckGracePeriod:
    def test_no_grace_period(self):
        """Should return inactive when no grace period exists."""
        mock_db = MagicMock()
        mock_db.fetchone.return_value = None

        result = check_grace_period(mock_db, "AAPL")
        assert result.is_active is False

    def test_active_grace_period(self):
        """Should return active status with remaining days."""
        future = (date.today() + timedelta(days=30)).isoformat()
        past = (date.today() - timedelta(days=150)).isoformat()
        mock_db = MagicMock()
        mock_db.fetchone.return_value = {
            "id": 1,
            "symbol": "AAPL",
            "reason": "Thesis intact, momentum fading",
            "started_at": past,
            "expires_at": future,
            "duration_days": 180,
            "is_active": 1,
        }

        result = check_grace_period(mock_db, "AAPL")
        assert result.is_active is True
        assert result.tier == 180
        assert result.days_remaining > 0
        assert result.grace_id == 1

    def test_expired_grace_period_auto_resolves(self):
        """Should auto-resolve and return inactive for expired periods."""
        yesterday = (date.today() - timedelta(days=1)).isoformat()
        mock_db = MagicMock()
        mock_db.fetchone.return_value = {
            "id": 1,
            "symbol": "AAPL",
            "reason": "test",
            "started_at": (date.today() - timedelta(days=200)).isoformat(),
            "expires_at": yesterday,
            "duration_days": 180,
            "is_active": 1,
        }

        result = check_grace_period(mock_db, "AAPL")
        assert result.is_active is False
        # Verify it called execute to expire the period
        mock_db.execute.assert_called_once()


# ---------------------------------------------------------------------------
# Tests: create_grace_period
# ---------------------------------------------------------------------------

class TestCreateGracePeriod:
    def test_creates_with_defaults(self):
        """Should create a 180-day grace period by default."""
        mock_db = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.lastrowid = 42
        mock_db.execute.return_value = mock_cursor

        gp_id = create_grace_period(mock_db, "AAPL", "Momentum fading")
        assert gp_id == 42
        mock_db.execute.assert_called_once()
        mock_db.conn.commit.assert_called_once()

        # Verify the args contain correct duration
        call_args = mock_db.execute.call_args
        assert call_args[0][1][3]  # expires_at is populated
        assert call_args[0][1][4] == 180  # duration_days

    def test_creates_90_day(self):
        """Should create a 90-day grace period when specified."""
        mock_db = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.lastrowid = 43
        mock_db.execute.return_value = mock_cursor

        gp_id = create_grace_period(mock_db, "AAPL", "Review", duration_days=90)
        assert gp_id == 43
        call_args = mock_db.execute.call_args
        assert call_args[0][1][4] == 90  # duration_days


# ---------------------------------------------------------------------------
# Tests: resolve_grace_period
# ---------------------------------------------------------------------------

class TestResolveGracePeriod:
    def test_resolves_active_periods(self):
        """Should resolve active grace periods."""
        mock_db = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.rowcount = 1
        mock_db.execute.return_value = mock_cursor

        result = resolve_grace_period(mock_db, "AAPL", "sold")
        assert result is True
        mock_db.conn.commit.assert_called_once()

    def test_returns_false_when_none(self):
        """Should return False when no active periods exist."""
        mock_db = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.rowcount = 0
        mock_db.execute.return_value = mock_cursor

        result = resolve_grace_period(mock_db, "AAPL")
        assert result is False


# ---------------------------------------------------------------------------
# Tests: list_active_grace_periods
# ---------------------------------------------------------------------------

class TestListActiveGracePeriods:
    def test_lists_active_periods(self):
        """Should list active grace periods with remaining days."""
        future = (date.today() + timedelta(days=30)).isoformat()
        mock_db = MagicMock()
        mock_db.fetchall.return_value = [
            {
                "id": 1,
                "symbol": "AAPL",
                "reason": "test",
                "duration_days": 180,
                "started_at": "2026-01-01",
                "expires_at": future,
            },
        ]

        result = list_active_grace_periods(mock_db)
        assert len(result) == 1
        assert result[0]["symbol"] == "AAPL"
        assert result[0]["days_remaining"] > 0

    def test_empty_when_none(self):
        """Should return empty list when no active periods."""
        mock_db = MagicMock()
        mock_db.fetchall.return_value = []

        result = list_active_grace_periods(mock_db)
        assert result == []


# ---------------------------------------------------------------------------
# Tests: expire_overdue_grace_periods
# ---------------------------------------------------------------------------

class TestExpireOverdue:
    def test_expires_overdue(self):
        """Should expire grace periods past their date."""
        mock_db = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.rowcount = 3
        mock_db.execute.return_value = mock_cursor

        count = expire_overdue_grace_periods(mock_db)
        assert count == 3
        mock_db.conn.commit.assert_called_once()

    def test_returns_zero_when_none_overdue(self):
        """Should return 0 when nothing to expire."""
        mock_db = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.rowcount = 0
        mock_db.execute.return_value = mock_cursor

        count = expire_overdue_grace_periods(mock_db)
        assert count == 0
