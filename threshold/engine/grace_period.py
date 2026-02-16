"""Grace period management for the scoring pipeline.

Implements two grace period tiers:
  - 90-day review: Standard review window when sell criteria start
    weakening but haven't triggered hard sells. Signals downgrade
    SELL to WATCH.
  - 180-day hold: Extended hold for positions where thesis is intact
    but momentum is fading. Signals downgrade SELL to HOLD.

Grace periods are stored in the ``grace_periods`` table (migration 001).
When a grace period is active for a ticker, sell signals are softened
in the scoring pipeline rather than suppressed entirely — the user
still sees the warning but the net_action changes.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class GracePeriodStatus:
    """Status of a ticker's grace period."""

    is_active: bool = False
    tier: int | None = None  # 90 or 180
    days_remaining: int = 0
    reason: str = ""
    started_at: str = ""
    expires_at: str = ""
    grace_id: int | None = None


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------

def check_grace_period(db: Any, ticker: str) -> GracePeriodStatus:
    """Check whether a ticker has an active grace period.

    Parameters
    ----------
    db : Database
        Open database connection.
    ticker : str
        Ticker symbol.

    Returns
    -------
    GracePeriodStatus
        Status object — ``is_active=True`` if in grace period.
    """
    row = db.fetchone(
        """SELECT * FROM grace_periods
        WHERE symbol = ? AND is_active = 1
        ORDER BY expires_at DESC LIMIT 1""",
        (ticker,),
    )
    if not row:
        return GracePeriodStatus()

    expires_at = row["expires_at"]
    today = date.today().isoformat()

    # Check if expired
    if expires_at < today:
        # Auto-expire
        _expire_grace_period(db, row["id"], "expired")
        return GracePeriodStatus()

    days_remaining = (
        datetime.fromisoformat(expires_at) - datetime.fromisoformat(today)
    ).days

    return GracePeriodStatus(
        is_active=True,
        tier=int(row["duration_days"]),
        days_remaining=max(days_remaining, 0),
        reason=row["reason"],
        started_at=row["started_at"],
        expires_at=expires_at,
        grace_id=row["id"],
    )


def create_grace_period(
    db: Any,
    ticker: str,
    reason: str,
    duration_days: int = 180,
) -> int:
    """Create a new grace period for a ticker.

    Parameters
    ----------
    db : Database
        Open database connection.
    ticker : str
        Ticker symbol.
    reason : str
        Why the grace period was created.
    duration_days : int
        Duration in days (typically 90 or 180).

    Returns
    -------
    int
        ID of the created grace period.
    """
    started = date.today().isoformat()
    expires = (date.today() + timedelta(days=duration_days)).isoformat()

    cursor = db.execute(
        """INSERT INTO grace_periods (symbol, reason, started_at, expires_at, duration_days)
        VALUES (?, ?, ?, ?, ?)""",
        (ticker, reason, started, expires, duration_days),
    )
    db.conn.commit()
    logger.info(
        "Created %d-day grace period for %s: %s (expires %s)",
        duration_days, ticker, reason, expires,
    )
    return cursor.lastrowid or 0


def resolve_grace_period(
    db: Any,
    ticker: str,
    resolution: str = "resolved",
) -> bool:
    """Resolve (close) all active grace periods for a ticker.

    Parameters
    ----------
    db : Database
        Open database connection.
    ticker : str
        Ticker symbol.
    resolution : str
        Resolution reason (e.g., 'sold', 'resolved', 'thesis_restored').

    Returns
    -------
    bool
        True if any grace periods were resolved.
    """
    cursor = db.execute(
        """UPDATE grace_periods
        SET is_active = 0, resolved_at = ?, resolution = ?
        WHERE symbol = ? AND is_active = 1""",
        (datetime.now().isoformat(), resolution, ticker),
    )
    db.conn.commit()
    return cursor.rowcount > 0


def list_active_grace_periods(db: Any) -> list[dict[str, Any]]:
    """List all active (non-expired) grace periods.

    Returns
    -------
    list[dict]
        Active grace periods with ticker, reason, days remaining.
    """
    today = date.today().isoformat()
    rows = db.fetchall(
        """SELECT * FROM grace_periods
        WHERE is_active = 1 AND expires_at >= ?
        ORDER BY expires_at ASC""",
        (today,),
    )

    result = []
    for row in rows:
        days_remaining = (
            datetime.fromisoformat(row["expires_at"])
            - datetime.fromisoformat(today)
        ).days
        result.append({
            "id": row["id"],
            "symbol": row["symbol"],
            "reason": row["reason"],
            "tier": row["duration_days"],
            "started_at": row["started_at"],
            "expires_at": row["expires_at"],
            "days_remaining": max(days_remaining, 0),
        })

    return result


def expire_overdue_grace_periods(db: Any) -> int:
    """Expire any grace periods that have passed their expiry date.

    Returns the number of grace periods expired.
    """
    today = date.today().isoformat()
    cursor = db.execute(
        """UPDATE grace_periods
        SET is_active = 0, resolved_at = ?, resolution = 'expired'
        WHERE is_active = 1 AND expires_at < ?""",
        (datetime.now().isoformat(), today),
    )
    db.conn.commit()
    return cursor.rowcount


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _expire_grace_period(db: Any, grace_id: int, resolution: str) -> None:
    """Expire a single grace period by ID."""
    db.execute(
        """UPDATE grace_periods
        SET is_active = 0, resolved_at = ?, resolution = ?
        WHERE id = ?""",
        (datetime.now().isoformat(), resolution, grace_id),
    )
    db.conn.commit()
