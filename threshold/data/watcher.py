"""File watcher — auto-triggers ticker onboarding on new SA exports.

Polls SA export directories for new .xlsx files and runs the onboarding
pipeline when fresh files are detected. State is stored in the database
(data_freshness table) rather than a JSON file.

Can be run as a cron job (--once) or as a polling daemon.
"""

from __future__ import annotations

import contextlib
import fcntl
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Default lock file location
_LOCK_FILE = Path("~/.threshold/.watcher_lock").expanduser()


@dataclass
class WatchResult:
    """Result of a watch cycle."""
    new_files: list[str] = field(default_factory=list)
    new_tickers: int = 0
    review_needed: int = 0
    positions_imported: int = 0
    errors: list[str] = field(default_factory=list)


def find_new_exports(
    dirs: list[str | Path],
    since_mtime: float,
) -> list[Path]:
    """Find .xlsx files newer than the given mtime.

    Parameters
    ----------
    dirs : list[str | Path]
        Directories to scan for .xlsx files.
    since_mtime : float
        Only return files modified after this time (epoch seconds).

    Returns
    -------
    list[Path]
        List of new export files, sorted by modification time.
    """
    new_files: list[Path] = []
    for directory in dirs:
        d = Path(directory)
        if not d.exists():
            continue
        for f in d.glob("*.xlsx"):
            if f.name.startswith("~$"):
                continue
            try:
                if f.stat().st_mtime > since_mtime:
                    new_files.append(f)
            except OSError:
                continue
    return sorted(new_files, key=lambda p: p.stat().st_mtime)


def get_last_processed_mtime(db: Any) -> float:
    """Get the last processed mtime from the data_freshness table.

    Parameters
    ----------
    db : Database
        Open database connection.

    Returns
    -------
    float
        Last processed mtime as epoch seconds, or 0 if no record.
    """
    from threshold.storage.queries import get_data_freshness

    freshness = get_data_freshness(db)
    watcher_record = freshness.get("watcher")
    if watcher_record and watcher_record.get("details"):
        try:
            import json
            details = json.loads(watcher_record["details"])
            return float(details.get("last_processed_mtime", 0))
        except (ValueError, TypeError, KeyError):
            pass
    return 0.0


def save_watcher_state(
    db: Any,
    last_mtime: float,
    files_processed: int,
    new_tickers: int,
) -> None:
    """Save watcher state to the data_freshness table.

    Parameters
    ----------
    db : Database
        Open database connection.
    last_mtime : float
        The max mtime of processed files.
    files_processed : int
        Number of files processed in this cycle.
    new_tickers : int
        Number of new tickers onboarded.
    """
    import json

    from threshold.storage.queries import update_data_freshness

    details = json.dumps({
        "last_processed_mtime": last_mtime,
        "files_processed": files_processed,
        "new_tickers": new_tickers,
    })
    update_data_freshness(db, "watcher", "ok", details)


def run_watch_cycle(
    db: Any,
    export_dir: str | Path,
    z_file_dirs: list[str | Path] | None = None,
) -> WatchResult:
    """Run one watch cycle: detect new exports, onboard tickers.

    Parameters
    ----------
    db : Database
        Open database connection.
    export_dir : str | Path
        Primary SA export directory.
    z_file_dirs : list | None
        Additional Z-file watchlist directories.

    Returns
    -------
    WatchResult
        Summary of what was done.
    """
    from threshold.data.onboarding import run_onboarding

    result = WatchResult()

    # Find new files
    dirs_to_scan = [Path(export_dir)]
    if z_file_dirs:
        dirs_to_scan.extend(Path(d) for d in z_file_dirs)

    last_mtime = get_last_processed_mtime(db)
    new_files = find_new_exports(dirs_to_scan, last_mtime)

    if not new_files:
        return result

    result.new_files = [f.name for f in new_files]
    logger.info("Found %d new export file(s)", len(new_files))

    # Run onboarding
    try:
        onboard_result = run_onboarding(
            db=db,
            export_dir=export_dir,
            z_file_dirs=z_file_dirs,
        )
        result.new_tickers = onboard_result.new_count
        result.review_needed = onboard_result.review_needed
    except Exception as e:
        logger.error("Onboarding failed: %s", e)
        result.errors.append(f"Onboarding: {e}")

    # Update state
    max_mtime = max(f.stat().st_mtime for f in new_files)
    save_watcher_state(
        db,
        last_mtime=max_mtime,
        files_processed=len(new_files),
        new_tickers=result.new_tickers,
    )

    return result


def acquire_lock() -> Any | None:
    """Acquire a file lock to prevent concurrent watcher runs.

    Returns the lock file descriptor if acquired, None if another
    instance is running.
    """
    _LOCK_FILE.parent.mkdir(parents=True, exist_ok=True)
    try:
        lock_fd = open(_LOCK_FILE, "w")  # noqa: SIM115
        fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return lock_fd
    except (BlockingIOError, OSError):
        return None


def release_lock(lock_fd: Any) -> None:
    """Release the file lock."""
    if lock_fd is None:
        return
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_UN)
        lock_fd.close()
    except OSError:
        pass
    with contextlib.suppress(OSError):
        os.unlink(_LOCK_FILE)


def run_daemon(
    db: Any,
    export_dir: str | Path,
    z_file_dirs: list[str | Path] | None = None,
    interval: int = 600,
) -> None:
    """Run the watcher as a polling daemon.

    Parameters
    ----------
    db : Database
        Open database connection.
    export_dir : str | Path
        Primary SA export directory.
    z_file_dirs : list | None
        Additional Z-file watchlist directories.
    interval : int
        Polling interval in seconds (default 600 = 10 minutes).
    """
    lock_fd = acquire_lock()
    if lock_fd is None:
        logger.warning("Another watcher instance is running — exiting")
        return

    try:
        logger.info("Watcher started (interval=%ds)", interval)
        while True:
            try:
                result = run_watch_cycle(db, export_dir, z_file_dirs)
                if result.new_files:
                    logger.info(
                        "Processed %d files: %d new tickers, %d review needed",
                        len(result.new_files),
                        result.new_tickers,
                        result.review_needed,
                    )
            except Exception as e:
                logger.error("Watch cycle error: %s", e)
            time.sleep(interval)
    except KeyboardInterrupt:
        logger.info("Watcher stopped")
    finally:
        release_lock(lock_fd)
