"""Tests for the file watcher — export detection and watch cycles."""

from __future__ import annotations

import json
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd

from threshold.data.watcher import (
    WatchResult,
    find_new_exports,
    get_last_processed_mtime,
    save_watcher_state,
)

# ---------------------------------------------------------------------------
# Helper: Create minimal .xlsx file
# ---------------------------------------------------------------------------

def _make_xlsx(filepath: Path, symbol: str = "AAPL") -> None:
    """Create a minimal .xlsx file."""
    df = pd.DataFrame({"Symbol": [symbol], "Weight": [1.0]})
    with pd.ExcelWriter(str(filepath), engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="Summary", index=False)


# ---------------------------------------------------------------------------
# Tests: WatchResult
# ---------------------------------------------------------------------------

class TestWatchResult:
    def test_default_values(self):
        """WatchResult should have sensible defaults."""
        result = WatchResult()
        assert result.new_files == []
        assert result.new_tickers == 0
        assert result.review_needed == 0
        assert result.positions_imported == 0
        assert result.errors == []

    def test_can_accumulate(self):
        """WatchResult fields should be mutable."""
        result = WatchResult()
        result.new_files.append("test.xlsx")
        result.new_tickers = 5
        result.errors.append("some error")
        assert len(result.new_files) == 1
        assert result.new_tickers == 5
        assert len(result.errors) == 1


# ---------------------------------------------------------------------------
# Tests: find_new_exports
# ---------------------------------------------------------------------------

class TestFindNewExports:
    def test_finds_new_files(self, tmp_path):
        """Should find xlsx files newer than the given mtime."""
        old_mtime = time.time() - 3600  # 1 hour ago

        f = tmp_path / "export.xlsx"
        _make_xlsx(f)

        result = find_new_exports([tmp_path], old_mtime)
        assert len(result) == 1
        assert result[0].name == "export.xlsx"

    def test_skips_old_files(self, tmp_path):
        """Should skip files older than the mtime threshold."""
        f = tmp_path / "old_export.xlsx"
        _make_xlsx(f)

        # Use a future timestamp
        future_mtime = time.time() + 3600

        result = find_new_exports([tmp_path], future_mtime)
        assert len(result) == 0

    def test_skips_temp_files(self, tmp_path):
        """Should skip ~$ temp files."""
        _make_xlsx(tmp_path / "normal.xlsx")
        _make_xlsx(tmp_path / "~$normal.xlsx")

        result = find_new_exports([tmp_path], 0)
        assert len(result) == 1
        assert result[0].name == "normal.xlsx"

    def test_skips_nonexistent_dirs(self):
        """Should silently skip non-existent directories."""
        result = find_new_exports(["/nonexistent/dir"], 0)
        assert result == []

    def test_multiple_dirs(self, tmp_path):
        """Should scan multiple directories."""
        dir1 = tmp_path / "exports"
        dir2 = tmp_path / "zfiles"
        dir1.mkdir()
        dir2.mkdir()

        _make_xlsx(dir1 / "export1.xlsx")
        _make_xlsx(dir2 / "zfile1.xlsx")

        result = find_new_exports([dir1, dir2], 0)
        assert len(result) == 2

    def test_sorted_by_mtime(self, tmp_path):
        """Results should be sorted by modification time."""
        f1 = tmp_path / "first.xlsx"
        _make_xlsx(f1)
        # Ensure second file has a later mtime
        time.sleep(0.05)
        f2 = tmp_path / "second.xlsx"
        _make_xlsx(f2)

        result = find_new_exports([tmp_path], 0)
        assert len(result) == 2
        assert result[0].name == "first.xlsx"
        assert result[1].name == "second.xlsx"

    def test_ignores_non_xlsx(self, tmp_path):
        """Should only find .xlsx files."""
        _make_xlsx(tmp_path / "export.xlsx")
        (tmp_path / "notes.txt").write_text("not a spreadsheet")
        (tmp_path / "data.csv").write_text("a,b,c")

        result = find_new_exports([tmp_path], 0)
        assert len(result) == 1
        assert result[0].name == "export.xlsx"


# ---------------------------------------------------------------------------
# Tests: get_last_processed_mtime / save_watcher_state
# ---------------------------------------------------------------------------

class TestWatcherState:
    def _make_mock_db(self, freshness_data: dict | None = None):
        """Create a mock database that simulates data_freshness table."""
        db = MagicMock()
        if freshness_data is None:
            freshness_data = {}
        return db, freshness_data

    def test_get_mtime_no_record(self):
        """Should return 0.0 when no watcher record exists."""
        with patch("threshold.storage.queries.get_data_freshness", return_value={}):
            result = get_last_processed_mtime(MagicMock())
        assert result == 0.0

    def test_get_mtime_with_record(self):
        """Should extract mtime from stored JSON details."""
        details = json.dumps({"last_processed_mtime": 1707920000.0})
        freshness = {"watcher": {"details": details}}

        with patch("threshold.storage.queries.get_data_freshness", return_value=freshness):
            result = get_last_processed_mtime(MagicMock())
        assert result == 1707920000.0

    def test_get_mtime_corrupt_details(self):
        """Should return 0.0 for corrupted details."""
        freshness = {"watcher": {"details": "not json"}}

        with patch("threshold.storage.queries.get_data_freshness", return_value=freshness):
            result = get_last_processed_mtime(MagicMock())
        assert result == 0.0

    def test_get_mtime_empty_details(self):
        """Should return 0.0 for empty/None details."""
        freshness = {"watcher": {"details": None}}

        with patch("threshold.storage.queries.get_data_freshness", return_value=freshness):
            result = get_last_processed_mtime(MagicMock())
        assert result == 0.0

    def test_save_state_calls_update(self):
        """save_watcher_state should call update_data_freshness."""
        mock_db = MagicMock()

        with patch("threshold.storage.queries.update_data_freshness") as mock_update:
            save_watcher_state(mock_db, last_mtime=1000.0, files_processed=3, new_tickers=2)
            mock_update.assert_called_once()

            # Verify the details JSON
            args = mock_update.call_args
            assert args[0][0] is mock_db
            assert args[0][1] == "watcher"
            assert args[0][2] == "ok"
            details = json.loads(args[0][3])
            assert details["last_processed_mtime"] == 1000.0
            assert details["files_processed"] == 3
            assert details["new_tickers"] == 2
