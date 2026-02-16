"""Tests for SA export reader — file I/O and XML stripping."""

from __future__ import annotations

import io
import os
import tempfile
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

import pandas as pd
import pytest

from threshold.data.adapters.sa_export_reader import (
    _strip_conditional_formatting,
    extract_sa_data_from_ratings,
    extract_tickers_from_summary,
    find_latest_export_per_account,
    read_all_sa_exports,
    read_sa_export,
)

# ---------------------------------------------------------------------------
# Helper: Create minimal valid .xlsx files for testing
# ---------------------------------------------------------------------------

def _make_minimal_xlsx(filepath: str | Path, sheets: dict[str, pd.DataFrame]) -> None:
    """Create a minimal .xlsx file from DataFrames."""
    with pd.ExcelWriter(str(filepath), engine="openpyxl") as writer:
        for name, df in sheets.items():
            df.to_excel(writer, sheet_name=name, index=False)


def _make_xlsx_with_conditional_formatting(filepath: str | Path) -> None:
    """Create an .xlsx with a <conditionalFormatting> element (simulating the SA bug)."""
    # First create a normal xlsx
    df = pd.DataFrame({"Symbol": ["AAPL", "MSFT"], "Weight": [0.5, 0.5]})
    tmp = filepath if isinstance(filepath, str) else str(filepath)
    with pd.ExcelWriter(tmp, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="Summary", index=False)

    # Now inject a conditionalFormatting element into the sheet XML
    ns = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
    with zipfile.ZipFile(tmp, "r") as zin:
        items = {}
        for item in zin.infolist():
            data = zin.read(item.filename)
            if item.filename.startswith("xl/worksheets/sheet") and item.filename.endswith(".xml"):
                tree = ET.parse(io.BytesIO(data))
                root = tree.getroot()
                # Add a conditionalFormatting element
                cf = ET.SubElement(root, f"{{{ns}}}conditionalFormatting")
                cf.set("sqref", "A1:A10")
                buf = io.BytesIO()
                tree.write(buf, xml_declaration=True, encoding="UTF-8")
                data = buf.getvalue()
            items[item] = data

    with zipfile.ZipFile(tmp, "w") as zout:
        for item, data in items.items():
            zout.writestr(item, data)


# ---------------------------------------------------------------------------
# Tests: _strip_conditional_formatting
# ---------------------------------------------------------------------------

class TestStripConditionalFormatting:
    def test_removes_cf_elements(self, tmp_path):
        """Conditional formatting elements should be stripped."""
        src = tmp_path / "with_cf.xlsx"
        dst = tmp_path / "clean.xlsx"

        _make_xlsx_with_conditional_formatting(src)

        # Verify CF exists in source
        ns = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
        with zipfile.ZipFile(str(src), "r") as z:
            for name in z.namelist():
                if name.startswith("xl/worksheets/sheet"):
                    tree = ET.parse(z.open(name))
                    assert len(tree.getroot().findall(f"{{{ns}}}conditionalFormatting")) > 0

        # Strip
        _strip_conditional_formatting(src, dst)

        # Verify CF removed in destination
        with zipfile.ZipFile(str(dst), "r") as z:
            for name in z.namelist():
                if name.startswith("xl/worksheets/sheet"):
                    tree = ET.parse(z.open(name))
                    assert len(tree.getroot().findall(f"{{{ns}}}conditionalFormatting")) == 0

    def test_preserves_data(self, tmp_path):
        """Data should be preserved after stripping."""
        src = tmp_path / "src.xlsx"
        dst = tmp_path / "dst.xlsx"

        _make_xlsx_with_conditional_formatting(src)
        _strip_conditional_formatting(src, dst)

        df = pd.read_excel(str(dst), sheet_name="Summary")
        assert len(df) == 2
        assert "AAPL" in df["Symbol"].values


# ---------------------------------------------------------------------------
# Tests: read_sa_export
# ---------------------------------------------------------------------------

class TestReadSAExport:
    def test_reads_basic_xlsx(self, tmp_path):
        """Should read a normal xlsx file."""
        filepath = tmp_path / "test.xlsx"
        _make_minimal_xlsx(filepath, {
            "Summary": pd.DataFrame({"Symbol": ["AAPL"], "Weight": [1.0]}),
            "Ratings": pd.DataFrame({"Symbol": ["AAPL"], "Quant": [4.5]}),
        })

        result = read_sa_export(filepath)
        assert "Summary" in result
        assert "Ratings" in result
        assert len(result["Summary"]) == 1

    def test_reads_xlsx_with_cf(self, tmp_path):
        """Should handle xlsx with conditional formatting (the SA bug)."""
        filepath = tmp_path / "sa_export.xlsx"
        _make_xlsx_with_conditional_formatting(filepath)

        result = read_sa_export(filepath)
        assert "Summary" in result
        assert len(result["Summary"]) == 2

    def test_raises_on_missing_file(self):
        """Should raise FileNotFoundError for missing files."""
        with pytest.raises(FileNotFoundError):
            read_sa_export("/nonexistent/file.xlsx")

    def test_cleans_up_temp_file(self, tmp_path):
        """Temp file should be cleaned up after reading."""
        filepath = tmp_path / "test.xlsx"
        _make_minimal_xlsx(filepath, {
            "Summary": pd.DataFrame({"Symbol": ["AAPL"]}),
        })

        temp_dir = tempfile.gettempdir()
        before = set(os.listdir(temp_dir))
        read_sa_export(filepath)
        after = set(os.listdir(temp_dir))
        # New temp files should have been cleaned up
        new_xlsx = {f for f in (after - before) if f.endswith(".xlsx")}
        assert len(new_xlsx) == 0


# ---------------------------------------------------------------------------
# Tests: read_all_sa_exports
# ---------------------------------------------------------------------------

class TestReadAllSAExports:
    def test_reads_multiple_files(self, tmp_path):
        """Should read all xlsx files in a folder."""
        for name in ["Account1.xlsx", "Account2.xlsx"]:
            _make_minimal_xlsx(tmp_path / name, {
                "Summary": pd.DataFrame({"Symbol": [name[:4]]}),
            })

        result = read_all_sa_exports(tmp_path)
        assert len(result) == 2

    def test_skips_temp_files(self, tmp_path):
        """Should skip ~$ temp files."""
        _make_minimal_xlsx(tmp_path / "normal.xlsx", {
            "Summary": pd.DataFrame({"Symbol": ["AAPL"]}),
        })
        _make_minimal_xlsx(tmp_path / "~$normal.xlsx", {
            "Summary": pd.DataFrame({"Symbol": ["TEMP"]}),
        })

        result = read_all_sa_exports(tmp_path)
        assert len(result) == 1
        assert "normal.xlsx" in result

    def test_raises_on_missing_folder(self):
        """Should raise FileNotFoundError for missing folders."""
        with pytest.raises(FileNotFoundError):
            read_all_sa_exports("/nonexistent/folder")


# ---------------------------------------------------------------------------
# Tests: extract_sa_data_from_ratings
# ---------------------------------------------------------------------------

class TestExtractSAData:
    def test_extracts_quant_and_grades(self):
        """Should extract quant score and letter grades."""
        df = pd.DataFrame({
            "Symbol": ["AAPL", "MSFT"],
            "Quant": [4.5, 3.8],
            "Momentum": ["A", "B+"],
            "Profitability": ["A-", "B"],
            "Revisions": ["B+", "C"],
            "Growth": ["B", "A-"],
            "Valuation": ["C+", "B-"],
        })

        result = extract_sa_data_from_ratings(df)
        assert "AAPL" in result
        assert result["AAPL"]["quantScore"] == 4.5
        assert result["AAPL"]["momentum"] == "A"
        assert result["AAPL"]["profitability"] == "A-"

    def test_skips_special_symbols(self):
        """Should skip TOTAL, CASH, etc."""
        df = pd.DataFrame({
            "Symbol": ["AAPL", "TOTAL", "CASH", ""],
            "Quant": [4.5, 0, 0, 0],
        })

        result = extract_sa_data_from_ratings(df)
        assert "AAPL" in result
        assert "TOTAL" not in result
        assert "CASH" not in result

    def test_handles_empty_df(self):
        """Should return empty dict for empty DataFrame."""
        result = extract_sa_data_from_ratings(pd.DataFrame())
        assert result == {}

    def test_handles_none_df(self):
        """Should return empty dict for None."""
        result = extract_sa_data_from_ratings(None)
        assert result == {}


# ---------------------------------------------------------------------------
# Tests: extract_tickers_from_summary
# ---------------------------------------------------------------------------

class TestExtractTickers:
    def test_extracts_symbols(self):
        """Should extract ticker symbols from Summary sheet."""
        df = pd.DataFrame({
            "Symbol": ["AAPL", "MSFT", "GOOGL"],
            "Weight": [0.4, 0.3, 0.3],
        })

        result = extract_tickers_from_summary(df)
        assert result == ["AAPL", "MSFT", "GOOGL"]

    def test_skips_special_values(self):
        """Should skip TOTAL, CASH, NaN, empty."""
        df = pd.DataFrame({
            "Symbol": ["AAPL", "TOTAL", "CASH", "", "NAN"],
        })

        result = extract_tickers_from_summary(df)
        assert result == ["AAPL"]

    def test_handles_none_and_empty(self):
        """Should handle None and empty DataFrames."""
        assert extract_tickers_from_summary(None) == []
        assert extract_tickers_from_summary(pd.DataFrame()) == []


# ---------------------------------------------------------------------------
# Tests: find_latest_export_per_account
# ---------------------------------------------------------------------------

class TestFindLatestExport:
    def test_finds_latest_by_prefix(self, tmp_path):
        """Should find the most recent file matching each account prefix."""
        accounts = [
            {"id": "individual", "sa_export_prefix": "Fidelity - Individual", "sa_export_prefix_old": ""},
        ]

        # Create two files with different dates
        _make_minimal_xlsx(
            tmp_path / "Fidelity - Individual 2026-02-01.xlsx",
            {"Summary": pd.DataFrame({"Symbol": ["OLD"]})},
        )
        _make_minimal_xlsx(
            tmp_path / "Fidelity - Individual 2026-02-15.xlsx",
            {"Summary": pd.DataFrame({"Symbol": ["NEW"]})},
        )

        result = find_latest_export_per_account(tmp_path, accounts)
        assert "individual" in result
        assert "2026-02-15" in result["individual"].name

    def test_uses_old_prefix_as_fallback(self, tmp_path):
        """Should fall back to old prefix if new prefix not found."""
        accounts = [
            {
                "id": "roth",
                "sa_export_prefix": "Fidelity - ROTH IRA",
                "sa_export_prefix_old": "Personal ROTH IRA",
            },
        ]

        _make_minimal_xlsx(
            tmp_path / "Personal ROTH IRA 2026-02-07.xlsx",
            {"Summary": pd.DataFrame({"Symbol": ["AAPL"]})},
        )

        result = find_latest_export_per_account(tmp_path, accounts)
        assert "roth" in result

    def test_returns_empty_for_missing_dir(self):
        """Should return empty dict for non-existent directory."""
        result = find_latest_export_per_account("/nonexistent", [])
        assert result == {}
