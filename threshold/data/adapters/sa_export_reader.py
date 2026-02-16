"""Seeking Alpha Excel Export Reader.

SA portfolio exports contain conditional formatting operators that openpyxl
and pandas cannot parse (the operator value is empty or non-standard, causing
a ValueError in ConditionalFormatting.from_tree).

This module strips the problematic <conditionalFormatting> XML elements from
the sheet files inside the .xlsx (which is just a ZIP archive), writes a
clean temporary copy, and reads it with pandas.

Sheets in each SA export:
    - Summary:   Price, change, weight, volume, 52W range, SA/WS ratings
    - Ratings:   Quant, SA Analyst, WS scores + letter grades
    - Holdings:  Shares, cost basis, today's gain, value
    - Dividends: Safety/Growth/Yield grades, ex-div dates, yield metrics
"""

from __future__ import annotations

import contextlib
import io
import logging
import os
import tempfile
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

# XML namespace for spreadsheetml
_NS = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
ET.register_namespace("", _NS)
ET.register_namespace(
    "r", "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
)
ET.register_namespace(
    "mc", "http://schemas.openxmlformats.org/markup-compatibility/2006"
)
ET.register_namespace(
    "x14ac", "http://schemas.microsoft.com/office/spreadsheetml/2009/9/ac"
)


def _strip_conditional_formatting(src_path: str | Path, dst_path: str | Path) -> None:
    """Copy an xlsx file, removing all <conditionalFormatting> elements.

    This is the proven workaround for the openpyxl bug where SA exports use
    conditional formatting operators that cannot be parsed.
    """
    with (
        zipfile.ZipFile(str(src_path), "r") as zin,
        zipfile.ZipFile(str(dst_path), "w") as zout,
    ):
        for item in zin.infolist():
            data = zin.read(item.filename)
            if item.filename.startswith("xl/worksheets/sheet") and item.filename.endswith(
                ".xml"
            ):
                tree = ET.parse(zin.open(item.filename))
                root = tree.getroot()
                for cf in root.findall(f"{{{_NS}}}conditionalFormatting"):
                    root.remove(cf)
                buf = io.BytesIO()
                tree.write(buf, xml_declaration=True, encoding="UTF-8")
                data = buf.getvalue()
            zout.writestr(item, data)


def read_sa_export(filepath: str | Path) -> dict[str, pd.DataFrame]:
    """Read a Seeking Alpha .xlsx export, returning {sheet_name: pd.DataFrame}.

    Handles the conditional formatting bug automatically by creating a
    temporary cleaned copy of the file.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"SA export not found: {filepath}")

    with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        _strip_conditional_formatting(str(filepath), tmp_path)
        return pd.read_excel(tmp_path, sheet_name=None)
    finally:
        with contextlib.suppress(OSError):
            os.unlink(tmp_path)


def read_all_sa_exports(folder: str | Path) -> dict[str, dict[str, pd.DataFrame]]:
    """Read every .xlsx in a folder (skipping temp ~$ files).

    Returns {filename: {sheet_name: pd.DataFrame}}.
    """
    folder = Path(folder)
    if not folder.is_dir():
        raise FileNotFoundError(f"Export folder not found: {folder}")

    results: dict[str, dict[str, pd.DataFrame]] = {}
    for f in sorted(folder.glob("*.xlsx")):
        if f.name.startswith("~$"):
            continue
        try:
            results[f.name] = read_sa_export(f)
        except Exception as e:
            logger.warning("Failed to read %s: %s", f.name, e)
    return results


def find_latest_export_per_account(
    export_dir: str | Path,
    accounts: list[dict],
) -> dict[str, Path]:
    """Find the most recent SA export file for each account.

    Parameters
    ----------
    export_dir : str | Path
        Directory containing SA export .xlsx files.
    accounts : list[dict]
        Account dicts with ``sa_export_prefix`` and ``sa_export_prefix_old``.

    Returns
    -------
    dict[str, Path]
        Mapping of account_id -> latest export file path.
    """
    export_dir = Path(export_dir)
    if not export_dir.is_dir():
        return {}

    result: dict[str, Path] = {}
    all_files = sorted(export_dir.glob("*.xlsx"), reverse=True)
    all_files = [f for f in all_files if not f.name.startswith("~$")]

    for acct in accounts:
        acct_id = acct.get("id", "")
        for prefix_key in ("sa_export_prefix", "sa_export_prefix_old"):
            prefix = acct.get(prefix_key, "")
            if not prefix:
                continue
            if acct_id in result:
                break
            for f in all_files:
                if f.name.startswith(prefix):
                    result[acct_id] = f
                    break

    return result


def extract_sa_data_from_ratings(
    ratings_df: pd.DataFrame,
) -> dict[str, dict]:
    """Extract ticker SA data from a Ratings sheet DataFrame.

    Returns {ticker: {quantScore, momentum, profitability, ...}}.
    """
    result: dict[str, dict] = {}

    if ratings_df is None or ratings_df.empty:
        return result

    # Find the symbol column
    sym_col = None
    for col in ratings_df.columns:
        if str(col).strip().lower() in ("symbol", "ticker"):
            sym_col = col
            break
    if sym_col is None and len(ratings_df.columns) > 0:
        sym_col = ratings_df.columns[0]

    if sym_col is None:
        return result

    # Column name mapping: SA export column -> our key
    # SA exports use "Quant Score", "Momentum Grade", etc. — must match exactly.
    grade_columns = {
        "Quant": "quantScore",
        "Quant Rating": "quantScore",
        "SA Quant Rating": "quantScore",
        "Quant Score": "quantScore",           # actual SA export column
        "SA Analysts Score": "saAnalysts",     # analyst consensus (numeric)
        "Wall St. Score": "wallStreet",        # wall street consensus (numeric)
    }
    letter_columns = {
        "Momentum": "momentum",
        "Mom": "momentum",
        "Momentum Grade": "momentum",          # actual SA export column
        "ETF Momentum": "momentum",            # ETF variant
        "Profitability": "profitability",
        "Prof": "profitability",
        "Profitability Grade": "profitability", # actual SA export column
        "EPS Revisions": "revisions",
        "Revisions": "revisions",
        "EPS Rev": "revisions",
        "EPS Revision Grade": "revisions",    # actual SA export column
        "Growth": "growth",
        "Growth Grade": "growth",              # actual SA export column
        "Valuation": "valuation",
        "Value": "valuation",
        "Val": "valuation",
        "Valuation Grade": "valuation",        # actual SA export column
    }

    for _, row in ratings_df.iterrows():
        symbol = str(row.get(sym_col, "")).strip().upper()
        if not symbol or symbol in ("TOTAL", "ACCOUNT TOTAL", "NAN", "", "CASH"):
            continue

        entry: dict = {}

        # Extract quant score (numeric)
        for col_name, key in grade_columns.items():
            for col in ratings_df.columns:
                if str(col).strip().lower() == col_name.lower():
                    try:
                        val = float(row[col])
                        if 0 < val <= 5:
                            entry[key] = val
                    except (ValueError, TypeError):
                        pass
                    break

        # Extract letter grades
        for col_name, key in letter_columns.items():
            for col in ratings_df.columns:
                if str(col).strip().lower() == col_name.lower():
                    val = str(row[col]).strip()
                    if val and val.lower() not in ("nan", "none", "-", ""):
                        entry[key] = val
                    break

        if entry:
            result[symbol] = entry

    return result


def extract_tickers_from_summary(
    summary_df: pd.DataFrame,
) -> list[str]:
    """Extract ticker symbols from a Summary sheet DataFrame."""
    if summary_df is None or summary_df.empty:
        return []

    sym_col = None
    for col in summary_df.columns:
        if str(col).strip().lower() in ("symbol", "ticker"):
            sym_col = col
            break
    if sym_col is None and len(summary_df.columns) > 0:
        sym_col = summary_df.columns[0]

    if sym_col is None:
        return []

    skip = {"TOTAL", "ACCOUNT TOTAL", "NAN", "", "CASH"}
    tickers = []
    for _, row in summary_df.iterrows():
        symbol = str(row.get(sym_col, "")).strip().upper()
        if symbol and symbol not in skip:
            tickers.append(symbol)

    return tickers
