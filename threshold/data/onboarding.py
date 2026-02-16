"""Ticker onboarding — auto-detect, enrich, and classify new tickers.

Scans SA export files and Z-file watchlists for tickers not yet in the
database, enriches them via yfinance, classifies their asset type and
Alden category, and registers them.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Tickers to skip (special rows, not real securities)
SKIP_TICKERS = frozenset({
    "CASH", "TOTAL", "ACCOUNT TOTAL", "NAN", "", "SPAXX",
    "CORE", "PENDING", "VYGVQ",  # Voyager bankruptcy remnant
})

# Keyword sets for ETF classification
_CRYPTO_KEYWORDS = frozenset({"bitcoin", "btc", "crypto", "blockchain", "ethereum", "eth"})
_GOLD_KEYWORDS = frozenset({"gold", "silver", "precious", "palladium", "platinum", "mining"})
_ENERGY_KEYWORDS = frozenset({"uranium", "copper", "resource", "commodity", "energy", "oil", "natural gas"})
_INTL_KEYWORDS = frozenset({
    "international", "emerging", "brazil", "korea", "peru", "latin",
    "chile", "mexico", "africa", "india", "china", "asia", "europe",
    "global", "world", "foreign",
})
_BOND_KEYWORDS = frozenset({"bond", "treasury", "tips", "income", "fixed income"})
_DIVIDEND_KEYWORDS = frozenset({"dividend", "quality", "value", "factor", "s&p 500"})
_SMALL_MID_KEYWORDS = frozenset({"small", "mid", "completion", "russell 2000"})

# Developed market countries
DEVELOPED_MARKETS = frozenset({
    "United Kingdom", "Japan", "Germany", "France", "Canada", "Australia",
    "Switzerland", "Netherlands", "Sweden", "Denmark", "Norway", "Finland",
    "Belgium", "Austria", "Ireland", "Italy", "Spain", "Portugal",
    "Singapore", "Hong Kong", "Israel", "New Zealand",
})


@dataclass
class OnboardingResult:
    """Result of a ticker onboarding run."""
    new_count: int = 0
    review_needed: int = 0
    review_tickers: list[str] = field(default_factory=list)
    new_tickers: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


def discover_tickers_from_exports(
    export_dir: str | Path,
    z_file_dirs: list[str | Path] | None = None,
) -> dict[str, str]:
    """Scan SA exports and Z-files to discover all tickers.

    Parameters
    ----------
    export_dir : str | Path
        Primary SA export directory.
    z_file_dirs : list | None
        Additional Z-file watchlist directories.

    Returns
    -------
    dict[str, str]
        Mapping of {ticker: source} where source describes where found.
    """
    from threshold.data.adapters.sa_export_reader import (
        extract_tickers_from_summary,
        read_sa_export,
    )

    tickers: dict[str, str] = {}
    dirs_to_scan = [Path(export_dir)] if export_dir else []
    if z_file_dirs:
        dirs_to_scan.extend(Path(d) for d in z_file_dirs)

    for scan_dir in dirs_to_scan:
        if not scan_dir.is_dir():
            continue
        source_label = "z-file" if scan_dir != Path(export_dir) else "portfolio"

        for xlsx_file in sorted(scan_dir.glob("*.xlsx")):
            if xlsx_file.name.startswith("~$"):
                continue
            try:
                sheets = read_sa_export(xlsx_file)
                summary = sheets.get("Summary")
                if summary is not None:
                    for ticker in extract_tickers_from_summary(summary):
                        if ticker not in SKIP_TICKERS and ticker not in tickers:
                            tickers[ticker] = f"{source_label}:{xlsx_file.name}"
            except Exception as e:
                logger.debug("Could not scan %s: %s", xlsx_file.name, e)

    return tickers


def find_new_tickers(
    db: Any,
    discovered: dict[str, str],
) -> dict[str, str]:
    """Compare discovered tickers against database, return unregistered ones.

    Parameters
    ----------
    db : Database
        Open database connection.
    discovered : dict[str, str]
        Mapping of {ticker: source} from discover_tickers_from_exports.

    Returns
    -------
    dict[str, str]
        Only tickers not already in the tickers table.
    """
    from threshold.storage.queries import get_ticker

    return {
        ticker: source
        for ticker, source in discovered.items()
        if ticker not in SKIP_TICKERS and get_ticker(db, ticker) is None
    }


def classify_etf(name: str, yf_info: dict | None = None) -> dict[str, Any]:
    """Classify an ETF based on its name and yfinance info.

    Returns a dict of classification fields to merge into the ticker record.
    """
    name_lower = name.lower()
    result: dict[str, Any] = {
        "type": "etf",
        "needs_review": False,
    }

    # Crypto ETFs
    if any(kw in name_lower for kw in _CRYPTO_KEYWORDS):
        result["is_crypto"] = True
        result["is_hard_money"] = True
        result["alden_category"] = "Hard Assets"
        return result

    # Gold / precious metals
    if any(kw in name_lower for kw in _GOLD_KEYWORDS):
        result["is_gold"] = True
        result["is_hard_money"] = True
        result["alden_category"] = "Hard Assets"
        return result

    # Energy / commodities
    if any(kw in name_lower for kw in _ENERGY_KEYWORDS):
        result["alden_category"] = "Hard Assets"
        return result

    # International / EM
    if any(kw in name_lower for kw in _INTL_KEYWORDS):
        result["is_international"] = True
        # Distinguish EM vs developed
        if any(kw in name_lower for kw in {"emerging", "brazil", "peru", "chile", "india", "china", "latin", "africa"}):
            result["alden_category"] = "Emerging Markets"
        else:
            result["alden_category"] = "Intl Developed"
        return result

    # Bonds / TIPS / income
    if any(kw in name_lower for kw in _BOND_KEYWORDS):
        result["is_cash"] = True
        result["is_war_chest"] = True
        result["alden_category"] = "Defensive/Income"
        return result

    # Dividend / quality / factor
    if any(kw in name_lower for kw in _DIVIDEND_KEYWORDS):
        result["alden_category"] = "US Large Cap"
        return result

    # Small/mid cap
    if any(kw in name_lower for kw in _SMALL_MID_KEYWORDS):
        result["alden_category"] = "US Small/Mid"
        return result

    # REITs
    if any(kw in name_lower for kw in {"reit", "real estate", "property"}):
        result["alden_category"] = "Defensive/Income"
        return result

    # Fallback — can't confidently classify
    result["needs_review"] = True
    result["alden_category"] = "Other"
    return result


def classify_stock(
    name: str,
    sector: str,
    country: str,
    market_cap: float,
) -> dict[str, Any]:
    """Classify a stock based on sector, country, and market cap.

    Returns a dict of classification fields to merge into the ticker record.
    """
    name_lower = name.lower()
    result: dict[str, Any] = {
        "type": "stock",
        "needs_review": False,
    }

    # Crypto-related equities
    if any(kw in name_lower for kw in _CRYPTO_KEYWORDS):
        result["is_crypto"] = True
        result["is_crypto_exempt"] = True
        result["alden_category"] = "Hard Assets"
        return result

    # Gold-related equities
    if any(kw in name_lower for kw in _GOLD_KEYWORDS):
        result["is_gold"] = True
        result["is_hard_money"] = True
        result["alden_category"] = "Hard Assets"
        return result

    # International stocks
    if country and country not in ("United States", "US", ""):
        result["is_international"] = True
        if country in DEVELOPED_MARKETS:
            result["alden_category"] = "Intl Developed"
        else:
            result["alden_category"] = "Emerging Markets"
        return result

    # US stocks: classify by market cap
    if market_cap and market_cap > 10_000_000_000:  # $10B+
        result["alden_category"] = "US Large Cap"
    elif market_cap and market_cap > 0:
        result["alden_category"] = "US Small/Mid"
    else:
        result["alden_category"] = "US Large Cap"  # Conservative default
        if not sector:
            result["needs_review"] = True

    return result


def enrich_and_classify(ticker: str) -> dict[str, Any]:
    """Enrich a ticker via yfinance and classify it.

    Returns a dict of fields suitable for upsert_ticker().
    """
    from threshold.data.adapters.yfinance_adapter import enrich_ticker

    enriched = enrich_ticker(ticker)
    if enriched is None:
        # yfinance failed — create a minimal record needing review
        return {
            "name": ticker,
            "type": "stock",
            "sector": None,
            "alden_category": "Other",
            "needs_review": True,
            "notes": "yfinance enrichment failed",
        }

    return enriched


def run_onboarding(
    db: Any,
    export_dir: str | Path,
    z_file_dirs: list[str | Path] | None = None,
    dry_run: bool = False,
    yf_delay: float = 0.3,
) -> OnboardingResult:
    """Detect new tickers, enrich, classify, and register.

    Parameters
    ----------
    db : Database
        Open database connection.
    export_dir : str | Path
        Primary SA export directory.
    z_file_dirs : list | None
        Additional Z-file watchlist directories.
    dry_run : bool
        If True, discover but don't write to database.
    yf_delay : float
        Seconds to wait between yfinance API calls.

    Returns
    -------
    OnboardingResult
        Summary of the onboarding run.
    """
    from threshold.storage.queries import upsert_ticker

    result = OnboardingResult()

    # Step 1: Discover all tickers
    discovered = discover_tickers_from_exports(export_dir, z_file_dirs)
    logger.info("Discovered %d tickers in exports", len(discovered))

    # Step 2: Find new ones
    new_tickers = find_new_tickers(db, discovered)
    if not new_tickers:
        logger.info("No new tickers to onboard")
        return result

    logger.info("Found %d new ticker(s) to onboard", len(new_tickers))

    # Step 3: Enrich and classify
    for i, (ticker, source) in enumerate(sorted(new_tickers.items()), 1):
        logger.info("[%d/%d] Enriching %s (%s)", i, len(new_tickers), ticker, source)

        try:
            enriched = enrich_and_classify(ticker)
        except Exception as e:
            logger.error("Failed to enrich %s: %s", ticker, e)
            result.errors.append(f"{ticker}: {e}")
            enriched = {
                "name": ticker,
                "type": "stock",
                "alden_category": "Other",
                "needs_review": True,
                "notes": f"Enrichment error: {e}",
            }

        if not dry_run:
            upsert_ticker(db, ticker, **enriched)

        result.new_tickers.append(ticker)
        result.new_count += 1

        if enriched.get("needs_review"):
            result.review_needed += 1
            result.review_tickers.append(ticker)

        if i < len(new_tickers):
            time.sleep(yf_delay)

    logger.info(
        "Onboarding complete: %d new, %d need review",
        result.new_count,
        result.review_needed,
    )
    return result
