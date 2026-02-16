"""yfinance data adapter for price data and ticker enrichment."""

from __future__ import annotations

import logging
from typing import Any

import yfinance as yf

logger = logging.getLogger(__name__)

# Sector mapping from yfinance quoteType/sector to our classification
_TYPE_MAP = {
    "EQUITY": "stock",
    "ETF": "etf",
    "MUTUALFUND": "fund",
    "CRYPTOCURRENCY": "crypto",
}

# Alden category heuristics based on sector and country
_ALDEN_BY_SECTOR = {
    "Technology": "US Large Cap",
    "Healthcare": "US Large Cap",
    "Financial Services": "US Large Cap",
    "Consumer Cyclical": "US Large Cap",
    "Consumer Defensive": "Defensive/Income",
    "Communication Services": "US Large Cap",
    "Industrials": "US Large Cap",
    "Energy": "Hard Assets",
    "Basic Materials": "Hard Assets",
    "Utilities": "Defensive/Income",
    "Real Estate": "Defensive/Income",
}


def enrich_ticker(symbol: str) -> dict[str, Any] | None:
    """Fetch ticker metadata from yfinance for registration.

    Returns a dict suitable for passing to upsert_ticker(), or None if
    the ticker cannot be found.
    """
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info

        if not info or info.get("regularMarketPrice") is None:
            logger.warning("No data found for %s", symbol)
            return None

        # Determine type
        quote_type = info.get("quoteType", "EQUITY")
        ticker_type = _TYPE_MAP.get(quote_type, "stock")

        # For ETFs, detect from quoteType
        if quote_type == "ETF":
            ticker_type = "etf"

        sector = info.get("sector", "")
        sector_detail = info.get("industry", "")
        name = info.get("longName") or info.get("shortName") or symbol
        country = info.get("country", "United States")

        # Determine Alden category
        is_international = country not in ("United States", "US", "")
        if ticker_type == "etf":
            alden_category = _classify_etf(symbol, name, info)
        elif is_international:
            alden_category = _classify_international(country, info)
        else:
            market_cap = info.get("marketCap", 0) or 0
            if market_cap < 10_000_000_000:  # < $10B
                alden_category = "US Small/Mid"
            else:
                alden_category = _ALDEN_BY_SECTOR.get(sector, "US Large Cap")

        # Detect special flags
        is_gold = any(kw in name.lower() for kw in ["gold", "mining", "precious"])
        is_crypto = ticker_type == "crypto" or any(
            kw in name.lower() for kw in ["bitcoin", "ethereum", "crypto"]
        )

        # yf_symbol override (e.g., BRK.B -> BRK-B)
        yf_symbol = None
        if "." in symbol:
            yf_symbol = symbol.replace(".", "-")

        needs_review = ticker_type == "etf"  # ETFs need manual category review

        return {
            "name": name,
            "type": ticker_type,
            "sector": sector or None,
            "sector_detail": sector_detail or None,
            "yf_symbol": yf_symbol,
            "alden_category": alden_category,
            "is_gold": is_gold,
            "is_hard_money": is_gold or is_crypto,
            "is_crypto": is_crypto,
            "is_crypto_exempt": False,
            "is_cash": False,
            "is_war_chest": False,
            "is_international": is_international,
            "is_amplifier_trim": False,
            "is_defensive_add": False,
            "dd_override": None,
            "verified_at": None,
            "needs_review": needs_review,
            "notes": "",
        }

    except Exception as e:
        logger.error("Failed to enrich %s: %s", symbol, e)
        return None


def _classify_etf(symbol: str, name: str, info: dict) -> str:
    """Heuristic classification for ETFs."""
    name_lower = name.lower()

    if any(kw in name_lower for kw in ["gold", "silver", "precious", "mining"]):
        return "Hard Assets"
    if any(kw in name_lower for kw in ["uranium", "copper", "resource", "commodity", "energy"]):
        return "Hard Assets"
    if any(kw in name_lower for kw in ["bitcoin", "ethereum", "crypto"]):
        return "Hard Assets"
    if any(kw in name_lower for kw in ["international", "emerging", "brazil", "korea", "peru", "latin"]):
        return "Emerging Markets"
    if any(kw in name_lower for kw in ["developed", "eafe", "europe", "japan"]):
        return "Intl Developed"
    if any(kw in name_lower for kw in ["small", "mid", "completion", "russell 2000"]):
        return "US Small/Mid"
    if any(kw in name_lower for kw in ["s&p 500", "large cap", "total market"]):
        return "US Large Cap"
    if any(kw in name_lower for kw in ["bond", "treasury", "tips", "income", "dividend"]):
        return "Defensive/Income"
    if any(kw in name_lower for kw in ["reit", "real estate", "property"]):
        return "Defensive/Income"

    return "Other"


def _classify_international(country: str, info: dict) -> str:
    """Classify international stocks."""
    emerging = {
        "Brazil", "China", "India", "Mexico", "South Korea", "Taiwan",
        "Indonesia", "Turkey", "Saudi Arabia", "South Africa", "Thailand",
        "Malaysia", "Philippines", "Colombia", "Chile", "Peru", "Argentina",
    }
    if country in emerging:
        return "Emerging Markets"
    return "Intl Developed"
