"""Tests for data adapters — yfinance enrichment and classification logic.

Uses mock yfinance data to avoid real API calls in CI.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch
from typing import Any

import pytest

from threshold.data.adapters.yfinance_adapter import (
    _classify_etf,
    _classify_international,
    enrich_ticker,
)


# ---------------------------------------------------------------------------
# Mock yfinance info dicts
# ---------------------------------------------------------------------------

def _mock_stock_info(
    symbol: str = "AAPL",
    name: str = "Apple Inc.",
    sector: str = "Technology",
    industry: str = "Consumer Electronics",
    country: str = "United States",
    market_cap: int = 3_000_000_000_000,
    price: float = 185.0,
    quote_type: str = "EQUITY",
) -> dict[str, Any]:
    """Build a mock yfinance .info dict for a stock."""
    return {
        "symbol": symbol,
        "longName": name,
        "shortName": name,
        "quoteType": quote_type,
        "sector": sector,
        "industry": industry,
        "country": country,
        "marketCap": market_cap,
        "regularMarketPrice": price,
    }


def _mock_etf_info(
    symbol: str = "SPY",
    name: str = "SPDR S&P 500 ETF Trust",
    price: float = 500.0,
) -> dict[str, Any]:
    """Build a mock yfinance .info dict for an ETF."""
    return {
        "symbol": symbol,
        "longName": name,
        "shortName": name,
        "quoteType": "ETF",
        "regularMarketPrice": price,
    }


# ---------------------------------------------------------------------------
# Classification tests (pure logic, no API calls)
# ---------------------------------------------------------------------------

class TestClassifyETF:
    """Test ETF classification heuristics."""

    def test_gold_etf(self):
        assert _classify_etf("GLD", "SPDR Gold Shares", {}) == "Hard Assets"

    def test_silver_etf(self):
        assert _classify_etf("SLV", "iShares Silver Trust", {}) == "Hard Assets"

    def test_uranium_etf(self):
        assert _classify_etf("URA", "Global X Uranium ETF", {}) == "Hard Assets"

    def test_copper_etf(self):
        assert _classify_etf("COPJ", "Sprott Junior Copper Miners ETF", {}) == "Hard Assets"

    def test_energy_etf(self):
        assert _classify_etf("XLE", "Energy Select Sector SPDR Fund", {}) == "Hard Assets"

    def test_bitcoin_etf(self):
        assert _classify_etf("FBTC", "Fidelity Wise Origin Bitcoin Fund", {}) == "Hard Assets"

    def test_emerging_markets_etf(self):
        assert _classify_etf("EEM", "iShares MSCI Emerging Markets ETF", {}) == "Emerging Markets"

    def test_korea_etf(self):
        assert _classify_etf("EWY", "iShares MSCI South Korea ETF", {}) == "Emerging Markets"

    def test_peru_etf(self):
        assert _classify_etf("EPU", "iShares MSCI Peru ETF", {}) == "Emerging Markets"

    def test_latin_america_etf(self):
        assert _classify_etf("ILF", "iShares Latin America 40 ETF", {}) == "Emerging Markets"

    def test_developed_markets_etf(self):
        assert _classify_etf("EFA", "iShares MSCI EAFE ETF", {}) == "Intl Developed"

    def test_europe_etf(self):
        assert _classify_etf("VGK", "Vanguard FTSE Europe ETF", {}) == "Intl Developed"

    def test_small_cap_etf(self):
        assert _classify_etf("IWM", "iShares Russell 2000 Small-Cap ETF", {}) == "US Small/Mid"

    def test_completion_etf(self):
        assert _classify_etf("VXF", "Vanguard Extended Market Completion Index", {}) == "US Small/Mid"

    def test_sp500_etf(self):
        assert _classify_etf("VOO", "Vanguard S&P 500 ETF", {}) == "US Large Cap"

    def test_bond_etf(self):
        assert _classify_etf("BND", "Vanguard Total Bond Market ETF", {}) == "Defensive/Income"

    def test_tips_etf(self):
        assert _classify_etf("STIP", "iShares 0-5 Year TIPS Bond ETF", {}) == "Defensive/Income"

    def test_dividend_etf(self):
        assert _classify_etf("SCHD", "Schwab U.S. Dividend Equity ETF", {}) == "Defensive/Income"

    def test_reit_etf(self):
        assert _classify_etf("VNQ", "Vanguard Real Estate ETF", {}) == "Defensive/Income"

    def test_unknown_etf(self):
        assert _classify_etf("XYZ", "Some Unknown ETF", {}) == "Other"


class TestClassifyInternational:
    """Test international stock classification."""

    def test_emerging_brazil(self):
        assert _classify_international("Brazil", {}) == "Emerging Markets"

    def test_emerging_china(self):
        assert _classify_international("China", {}) == "Emerging Markets"

    def test_emerging_india(self):
        assert _classify_international("India", {}) == "Emerging Markets"

    def test_emerging_korea(self):
        assert _classify_international("South Korea", {}) == "Emerging Markets"

    def test_emerging_argentina(self):
        assert _classify_international("Argentina", {}) == "Emerging Markets"

    def test_developed_uk(self):
        assert _classify_international("United Kingdom", {}) == "Intl Developed"

    def test_developed_japan(self):
        assert _classify_international("Japan", {}) == "Intl Developed"

    def test_developed_germany(self):
        assert _classify_international("Germany", {}) == "Intl Developed"

    def test_developed_canada(self):
        assert _classify_international("Canada", {}) == "Intl Developed"

    def test_developed_france(self):
        assert _classify_international("France", {}) == "Intl Developed"


# ---------------------------------------------------------------------------
# enrich_ticker() with mocked yfinance
# ---------------------------------------------------------------------------

class TestEnrichTicker:
    """Test enrich_ticker() with mocked yfinance responses."""

    @patch("threshold.data.adapters.yfinance_adapter.yf")
    def test_enrich_us_large_cap(self, mock_yf):
        """US large-cap stock should classify correctly."""
        mock_ticker = MagicMock()
        mock_ticker.info = _mock_stock_info(
            symbol="AAPL", name="Apple Inc.",
            sector="Technology", country="United States",
            market_cap=3_000_000_000_000,
        )
        mock_yf.Ticker.return_value = mock_ticker

        result = enrich_ticker("AAPL")
        assert result is not None
        assert result["name"] == "Apple Inc."
        assert result["type"] == "stock"
        assert result["sector"] == "Technology"
        assert result["alden_category"] == "US Large Cap"
        assert result["is_international"] is False

    @patch("threshold.data.adapters.yfinance_adapter.yf")
    def test_enrich_small_cap(self, mock_yf):
        """Small-cap stock (< $10B) should classify as US Small/Mid."""
        mock_ticker = MagicMock()
        mock_ticker.info = _mock_stock_info(
            symbol="WLDN", name="Willdan Group",
            sector="Industrials", country="United States",
            market_cap=800_000_000,
        )
        mock_yf.Ticker.return_value = mock_ticker

        result = enrich_ticker("WLDN")
        assert result is not None
        assert result["alden_category"] == "US Small/Mid"

    @patch("threshold.data.adapters.yfinance_adapter.yf")
    def test_enrich_international_emerging(self, mock_yf):
        """International stock from emerging market."""
        mock_ticker = MagicMock()
        mock_ticker.info = _mock_stock_info(
            symbol="CIB", name="Grupo Cibest",
            sector="Financials", country="Colombia",
            market_cap=5_000_000_000,
        )
        mock_yf.Ticker.return_value = mock_ticker

        result = enrich_ticker("CIB")
        assert result is not None
        assert result["is_international"] is True
        assert result["alden_category"] == "Emerging Markets"

    @patch("threshold.data.adapters.yfinance_adapter.yf")
    def test_enrich_international_developed(self, mock_yf):
        """International stock from developed market."""
        mock_ticker = MagicMock()
        mock_ticker.info = _mock_stock_info(
            symbol="DBSDY", name="DBS Group Holdings",
            sector="Financial Services", country="Singapore",
            market_cap=80_000_000_000,
        )
        mock_yf.Ticker.return_value = mock_ticker

        result = enrich_ticker("DBSDY")
        assert result is not None
        assert result["is_international"] is True
        assert result["alden_category"] == "Intl Developed"

    @patch("threshold.data.adapters.yfinance_adapter.yf")
    def test_enrich_etf(self, mock_yf):
        """ETF should be classified with heuristics."""
        mock_ticker = MagicMock()
        mock_ticker.info = _mock_etf_info("URA", "Global X Uranium ETF")
        mock_yf.Ticker.return_value = mock_ticker

        result = enrich_ticker("URA")
        assert result is not None
        assert result["type"] == "etf"
        assert result["alden_category"] == "Hard Assets"
        assert result["needs_review"] is True  # ETFs always flagged

    @patch("threshold.data.adapters.yfinance_adapter.yf")
    def test_enrich_gold_stock(self, mock_yf):
        """Gold mining stock should set is_gold flag."""
        mock_ticker = MagicMock()
        mock_ticker.info = _mock_stock_info(
            symbol="NEM", name="Newmont Mining Corporation",
            sector="Basic Materials", country="United States",
            market_cap=50_000_000_000,
        )
        mock_yf.Ticker.return_value = mock_ticker

        result = enrich_ticker("NEM")
        assert result is not None
        assert result["is_gold"] is True
        assert result["is_hard_money"] is True
        assert result["alden_category"] == "Hard Assets"

    @patch("threshold.data.adapters.yfinance_adapter.yf")
    def test_enrich_dot_symbol(self, mock_yf):
        """Symbols with dots should get yf_symbol override."""
        mock_ticker = MagicMock()
        mock_ticker.info = _mock_stock_info(
            symbol="BRK.B", name="Berkshire Hathaway Inc Class B",
            sector="Financial Services",
        )
        mock_yf.Ticker.return_value = mock_ticker

        result = enrich_ticker("BRK.B")
        assert result is not None
        assert result["yf_symbol"] == "BRK-B"

    @patch("threshold.data.adapters.yfinance_adapter.yf")
    def test_enrich_no_dot_symbol(self, mock_yf):
        """Normal symbols should not have yf_symbol override."""
        mock_ticker = MagicMock()
        mock_ticker.info = _mock_stock_info()
        mock_yf.Ticker.return_value = mock_ticker

        result = enrich_ticker("AAPL")
        assert result is not None
        assert result["yf_symbol"] is None

    @patch("threshold.data.adapters.yfinance_adapter.yf")
    def test_enrich_returns_none_on_no_data(self, mock_yf):
        """Should return None when no data found."""
        mock_ticker = MagicMock()
        mock_ticker.info = {"regularMarketPrice": None}
        mock_yf.Ticker.return_value = mock_ticker

        result = enrich_ticker("NONEXISTENT")
        assert result is None

    @patch("threshold.data.adapters.yfinance_adapter.yf")
    def test_enrich_returns_none_on_empty_info(self, mock_yf):
        """Should return None when info dict is empty."""
        mock_ticker = MagicMock()
        mock_ticker.info = {}
        mock_yf.Ticker.return_value = mock_ticker

        result = enrich_ticker("EMPTY")
        assert result is None

    @patch("threshold.data.adapters.yfinance_adapter.yf")
    def test_enrich_handles_exception(self, mock_yf):
        """Should return None on yfinance exception."""
        mock_yf.Ticker.side_effect = Exception("API error")

        result = enrich_ticker("ERROR")
        assert result is None


class TestEnrichDefaults:
    """Verify default values in enriched output."""

    @patch("threshold.data.adapters.yfinance_adapter.yf")
    def test_default_boolean_flags(self, mock_yf):
        """Boolean flags should default to False."""
        mock_ticker = MagicMock()
        mock_ticker.info = _mock_stock_info()
        mock_yf.Ticker.return_value = mock_ticker

        result = enrich_ticker("AAPL")
        assert result is not None
        assert result["is_crypto_exempt"] is False
        assert result["is_cash"] is False
        assert result["is_war_chest"] is False
        assert result["is_amplifier_trim"] is False
        assert result["is_defensive_add"] is False
        assert result["dd_override"] is None

    @patch("threshold.data.adapters.yfinance_adapter.yf")
    def test_energy_sector_classification(self, mock_yf):
        """Energy sector stocks should classify as Hard Assets."""
        mock_ticker = MagicMock()
        mock_ticker.info = _mock_stock_info(
            sector="Energy", name="Exxon Mobil",
            market_cap=400_000_000_000,
        )
        mock_yf.Ticker.return_value = mock_ticker

        result = enrich_ticker("XOM")
        assert result is not None
        assert result["alden_category"] == "Hard Assets"

    @patch("threshold.data.adapters.yfinance_adapter.yf")
    def test_utility_sector_classification(self, mock_yf):
        """Utility stocks should classify as Defensive/Income."""
        mock_ticker = MagicMock()
        mock_ticker.info = _mock_stock_info(
            sector="Utilities", name="NextEra Energy",
            market_cap=150_000_000_000,
        )
        mock_yf.Ticker.return_value = mock_ticker

        result = enrich_ticker("NEE")
        assert result is not None
        assert result["alden_category"] == "Defensive/Income"

    @patch("threshold.data.adapters.yfinance_adapter.yf")
    def test_real_estate_sector_classification(self, mock_yf):
        """Real estate stocks should classify as Defensive/Income."""
        mock_ticker = MagicMock()
        mock_ticker.info = _mock_stock_info(
            sector="Real Estate", name="Prologis Inc",
            market_cap=120_000_000_000,
        )
        mock_yf.Ticker.return_value = mock_ticker

        result = enrich_ticker("PLD")
        assert result is not None
        assert result["alden_category"] == "Defensive/Income"
