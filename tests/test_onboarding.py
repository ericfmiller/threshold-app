"""Tests for ticker onboarding — classification and discovery."""

from __future__ import annotations

from threshold.data.onboarding import (
    SKIP_TICKERS,
    OnboardingResult,
    classify_etf,
    classify_stock,
)

# ---------------------------------------------------------------------------
# Tests: classify_etf
# ---------------------------------------------------------------------------

class TestClassifyETF:
    def test_gold_etf(self):
        """Gold ETFs should be Hard Assets."""
        result = classify_etf("SPDR Gold Trust")
        assert result["alden_category"] == "Hard Assets"
        assert result["is_gold"] is True
        assert result["is_hard_money"] is True

    def test_crypto_etf(self):
        """Crypto ETFs should be Hard Assets."""
        result = classify_etf("Fidelity Wise Origin Bitcoin Fund")
        assert result["alden_category"] == "Hard Assets"
        assert result["is_crypto"] is True

    def test_uranium_etf(self):
        """Uranium/energy ETFs should be Hard Assets."""
        result = classify_etf("Global X Uranium ETF")
        assert result["alden_category"] == "Hard Assets"

    def test_emerging_market_etf(self):
        """Emerging market ETFs should be EM."""
        result = classify_etf("iShares MSCI Emerging Markets")
        assert result["alden_category"] == "Emerging Markets"
        assert result["is_international"] is True

    def test_latin_america_etf(self):
        """Latin America ETFs should be EM."""
        result = classify_etf("iShares Latin America 40 ETF")
        assert result["alden_category"] == "Emerging Markets"

    def test_developed_intl_etf(self):
        """Developed international ETFs should be Intl Developed."""
        result = classify_etf("Vanguard FTSE Europe ETF")
        assert result["alden_category"] == "Intl Developed"

    def test_bond_etf(self):
        """Bond ETFs should be Defensive/Income."""
        result = classify_etf("iShares TIPS Bond ETF")
        assert result["alden_category"] == "Defensive/Income"
        assert result["is_cash"] is True
        assert result["is_war_chest"] is True

    def test_dividend_etf(self):
        """Dividend ETFs should be US Large Cap."""
        result = classify_etf("Schwab US Dividend Equity ETF")
        assert result["alden_category"] == "US Large Cap"

    def test_small_cap_etf(self):
        """Small cap ETFs should be US Small/Mid."""
        result = classify_etf("Vanguard Small Cap Index")
        assert result["alden_category"] == "US Small/Mid"

    def test_reit_etf(self):
        """REIT ETFs should be Defensive/Income."""
        result = classify_etf("Vanguard Real Estate ETF")
        assert result["alden_category"] == "Defensive/Income"

    def test_unknown_etf_needs_review(self):
        """Unclassifiable ETFs should need review."""
        result = classify_etf("Some Obscure Leveraged 3x Product")
        assert result["needs_review"] is True
        assert result["alden_category"] == "Other"

    def test_all_etfs_have_type(self):
        """All ETF classifications should set type=etf."""
        for name in ["Gold ETF", "Bitcoin Fund", "S&P 500 ETF"]:
            result = classify_etf(name)
            assert result["type"] == "etf"


# ---------------------------------------------------------------------------
# Tests: classify_stock
# ---------------------------------------------------------------------------

class TestClassifyStock:
    def test_us_large_cap(self):
        """US stocks with large market cap should be US Large Cap."""
        result = classify_stock("Apple Inc.", "Technology", "United States", 3_000_000_000_000)
        assert result["alden_category"] == "US Large Cap"

    def test_us_small_cap(self):
        """US stocks with small market cap should be US Small/Mid."""
        result = classify_stock("Small Corp", "Technology", "United States", 500_000_000)
        assert result["alden_category"] == "US Small/Mid"

    def test_international_developed(self):
        """Stocks from developed markets should be Intl Developed."""
        result = classify_stock("DBS Group", "Financial Services", "Singapore", 50_000_000_000)
        assert result["alden_category"] == "Intl Developed"
        assert result["is_international"] is True

    def test_emerging_market(self):
        """Stocks from emerging markets should be Emerging Markets."""
        result = classify_stock("Grupo Cibest", "Financial Services", "Colombia", 20_000_000_000)
        assert result["alden_category"] == "Emerging Markets"
        assert result["is_international"] is True

    def test_crypto_stock(self):
        """Bitcoin-related stocks should be Hard Assets + crypto exempt."""
        result = classify_stock("Strategy Inc Bitcoin Treasury", "Technology", "United States", 10_000_000_000)
        assert result["is_crypto"] is True
        assert result["is_crypto_exempt"] is True
        assert result["alden_category"] == "Hard Assets"

    def test_gold_stock(self):
        """Gold mining stocks should be Hard Assets."""
        result = classify_stock("Newmont Mining Corp", "Basic Materials", "United States", 50_000_000_000)
        assert result["is_gold"] is True
        assert result["is_hard_money"] is True
        assert result["alden_category"] == "Hard Assets"

    def test_no_market_cap_defaults_safely(self):
        """Stocks with no market cap should default to US Large Cap."""
        result = classify_stock("Unknown Corp", "", "", 0)
        assert result["alden_category"] == "US Large Cap"

    def test_no_sector_no_market_cap_needs_review(self):
        """Stocks with no sector and no market cap should need review."""
        result = classify_stock("Mystery Corp", "", "", 0)
        assert result["needs_review"] is True


# ---------------------------------------------------------------------------
# Tests: SKIP_TICKERS
# ---------------------------------------------------------------------------

class TestSkipTickers:
    def test_contains_expected_values(self):
        """SKIP_TICKERS should contain known special values."""
        assert "CASH" in SKIP_TICKERS
        assert "TOTAL" in SKIP_TICKERS
        assert "ACCOUNT TOTAL" in SKIP_TICKERS
        assert "" in SKIP_TICKERS

    def test_real_tickers_not_skipped(self):
        """Real tickers should not be in SKIP_TICKERS."""
        assert "AAPL" not in SKIP_TICKERS
        assert "MSFT" not in SKIP_TICKERS
        assert "SPY" not in SKIP_TICKERS


# ---------------------------------------------------------------------------
# Tests: OnboardingResult
# ---------------------------------------------------------------------------

class TestOnboardingResult:
    def test_default_values(self):
        """Default OnboardingResult should have zero counts."""
        result = OnboardingResult()
        assert result.new_count == 0
        assert result.review_needed == 0
        assert result.new_tickers == []
        assert result.errors == []
