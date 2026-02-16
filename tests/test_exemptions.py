"""Tests for ticker exemptions — crypto halving cycle and cash/war chest."""

from __future__ import annotations

from unittest.mock import MagicMock

from threshold.engine.exemptions import (
    ExemptionResult,
    get_exempt_tickers,
    is_exempt_from_sell,
)

# ---------------------------------------------------------------------------
# Tests: ExemptionResult
# ---------------------------------------------------------------------------

class TestExemptionResult:
    def test_default_not_exempt(self):
        """Default ExemptionResult should not be exempt."""
        result = ExemptionResult()
        assert result.is_exempt is False
        assert result.exemption_type == ""


# ---------------------------------------------------------------------------
# Tests: is_exempt_from_sell
# ---------------------------------------------------------------------------

class TestIsExemptFromSell:
    def test_crypto_exempt(self):
        """Crypto-exempt tickers should be exempt."""
        ticker = {"symbol": "FBTC", "is_crypto_exempt": True, "is_cash": False}
        result = is_exempt_from_sell(ticker)
        assert result.is_exempt is True
        assert result.exemption_type == "crypto_halving"

    def test_cash_exempt(self):
        """Cash/war chest tickers should be exempt."""
        ticker = {"symbol": "STIP", "is_crypto_exempt": False, "is_cash": True}
        result = is_exempt_from_sell(ticker)
        assert result.is_exempt is True
        assert result.exemption_type == "cash"

    def test_cash_takes_priority(self):
        """Cash exemption should take priority over crypto."""
        ticker = {"symbol": "STIP", "is_crypto_exempt": True, "is_cash": True}
        result = is_exempt_from_sell(ticker)
        assert result.exemption_type == "cash"

    def test_normal_ticker_not_exempt(self):
        """Normal tickers should not be exempt."""
        ticker = {"symbol": "AAPL", "is_crypto_exempt": False, "is_cash": False}
        result = is_exempt_from_sell(ticker)
        assert result.is_exempt is False
        assert result.exemption_type == "none"

    def test_crypto_with_expiry_active(self):
        """Crypto exemption with future expiry should be active."""
        ticker = {"symbol": "MSTR", "is_crypto_exempt": True, "is_cash": False}
        config = MagicMock()
        config.scoring.crypto_exempt_expiry = "2030-12-31"

        result = is_exempt_from_sell(ticker, config)
        assert result.is_exempt is True
        assert result.expires_at == "2030-12-31"

    def test_crypto_with_expiry_expired(self):
        """Crypto exemption with past expiry should not be active."""
        ticker = {"symbol": "MSTR", "is_crypto_exempt": True, "is_cash": False}
        config = MagicMock()
        config.scoring.crypto_exempt_expiry = "2020-01-01"

        result = is_exempt_from_sell(ticker, config)
        assert result.is_exempt is False
        assert result.is_expired is True

    def test_crypto_no_config(self):
        """Crypto exemption without config should be exempt (no expiry)."""
        ticker = {"symbol": "FETH", "is_crypto_exempt": True, "is_cash": False}
        result = is_exempt_from_sell(ticker, config=None)
        assert result.is_exempt is True
        assert result.expires_at == ""

    def test_missing_flags_default_to_false(self):
        """Tickers with missing flags should not be exempt."""
        ticker = {"symbol": "XYZ"}
        result = is_exempt_from_sell(ticker)
        assert result.is_exempt is False


# ---------------------------------------------------------------------------
# Tests: get_exempt_tickers
# ---------------------------------------------------------------------------

class TestGetExemptTickers:
    def test_returns_exempt_tickers(self):
        """Should return dict of tickers with exemptions."""
        tickers = [
            {"symbol": "FBTC", "is_crypto_exempt": True, "is_cash": False},
            {"symbol": "AAPL", "is_crypto_exempt": False, "is_cash": False},
            {"symbol": "STIP", "is_crypto_exempt": False, "is_cash": True},
        ]
        result = get_exempt_tickers(tickers)
        assert "FBTC" in result
        assert "STIP" in result
        assert "AAPL" not in result

    def test_empty_tickers(self):
        """Should return empty dict for empty input."""
        result = get_exempt_tickers([])
        assert result == {}

    def test_no_exempt_tickers(self):
        """Should return empty dict when no exemptions."""
        tickers = [
            {"symbol": "AAPL", "is_crypto_exempt": False, "is_cash": False},
            {"symbol": "MSFT", "is_crypto_exempt": False, "is_cash": False},
        ]
        result = get_exempt_tickers(tickers)
        assert result == {}
