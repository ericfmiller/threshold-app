"""Tests for SA API fetcher — rating extraction and batch fetching."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from threshold.data.adapters.sa_api_fetcher import (
    _SA_API_BASE,
    fetch_all_ratings,
    fetch_ticker_rating,
)

# ---------------------------------------------------------------------------
# Mock responses
# ---------------------------------------------------------------------------

def _make_sa_response(quant_score=4.5, grades=None):
    """Build a mock SA API JSON response."""
    if grades is None:
        grades = {
            "momentum": {"grade": "A"},
            "profitability": {"grade": "B+"},
            "revisions": {"grade": "A-"},
            "growth": {"grade": "B"},
            "valuation": {"grade": "C+"},
        }
    attrs = {"quant": {"score": quant_score}}
    attrs.update(grades)
    return {"data": {"attributes": attrs}}


# ---------------------------------------------------------------------------
# Tests: fetch_ticker_rating
# ---------------------------------------------------------------------------

class TestFetchTickerRating:
    def test_extracts_quant_score(self):
        """Should extract numeric quant score from response."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = _make_sa_response(4.85)

        with patch("requests.get", return_value=mock_resp):
            result = fetch_ticker_rating("AAPL", cookies="fake_cookies")

        assert result is not None
        assert result["quantScore"] == 4.85

    def test_extracts_factor_grades(self):
        """Should extract all 5 factor grades."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = _make_sa_response()

        with patch("requests.get", return_value=mock_resp):
            result = fetch_ticker_rating("AAPL", cookies="fake_cookies")

        assert result["momentum"] == "A"
        assert result["profitability"] == "B+"
        assert result["revisions"] == "A-"
        assert result["growth"] == "B"
        assert result["valuation"] == "C+"

    def test_handles_numeric_quant_directly(self):
        """Should handle quant as a raw number (not a dict)."""
        response = {"data": {"attributes": {"quant": 3.72}}}
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = response

        with patch("requests.get", return_value=mock_resp):
            result = fetch_ticker_rating("MSFT", cookies="fake_cookies")

        assert result is not None
        assert result["quantScore"] == 3.72

    def test_handles_string_grades(self):
        """Should handle grades as direct strings (not dicts)."""
        response = {
            "data": {
                "attributes": {
                    "quant": {"score": 4.0},
                    "momentum": "B",
                    "growth": "A-",
                }
            }
        }
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = response

        with patch("requests.get", return_value=mock_resp):
            result = fetch_ticker_rating("TEST", cookies="fake_cookies")

        assert result is not None
        assert result["momentum"] == "B"
        assert result["growth"] == "A-"

    def test_returns_none_on_404(self):
        """Should return None when API returns non-200."""
        mock_resp = MagicMock()
        mock_resp.status_code = 404

        with patch("requests.get", return_value=mock_resp):
            result = fetch_ticker_rating("BAD", cookies="fake_cookies")

        assert result is None

    def test_returns_none_on_empty_attributes(self):
        """Should return None when response has no attributes."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"data": {"attributes": {}}}

        with patch("requests.get", return_value=mock_resp):
            result = fetch_ticker_rating("EMPTY", cookies="fake_cookies")

        assert result is None

    def test_returns_none_on_missing_data(self):
        """Should return None when response has no data key."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {}

        with patch("requests.get", return_value=mock_resp):
            result = fetch_ticker_rating("NODATA", cookies="fake_cookies")

        assert result is None

    def test_returns_none_on_no_cookies(self):
        """Should return None when no cookies available."""
        with patch("threshold.data.adapters.sa_api_fetcher._get_chrome_cookies", return_value=None):
            result = fetch_ticker_rating("AAPL")

        assert result is None

    def test_returns_none_on_exception(self):
        """Should return None when request throws exception."""
        with patch("requests.get", side_effect=ConnectionError("Network error")):
            result = fetch_ticker_rating("AAPL", cookies="fake_cookies")

        assert result is None

    def test_uses_correct_url(self):
        """Should construct the correct API URL."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = _make_sa_response()

        with patch("requests.get", return_value=mock_resp) as mock_get:
            fetch_ticker_rating("AAPL", cookies="fake_cookies")

            url = mock_get.call_args[0][0]
            assert url == f"{_SA_API_BASE}/symbols/AAPL/quant"

    def test_rounds_quant_to_2_decimals(self):
        """Quant score should be rounded to 2 decimal places."""
        response = {"data": {"attributes": {"quant": {"score": 4.123456}}}}
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = response

        with patch("requests.get", return_value=mock_resp):
            result = fetch_ticker_rating("AAPL", cookies="fake_cookies")

        assert result["quantScore"] == 4.12


# ---------------------------------------------------------------------------
# Tests: fetch_all_ratings
# ---------------------------------------------------------------------------

class TestFetchAllRatings:
    def test_returns_successful_only(self):
        """Should only include tickers that succeeded."""
        responses = {
            "AAPL": {"quantScore": 4.5},
            "MSFT": None,
            "GOOGL": {"quantScore": 4.2},
        }

        def mock_fetch(ticker, cookies=None, api_url=_SA_API_BASE):
            return responses.get(ticker)

        with (
            patch("threshold.data.adapters.sa_api_fetcher.fetch_ticker_rating", side_effect=mock_fetch),
            patch("threshold.data.adapters.sa_api_fetcher.time"),
        ):
            result = fetch_all_ratings(["AAPL", "MSFT", "GOOGL"], cookies="cookies", rate_delay=0)

        assert "AAPL" in result
        assert "MSFT" not in result
        assert "GOOGL" in result

    def test_returns_empty_when_no_cookies(self):
        """Should return empty dict when Chrome cookies unavailable."""
        with patch("threshold.data.adapters.sa_api_fetcher._get_chrome_cookies", return_value=None):
            result = fetch_all_ratings(["AAPL", "MSFT"])

        assert result == {}

    def test_empty_ticker_list(self):
        """Should handle empty ticker list."""
        result = fetch_all_ratings([], cookies="cookies")
        assert result == {}

    def test_respects_rate_delay(self):
        """Should sleep between API calls."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = _make_sa_response()

        with (
            patch("requests.get", return_value=mock_resp),
            patch("threshold.data.adapters.sa_api_fetcher.time") as mock_time,
        ):
            fetch_all_ratings(["AAPL", "MSFT", "GOOGL"], cookies="cookies", rate_delay=1.5)

            # Should sleep between calls (2 sleeps for 3 tickers)
            assert mock_time.sleep.call_count == 2
            mock_time.sleep.assert_called_with(1.5)
