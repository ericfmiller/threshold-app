"""Seeking Alpha API data fetcher using browser cookies.

Fetches fresh quant scores and factor grades from the SA API for tickers
whose data is older than a configurable threshold. Requires Chrome to be
logged into Seeking Alpha with an active SA Premium subscription.

Uses browser-cookie3 to extract Chrome cookies automatically.
"""

from __future__ import annotations

import logging
import time
from typing import Any

logger = logging.getLogger(__name__)

# SA API endpoints
_SA_API_BASE = "https://seekingalpha.com/api/v3"
_SA_QUANT_ENDPOINT = "/symbols/{ticker}/quant"

# Headers to mimic a browser request
_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json",
    "Referer": "https://seekingalpha.com/",
}


def _get_chrome_cookies() -> Any:
    """Get Chrome cookies for seekingalpha.com.

    Returns a cookie jar or None if browser-cookie3 is not available
    or Chrome is not running/accessible.
    """
    try:
        import browser_cookie3  # type: ignore[import-untyped]

        return browser_cookie3.chrome(domain_name=".seekingalpha.com")
    except Exception as e:
        logger.warning("Could not get Chrome cookies: %s", e)
        return None


def fetch_ticker_rating(
    ticker: str,
    cookies: Any = None,
    api_url: str = _SA_API_BASE,
) -> dict[str, Any] | None:
    """Fetch quant rating and factor grades for a single ticker.

    Parameters
    ----------
    ticker : str
        The ticker symbol to fetch.
    cookies : Any
        Cookie jar from browser-cookie3. If None, will attempt to get from Chrome.
    api_url : str
        Base URL for the SA API.

    Returns
    -------
    dict | None
        Dict with {quantScore, momentum, profitability, revisions, growth, valuation}
        or None if the fetch fails.
    """
    import requests

    if cookies is None:
        cookies = _get_chrome_cookies()
        if cookies is None:
            return None

    url = f"{api_url}/symbols/{ticker}/quant"

    try:
        resp = requests.get(url, cookies=cookies, headers=_HEADERS, timeout=15)
        if resp.status_code != 200:
            logger.debug("SA API returned %d for %s", resp.status_code, ticker)
            return None

        data = resp.json()
        attrs = data.get("data", {}).get("attributes", {})
        if not attrs:
            return None

        # Extract the fields we need
        result: dict[str, Any] = {}

        # Quant score (numeric 1-5)
        quant = attrs.get("quant", {})
        if isinstance(quant, dict):
            score = quant.get("score")
            if score is not None:
                result["quantScore"] = round(float(score), 2)
        elif isinstance(quant, (int, float)):
            result["quantScore"] = round(float(quant), 2)

        # Factor grades (letter grades)
        grade_map = {
            "momentum": "momentum",
            "profitability": "profitability",
            "revisions": "revisions",
            "growth": "growth",
            "valuation": "valuation",
        }

        for api_key, our_key in grade_map.items():
            factor = attrs.get(api_key, {})
            if isinstance(factor, dict):
                grade = factor.get("grade")
                if grade:
                    result[our_key] = str(grade)
            elif isinstance(factor, str) and factor:
                result[our_key] = factor

        return result if result else None

    except Exception as e:
        logger.debug("Failed to fetch SA rating for %s: %s", ticker, e)
        return None


def fetch_all_ratings(
    tickers: list[str],
    cookies: Any = None,
    rate_delay: float = 1.0,
) -> dict[str, dict[str, Any]]:
    """Batch fetch SA ratings for multiple tickers with rate limiting.

    Parameters
    ----------
    tickers : list[str]
        List of ticker symbols to fetch.
    cookies : Any
        Cookie jar from browser-cookie3. If None, will attempt to get from Chrome.
    rate_delay : float
        Seconds to wait between API calls (default 1.0).

    Returns
    -------
    dict[str, dict]
        Mapping of {ticker: rating_data}. Only includes tickers that succeeded.
    """
    if cookies is None:
        cookies = _get_chrome_cookies()
        if cookies is None:
            logger.warning("No Chrome cookies available for SA API")
            return {}

    results: dict[str, dict[str, Any]] = {}
    success = 0
    failed = 0

    for i, ticker in enumerate(tickers):
        rating = fetch_ticker_rating(ticker, cookies=cookies)
        if rating:
            results[ticker] = rating
            success += 1
        else:
            failed += 1

        if i < len(tickers) - 1:
            time.sleep(rate_delay)

    logger.info(
        "SA API fetch: %d/%d succeeded, %d failed",
        success,
        len(tickers),
        failed,
    )
    return results
