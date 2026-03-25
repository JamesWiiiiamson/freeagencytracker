"""
Shared Sportradar HTTP client with rate limiting and retry logic.

Provides a single get() helper that:
  - Appends the API key to every request
  - Enforces a configurable per-request delay (trial keys = 1 req/sec)
  - Retries on transient HTTP errors (429, 5xx) with exponential back-off
  - Raises immediately on permanent errors (4xx other than 429)
"""

import time
import requests
from config import SPORTRADAR_BASE, SPORTRADAR_TRANSACTIONS_KEY, RATE_LIMIT_DELAY

# How many times to retry on transient failures before giving up.
_MAX_RETRIES = 3
# Base delay (seconds) for exponential back-off: 2, 4, 8 ...
_BACKOFF_BASE = 2


class SportradarClient:
    """Thin wrapper around requests that handles auth, rate limiting, and retries."""

    def __init__(
        self,
        api_key: str = SPORTRADAR_TRANSACTIONS_KEY,
        base_url: str = SPORTRADAR_BASE,
        rate_limit_delay: float = RATE_LIMIT_DELAY,
    ) -> None:
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._rate_limit_delay = rate_limit_delay
        self._session = requests.Session()

    def get(self, path: str, **params) -> dict:
        """
        Perform a GET request to the Sportradar API.

        Args:
            path: Path relative to the base URL, e.g.
                  '/league/2026/03/12/transactions.json'
            **params: Additional query parameters (api_key is added automatically).

        Returns:
            Parsed JSON response as a dict.

        Raises:
            requests.HTTPError: On non-retryable HTTP errors.
            RuntimeError: If all retries are exhausted.
        """
        url = f"{self._base_url}/{path.lstrip('/')}"
        params["api_key"] = self._api_key

        for attempt in range(1, _MAX_RETRIES + 1):
            # Enforce rate limit before every request (incl. retries).
            time.sleep(self._rate_limit_delay)

            response = self._session.get(url, params=params, timeout=30)

            if response.status_code == 200:
                return response.json()

            if response.status_code == 429 or response.status_code >= 500:
                # Transient – retry with exponential back-off.
                wait = _BACKOFF_BASE ** attempt
                print(
                    f"[SportradarClient] HTTP {response.status_code} on attempt "
                    f"{attempt}/{_MAX_RETRIES}. Retrying in {wait}s …"
                )
                time.sleep(wait)
                continue

            # Permanent error (404, 401, etc.) – fail fast.
            response.raise_for_status()

        raise RuntimeError(
            f"[SportradarClient] All {_MAX_RETRIES} retries exhausted for {url}"
        )
