"""HTTP helper for the Open-Meteo APIs."""
from __future__ import annotations

import time

import requests

RETRY_STATUS = {429, 500, 502, 503, 504}


def get_json(url: str, params: dict, timeout: int = 120, retries: int = 5) -> dict:
    """GET JSON, retrying connection errors, timeouts, rate limits and 5xx with backoff.
    Other 4xx errors (bad parameters) raise immediately."""
    for attempt in range(1, retries + 1):
        try:
            r = requests.get(url, params=params, timeout=timeout)
            if r.status_code not in RETRY_STATUS:
                r.raise_for_status()
                return r.json()
            err = f"HTTP {r.status_code}"
        except (requests.ConnectionError, requests.Timeout) as exc:
            err = type(exc).__name__
        if attempt == retries:
            raise RuntimeError(f"GET {url} failed after {retries} attempts ({err})")
        wait = 5 * 2 ** (attempt - 1)
        print(f"{err}, retrying in {wait}s")
        time.sleep(wait)
