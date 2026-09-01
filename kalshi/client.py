"""Signed HTTP client for the Kalshi trade API (read-only usage here).

This client only calls GET endpoints. It never places, cancels, or modifies
orders — the agent is recommend-only by design.
"""
from __future__ import annotations

import logging
import time
from typing import Any, Iterator, Optional

import requests

from config.credentials import KalshiCredentials, load_credentials
from config.settings import Settings, DEFAULTS
from kalshi.auth import build_headers

log = logging.getLogger("kalshi.client")


class KalshiError(RuntimeError):
    pass


class KalshiClient:
    def __init__(self, settings: Settings = DEFAULTS, creds: Optional[KalshiCredentials] = None):
        self.settings = settings
        self.creds = creds or load_credentials()
        self.base = settings.base_url()
        # The path prefix that must be part of the signed path.
        self.path_prefix = "/trade-api/v2"
        self.session = requests.Session()
        self._min_interval = 0.12  # ~8 req/s, under the basic-tier ceiling
        self._last_call = 0.0

    # --- low level -------------------------------------------------------
    def _throttle(self) -> None:
        dt = time.time() - self._last_call
        if dt < self._min_interval:
            time.sleep(self._min_interval - dt)
        self._last_call = time.time()

    def _get(self, endpoint: str, params: Optional[dict] = None, retries: int = 3) -> dict[str, Any]:
        """GET {base}{endpoint}. `endpoint` starts with '/', e.g. '/markets'."""
        path = self.path_prefix + endpoint
        url = self.base + endpoint
        for attempt in range(retries + 1):
            self._throttle()
            headers = build_headers(self.creds.key_id, self.creds.private_key, "GET", path)
            try:
                resp = self.session.get(url, headers=headers, params=params, timeout=20)
            except requests.RequestException as e:
                if attempt < retries:
                    time.sleep(1.5 * (attempt + 1))
                    continue
                raise KalshiError(f"Network error on {endpoint}: {e}") from e

            if resp.status_code == 200:
                return resp.json()
            if resp.status_code in (429, 500, 502, 503, 504) and attempt < retries:
                wait = 2.0 * (attempt + 1)
                log.warning("HTTP %s on %s; retrying in %.1fs", resp.status_code, endpoint, wait)
                time.sleep(wait)
                continue
            raise KalshiError(f"HTTP {resp.status_code} on {endpoint}: {resp.text[:400]}")
        raise KalshiError(f"Exhausted retries on {endpoint}")

    def _paginate(self, endpoint: str, params: dict, key: str, max_pages: int = 40) -> Iterator[dict]:
        params = dict(params)
        pages = 0
        while True:
            data = self._get(endpoint, params)
            for item in data.get(key, []) or []:
                yield item
            cursor = data.get("cursor")
            pages += 1
            if not cursor or pages >= max_pages:
                break
            params["cursor"] = cursor

    # --- public endpoints ------------------------------------------------
    def exchange_status(self) -> dict:
        return self._get("/exchange/status")

    def balance(self) -> dict:
        """Portfolio balance — used as the auth smoke test."""
        return self._get("/portfolio/balance")

    def get_events(self, series_ticker: Optional[str] = None, status: str = "open",
                   with_nested_markets: bool = False, limit: int = 200) -> list[dict]:
        params: dict[str, Any] = {"limit": limit, "status": status}
        if series_ticker:
            params["series_ticker"] = series_ticker
        if with_nested_markets:
            params["with_nested_markets"] = "true"
        return list(self._paginate("/events", params, "events"))

    def get_markets(self, series_ticker: Optional[str] = None, event_ticker: Optional[str] = None,
                    status: Optional[str] = "open", limit: int = 200) -> list[dict]:
        params: dict[str, Any] = {"limit": limit}
        if status:
            params["status"] = status
        if series_ticker:
            params["series_ticker"] = series_ticker
        if event_ticker:
            params["event_ticker"] = event_ticker
        return list(self._paginate("/markets", params, "markets"))

    def get_market(self, ticker: str) -> dict:
        return self._get(f"/markets/{ticker}").get("market", {})

    def get_orderbook(self, ticker: str, depth: int = 10) -> dict:
        return self._get(f"/markets/{ticker}/orderbook", {"depth": depth}).get("orderbook", {})

    def get_trades(self, ticker: str, limit: int = 100) -> list[dict]:
        data = self._get("/markets/trades", {"ticker": ticker, "limit": limit})
        return data.get("trades", []) or []

    # --- read-only portfolio endpoints (no execution) ---
    def get_positions(self) -> dict:
        """Current market + event positions."""
        return self._get("/portfolio/positions")

    def get_orders(self, status: Optional[str] = None) -> list[dict]:
        """Resting/executed orders. status in {resting, canceled, executed} or None for all."""
        params = {"status": status} if status else None
        return self._get("/portfolio/orders", params).get("orders", []) or []

    def get_fills(self, limit: int = 100) -> list[dict]:
        """Recent fills (executed trades on the account)."""
        return self._get("/portfolio/fills", {"limit": limit}).get("fills", []) or []
