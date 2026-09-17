"""Signed HTTP client for the Kalshi trade API.

Mostly GET endpoints for market/account data. `create_order` is the one write
endpoint, added 2026-09-17 to back live auto-execution (engine/execution.py,
engine/live.py) — see those modules for the safety gating around it. This
client itself does no gating; it just sends what it's given.
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

    def _post(self, endpoint: str, body: dict, retries: int = 2) -> dict[str, Any]:
        """POST {base}{endpoint} with a JSON body, signed the same way as GET
        (Kalshi signs timestamp+METHOD+path only, never the body).

        Retried on network/5xx/429 like `_get`. Safe to retry an order POST
        specifically because callers pass a stable `client_order_id` — Kalshi
        dedups on it, so a retry after a timeout returns the original order
        instead of creating a second one."""
        path = self.path_prefix + endpoint
        url = self.base + endpoint
        for attempt in range(retries + 1):
            self._throttle()
            headers = build_headers(self.creds.key_id, self.creds.private_key, "POST", path)
            try:
                resp = self.session.post(url, headers=headers, json=body, timeout=20)
            except requests.RequestException as e:
                if attempt < retries:
                    time.sleep(1.5 * (attempt + 1))
                    continue
                raise KalshiError(f"Network error on {endpoint}: {e}") from e

            if resp.status_code in (200, 201):
                return resp.json()
            if resp.status_code in (429, 500, 502, 503, 504) and attempt < retries:
                wait = 2.0 * (attempt + 1)
                log.warning("HTTP %s on %s; retrying in %.1fs", resp.status_code, endpoint, wait)
                time.sleep(wait)
                continue
            raise KalshiError(f"HTTP {resp.status_code} on {endpoint}: {resp.text[:400]}")
        raise KalshiError(f"Exhausted retries on {endpoint}")

    def _delete(self, endpoint: str, retries: int = 2) -> dict[str, Any]:
        """DELETE {base}{endpoint}, signed like GET/POST."""
        path = self.path_prefix + endpoint
        url = self.base + endpoint
        for attempt in range(retries + 1):
            self._throttle()
            headers = build_headers(self.creds.key_id, self.creds.private_key, "DELETE", path)
            try:
                resp = self.session.delete(url, headers=headers, timeout=20)
            except requests.RequestException as e:
                if attempt < retries:
                    time.sleep(1.5 * (attempt + 1))
                    continue
                raise KalshiError(f"Network error on {endpoint}: {e}") from e

            if resp.status_code in (200, 201, 204):
                return resp.json() if resp.content else {}
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

    def create_order(self, ticker: str, side: str, action: str, count: int,
                     limit_price: float, client_order_id: str,
                     time_in_force: str = "good_till_canceled",
                     self_trade_prevention_type: str = "taker_at_cross") -> dict:
        """POST /portfolio/events/orders (v2) — places a REAL limit order.

        Kalshi deprecated the old /portfolio/orders endpoint (confirmed 2026-09-17,
        HTTP 410 "deprecated_v1_order_endpoint") in favor of this one, which quotes
        EVERYTHING from the YES leg -- there is no separate NO leg in this endpoint's
        schema at all. Verbatim from the `side` field's own description (both
        create-order-v2 and batch-create-orders-v2 docs, same wording):
            "Side of the book for an order or trade. For event markets, this refers
            to the YES leg only: `bid` means buy YES, `ask` means sell YES. (Selling
            YES is economically equivalent to buying NO at `1 - price`, but this
            endpoint quotes everything from the YES side.)"
        That is the exact, textual conversion rule this method applies for side="no":
        book_side="ask", price=(1 - limit_price). This mirrors Kalshi's own legacy
        action/side -> outcome_side/book_side table (buy no -> book_side=ask) and is
        internally self-consistent with a resting order's fill behavior (a deeply
        unattractive NO limit, e.g. buy-no-at-2c when NO trades near 85c, becomes a
        deeply unattractive YES ask at 98c when YES trades near 12c -- same distance
        from market on both sides). Smoke-tested live 2026-09-17 for the YES side
        (1 contract, rested unfilled, canceled cleanly); the NO-side conversion
        itself should get the same live rest-then-cancel check before being trusted
        unattended overnight -- see CLAUDE.md's "Live order execution" section.
        """
        if side not in ("yes", "no"):
            raise ValueError(f"invalid side {side!r}")
        if action != "buy":
            raise NotImplementedError(f"action={action!r} not implemented for the v2 order "
                                      f"endpoint (only 'buy' is verified/wired).")
        book_side = "bid" if side == "yes" else "ask"
        v2_price = limit_price if side == "yes" else (1.0 - limit_price)
        body: dict[str, Any] = {
            "ticker": ticker,
            "client_order_id": client_order_id,
            "side": book_side,
            "count": f"{count:.2f}",
            "price": f"{v2_price:.2f}",
            "time_in_force": time_in_force,
            "self_trade_prevention_type": self_trade_prevention_type,
        }
        return self._post("/portfolio/events/orders", body)

    def cancel_order(self, order_id: str) -> dict:
        """DELETE /portfolio/events/orders/{order_id} (v2) — cancels a resting
        order (any unfilled remainder). Already-filled quantity stays filled."""
        return self._delete(f"/portfolio/events/orders/{order_id}")

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
