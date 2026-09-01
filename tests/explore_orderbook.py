"""Dump the RAW orderbook + trades payloads to learn the exact schema."""
from __future__ import annotations

import json

from kalshi.client import KalshiClient


def main() -> None:
    c = KalshiClient()
    # Find a market that actually has resting size per the market object.
    markets = c.get_markets(series_ticker="KXMLBGAME", status="open", limit=200)

    def size(m):
        try:
            return float(m.get("yes_bid_size_fp") or 0) + float(m.get("yes_ask_size_fp") or 0)
        except (TypeError, ValueError):
            return 0.0

    markets.sort(key=size, reverse=True)
    for m in markets[:3]:
        tk = m["ticker"]
        print(f"\n===== {tk}  (bid_size={m.get('yes_bid_size_fp')} ask_size={m.get('yes_ask_size_fp')}) =====")
        raw = c._get(f"/markets/{tk}/orderbook", {"depth": 10})
        print("RAW orderbook response:")
        print(json.dumps(raw, indent=2, default=str))


if __name__ == "__main__":
    main()
