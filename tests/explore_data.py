"""Inspect real Kalshi MLB data shapes so downstream code matches reality."""
from __future__ import annotations

import json

from kalshi.client import KalshiClient


def main() -> None:
    c = KalshiClient()
    markets = c.get_markets(series_ticker="KXMLBGAME", status="open", limit=200)
    print(f"Total open KXMLBGAME markets: {len(markets)}")

    # Rank by volume to find the most active (money is flowing there).
    def vol(m):
        return m.get("volume") or 0
    active = sorted(markets, key=vol, reverse=True)

    print("\n--- Top 8 by volume ---")
    for m in active[:8]:
        print(f"{m.get('ticker'):40} vol={m.get('volume')} oi={m.get('open_interest')} "
              f"yb={m.get('yes_bid')} ya={m.get('yes_ask')} last={m.get('last_price')} "
              f"close={m.get('close_time')}")

    # Full field dump of the single most active market.
    top = active[0]
    print("\n--- Full field dump of most active market ---")
    print(json.dumps(top, indent=2, default=str))

    tk = top["ticker"]
    print(f"\n--- Orderbook for {tk} ---")
    print(json.dumps(c.get_orderbook(tk, depth=10), indent=2, default=str))

    print(f"\n--- Recent trades for {tk} (up to 5) ---")
    trades = c.get_trades(tk, limit=5)
    print(f"count={len(trades)}")
    print(json.dumps(trades[:5], indent=2, default=str))


if __name__ == "__main__":
    main()
