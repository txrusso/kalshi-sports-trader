"""Auth smoke test: prove signed requests work against the live Kalshi API.

Run from the project root:  py -3 -m tests.smoke_auth
"""
from __future__ import annotations

import json

from kalshi.client import KalshiClient


def main() -> None:
    client = KalshiClient()
    print("Key id:", client.creds.key_id[:8], "... (loaded OK)")

    print("\n[1] Exchange status (public):")
    print(json.dumps(client.exchange_status(), indent=2))

    print("\n[2] Portfolio balance (requires valid signature):")
    bal = client.balance()
    print(json.dumps(bal, indent=2))
    cents = bal.get("balance")
    if cents is not None:
        print(f"    -> Balance: ${cents/100:,.2f}")

    print("\n[3] Sample MLB markets:")
    markets = client.get_markets(series_ticker="KXMLBGAME", status="open", limit=5)
    print(f"    Found {len(markets)} markets (showing up to 3):")
    for m in markets[:3]:
        print(f"    - {m.get('ticker')}: {m.get('title')} "
              f"yes_bid={m.get('yes_bid')} yes_ask={m.get('yes_ask')} vol={m.get('volume')}")

    print("\nAUTH OK ✅" if "balance" in bal else "\nAuth returned unexpected payload")


if __name__ == "__main__":
    main()
