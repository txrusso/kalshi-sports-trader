"""Offline lifecycle tests for engine/maker.py -- no network, no real money.

This path places REAL orders, so the cases that matter are the ones that cost
money when they go wrong: double-placing after a re-peg, losing a partial fill,
recording a bet that never filled, and leaving an order resting past game time.

Run:  .venv\\Scripts\\python.exe -m tests.test_maker
"""
from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta, timezone

from config.settings import DEFAULTS
from engine.maker import (CANCELED, FALLBACK_FILLED, FILLED, MakerManager,
                          ManagedOrder, maker_price)
from kalshi.normalize import OrderBook

SETTINGS = replace(DEFAULTS, maker_mode=True, maker_improve_cents=1.0,
                   maker_taker_fallback_min=5.0, max_order_cost_usd=1000.0)


class FakeClient:
    """Minimal stand-in for KalshiClient: scripted book, scripted fills."""

    def __init__(self, yes_levels, no_levels, balance_usd=100.0):
        self.yes_levels = list(yes_levels)
        self.no_levels = list(no_levels)
        self.balance_usd = balance_usd
        self.orders: list[dict] = []
        self.canceled: list[str] = []
        self.fills_by_order: dict[str, list[dict]] = {}
        self.resting: list[dict] = []
        self._n = 0

    # --- reads ---
    def balance(self):
        return {"balance": self.balance_usd * 100}

    def get_orderbook(self, ticker, depth=10):
        return {"yes_dollars": [[p, s] for p, s in self.yes_levels],
                "no_dollars": [[p, s] for p, s in self.no_levels]}

    def get_market(self, ticker):
        return {"ticker": ticker, "status": "active",
                "yes_bid_dollars": "0.46", "yes_ask_dollars": "0.50",
                "no_bid_dollars": "0.50", "no_ask_dollars": "0.54",
                "volume_fp": "100", "open_interest_fp": "100"}

    def get_fills(self, limit=100, order_id=None, ticker=None):
        return list(self.fills_by_order.get(order_id, []))

    def get_orders(self, status=None):
        return list(self.resting)

    # --- writes ---
    def create_order(self, ticker, side, action, count, limit_price,
                     client_order_id, post_only=False, **kw):
        self._n += 1
        oid = f"order-{self._n}"
        self.orders.append({"order_id": oid, "ticker": ticker, "side": side,
                            "count": count, "price": limit_price,
                            "post_only": post_only, "coid": client_order_id})
        return {"order_id": oid, "client_order_id": client_order_id,
                "fill_count": 0, "remaining_count": count}

    def cancel_order(self, order_id):
        self.canceled.append(order_id)
        return {}

    # --- test helper ---
    def fill(self, order_id, count, price, is_taker=False):
        self.fills_by_order.setdefault(order_id, []).append(
            {"count_fp": str(count), "outcome_side": "yes",
             "yes_price_dollars": str(price), "no_price_dollars": str(round(1 - price, 2)),
             "fee_cost": "0.000000", "is_taker": is_taker, "order_id": order_id})


def _bet(event_key="EV1"):
    return {"event_key": event_key, "ticker": "KXMLBTOTAL-X-8", "side": "YES",
            "label": "A vs B Over 7.5", "entry_price": 0.50, "contracts": 4.0,
            "first_pitch": "2026-09-22T23:00:00+00:00"}


def _mgr(client, recorded):
    return MakerManager(client, SETTINGS, on_fill=lambda o, b: recorded.append(b))


results = []


def check(name, cond, detail=""):
    results.append((name, bool(cond)))
    print(f"  [{'ok' if cond else 'FAIL'}] {name}" + (f"  -- {detail}" if detail and not cond else ""))


def main() -> None:
    now = datetime(2026, 9, 22, 22, 0, tzinfo=timezone.utc)
    later = now + timedelta(minutes=60)

    print("pricing")
    b = OrderBook(yes_levels=[(0.40, 50), (0.46, 12)], no_levels=[(0.48, 20), (0.50, 40)])
    check("rests 1c above the bid", maker_price(b, "yes", 0.55) == 0.47)
    check("never crosses the ask", maker_price(b, "yes", 0.55) < b.best_yes_ask)
    # The cap binds only at or below the bid, since the quote is always bid+1c.
    check("respects the model cap", maker_price(b, "yes", 0.46) == 0.46)
    check("cap below the bid -> None (would never fill)",
          maker_price(b, "yes", 0.45) is None)
    tight = OrderBook(yes_levels=[(0.49, 10)], no_levels=[(0.50, 10)])
    check("joins the bid on a 1c spread", maker_price(tight, "yes", 0.60) == 0.49)
    check("far-below cap -> None", maker_price(b, "yes", 0.20) is None)
    check("empty book -> None", maker_price(OrderBook(), "yes", 0.50) is None)

    print("placement")
    c = FakeClient([(0.46, 12)], [(0.50, 40)])
    recorded = []
    m = _mgr(c, recorded)
    o = m.place("KXMLBTOTAL-X-8", "yes", 4.0, 0.50, later, _bet(), "EV1")
    check("places one post-only order", len(c.orders) == 1 and c.orders[0]["post_only"] is True)
    check("priced at bid+1c", c.orders[0]["price"] == 0.47)
    check("reserves capital", m.reserved_usd() == round(4.0 * 0.47, 4))

    print("idempotency")
    m.place("KXMLBTOTAL-X-8", "yes", 4.0, 0.50, later, _bet(), "EV1")
    check("re-offering the same event does not double-place", len(c.orders) == 1)

    print("partial fill then re-peg")
    c.fill("order-1", 1.5, 0.47)
    c.yes_levels = [(0.48, 30)]                      # book moved up -> re-peg to 0.49
    m.poll(now)
    check("partial fill tracked", o.filled_contracts == 1.5)
    check("canceled before re-placing", "order-1" in c.canceled)
    check("re-pegged to the new price", len(c.orders) == 2 and c.orders[1]["price"] == 0.49)
    check("re-places only the REMAINDER", c.orders[1]["count"] == 2.5,
          f"got {c.orders[1]['count']}")
    check("nothing recorded while still working", recorded == [])

    print("completion")
    c.fill("order-2", 2.5, 0.49)
    m.poll(now)
    check("reaches FILLED", o.state == FILLED)
    check("records exactly one ledger row", len(recorded) == 1)
    row = recorded[0] if recorded else {}
    check("records the REAL filled size", row.get("contracts") == 4.0)
    check("records the blended avg price",
          abs(row.get("entry_price", 0) - (1.5 * 0.47 + 2.5 * 0.49) / 4.0) < 1e-6,
          f"got {row.get('entry_price')}")
    check("marks execution as maker", row.get("execution") == "maker")
    check("no capital left reserved", m.reserved_usd() == 0.0)

    print("deadline -> taker fallback")
    c2 = FakeClient([(0.46, 12)], [(0.50, 40)])
    rec2 = []
    m2 = _mgr(c2, rec2)
    o2 = m2.place("KXMLBTOTAL-Y-8", "yes", 2.0, 0.50, now + timedelta(minutes=1), _bet("EV2"), "EV2")
    m2.poll(now + timedelta(minutes=2))              # past the deadline
    check("crosses at the deadline", len(c2.orders) == 2)
    check("fallback priced at the ask, capped", c2.orders[1]["price"] <= 0.50)
    check("fallback is NOT post-only", c2.orders[1]["post_only"] is False)

    print("no fill at all -> nothing recorded")
    c3 = FakeClient([(0.46, 12)], [(0.50, 40)])
    rec3 = []
    m3 = _mgr(c3, rec3)
    m3.place("KXMLBTOTAL-Z-8", "yes", 2.0, 0.50, now + timedelta(minutes=1), _bet("EV3"), "EV3")
    m3.poll(now + timedelta(minutes=2))              # crosses, but FakeClient reports no fills
    check("an unfilled order writes no ledger row", rec3 == [])

    print("capital guard")
    c4 = FakeClient([(0.46, 12)], [(0.50, 40)], balance_usd=1.0)
    m4 = _mgr(c4, [])
    m4.place("KXMLBTOTAL-Q-8", "yes", 100.0, 0.50, later, _bet("EV4"), "EV4")
    check("refuses an order the balance can't cover", len(c4.orders) == 0)

    print("crash recovery / adoption")
    import uuid as _uuid
    from engine.maker import _ORDER_NAMESPACE
    c6 = FakeClient([(0.46, 12)], [(0.50, 40)])
    # Simulate an order left resting by a previous process for event EV6.
    prior_coid = str(_uuid.uuid5(_ORDER_NAMESPACE, "EV6:1"))
    c6.resting = [{"order_id": "old-99", "client_order_id": prior_coid,
                   "ticker": "KXMLBTOTAL-S-8", "yes_price_dollars": "0.47"}]
    c6.fill("old-99", 1.0, 0.47)
    rec6 = []
    m6 = _mgr(c6, rec6)
    o6 = m6.place("KXMLBTOTAL-S-8", "yes", 3.0, 0.50, later, _bet("EV6"), "EV6")
    check("adopts the resting order instead of duplicating", len(c6.orders) == 0)
    check("adopted order keeps its id", o6 is not None and o6.order_id == "old-99")
    check("adopted order recovers prior fills", o6 is not None and o6.filled_contracts == 1.0)

    print("shutdown drain")
    c5 = FakeClient([(0.46, 12)], [(0.50, 40)])
    rec5 = []
    m5 = _mgr(c5, rec5)
    m5.place("KXMLBTOTAL-R-8", "yes", 2.0, 0.50, later, _bet("EV5"), "EV5")
    m5.stop(drain=True)
    check("cancels resting orders on shutdown", len(c5.canceled) == 1)
    check("no phantom ledger row on shutdown", rec5 == [])

    bad = [n for n, ok in results if not ok]
    print("\n%d/%d passed" % (len(results) - len(bad), len(results)))
    if bad:
        print("FAILED: " + ", ".join(bad))
        raise SystemExit(1)
    print("all maker lifecycle tests passed")


if __name__ == "__main__":
    main()
