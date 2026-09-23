"""Every real-money bet, as it actually filled -- the one ledger view that
reporting should grade.

Two things were wrong with grading straight off a ledger file:

1. WHICH ledger. Since 2026-09-17 Stakey places its own orders and records them
   to output/live_ledger.jsonl; the paper ledger (output/paper_ledger.jsonl)
   stopped growing on 2026-09-16. `cli.py settle` -- and so the nightly results
   text -- plus `ledger_stats` and the PNG dashboard all still read ONLY the
   paper ledger, so none of the live bets were in them. The paper ledger is not
   paper money either: those were real orders placed by hand from the alert.
   Both ledgers are real bets; only who clicked buy differs.

2. WHAT FILLED. engine/live.py records a live bet at SUBMIT time. An order that
   rests on the book and is later canceled unfilled still sits in the ledger
   looking like a placed bet, and was graded as a win or loss on a position
   that never existed (two such rows, CLE@BOS 2026-09-22). This checks every
   live row whose submit-time status wasn't `filled` against Kalshi's own
   fills: never-filled rows are dropped, partial fills are graded on what
   filled.

Read-only: only GETs, and never writes a ledger.
"""
from __future__ import annotations

import logging
from typing import Optional

log = logging.getLogger("engine.real_bets")

# order_id -> (status, filled contracts). Only terminal states are cached --
# an executed or canceled order can never change again. In-process only, so
# the long-running loop and dashboard look each order up once, and nothing is
# written to disk.
_FILL_CACHE: dict[str, tuple[str, float]] = {}


def _order_fill(order_id: str, client) -> Optional[tuple[str, float]]:
    """(status, filled contracts) for one order, or None if it can't be read."""
    if order_id in _FILL_CACHE:
        return _FILL_CACHE[order_id]
    try:
        status = (client.get_order(order_id) or {}).get("status") or "?"
        fills = client.get_fills(limit=100, order_id=order_id)
        filled = sum(float(f.get("count_fp") or f.get("count") or 0) for f in fills)
    except Exception:
        log.warning("Fill lookup failed for order %s; grading it as recorded.",
                    order_id, exc_info=True)
        return None
    if status in ("executed", "canceled"):
        _FILL_CACHE[order_id] = (status, filled)
    return status, filled


def apply_fill(bet: dict, status: str, filled: float) -> dict:
    """The bet as it actually filled.

    Sets `_fill_status`. A terminal order with nothing filled gets
    `_nofill=True` (the caller decides whether to show or drop it). A partial
    fill has `contracts` and `wager_usd` cut to what filled, keeping the
    original as `requested_contracts`. A still-resting order is left as
    recorded -- it may yet fill.
    """
    out = {**bet, "_fill_status": status}
    requested = float(bet.get("contracts") or 0)
    if status not in ("executed", "canceled") or filled >= requested - 1e-9:
        return out
    if filled <= 1e-9:
        out["_nofill"] = True
        return out
    price = float(bet.get("entry_price") or 0)
    out.update(requested_contracts=requested, contracts=round(filled, 2),
               wager_usd=round(price * filled, 2))
    return out


def load_real_bets(client=None, include_unfilled: bool = False) -> list[dict]:
    """Both ledgers, each row tagged `_src` ("live" / "paper"), live rows
    corrected to what actually filled.

    Never-filled orders are dropped unless `include_unfilled`, in which case
    they are kept and marked `_nofill=True` (the dashboard lists them as "no
    fill"). `client` is created on demand only if some row needs a lookup; if
    that fails, rows are graded as recorded rather than the report failing.
    """
    from engine.live import LiveLedger
    from engine.paper import PaperLedger

    bets = [{**b, "_src": "paper"} for b in PaperLedger().load()]
    live = LiveLedger().load()
    needs = [b for b in live if b.get("order_id") and b.get("order_status") != "filled"
             and b["order_id"] not in _FILL_CACHE]
    if needs and client is None:
        try:
            from config.settings import DEFAULTS
            from kalshi.client import KalshiClient
            client = KalshiClient(DEFAULTS)
        except Exception:
            log.warning("No Kalshi client for fill checks; grading live rows as recorded.",
                        exc_info=True)

    for b in live:
        row = {**b, "_src": "live"}
        if b.get("order_id") and b.get("order_status") != "filled":
            got = _FILL_CACHE.get(b["order_id"])
            if got is None and client is not None:
                got = _order_fill(b["order_id"], client)
            if got is not None:
                row = apply_fill(row, *got)
        if row.get("_nofill") and not include_unfilled:
            continue
        bets.append(row)
    return bets
