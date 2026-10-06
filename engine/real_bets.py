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
import time
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

    # Kalshi AGES ORDERS OUT of /portfolio/orders/{id} -- it 404s on orders
    # only days old (5 of 29 checked on 2026-09-29, the oldest 7 days). That
    # used to abort this whole lookup, and a None return means "grade it as
    # recorded", i.e. ASSUME IT FILLED. So this regressed silently back into
    # the exact bug the module exists to fix: the two never-filled CLE@BOS
    # rows of 2026-09-22 that the original fix correctly dropped came back as
    # a graded win and loss once their order records expired, along with three
    # more from 09-25. Five phantom bets, $4.59 of wager that never happened,
    # in the record and P&L of every report.
    #
    # Fills OUTLIVE the order record (verified: every still-readable order's
    # fills came back, back to 09-18), so ask them separately and let them be
    # the authority. An order Kalshi no longer serves cannot still be resting,
    # so "unknown order + zero fills" is a real no-fill, not an unknown -- and
    # it was confirmed independently for all five at the ACCOUNT level, where
    # get_fills(ticker=...) returns nothing on those tickers either.
    status = None
    try:
        status = (client.get_order(order_id) or {}).get("status") or "?"
    except Exception:
        log.info("Order %s is gone from /portfolio/orders (aged out); "
                 "grading it from its fills instead.", order_id)
    try:
        fills = client.get_fills(limit=100, order_id=order_id)
        filled = sum(float(f.get("count_fp") or f.get("count") or 0) for f in fills)
    except Exception:
        # Only now is the bet genuinely unverifiable.
        log.warning("Fill lookup failed for order %s; grading it as recorded.",
                    order_id, exc_info=True)
        return None
    if status is None:
        # Derived, not reported. Both are terminal, so this caches and
        # apply_fill() treats it exactly as a real terminal order.
        status = "executed" if filled > 1e-9 else "canceled"
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


# ticker -> {"contracts", "price", "fees"} from the account's real fills.
# TTL'd rather than built once per process: the dashboard and the loop both
# run for hours, and a fill that lands mid-session has to show up without a
# restart. One paginated fetch per TTL against hundreds of graded rows.
_TICKER_FILLS: Optional[dict] = None
_TICKER_FILLS_AT: float = 0.0
_TICKER_FILLS_TTL = 60.0


def reset_fill_cache() -> None:
    """Drop the cached fills. For tests, and for any caller that needs a
    guaranteed-fresh read."""
    global _TICKER_FILLS, _TICKER_FILLS_AT
    _TICKER_FILLS, _TICKER_FILLS_AT = None, 0.0


def _ticker_fills(client) -> dict:
    """What actually traded, per ticker, from Kalshi's own fills.

    WHY THIS EXISTS (found 2026-10-06 reconciling the dashboard against the
    balance). A live ledger row is written at SUBMIT time and its
    `entry_price` is the LIMIT we sent, not the price we got. Kalshi fills at
    the best available price, which is routinely better: on
    KXMLBGAME-26SEP232040AZCOL-COL the ledger recorded 0.31 and the fill was
    0.13. Grading off the limit overstates every loss and understates every
    win, and across the live ledger it understated realized P&L by $4.63 --
    the dashboard read +$1.85 where the account was really +$6.04.

    `apply_fill` below already trimmed CONTRACTS for a partial fill; it never
    touched price, and it only ran when the submit-time status was not
    "filled", so a fully-filled order was always taken at its limit.
    """
    global _TICKER_FILLS, _TICKER_FILLS_AT
    if _TICKER_FILLS is not None and (time.monotonic() - _TICKER_FILLS_AT) < _TICKER_FILLS_TTL:
        return _TICKER_FILLS
    if client is None or not hasattr(client, "get_all_fills"):
        return {}
    try:
        fills = client.get_all_fills()
    except Exception:
        log.warning("Could not fetch fills; grading rows at their recorded price.",
                    exc_info=True)
        return {}
    agg: dict = {}
    for f in fills:
        # Do NOT filter on `action`. Buying NO is submitted as a YES ASK (see
        # CLAUDE.md's "Live order execution"), so Kalshi records a NO bet as
        # action="sell" with outcome_side="no" -- filtering to action=="buy"
        # silently drops every NO-side bet and leaves it priced at its limit.
        # This agent never sells to close; it holds to settlement, so every
        # fill on the account is a position being opened. `outcome_side` is
        # the field that says which side, and it is what we key on.
        t = f.get("ticker") or f.get("market_ticker")
        try:
            ct = float(f.get("count_fp") or 0)
        except (TypeError, ValueError):
            continue
        if not t or ct <= 0:
            continue
        side = (f.get("outcome_side") or f.get("side") or "yes").lower()
        key = "no_price_dollars" if side == "no" else "yes_price_dollars"
        try:
            px = float(f.get(key) or 0)
        except (TypeError, ValueError):
            continue
        a = agg.setdefault(t, {"contracts": 0.0, "cost": 0.0, "fees": 0.0})
        a["contracts"] += ct
        a["cost"] += px * ct
        a["fees"] += float(f.get("fee_cost") or 0)
    for a in agg.values():
        a["price"] = a["cost"] / a["contracts"] if a["contracts"] else 0.0
    _TICKER_FILLS, _TICKER_FILLS_AT = agg, time.monotonic()
    return agg


def apply_real_fill(bet: dict, got: dict) -> dict:
    """Re-price and re-size one row to what actually traded, keeping the
    recorded values as `recorded_*` so nothing is lost."""
    price = round(float(got["price"]), 4)
    ct = round(float(got["contracts"]), 2)
    out = {**bet}
    if abs(price - float(bet.get("entry_price") or 0)) > 1e-9:
        out["recorded_entry_price"] = bet.get("entry_price")
        out["entry_price"] = price
    if abs(ct - float(bet.get("contracts") or 0)) > 1e-9:
        out.setdefault("requested_contracts", bet.get("contracts"))
        out["contracts"] = ct
    out["wager_usd"] = round(price * ct, 4)
    out["fees_usd"] = round(float(got["fees"]), 6)   # real fee, not the model
    out["_repriced"] = True
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

    # Re-price EVERY row (both ledgers) against the real fills. The paper rows
    # were placed by hand and carry the alert's price, which is no more the
    # executed price than the live rows' limit was.
    fills = _ticker_fills(client)
    if fills:
        out = []
        for b in bets:
            got = fills.get(b.get("ticker"))
            if got:
                out.append(apply_real_fill(b, got))
            elif b.get("_src") == "paper":
                # No fill anywhere on the account: this alert was never acted
                # on. Keep it only when explicitly asked for, the same way a
                # never-filled live order is handled.
                b = {**b, "_nofill": True, "_fill_status": "no fill on account"}
                if include_unfilled:
                    out.append(b)
            else:
                out.append(b)
        bets = out
    return bets
