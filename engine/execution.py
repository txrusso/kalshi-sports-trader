"""Order preview + validation, and a GATED submit path.

Design boundary (deliberate):
  * preview_order() is fully implemented. It is READ-ONLY: it validates an order
    against price/size caps, market status, and account balance, and renders a
    human-readable summary. Use it freely to see exactly what would be sent.
  * submit_order() does NOT place live orders in this build. Live money-order
    execution is intentionally not implemented here — the agent is recommend-only
    and the operator (you) owns the trigger. It raises ExecutionDisabled with
    instructions instead.

If you want live placement, implement it yourself in a module you control (see
`ENABLING LIVE ORDERS` at the bottom) — Kalshi's order endpoint and the required
request signing are documented there so you can wire it up deliberately.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from config.settings import Settings, DEFAULTS
from kalshi.client import KalshiClient, KalshiError
from kalshi.normalize import parse_market


class ExecutionDisabled(RuntimeError):
    """Raised when a live order submission is attempted but not enabled."""


@dataclass
class OrderRequest:
    ticker: str
    side: str                 # "yes" | "no"
    action: str = "buy"       # "buy" | "sell"
    count: int = 0
    limit_price: float = 0.0  # dollars, 0..1


@dataclass
class Preview:
    ok: bool
    lines: list[str] = field(default_factory=list)

    def render(self) -> str:
        head = "ORDER PREVIEW (dry run — nothing sent)"
        return "\n".join([head, "=" * len(head), *self.lines])


def preview_order(client: KalshiClient, req: OrderRequest, settings: Settings = DEFAULTS) -> Preview:
    lines: list[str] = []
    ok = True

    def check(cond: bool, ok_msg: str, fail_msg: str) -> None:
        nonlocal ok
        lines.append(("  [ok] " if cond else "  [X]  ") + (ok_msg if cond else fail_msg))
        if not cond:
            ok = False

    cents = round(req.limit_price * 100)
    cost = req.count * req.limit_price

    lines.append(f"  {req.action.upper()} {req.count} × {req.ticker} {req.side.upper()} "
                 f"@ ${req.limit_price:.2f}  (limit {cents}c)  est. cost ${cost:,.2f}")
    lines.append("")

    check(req.side in ("yes", "no"), f"side={req.side}", f"invalid side '{req.side}'")
    check(req.action in ("buy", "sell"), f"action={req.action}", f"invalid action '{req.action}'")
    check(req.count > 0, "count > 0", "count must be positive")
    check(req.count <= settings.max_order_contracts,
          f"count within cap ({settings.max_order_contracts})",
          f"count {req.count} exceeds cap {settings.max_order_contracts}")
    check(1 <= cents <= 99, f"price {cents}c in 1..99", f"price {cents}c out of range 1..99")
    check(cost <= settings.max_order_cost_usd,
          f"cost within cap (${settings.max_order_cost_usd:,.0f})",
          f"cost ${cost:,.2f} exceeds cap ${settings.max_order_cost_usd:,.0f}")

    # Market must exist and be tradeable.
    try:
        m = client.get_market(req.ticker)
        q = parse_market(m) if m else None
        check(bool(q and q.status in ("active", "open")),
              f"market status={getattr(q, 'status', None)}",
              f"market not tradeable (status={getattr(q, 'status', 'missing')})")
        if q:
            ref = q.yes_ask if req.side == "yes" else q.no_ask
            if ref:
                lines.append(f"       current {req.side} ask ≈ ${ref:.2f}; your limit ${req.limit_price:.2f}")
    except KalshiError as e:
        check(False, "", f"could not fetch market: {e}")

    # Funds check (buy only).
    if req.action == "buy":
        try:
            bal = client.balance()
            bal_usd = float(str(bal.get("balance") or 0)) / 100.0
            check(cost <= bal_usd, f"balance ${bal_usd:,.2f} covers cost",
                  f"insufficient balance: need ${cost:,.2f}, have ${bal_usd:,.2f}")
        except KalshiError as e:
            check(False, "", f"could not fetch balance: {e}")

    return Preview(ok=ok, lines=lines)


def submit_order(client: KalshiClient, req: OrderRequest, settings: Settings = DEFAULTS):
    """Live submission is intentionally not enabled in this build."""
    raise ExecutionDisabled(
        "Live order placement is not implemented in this agent. It is recommend-only. "
        "Use the preview to see the exact order, then place it yourself in the Kalshi "
        "app/UI, or wire up your own execution module (see engine/execution.py header)."
    )


# --------------------------------------------------------------------------- #
# ENABLING LIVE ORDERS (do this yourself, deliberately)
# --------------------------------------------------------------------------- #
# Kalshi places orders via:
#     POST /trade-api/v2/portfolio/orders
# Body (limit buy example):
#     {"ticker": ..., "client_order_id": <uuid4>, "side": "yes"|"no",
#      "action": "buy"|"sell", "count": <int>, "type": "limit",
#      "yes_price": <1..99>}   # or "no_price" for the NO side
# Auth: the SAME signing already in kalshi/auth.build_headers works for POST
#   (Kalshi signs timestamp+METHOD+path only, not the body). Send the JSON body
#   with the signed headers. Always pass a unique client_order_id so retries can't
#   double-submit. Start on the demo API (--demo) with tiny size.
