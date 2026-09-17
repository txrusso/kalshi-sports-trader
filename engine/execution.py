"""Order preview + live submission.

  * preview_order() validates an order against price/size caps, market status,
    and account balance, and renders a human-readable summary. Read-only.
  * submit_order() places a REAL order (live or demo, per Settings.use_demo).
    It always re-runs preview_order()'s checks first and refuses to submit if
    any fail — those per-order caps (engine/execution.py's `max_order_contracts`
    / `max_order_cost_usd` in config/settings.py) are the only gate on this
    path; there is no daily loss cap, position-count cap, or confirmation
    step. That's a deliberate choice made by the user 2026-09-17.

Called from two places:
  * `cli.py place --confirm` — a manual, one-off order you type in yourself.
  * `engine/live.py` — the automatic path, invoked by the loop's T-minus
    trigger (Settings.live_trade / `loop --live`) with no human step.

Pass a stable `client_order_id` (see kalshi/client.py's create_order) so a
retried submission after a network hiccup lands on the original order instead
of duplicating it.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Optional

from config.settings import Settings, DEFAULTS
from kalshi.client import KalshiClient, KalshiError
from kalshi.normalize import parse_market


class ExecutionDisabled(RuntimeError):
    """Raised when a live order submission fails preview_order()'s safety checks."""


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


@dataclass
class OrderResult:
    order_id: Optional[str]
    status: Optional[str]
    client_order_id: str
    raw: dict


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


def submit_order(client: KalshiClient, req: OrderRequest, settings: Settings = DEFAULTS,
                  client_order_id: Optional[str] = None) -> OrderResult:
    """Places a REAL order. Raises ExecutionDisabled if preview_order() finds
    any problem (bad side/action/count, price out of range, over a cap,
    market not tradeable, or insufficient balance) — nothing is sent in that
    case. Raises KalshiError on a network/API failure during submission.

    `client_order_id` should be stable per intended order (not random) when
    the caller might retry — e.g. engine/live.py derives it from the game's
    event key so a crash-and-restart between "order sent" and "recorded"
    can't double-place the same bet.
    """
    preview = preview_order(client, req, settings)
    if not preview.ok:
        raise ExecutionDisabled("Order failed a safety check:\n" + preview.render())

    coid = client_order_id or str(uuid.uuid4())
    raw = client.create_order(
        ticker=req.ticker, side=req.side, action=req.action,
        count=req.count, limit_price=req.limit_price, client_order_id=coid,
    )
    # v2 create-order response is flat -- {order_id, client_order_id, fill_count,
    # remaining_count, ts_ms} -- and carries no "status" field (unlike the old,
    # now-deprecated /portfolio/orders response this was originally written against).
    # Derive a status label from fill_count/remaining_count instead of guessing at one.
    try:
        filled = float(raw.get("fill_count") or 0)
        remaining = float(raw.get("remaining_count") or 0)
    except (TypeError, ValueError):
        filled = remaining = None
    if filled is None:
        status = None
    elif filled <= 0:
        status = "resting"
    elif remaining <= 0:
        status = "filled"
    else:
        status = "partially_filled"
    return OrderResult(order_id=raw.get("order_id"), status=status,
                       client_order_id=coid, raw=raw)
