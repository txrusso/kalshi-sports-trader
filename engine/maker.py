"""Maker-side order lifecycle: rest on the book, re-peg, fall back to taking.

The taker path (engine/live.py) sends one order at the ask and is done in a
single call. A maker order is a process: it has to be priced against the book,
watched for partial fills, re-priced as the book moves, and eventually either
filled or abandoned. That process is what this module owns.

    place_maker()  -> price from the book, submit post-only, start tracking
    MakerManager.poll() -> one lifecycle tick over every tracked order
    MakerManager.start()/stop() -> run poll() on a background thread

WHY A THREAD. The scan loop sleeps `scan_interval_seconds` (1800s) between
cycles, so it cannot service a 30-second re-peg cadence. The manager therefore
runs its own daemon thread, independent of the scan cycle, and is the only
thing in this project that touches Kalshi outside a scan.

PRICING (`maker_price`). Rest at best_bid + `maker_improve_cents` -- in front
of the existing queue but still strictly inside the spread, so it rests rather
than crosses. Two hard guards on top:
  * never price at or above the best ask (that would take, not make), and
  * never price above `limit_cap`, the most the model would pay -- which is the
    taker ask it would otherwise have paid. A maker order is only ever an
    improvement on the taker price, never a worse one.
If the spread is 1c wide there is no room to improve without crossing, so we
join the bid instead of jumping it.

POST-ONLY. `post_only=True` makes the exchange reject an order that would cross
and execute as a taker. Our own book read is a few hundred ms stale, so this is
the only way to GUARANTEE maker status. Kalshi's create-order-v2 accepts the
field (confirmed 2026-09-22). If a post-only order is rejected for crossing,
that means the book moved in our favour -- we re-read and re-price rather than
silently taking. `maker_require_post_only` decides what happens if the field
turns out not to be honored at all: False (default) emulates it by pricing
strictly below the ask, True refuses to rest.

THE DEADLINE. Every managed order carries a `deadline` -- `maker_taker_fallback_min`
minutes before the game starts. At the deadline any unfilled remainder is
canceled and re-submitted as a plain taker order at the ask, so a bet the model
wanted is never lost just because nobody lifted our bid. This is what makes
maker mode a strict improvement on price rather than a gamble on filling: worst
case we pay what the taker path would have paid anyway, minus a few minutes.

IDEMPOTENCY, AND WHERE IT'S WEAKER THAN THE TAKER PATH. engine/live.py derives
one deterministic client_order_id per event so a crash between "sent" and
"recorded" can't double-place. Re-pegging inherently places multiple orders per
event, so the id is `uuid5(ns, "<event_key>:<attempt>")` -- deterministic per
ATTEMPT. A crash-retry within an attempt is still deduped by Kalshi; a crash
between cancel and re-place is reconciled by `_adopt_existing()`: because the
ids are a pure function of (event_key, attempt), a restarted process can
regenerate the same candidate ids and recognise its own resting order on the
book, resuming management of it instead of placing a second one.

CAPITAL. `reserved_usd()` reports the notional resting across all tracked
orders. Kalshi already withholds buying power for resting orders, so the
balance check in execution.preview_order() is not double-counting -- but the
loop can trigger many games in one cycle, so `can_afford()` is checked against
(live balance - our own resting notional) before each new order to keep several
simultaneous maker orders from collectively overcommitting the bankroll.
"""
from __future__ import annotations

import logging
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Callable, Optional

from config.fees import fee_for, maker_saving_cents
from config.settings import Settings
from engine.execution import ExecutionDisabled, OrderRequest, submit_order
from kalshi.client import KalshiClient, KalshiError
from kalshi.normalize import OrderBook

log = logging.getLogger("engine.maker")

_ORDER_NAMESPACE = uuid.UUID("d9a1f6f0-6e0a-4e8b-9c3a-2f6b6a2b6a11")

RESTING = "resting"
FILLED = "filled"
FALLBACK_FILLED = "fallback_filled"
CANCELED = "canceled"
FAILED = "failed"
_TERMINAL = {FILLED, FALLBACK_FILLED, CANCELED, FAILED}


def _f(v, default: float = 0.0) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def maker_price(book: OrderBook, side: str, limit_cap: float,
                improve_cents: float = 1.0) -> Optional[float]:
    """Price for a resting BUY of `side`, or None if the book gives no room.

    best_bid + improve_cents, clamped strictly below the ask (so it rests) and
    at or below `limit_cap` (so it never costs more than taking would have)."""
    if side == "yes":
        best_bid, best_ask = book.best_yes_bid, book.best_yes_ask
    else:
        best_bid, best_ask = book.best_no_bid, book.best_no_ask
    if best_ask <= 0:
        return None

    improve = improve_cents / 100.0
    px = round(best_bid + improve, 2) if best_bid > 0 else round(best_ask - improve, 2)

    # Never cross. If improving would reach the ask, sit on the bid instead --
    # with a 1c spread there is simply no room to jump the queue and stay a maker.
    if px >= best_ask - 1e-9:
        px = round(best_ask - 0.01, 2)
        if best_bid > 0 and px >= best_ask - 1e-9:
            px = round(best_bid, 2)

    px = min(px, round(limit_cap, 2))
    px = round(px, 2)
    if not (0.01 <= px <= 0.99) or px >= best_ask - 1e-9:
        return None
    # A cap below the current best bid can't compete for the queue at all -- it
    # would rest where it will never fill. Report "no room" so the caller skips
    # it rather than parking a dead order on the book.
    if best_bid > 0 and px < best_bid - 1e-9:
        return None
    return px


@dataclass
class ManagedOrder:
    """One game's maker order, across however many re-pegs it takes."""
    event_key: str
    ticker: str
    side: str                       # "yes" | "no"
    target_contracts: float         # what the model asked for
    limit_cap: float                # taker ask at decision time -- our price ceiling
    deadline: datetime              # cross-to-taker time (UTC)
    bet: dict                       # ledger row template, recorded on fill
    order_id: Optional[str] = None
    # Every order id this bet has used. A re-peg cancels one order and opens
    # another, but fills already taken on the OLD id still count toward this
    # bet -- summing only the live id would forget them and re-place too many
    # contracts, over-buying the position.
    order_ids: list[str] = field(default_factory=list)
    client_order_id: Optional[str] = None
    resting_price: float = 0.0
    filled_contracts: float = 0.0
    fill_cost: float = 0.0          # dollars actually spent on fills
    fees_paid: float = 0.0
    attempt: int = 0
    repegs: int = 0
    state: str = RESTING
    last_error: Optional[str] = None

    @property
    def remaining(self) -> float:
        return max(0.0, round(self.target_contracts - self.filled_contracts, 2))

    @property
    def avg_fill_price(self) -> float:
        return round(self.fill_cost / self.filled_contracts, 4) if self.filled_contracts else 0.0

    @property
    def resting_usd(self) -> float:
        return self.remaining * self.resting_price if self.state == RESTING else 0.0


class MakerManager:
    """Tracks resting maker orders and drives them to a terminal state.

    `on_fill(order, bet)` is called once per order that ends up with any fill,
    with `bet` already updated to the REAL filled quantity and average price --
    that callback is what writes the ledger row (engine/live.py passes
    LiveLedger.record), so the ledger records what actually traded rather than
    what was requested."""

    def __init__(self, client: KalshiClient, settings: Settings,
                 on_fill: Optional[Callable[[ManagedOrder, dict], None]] = None):
        self.client = client
        self.settings = settings
        self.on_fill = on_fill
        self.orders: dict[str, ManagedOrder] = {}
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    # --- capital ---------------------------------------------------------
    def reserved_usd(self) -> float:
        """Notional currently resting across all tracked orders."""
        with self._lock:
            return round(sum(o.resting_usd for o in self.orders.values()), 4)

    def can_afford(self, cost_usd: float) -> bool:
        """Does the live balance cover `cost_usd` on top of what we already have
        resting? Guards the case where many games trigger in one cycle and each
        order individually passes preview_order()'s balance check."""
        try:
            bal = self.client.balance()
            available = _f(bal.get("balance")) / 100.0
        except KalshiError:
            log.warning("Balance fetch failed; allowing the order and letting "
                        "preview_order()'s own check decide.", exc_info=True)
            return True
        return cost_usd <= (available - self.reserved_usd()) + 1e-9

    # --- placement -------------------------------------------------------
    def place(self, ticker: str, side: str, contracts: float, limit_cap: float,
              deadline: datetime, bet: dict, event_key: str) -> Optional[ManagedOrder]:
        """Price against the live book and rest a post-only order. Returns the
        tracked order, or None if it couldn't be priced/placed."""
        with self._lock:
            if event_key in self.orders and self.orders[event_key].state not in _TERMINAL:
                return self.orders[event_key]        # already working this game

        book = self._book(ticker)
        if book is None:
            log.warning("MAKER %s: no order book; skipping.", ticker)
            return None
        px = maker_price(book, side, limit_cap, self.settings.maker_improve_cents)
        if px is None:
            log.info("MAKER %s: no room to rest inside the spread (cap %.2f); "
                     "leaving it to the taker fallback at the deadline.", ticker, limit_cap)
            return None
        if not self.can_afford(contracts * px):
            log.warning("MAKER %s: skipped -- $%.2f would overcommit the bankroll "
                        "on top of $%.2f already resting.", ticker, contracts * px,
                        self.reserved_usd())
            return None

        o = ManagedOrder(event_key=event_key, ticker=ticker, side=side,
                         target_contracts=contracts, limit_cap=limit_cap,
                         deadline=deadline, bet=bet)

        # Crash recovery: if a previous process already rested an order for this
        # event, adopt it rather than placing a second one on top.
        if self._adopt_existing(o):
            with self._lock:
                self.orders[event_key] = o
            log.info("MAKER %s: adopted an order left resting by an earlier run "
                     "(order_id=%s, %.2f already filled).", ticker, o.order_id,
                     o.filled_contracts)
            return o

        if not self._submit_resting(o, px):
            return None
        with self._lock:
            self.orders[event_key] = o
        saving = maker_saving_cents(px, ticker, self.client) * contracts
        log.info("MAKER resting %.2f x %s %s @ %.2f (taker ask was %.2f, saves ~$%.4f in fees) "
                 "order_id=%s", contracts, ticker, side.upper(), px, limit_cap, saving, o.order_id)
        return o

    def _adopt_existing(self, o: ManagedOrder) -> bool:
        """Find and re-attach to an order this bet already has resting.

        The client_order_id is `uuid5(ns, "<event_key>:<attempt>")` -- a pure
        function of the event and attempt number -- so a restarted process can
        regenerate the same ids and match them against the live resting book.
        That closes the one idempotency gap re-pegging opens: a crash between
        "order placed" and "order recorded" would otherwise place a duplicate on
        the next cycle."""
        try:
            resting = self.client.get_orders(status="resting")
        except KalshiError:
            log.warning("MAKER %s: could not list resting orders; proceeding without "
                        "adoption (a duplicate is possible if a previous run crashed "
                        "mid-place).", o.ticker, exc_info=True)
            return False
        if not resting:
            return False
        wanted = {str(uuid.uuid5(_ORDER_NAMESPACE, f"{o.event_key}:{n}")): n
                  for n in range(1, self.settings.maker_max_repegs + 2)}
        for row in resting:
            coid = row.get("client_order_id")
            if coid not in wanted or row.get("ticker") != o.ticker:
                continue
            o.order_id = row.get("order_id")
            if o.order_id:
                o.order_ids.append(o.order_id)
            o.client_order_id = coid
            o.attempt = wanted[coid]
            o.resting_price = _f(row.get("yes_price_dollars") if o.side == "yes"
                                 else row.get("no_price_dollars"))
            o.state = RESTING
            self._refresh_fills(o)
            return True
        return False

    def _submit_resting(self, o: ManagedOrder, price: float) -> bool:
        """Submit (or re-submit) `o`'s remaining size at `price`. Returns success."""
        o.attempt += 1
        coid = str(uuid.uuid5(_ORDER_NAMESPACE, f"{o.event_key}:{o.attempt}"))
        req = OrderRequest(ticker=o.ticker, side=o.side, action="buy",
                           count=o.remaining, limit_price=price)
        try:
            res = submit_order(self.client, req, self.settings, client_order_id=coid,
                               post_only=self.settings.maker_post_only)
        except ExecutionDisabled as e:
            o.last_error = str(e)
            log.error("MAKER %s blocked by a safety check: %s", o.ticker, e)
            o.state = FAILED
            return False
        except KalshiError as e:
            msg = str(e)
            o.last_error = msg
            # post_only rejection = the book moved and our price would now take.
            # Not an error: re-read and re-price on the next tick.
            if "post_only" in msg.lower() or "would_match" in msg.lower() or "cross" in msg.lower():
                log.info("MAKER %s: post-only rejected (book moved); will re-price.", o.ticker)
                return False
            log.error("MAKER %s submission failed: %s", o.ticker, e)
            return False
        except Exception:
            o.last_error = "unexpected"
            log.exception("MAKER %s submission crashed.", o.ticker)
            return False

        o.order_id = res.order_id
        if res.order_id and res.order_id not in o.order_ids:
            o.order_ids.append(res.order_id)
        o.client_order_id = res.client_order_id
        o.resting_price = price
        o.state = RESTING
        return True

    # --- lifecycle -------------------------------------------------------
    def poll(self, now: Optional[datetime] = None) -> None:
        """One lifecycle tick: refresh fills, re-peg stale prices, and cross at
        the deadline. Never raises -- a failure on one order must not stop the
        thread or the others."""
        now = now or datetime.now(timezone.utc)
        with self._lock:
            working = [o for o in self.orders.values() if o.state not in _TERMINAL]
        for o in working:
            try:
                self._tick(o, now)
            except Exception:
                log.exception("MAKER %s: lifecycle tick failed; retrying next poll.", o.ticker)

    def _tick(self, o: ManagedOrder, now: datetime) -> None:
        self._refresh_fills(o)
        if o.remaining <= 0:
            self._finish(o, FILLED)
            return

        # Deadline: stop making, start taking, for whatever is left.
        if now >= o.deadline:
            self._cross(o)
            return

        book = self._book(o.ticker)
        if book is None:
            return
        want = maker_price(book, o.side, o.limit_cap, self.settings.maker_improve_cents)
        if want is None or abs(want - o.resting_price) < 0.005:
            return                                    # still the right price
        if o.repegs >= self.settings.maker_max_repegs:
            log.info("MAKER %s: hit the re-peg cap (%d); holding at %.2f until the deadline.",
                     o.ticker, self.settings.maker_max_repegs, o.resting_price)
            return

        # Re-peg: cancel, then re-rest the REMAINDER at the new price. Cancel
        # first so a fill landing mid-swap can't leave us doubled up.
        if not self._cancel(o):
            return
        self._refresh_fills(o)                        # a fill may have landed during cancel
        if o.remaining <= 0:
            self._finish(o, FILLED)
            return
        o.repegs += 1
        if self._submit_resting(o, want):
            log.info("MAKER %s: re-pegged %.2f -> %.2f (re-peg %d, %.2f left)",
                     o.ticker, o.resting_price, want, o.repegs, o.remaining)

    def _refresh_fills(self, o: ManagedOrder) -> None:
        """Pull this order's fills and update filled qty / cost / fees.

        Reads /portfolio/fills rather than the order's own counters because
        fills carry the real `fee_cost` and `is_taker` -- which is how a maker
        fill's actual fee gets recorded (and how config/fees.py's unverified
        maker rate finally gets checked against reality)."""
        ids = o.order_ids or ([o.order_id] if o.order_id else [])
        if not ids:
            return
        fills: list[dict] = []
        for oid in ids:
            try:
                fills.extend(self.client.get_fills(limit=100, order_id=oid))
            except KalshiError:
                # Partial data would UNDERCOUNT fills and make us re-place too
                # much, so bail out and keep the last known-good total instead.
                log.warning("MAKER %s: fill fetch failed for %s; keeping the last "
                            "known fill total and retrying next poll.", o.ticker, oid,
                            exc_info=True)
                return
        qty = cost = fees = 0.0
        makers = takers = 0
        for f in fills:
            c = _f(f.get("count_fp"))
            side = (f.get("outcome_side") or o.side).lower()
            px = _f(f.get("yes_price_dollars") if side == "yes" else f.get("no_price_dollars"))
            qty += c
            cost += c * px
            fees += _f(f.get("fee_cost"))
            if f.get("is_taker"):
                takers += 1
            else:
                makers += 1
        if qty > o.filled_contracts:
            log.info("MAKER %s: fill %.2f -> %.2f of %.2f (%d maker / %d taker fills, fees $%.4f)",
                     o.ticker, o.filled_contracts, qty, o.target_contracts, makers, takers, fees)
        o.filled_contracts = round(qty, 2)
        o.fill_cost = cost
        o.fees_paid = fees

    def _cancel(self, o: ManagedOrder) -> bool:
        if not o.order_id:
            return True
        try:
            self.client.cancel_order(o.order_id)
            return True
        except KalshiError as e:
            # Already gone (filled or canceled) -- refresh and let _tick re-read.
            log.info("MAKER %s: cancel returned %s (likely already filled/canceled).",
                     o.ticker, e)
            return False

    def _cross(self, o: ManagedOrder) -> None:
        """Deadline reached: cancel the rest and take the ask, so the model's bet
        is never lost merely because nobody lifted our bid."""
        self._cancel(o)
        self._refresh_fills(o)
        if o.remaining <= 0:
            self._finish(o, FILLED)
            return

        book = self._book(o.ticker)
        ask = (book.best_yes_ask if o.side == "yes" else book.best_no_ask) if book else 0.0
        # If the ask has run ABOVE what the model was ever willing to pay, the
        # edge that justified this bet is gone. Crossing at limit_cap would just
        # rest an unfillable order on the book with nothing left to manage it, so
        # take the no-bet outcome instead.
        if ask > 0 and round(ask, 2) > round(o.limit_cap, 2) + 1e-9:
            log.info("MAKER %s: ask %.2f ran past the model's cap %.2f by the deadline; "
                     "abandoning %.2f contracts rather than overpaying.",
                     o.ticker, ask, o.limit_cap, o.remaining)
            self._finish(o, FILLED if o.filled_contracts else CANCELED)
            return
        price = min(round(ask, 2), round(o.limit_cap, 2)) if ask > 0 else round(o.limit_cap, 2)
        if not (0.01 <= price <= 0.99):
            log.warning("MAKER %s: no usable ask at the deadline; abandoning %.2f contracts.",
                        o.ticker, o.remaining)
            self._finish(o, CANCELED if o.filled_contracts else FAILED)
            return

        o.attempt += 1
        coid = str(uuid.uuid5(_ORDER_NAMESPACE, f"{o.event_key}:taker:{o.attempt}"))
        req = OrderRequest(ticker=o.ticker, side=o.side, action="buy",
                           count=o.remaining, limit_price=price)
        log.info("MAKER %s: deadline reached with %.2f unfilled -- crossing at %.2f.",
                 o.ticker, o.remaining, price)
        try:
            submit_order(self.client, req, self.settings, client_order_id=coid)
        except (ExecutionDisabled, KalshiError) as e:
            o.last_error = str(e)
            log.error("MAKER %s: taker fallback failed: %s", o.ticker, e)
            self._finish(o, FILLED if o.filled_contracts else FAILED)
            return
        except Exception:
            log.exception("MAKER %s: taker fallback crashed.", o.ticker)
            self._finish(o, FILLED if o.filled_contracts else FAILED)
            return
        self._refresh_fills(o)
        self._finish(o, FALLBACK_FILLED if o.filled_contracts else FAILED)

    def _finish(self, o: ManagedOrder, state: str) -> None:
        o.state = state
        if o.filled_contracts <= 0:
            log.info("MAKER %s: finished with no fill (%s).", o.ticker, state)
            return
        # Record what ACTUALLY traded, not what was requested.
        bet = dict(o.bet)
        bet["contracts"] = o.filled_contracts
        bet["entry_price"] = o.avg_fill_price
        bet["wager_usd"] = round(o.fill_cost, 4)
        bet["fees_usd"] = round(o.fees_paid, 6)
        bet["order_id"] = o.order_id
        bet["order_status"] = state
        bet["client_order_id"] = o.client_order_id
        bet["execution"] = "maker" if state == FILLED else "maker_taker_fallback"
        bet["repegs"] = o.repegs
        if o.filled_contracts + 0.005 < o.target_contracts:
            bet["partial_fill"] = True
            bet["requested_contracts"] = o.target_contracts
        log.info("MAKER %s: DONE %s -- %.2f/%.2f @ avg %.4f, fees $%.4f",
                 o.ticker, state, o.filled_contracts, o.target_contracts,
                 o.avg_fill_price, o.fees_paid)
        if self.on_fill:
            try:
                self.on_fill(o, bet)
            except Exception:
                log.exception("MAKER %s: on_fill callback failed; the fill is REAL and "
                              "unrecorded -- reconcile against /portfolio/fills.", o.ticker)

    def _book(self, ticker: str) -> Optional[OrderBook]:
        try:
            return OrderBook.from_raw({"orderbook": self.client.get_orderbook(ticker)})
        except KalshiError:
            log.warning("MAKER %s: order book fetch failed.", ticker, exc_info=True)
            return None

    # --- thread ----------------------------------------------------------
    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="maker-manager", daemon=True)
        self._thread.start()
        log.info("Maker order manager started (poll every %.0fs, cross at T-%.0f min).",
                 self.settings.maker_poll_seconds, self.settings.maker_taker_fallback_min)

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                self.poll()
            except Exception:
                log.exception("Maker manager poll failed; continuing.")
            self._stop.wait(self.settings.maker_poll_seconds)

    def stop(self, drain: bool = True) -> None:
        """Stop polling. `drain` cancels anything still resting first, so a loop
        shutdown doesn't leave unmanaged orders on the book."""
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5.0)
        if not drain:
            return
        with self._lock:
            working = [o for o in self.orders.values() if o.state not in _TERMINAL]
        for o in working:
            log.info("MAKER %s: canceling on shutdown (%.2f resting).", o.ticker, o.remaining)
            self._cancel(o)
            try:
                self._refresh_fills(o)
                self._finish(o, FILLED if o.filled_contracts else CANCELED)
            except Exception:
                log.exception("MAKER %s: shutdown reconcile failed.", o.ticker)

    def summary(self) -> str:
        with self._lock:
            os_ = list(self.orders.values())
        if not os_:
            return "maker: no orders"
        resting = [o for o in os_ if o.state == RESTING]
        done = [o for o in os_ if o.state in (FILLED, FALLBACK_FILLED)]
        return (f"maker: {len(resting)} resting (${self.reserved_usd():.2f}), "
                f"{len(done)} filled, {len(os_) - len(resting) - len(done)} other")
