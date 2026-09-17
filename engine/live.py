"""Live-order trigger: the real-money counterpart to paper.py.

At the same T-minus-first-pitch window paper.py uses (`iter_trigger_candidates`),
this submits a REAL order through Kalshi's API (engine.execution.submit_order)
instead of just recording a paper bet, then notifies the user AFTER the order
is placed — no confirmation step, no human in the loop.

Enabled via `loop --live` (Settings.live_trade). Safety is whatever
engine.execution.preview_order() enforces: per-order contract/cost caps, price
range, market status, account balance. Nothing more — no daily loss cap, no
cap on concurrent positions. That is a deliberate choice, made explicitly by
the user on 2026-09-17: fully automatic submission, notify-after-the-fact,
straight to the live (non-demo) account, no limits beyond the existing
per-order caps in config/settings.py. Reconsider before raising
max_order_contracts / max_order_cost_usd, since those caps are now the only
brake on this path.

Each order's client_order_id is deterministic (uuid5 of the game's event key,
not a random uuid4), so if the loop crashes or restarts between "order sent"
and "recorded to the live ledger", the retry on the next cycle reuses the same
client_order_id — Kalshi dedups on it, returning the original order instead of
placing a second one.

Both YES and NO recommendations trade live. Kalshi deprecated the order-creation
endpoint this originally shipped against (caught live 2026-09-17 by a smoke-test
order before any money moved). The replacement v2 endpoint quotes everything from
the YES leg (no separate NO field at all); kalshi/client.py::create_order converts
a NO buy into a YES ask at (1 - price), per that endpoint's own field description.
Both directions were smoke-tested live (1 contract each, rested unfilled, canceled
cleanly) and the NO-side conversion was independently confirmed by reading the
resulting resting order back from Kalshi (GET /portfolio/orders showed
outcome_side="no", no_price_dollars matching the intended price exactly) before
being trusted here. See CLAUDE.md's "Live order execution" section.
"""
from __future__ import annotations

import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

from config.settings import OUTPUT_DIR, Settings
from engine.execution import ExecutionDisabled, OrderRequest, submit_order
from engine.paper import PaperLedger, iter_trigger_candidates
from kalshi.client import KalshiClient, KalshiError
from signals.recommendation import Recommendation

log = logging.getLogger("engine.live")

# Fixed namespace so uuid5(event_key) is stable across process restarts.
_ORDER_NAMESPACE = uuid.UUID("d9a1f6f0-6e0a-4e8b-9c3a-2f6b6a2b6a11")


class LiveLedger(PaperLedger):
    """Same file-backed, event-deduped ledger shape as PaperLedger, pointed at
    a separate file so real fills never mix with the paper track record."""

    def __init__(self, path: Optional[Path] = None):
        super().__init__(path or (OUTPUT_DIR / "live_ledger.jsonl"))


def run_live_trigger(recs: list[Recommendation], clients: dict, ledger: LiveLedger,
                     client: KalshiClient, settings: Settings, window_minutes: float,
                     now: Optional[datetime] = None) -> list[dict]:
    """Submit a real order for any recommended game that starts within
    `window_minutes` and hasn't been bet yet. Returns the newly placed
    (successfully submitted) bets, each with order_id/order_status/
    client_order_id added.

    A submission that fails (safety check or API/network error) is logged and
    skipped WITHOUT recording to `ledger`, so it's reconsidered next cycle
    rather than being silently lost."""
    placed: list[dict] = []
    for r, bet, _g in iter_trigger_candidates(recs, clients, ledger, window_minutes, now):
        coid = str(uuid.uuid5(_ORDER_NAMESPACE, bet["event_key"]))
        req = OrderRequest(ticker=r.ticker, side=r.side.lower(), action="buy",
                           count=r.suggested_contracts, limit_price=r.entry_price)
        try:
            result = submit_order(client, req, settings, client_order_id=coid)
        except ExecutionDisabled as e:
            log.error("LIVE ORDER blocked by safety check for %s: %s", r.ticker, e)
            continue
        except KalshiError as e:
            log.error("LIVE ORDER submission failed for %s (%s); will retry next cycle "
                      "(client_order_id=%s is stable, so a duplicate can't land).",
                      r.ticker, e, coid)
            continue
        except Exception:
            log.exception("LIVE ORDER submission crashed for %s; will retry next cycle.", r.ticker)
            continue

        bet["order_id"] = result.order_id
        bet["order_status"] = result.status
        bet["client_order_id"] = result.client_order_id
        ledger.record(bet)
        placed.append(bet)
        log.info("LIVE ORDER placed (%.0f min before start): %s %s @ %.2f  order_id=%s status=%s",
                 bet["minutes_before"], r.side, bet["label"], r.entry_price,
                 result.order_id, result.status)
    return placed
