"""The model inputs behind each PLACED bet, kept only while the bet is pending.

When the loop places a bet it saves that recommendation exactly as
output/recommendations_latest.json carries it (the same `r.__dict__`, incl. the
`debug` block with the money-flow and fair-value inputs). The dashboard's Model
tab explains a Pending bet from this -- the numbers Stakey actually saw when it
bet, not today's. `cli.py settle` (run nightly by the KalshiSettle task) deletes
the entry once the bet has settled.

Two rules, because this sits in the path that places real orders:
- save() NEVER raises. A failure here costs an explanation, never an order or a
  ledger row; callers invoke it only AFTER the bet is already recorded.
- Writes are atomic (temp file + os.replace), so the dashboard can never read a
  half-written file.

One file, keyed by event_key (the same key the ledgers dedupe on):
    {event_key: {"ticker", "placed_ts", "rec": {...}}}
"""
from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable, Optional

from config.settings import OUTPUT_DIR

log = logging.getLogger("engine.bet_inputs")

PATH = OUTPUT_DIR / "bet_inputs.json"
MAX_AGE_DAYS = 7      # safety net: drop anything this old even if it never graded


def load(path: Optional[Path] = None) -> dict[str, dict]:
    """All saved entries; {} if the file is missing or unreadable."""
    path = path or PATH          # resolved at call time, not bound at import
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def _write(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, indent=1, default=str), encoding="utf-8")
    # On Windows os.replace fails while another process has the target open
    # (the dashboard reading it); that window is milliseconds, so retry briefly.
    for attempt in range(5):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            if attempt == 4:
                raise
            time.sleep(0.1)


def save(bet: dict, rec, path: Optional[Path] = None) -> None:
    """Store `rec`'s inputs under the bet's event_key. Never raises."""
    try:
        path = path or PATH
        key = bet.get("event_key")
        if not key:
            return
        rec_dict = rec if isinstance(rec, dict) else rec.__dict__
        data = load(path)
        data[key] = {
            "ticker": bet.get("ticker"),
            "placed_ts": bet.get("ts") or datetime.now(timezone.utc).isoformat(),
            # Round-trip through JSON exactly as the scan file does (default=str).
            "rec": json.loads(json.dumps(rec_dict, default=str)),
        }
        _write(data, path)
    except Exception:
        log.warning("Could not save model inputs for %s (the bet itself is unaffected).",
                    bet.get("ticker"), exc_info=True)


def prune(is_settled: Callable[[str], bool], path: Optional[Path] = None,
          now: Optional[datetime] = None) -> int:
    """Delete entries whose bet has settled, or that are older than MAX_AGE_DAYS.
    Returns how many were removed."""
    path = path or PATH
    data = load(path)
    if not data:
        return 0
    now = now or datetime.now(timezone.utc)
    keep = {}
    for key, entry in data.items():
        try:
            placed = datetime.fromisoformat(str(entry.get("placed_ts")).replace("Z", "+00:00"))
            if placed.tzinfo is None:
                placed = placed.replace(tzinfo=timezone.utc)
        except ValueError:
            placed = None
        too_old = placed is not None and placed < now - timedelta(days=MAX_AGE_DAYS)
        if too_old or is_settled(entry.get("ticker") or ""):
            continue
        keep[key] = entry
    removed = len(data) - len(keep)
    if removed:
        _write(keep, path)
    return removed


def prune_settled(clients: dict, path: Optional[Path] = None) -> int:
    """prune() using real outcomes -- what `cli.py settle` calls nightly."""
    path = path or PATH
    data = load(path)
    if not data:
        return 0
    from data.games import resolve_outcomes
    outcomes = resolve_outcomes({e.get("ticker") for e in data.values() if e.get("ticker")},
                                clients)
    return prune(lambda t: outcomes.get(t) is not None, path)
