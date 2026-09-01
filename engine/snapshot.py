"""Snapshot persistence.

Two artifacts:
  * snapshots/_last_state.json  -- most recent mid/OI per ticker (for OI momentum).
  * snapshots/YYYY-MM-DD.jsonl  -- full per-cycle rows (for backtesting the signal).

Building this history is the piece most likely missing from a first attempt: you
cannot validate a 'follow the money' thesis without recorded flow-vs-outcome data.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from config.settings import SNAPSHOTS_DIR

log = logging.getLogger("engine.snapshot")


class SnapshotStore:
    def __init__(self, directory: Path = SNAPSHOTS_DIR):
        self.dir = Path(directory)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.state_path = self.dir / "_last_state.json"

    def load_prev_state(self) -> dict[str, dict]:
        if not self.state_path.exists():
            return {}
        try:
            return json.loads(self.state_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            log.warning("Could not read prev state; starting fresh.")
            return {}

    def write_state(self, state: dict[str, dict]) -> None:
        tmp = self.state_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(state), encoding="utf-8")
        tmp.replace(self.state_path)

    def append_rows(self, rows: list[dict[str, Any]], cycle_ts: Optional[datetime] = None) -> Path:
        cycle_ts = cycle_ts or datetime.now(timezone.utc)
        path = self.dir / f"{cycle_ts:%Y-%m-%d}.jsonl"
        with path.open("a", encoding="utf-8") as f:
            for r in rows:
                r = {"cycle_ts": cycle_ts.isoformat(), **r}
                f.write(json.dumps(r, default=str) + "\n")
        return path
