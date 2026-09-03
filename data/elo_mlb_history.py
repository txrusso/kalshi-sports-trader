"""Builds and disk-caches MLB Elo training history (signals/elo_mlb.py) for live use
in data/fair_value.py. Refetches at most once/day -- KalshiPaperLoop restarts daily
anyway (see docs/research-log.md), and a day-old completed-games snapshot is all rating_before()
ever needs to answer a pregame question (only games strictly before the query date
matter). Ratings are built once per process and reused for every scan cycle, the same
process-lifetime-cache treatment data/mlb_stats.py's team_run_rates() already gets.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

from data.mlb_stats import fetch_historical_games
from signals.elo_mlb import EloRatings

log = logging.getLogger("data.elo_mlb_history")

CACHE_PATH = Path(__file__).resolve().parent / "elo_mlb_history_cache.json"
HISTORY_START = "2023-01-01"   # ~3.5 seasons back -- ratings converged before any live game
MAX_AGE_HOURS = 20

_cached: Optional[EloRatings] = None


def get_elo_ratings(force_refresh: bool = False) -> EloRatings:
    global _cached
    if _cached is not None and not force_refresh:
        return _cached
    games = None if force_refresh else _load_cache_if_fresh()
    if games is None:
        games = _fetch_and_cache()
    _cached = EloRatings(games)
    return _cached


def _load_cache_if_fresh() -> Optional[list]:
    if not CACHE_PATH.exists():
        return None
    try:
        with open(CACHE_PATH, encoding="utf-8") as f:
            data = json.load(f)
        built_at = datetime.fromisoformat(data["built_at"])
        if datetime.now(timezone.utc) - built_at > timedelta(hours=MAX_AGE_HOURS):
            return None
        return data["games"]
    except (OSError, ValueError, KeyError):
        return None


def _fetch_and_cache() -> list:
    end = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    log.info("Fetching MLB Elo history %s..%s (network-bound, ~1 min)...", HISTORY_START, end)
    games = fetch_historical_games(HISTORY_START, end)
    try:
        with open(CACHE_PATH, "w", encoding="utf-8") as f:
            json.dump({"games": games, "built_at": datetime.now(timezone.utc).isoformat()}, f)
    except OSError as e:
        log.warning("Could not write Elo history cache: %s", e)
    return games
