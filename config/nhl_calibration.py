"""Live NHL totals calibration -- the two swept constants, re-derived nightly.

WHY THIS EXISTS. data/scoring_environment_nhl.py::TEAM_SIGNAL_SHRINK and
data/fair_value_nhl_totals.py::NHL_TOTALS_DISPERSION were swept once, on
2026-09-30, against two completed seasons. The 2026-27 season adds real games
every night, so backtest/nhl_recalibrate.py re-runs that sweep nightly and
writes the result here; the models read it, falling back to the code constants
when the file is missing, unreadable or stale-shaped.

THE CODE CONSTANTS REMAIN THE SOURCE OF TRUTH FOR "what shipped". This file
only ever holds a value that beat them on BOTH walk-forward splits -- the
project's standing bar (see CLAUDE.md: every production constant here moved
on a both-splits win, never on a single-split or full-sample result). A
corrupt, partial or absent file degrades to the shipped constants rather than
to anything unvalidated.

VERSIONING MATTERS MORE THAN USUAL HERE. NHL is shadow-traded, and the whole
point of that ledger is to decide whether NHL earns its way back to real
money. A model that silently re-tunes itself makes that record measure a
moving target, so every applied change bumps `version`, and
engine/paper.py::ShadowLedger stamps it onto each simulated bet. The track
record stays segmentable by exactly the model that produced it.
"""
from __future__ import annotations

import json
import logging
import os
from typing import Any, Optional

from config.settings import OUTPUT_DIR

log = logging.getLogger("config.nhl_calibration")

CALIBRATION_PATH = OUTPUT_DIR / "nhl_calibration.json"

# Sanity bounds. A nightly job that somehow produces a value outside these is
# rejected rather than trusted -- shrink at 1.0 would restore the phantom-edge
# bug this whole exercise removed, and a dispersion at/above 1.0 undoes the
# measured under-dispersion.
SHRINK_RANGE = (0.0, 0.5)
DISPERSION_RANGE = (0.5, 1.0)

_cache: Optional[dict] = None
_cache_mtime: Optional[float] = None


def _shipped() -> dict:
    """The code constants -- imported lazily to avoid an import cycle, since
    both modules below import settings, which this module also imports."""
    from data.fair_value_nhl_totals import NHL_TOTALS_DISPERSION
    from data.scoring_environment_nhl import TEAM_SIGNAL_SHRINK
    return {"version": "shipped-2026-09-30",
            "team_signal_shrink": TEAM_SIGNAL_SHRINK,
            "dispersion": NHL_TOTALS_DISPERSION,
            "source": "code constants"}


def _valid(d: Any) -> bool:
    if not isinstance(d, dict):
        return False
    try:
        s, p = float(d["team_signal_shrink"]), float(d["dispersion"])
    except (KeyError, TypeError, ValueError):
        return False
    return (SHRINK_RANGE[0] <= s <= SHRINK_RANGE[1]
            and DISPERSION_RANGE[0] <= p <= DISPERSION_RANGE[1]
            and isinstance(d.get("version"), str) and d["version"])


def load() -> dict:
    """Effective calibration. Re-reads only when the file's mtime changes, so
    a nightly rewrite is picked up without a per-market file read."""
    global _cache, _cache_mtime
    try:
        mtime = os.path.getmtime(CALIBRATION_PATH)
    except OSError:
        return _shipped()                      # no file yet -- normal, not an error
    if _cache is not None and _cache_mtime == mtime:
        return _cache
    try:
        data = json.loads(CALIBRATION_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        log.warning("NHL calibration file unreadable (%s); using shipped constants.", e)
        return _shipped()
    if not _valid(data):
        log.warning("NHL calibration file failed validation; using shipped constants.")
        return _shipped()
    _cache, _cache_mtime = data, mtime
    return data


def shrink() -> float:
    return float(load()["team_signal_shrink"])


def dispersion() -> float:
    return float(load()["dispersion"])


def version() -> str:
    return str(load()["version"])


def save(team_signal_shrink: float, dispersion_: float, version_: str, **meta) -> dict:
    """Write a new calibration. Refuses anything outside the sanity bounds --
    this writes the numbers a live (if simulated) model will price with, so a
    bad sweep must fail loudly rather than quietly ship."""
    payload = {"version": version_, "team_signal_shrink": round(float(team_signal_shrink), 4),
               "dispersion": round(float(dispersion_), 4), **meta}
    if not _valid(payload):
        raise ValueError(f"refusing to save an out-of-range NHL calibration: {payload}")
    CALIBRATION_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = CALIBRATION_PATH.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(tmp, CALIBRATION_PATH)          # atomic: a reader never sees a half file
    global _cache, _cache_mtime
    _cache = _cache_mtime = None
    return payload
