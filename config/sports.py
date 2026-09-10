"""Ticker-prefix -> (sport, market kind) registry.

The single source of truth for "which sport is this ticker, and is it a
winner market or a totals market" -- previously this was a `ticker.startswith
("KXMLBTOTAL")` literal duplicated independently across engine/scanner.py,
engine/paper.py, signals/recommendation.py, signals/calibration.py,
output/reporter.py, backtest/evaluate.py, and cli.py. Adding a new sport
(e.g. NFL, confirmed live via KXNFLGAME/KXNFLTOTAL) means adding its prefixes
here once; every caller picks it up automatically.
"""
from __future__ import annotations

# prefix -> (sport, market kind: "winner" | "total")
SPORT_PREFIXES: dict[str, tuple[str, str]] = {
    "KXMLBGAME": ("mlb", "winner"),
    "KXMLBTOTAL": ("mlb", "total"),
    "KXNFLGAME": ("nfl", "winner"),
    "KXNFLTOTAL": ("nfl", "total"),
    "KXNBAGAME": ("nba", "winner"),
    "KXNBATOTAL": ("nba", "total"),
}


def _lookup(ticker: str) -> tuple[str, str]:
    for prefix, (sport, kind) in SPORT_PREFIXES.items():
        if ticker.startswith(prefix):
            return sport, kind
    return "unknown", "winner"


def sport_of(ticker: str) -> str:
    return _lookup(ticker)[0]


def market_kind(ticker: str) -> str:
    """'winner' or 'total'."""
    return _lookup(ticker)[1]


def is_total(ticker: str) -> bool:
    return market_kind(ticker) == "total"


def calibration_bucket(ticker: str) -> str:
    """Sport-dimensioned calibration bucket key, e.g. 'mlb:winner'.

    Without the sport prefix, bets from different sports sharing the same
    market kind would silently pool into one calibration bucket and distort
    each other's multiplier. Must ship before any second sport starts
    accumulating settled bets (paper or backtest).
    """
    sport, kind = _lookup(ticker)
    return f"{sport}:{kind}"
