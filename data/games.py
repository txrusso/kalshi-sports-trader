"""Cross-sport game lookup + outcome resolution, dispatched by
config.sports.sport_of(). Replaces the MLB-only `_match_game`/`resolve_outcomes`
logic that used to be duplicated across engine/paper.py and
backtest/evaluate.py -- now that there are multiple sports, a single dispatch
point is a real de-duplication, not a speculative abstraction.

`clients` is always {"mlb": MlbStatsClient(), "nfl": NflDataClient(),
"nba": NbaDataClient(), "nhl": NhlDataClient()}; callers build it once (per
CLI invocation / loop lifetime) and pass it through.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from config.sports import is_total, sport_of
from data.fair_value import parse_ticker as parse_mlb_ticker
from data.fair_value_totals import parse_total_ticker as parse_mlb_total_ticker
from data.fair_value_nfl import parse_ticker as parse_nfl_ticker
from data.fair_value_nfl_totals import parse_total_ticker as parse_nfl_total_ticker
from data.fair_value_nba import parse_ticker as parse_nba_ticker
from data.fair_value_nba_totals import parse_total_ticker as parse_nba_total_ticker
from data.fair_value_nhl import parse_ticker as parse_nhl_ticker
from data.fair_value_nhl_totals import parse_total_ticker as parse_nhl_total_ticker
from data.mlb_stats import MlbStatsClient
from data.nfl_data import NflDataClient
from data.nba_data import NbaDataClient
from data.nhl_data import NhlDataClient


@dataclass
class Game:
    sport: str
    away_abbr: str
    home_abbr: str
    game_datetime: Optional[datetime]   # UTC start time; None when unavailable
    state: str                           # "Preview" / "Live" / "Final"
    winner_abbr: Optional[str]
    total_score: Optional[float]         # runs or points, when Final


def build_clients() -> dict:
    """The standard {"mlb": ..., "nfl": ..., "nba": ..., "nhl": ...} client bundle
    every caller needs."""
    return {"mlb": MlbStatsClient(), "nfl": NflDataClient(), "nba": NbaDataClient(),
            "nhl": NhlDataClient()}


def _match_mlb(ticker: str, mlb: MlbStatsClient, sched_cache: dict) -> Optional[Game]:
    total = is_total(ticker)
    pt = parse_mlb_total_ticker(ticker) if total else parse_mlb_ticker(ticker)
    if not pt:
        return None
    key = pt.date.strftime("%Y-%m-%d")
    if key not in sched_cache:
        sched_cache[key] = mlb.schedule(pt.date)
    for g in sched_cache[key]:
        matched = (pt.teams in (g.away_abbr + g.home_abbr, g.home_abbr + g.away_abbr) if total
                  else (pt.yes_team in g.teams() and (not pt.opponent or pt.opponent in g.teams())))
        if matched:
            return Game("mlb", g.away_abbr, g.home_abbr, g.game_datetime, g.state,
                       g.winner_abbr, g.total_runs)
    return None


def _match_nfl(ticker: str, nfl: NflDataClient) -> Optional[Game]:
    if is_total(ticker):
        pt = parse_nfl_total_ticker(ticker)
        if not pt:
            return None
        g = nfl.find_game(pt.date_str, pt.away_abbr, pt.home_abbr)
    else:
        pt = parse_nfl_ticker(ticker)
        if not pt:
            return None
        away = pt.opponent if pt.yes_is_home else pt.yes_team
        home = pt.yes_team if pt.yes_is_home else pt.opponent
        g = nfl.find_game(pt.date_str, away, home)
    if not g:
        return None
    return Game("nfl", g.away_abbr, g.home_abbr, g.kickoff_utc, g.state,
                g.winner_abbr, g.total_points)


def _match_nba(ticker: str, nba: NbaDataClient) -> Optional[Game]:
    if is_total(ticker):
        pt = parse_nba_total_ticker(ticker)
        if not pt:
            return None
        g = nba.find_game(pt.date_str, pt.away_abbr, pt.home_abbr)
    else:
        pt = parse_nba_ticker(ticker)
        if not pt:
            return None
        away = pt.opponent if pt.yes_is_home else pt.yes_team
        home = pt.yes_team if pt.yes_is_home else pt.opponent
        g = nba.find_game(pt.date_str, away, home)
    if not g:
        return None
    # No time-of-day in this data source (see data/nba_data.py) -- game_datetime
    # is always None for NBA; callers fall back to the market's own
    # occurrence_datetime (same last-resort NFL already documents).
    return Game("nba", g.away_abbr, g.home_abbr, None, g.state,
                g.winner_abbr, g.total_points)


def _match_nhl(ticker: str, nhl: NhlDataClient) -> Optional[Game]:
    if is_total(ticker):
        pt = parse_nhl_total_ticker(ticker)
        if not pt:
            return None
        g = nhl.find_game(pt.date_str, pt.away_abbr, pt.home_abbr)
    else:
        pt = parse_nhl_ticker(ticker)
        if not pt:
            return None
        away = pt.opponent if pt.yes_is_home else pt.yes_team
        home = pt.yes_team if pt.yes_is_home else pt.opponent
        g = nhl.find_game(pt.date_str, away, home)
    if not g:
        return None
    # UNLIKE NBA, this data source DOES carry a real start time -- see
    # NhlGame.puck_drop_utc / data/nhl_data.py's module docstring.
    return Game("nhl", g.away_abbr, g.home_abbr, g.puck_drop_utc, g.state,
                g.winner_abbr, g.total_goals)


_TOTAL_PARSERS = {"mlb": parse_mlb_total_ticker, "nfl": parse_nfl_total_ticker,
                  "nba": parse_nba_total_ticker, "nhl": parse_nhl_total_ticker}
_WINNER_PARSERS = {"mlb": parse_mlb_ticker, "nfl": parse_nfl_ticker, "nba": parse_nba_ticker,
                   "nhl": parse_nhl_ticker}


def match_game(ticker: str, clients: dict, sched_cache: dict) -> Optional[Game]:
    """The game a market's ticker refers to, or None. `sched_cache` is a
    caller-owned dict (MLB's per-date schedule cache) reused across many
    tickers in one call -- pass the same dict across a batch (e.g.
    engine/paper.py's per-cycle trigger loop, resolve_outcomes below) so MLB's
    schedule isn't re-fetched per ticker."""
    sport = sport_of(ticker)
    if sport == "mlb":
        return _match_mlb(ticker, clients["mlb"], sched_cache)
    if sport == "nfl":
        return _match_nfl(ticker, clients["nfl"])
    if sport == "nba":
        return _match_nba(ticker, clients["nba"])
    if sport == "nhl":
        return _match_nhl(ticker, clients["nhl"])
    return None


def resolve_outcomes(tickers: set[str], clients: dict) -> dict[str, bool]:
    """ticker -> did YES resolve true? (winner: YES team won; total: over hit).
    Skips games not yet Final."""
    out: dict[str, bool] = {}
    sched_cache: dict = {}
    for tk in tickers:
        g = match_game(tk, clients, sched_cache)
        if not g or g.state != "Final":
            continue
        sport = sport_of(tk)
        if is_total(tk):
            if g.total_score is None:
                continue
            line = _TOTAL_PARSERS[sport](tk)
            if not line:
                continue
            out[tk] = (g.total_score > line.line)
        else:
            if not g.winner_abbr:
                continue
            pt = _WINNER_PARSERS[sport](tk)
            if not pt:
                continue
            out[tk] = (g.winner_abbr == pt.yes_team)
    return out
