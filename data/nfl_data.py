"""nflverse game data (games.csv): schedule + results, 1999-present.

Free, public, no key required -- the NFL analog of data/mlb_stats.py, and covers
in one CSV what MLB needed three separate endpoints for (schedule, team records,
run rates): https://github.com/nflverse/nfldata (the "games" file). Confirmed
live 2026-08-29: 7,548 rows, 1999 through the (not-yet-played) end of the 2026
season, including starting QBs (away_qb_name/home_qb_name -- the NFL analog of
MLB's probable pitcher) and roof/surface/temp/wind.

Not included: preseason games (games.csv is regular-season + playoffs only, per
game_type). Kalshi does list preseason KXNFLGAME/KXNFLTOTAL markets -- those
simply get no fair-value estimate (money-flow-only), same graceful-degradation
path MLB takes for any market it can't match to a real game.
"""
from __future__ import annotations

import csv
import io
import logging
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import requests

from config.settings import PROJECT_ROOT

log = logging.getLogger("data.nfl")

GAMES_CSV_URL = "https://github.com/nflverse/nfldata/raw/master/data/games.csv"
CACHE_PATH = PROJECT_ROOT / "data" / "nfl_games_cache.csv"
# NFL data moves slowly (games cluster Sun/Mon/Thu; no live-loop freshness
# pressure the way MLB's daily slate has) -- a long TTL avoids hitting GitHub
# every scan cycle.
CACHE_TTL_SECONDS = 12 * 3600

# Kalshi's ticker team codes vs nflverse's -- confirmed by diffing the full
# 32-team roster from live Kalshi KXNFLGAME markets against games.csv's team
# codes (2026-08-29): every code matches except these two.
KALSHI_TO_NFLVERSE = {"JAC": "JAX", "LAR": "LA"}
NFLVERSE_TO_KALSHI = {v: k for k, v in KALSHI_TO_NFLVERSE.items()}

# All 32 current Kalshi-style team codes (confirmed live via KXNFLGAME 2026-08-29).
TEAM_CODES = frozenset({
    "ARI", "ATL", "BAL", "BUF", "CAR", "CHI", "CIN", "CLE", "DAL", "DEN", "DET",
    "GB", "HOU", "IND", "JAC", "KC", "LAC", "LAR", "LV", "MIA", "MIN", "NE", "NO",
    "NYG", "NYJ", "PHI", "PIT", "SEA", "SF", "TB", "TEN", "WAS",
})


def _to_kalshi(code: str) -> str:
    return NFLVERSE_TO_KALSHI.get(code, code)


def split_team_codes(blob: str) -> Optional[tuple[str, str]]:
    """Split a concatenated ticker team-blob ('CHITEN') into (first, second)
    using the fixed 32-code roster -- codes are 2-3 chars so this is unambiguous
    in practice. Blob order is AWAY-then-HOME, confirmed against games.csv's
    game_id convention (e.g. '2026_02_NYG_LA' for Kalshi ticker blob 'NYGLAR')."""
    for i in (2, 3):
        a, b = blob[:i], blob[i:]
        if a in TEAM_CODES and b in TEAM_CODES:
            return a, b
    return None


def season_for_date(date_str: str) -> int:
    """NFL season year for a calendar date: Jan/Feb games belong to the season
    that started the previous September (matches nflverse's `season` column).
    Shared by signals/elo_nfl.py and data/scoring_environment_nfl.py."""
    year, month = int(date_str[:4]), int(date_str[5:7])
    return year if month >= 3 else year - 1


def _parse_int(v: Optional[str]) -> Optional[int]:
    try:
        return int(float(v)) if v not in (None, "") else None
    except (TypeError, ValueError):
        return None


@dataclass
class NflGame:
    game_id: str
    season: int
    week: int
    game_type: str          # "REG", "WC", "DIV", "CON", "SB"
    date_str: str            # YYYY-MM-DD (gameday)
    away_abbr: str           # Kalshi-style code
    home_abbr: str
    away_score: Optional[int]
    home_score: Optional[int]
    away_qb: str = ""
    home_qb: str = ""
    roof: str = ""
    surface: str = ""

    def teams(self) -> set[str]:
        return {self.away_abbr, self.home_abbr}

    @property
    def state(self) -> str:
        return "Final" if self.away_score is not None and self.home_score is not None else "Preview"

    @property
    def winner_abbr(self) -> Optional[str]:
        if self.away_score is None or self.home_score is None or self.away_score == self.home_score:
            return None
        return self.home_abbr if self.home_score > self.away_score else self.away_abbr

    @property
    def total_points(self) -> Optional[int]:
        if self.away_score is None or self.home_score is None:
            return None
        return self.away_score + self.home_score


class NflDataClient:
    """Fetches + caches nflverse's games.csv, keyed by Kalshi-style team codes."""

    def __init__(self, cache_path: Path = CACHE_PATH, ttl_seconds: int = CACHE_TTL_SECONDS):
        self.cache_path = cache_path
        self.ttl_seconds = ttl_seconds
        self._games: Optional[list[NflGame]] = None
        self._by_team: Optional[dict[str, list[NflGame]]] = None

    def _fetch_fresh(self) -> Optional[str]:
        try:
            r = requests.get(GAMES_CSV_URL, timeout=30)
            if r.status_code == 200:
                return r.text
            log.warning("nflverse games.csv -> HTTP %s", r.status_code)
        except requests.RequestException as e:
            log.warning("nflverse games.csv fetch failed: %s", e)
        return None

    def _load_text(self) -> str:
        stale = True
        if self.cache_path.exists():
            age = time.time() - self.cache_path.stat().st_mtime
            stale = age > self.ttl_seconds
        if stale:
            text = self._fetch_fresh()
            if text is not None:
                self.cache_path.parent.mkdir(parents=True, exist_ok=True)
                self.cache_path.write_text(text, encoding="utf-8")
                return text
            log.warning("nflverse fetch failed; falling back to cache if any.")
        if self.cache_path.exists():
            return self.cache_path.read_text(encoding="utf-8")
        raise RuntimeError("No nflverse games.csv available (fetch failed, no cache).")

    def games(self) -> list[NflGame]:
        if self._games is not None:
            return self._games
        text = self._load_text()
        out: list[NflGame] = []
        for row in csv.DictReader(io.StringIO(text)):
            away = _to_kalshi((row.get("away_team") or "").upper())
            home = _to_kalshi((row.get("home_team") or "").upper())
            if not away or not home:
                continue
            out.append(NflGame(
                game_id=row.get("game_id", ""),
                season=_parse_int(row.get("season")) or 0,
                week=_parse_int(row.get("week")) or 0,
                game_type=row.get("game_type", ""),
                date_str=row.get("gameday", ""),
                away_abbr=away, home_abbr=home,
                away_score=_parse_int(row.get("away_score")),
                home_score=_parse_int(row.get("home_score")),
                away_qb=row.get("away_qb_name", "") or "",
                home_qb=row.get("home_qb_name", "") or "",
                roof=row.get("roof", "") or "",
                surface=row.get("surface", "") or "",
            ))
        out.sort(key=lambda g: g.date_str)
        self._games = out
        return out

    def _team_index(self) -> dict[str, list[NflGame]]:
        if self._by_team is None:
            idx: dict[str, list[NflGame]] = {}
            for g in self.games():
                idx.setdefault(g.away_abbr, []).append(g)
                idx.setdefault(g.home_abbr, []).append(g)
            self._by_team = idx
        return self._by_team

    def team_games_before(self, team_abbr: str, as_of_date: str) -> list[NflGame]:
        """Completed games for `team_abbr` strictly before `as_of_date` (YYYY-MM-DD),
        chronological. The leak-safety primitive: Elo and the scoring-rate model
        both build off this instead of any 'as of today' field from an external
        source -- the exact leakage class MLB's season_backtest.py found and fixed
        for the MLB Stats API's leagueRecord (see its module docstring)."""
        return [g for g in self._team_index().get(team_abbr, [])
                if g.date_str < as_of_date and g.state == "Final"]

    def find_game(self, date_str: str, team_a: str, team_b: str,
                  window_days: int = 3) -> Optional[NflGame]:
        """Best-effort game lookup by approximate date + team pair (NFL kickoffs
        can shift a day for TV scheduling vs. what a ticker's date segment
        implies; mirrors season_backtest.py's SeasonLedger.find_game +/- buffer)."""
        want = {team_a, team_b}
        target = datetime.strptime(date_str, "%Y-%m-%d")
        best, best_dist = None, None
        for g in self._team_index().get(team_a, []):
            if g.teams() != want:
                continue
            dist = abs((datetime.strptime(g.date_str, "%Y-%m-%d") - target).days)
            if dist <= window_days and (best_dist is None or dist < best_dist):
                best, best_dist = g, dist
        return best

    def clear_cache(self) -> None:
        self._games = None
        self._by_team = None
