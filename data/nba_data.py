"""NBA game-log data: nba_stats_schedule_<season>.csv releases published by
sportsdataverse-data (github.com/sportsdataverse/sportsdataverse-data), the NBA
analog of data/nfl_data.py's nflverse games.csv. Free, public, no key required.

Structural difference from nflverse: NBA's mirror ships ONE FILE PER SEASON
(keyed by the season's STARTING calendar year, e.g. "2025" = the 2025-26
season), 1996-present, rather than one cumulative file -- so this client
fetches and merges a rolling window of the last NBA_HISTORY_SEASONS season
files instead of one file. Each file is also shaped as a per-team-per-game log
(exactly 2 rows per game_id, one per team, home/away parsed from the
`matchup` column: "TEAM vs. OPP" = TEAM home, "TEAM @ OPP" = TEAM away) rather
than nflverse's one-row-per-game-with-home/away-columns shape -- rows are
paired by game_id into one NbaGame each in games().

Confirmed live 2026-09-10:
  - Kalshi's 30 team tricodes match this source EXACTLY (spot-checked against
    Kalshi's KXNBA-27-<TEAM> championship futures, all 30 teams) -- no
    NFL-style JAX/LAR remap needed.
  - season_type is only ever "regular-season"/"playoffs" (checked the full
    2025-26 file) -- preseason is excluded entirely, same free "no game
    found => likely preseason" refusal signal nflverse gives NFL's model.
  - The file for a season that hasn't started yet does not exist yet
    (nba_stats_schedule_2026.csv 404s, five-plus weeks before Kalshi's own
    listed 2026-27 openers on Oct 20). The repo's automation runs "daily,
    late Oct-mid Jul" per its README, so this should resolve itself before
    the season actually starts trading -- but until that file appears, every
    2026-27 game looks like "no game found" and NBA gets zero fair-value
    estimates. Worth a live recheck in early-mid October before relying on
    this in production.
  - UNLIKE nflverse, this source carries no tip-off time-of-day, only
    game_date -- so there is no independent way to cross-check Kalshi's own
    `occurrence_datetime` the way NFL's model caught (and corrected) a
    systematic +3h skew there (see engine/paper.py's module note). NBA's
    fair-value model and the paper-trigger both have to trust
    occurrence_datetime as-is for now. Recommend a manual spot check of a
    real Kalshi occurrence_datetime against the actual broadcast tip-off time
    before the paper loop goes live for NBA -- if the same skew exists here,
    it would silently misfire the trigger window the same way it did for NFL
    until that was caught.
"""
from __future__ import annotations

import csv
import io
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import requests

from config.settings import PROJECT_ROOT

log = logging.getLogger("data.nba")

SCHEDULE_URL_TMPL = ("https://github.com/sportsdataverse/sportsdataverse-data/releases/"
                     "download/nba_stats_schedules/nba_stats_schedule_{season}.csv")
CACHE_DIR = PROJECT_ROOT / "data" / "nba_games_cache"
# How many past-plus-current season files to load for Elo/scoring-rate history.
# Each is a separate HTTP fetch (unlike nflverse's one cumulative file), so this
# is a real cost/depth tradeoff -- 4 seasons is ~5,200 games, comparable in scale
# to MLB's Elo history window (~9,300 games back to 2023, i.e. ~3.8 MLB seasons).
NBA_HISTORY_SEASONS = 4
# NBA moves slowly like NFL (games cluster a few nights/week, no MLB-style daily
# slate) -- a long TTL avoids hitting GitHub every scan cycle.
CACHE_TTL_SECONDS = 12 * 3600

# All 30 current Kalshi-style team codes (confirmed live via KXNBA-27-<TEAM>
# championship futures 2026-09-10 -- identical to this data source's own codes).
TEAM_CODES = frozenset({
    "ATL", "BKN", "BOS", "CHA", "CHI", "CLE", "DAL", "DEN", "DET", "GSW", "HOU",
    "IND", "LAC", "LAL", "MEM", "MIA", "MIL", "MIN", "NOP", "NYK", "OKC", "ORL",
    "PHI", "PHX", "POR", "SAC", "SAS", "TOR", "UTA", "WAS",
})


def split_team_codes(blob: str) -> Optional[tuple[str, str]]:
    """Split a concatenated ticker team-blob ('BOSDET') into (first, second)
    using the fixed 30-code roster -- mirrors data/nfl_data.py's identical
    helper. Blob order is AWAY-then-HOME (confirmed live 2026-09-10 against
    Kalshi's own KXNBAGAME markets, same convention as MLB/NFL)."""
    for i in (3,):
        a, b = blob[:i], blob[i:]
        if a in TEAM_CODES and b in TEAM_CODES:
            return a, b
    return None


def season_for_date(date_str: str) -> int:
    """NBA season year (its STARTING calendar year) for a calendar date:
    Jan-Jul games belong to the season that started the previous
    October/fall. Matches this data source's own `season` column (checked:
    an April 2026 game carried season=2025, i.e. the "2025-26" season).
    Shared by signals/elo_nba.py and data/scoring_environment_nba.py."""
    year, month = int(date_str[:4]), int(date_str[5:7])
    return year if month >= 8 else year - 1


def _parse_int(v: Optional[str]) -> Optional[int]:
    try:
        return int(float(v)) if v not in (None, "") else None
    except (TypeError, ValueError):
        return None


def _parse_matchup(team: str, matchup: str) -> Optional[tuple[str, bool]]:
    """('OPP', is_home) from a row's own team + its `matchup` string
    ("BOS vs. DET" -> home; "BOS @ DET" -> away)."""
    if " vs. " in matchup:
        opp = matchup.split(" vs. ")[-1].strip()
        return opp, True
    if " @ " in matchup:
        opp = matchup.split(" @ ")[-1].strip()
        return opp, False
    return None


@dataclass
class NbaGame:
    game_id: str
    season: int
    season_type: str          # "regular-season" | "playoffs"
    date_str: str              # YYYY-MM-DD
    away_abbr: str
    home_abbr: str
    away_score: Optional[int]
    home_score: Optional[int]

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


class NbaDataClient:
    """Fetches + caches a rolling window of nba_stats_schedule_<season>.csv
    files, keyed by Kalshi-style team codes (identity mapping -- no remap
    table needed, see module docstring)."""

    def __init__(self, cache_dir: Path = CACHE_DIR, ttl_seconds: int = CACHE_TTL_SECONDS,
                history_seasons: int = NBA_HISTORY_SEASONS):
        self.cache_dir = cache_dir
        self.ttl_seconds = ttl_seconds
        self.history_seasons = history_seasons
        self._games: Optional[list[NbaGame]] = None
        self._by_team: Optional[dict[str, list[NbaGame]]] = None

    def _fetch_fresh(self, season: int) -> Optional[str]:
        try:
            r = requests.get(SCHEDULE_URL_TMPL.format(season=season), timeout=30)
            if r.status_code == 200:
                return r.text
            if r.status_code != 404:   # 404 = season hasn't started yet; not a warning
                log.warning("nba_stats_schedule_%s.csv -> HTTP %s", season, r.status_code)
        except requests.RequestException as e:
            log.warning("nba_stats_schedule_%s.csv fetch failed: %s", season, e)
        return None

    def _load_season_text(self, season: int) -> Optional[str]:
        path = self.cache_dir / f"schedule_{season}.csv"
        stale = True
        if path.exists():
            age = time.time() - path.stat().st_mtime
            stale = age > self.ttl_seconds
        if stale:
            text = self._fetch_fresh(season)
            if text is not None:
                self.cache_dir.mkdir(parents=True, exist_ok=True)
                path.write_text(text, encoding="utf-8")
                return text
        if path.exists():
            return path.read_text(encoding="utf-8")
        return None   # no cache, no fresh fetch (e.g. season hasn't started) -- not an error

    def _parse_season(self, text: str) -> list[NbaGame]:
        by_game: dict[str, dict] = {}
        for row in csv.DictReader(io.StringIO(text)):
            team = (row.get("team_abbreviation") or "").upper()
            gid = row.get("game_id", "")
            parsed = _parse_matchup(team, row.get("matchup", "") or "")
            if not team or not gid or not parsed:
                continue
            opp, is_home = parsed
            pts = _parse_int(row.get("pts"))
            entry = by_game.setdefault(gid, {
                "game_id": gid, "date_str": row.get("game_date", ""),
                "season": _parse_int(row.get("season")) or 0,
                "season_type": row.get("season_type", "") or "",
            })
            if is_home:
                entry["home_abbr"], entry["home_score"] = team, pts
            else:
                entry["away_abbr"], entry["away_score"] = team, pts

        out: list[NbaGame] = []
        for entry in by_game.values():
            if "home_abbr" not in entry or "away_abbr" not in entry:
                continue   # incomplete pair (missing/malformed row on one side)
            out.append(NbaGame(
                game_id=entry["game_id"], season=entry["season"],
                season_type=entry["season_type"], date_str=entry["date_str"],
                away_abbr=entry["away_abbr"], home_abbr=entry["home_abbr"],
                away_score=entry.get("away_score"), home_score=entry.get("home_score"),
            ))
        return out

    def games(self) -> list[NbaGame]:
        if self._games is not None:
            return self._games
        import datetime as _dt
        current_season = season_for_date(_dt.datetime.now().strftime("%Y-%m-%d"))
        out: list[NbaGame] = []
        for season in range(current_season - self.history_seasons + 1, current_season + 1):
            text = self._load_season_text(season)
            if text is None:
                continue   # e.g. the current season hasn't started yet -- see module docstring
            out.extend(self._parse_season(text))
        out.sort(key=lambda g: g.date_str)
        self._games = out
        return out

    def _team_index(self) -> dict[str, list[NbaGame]]:
        if self._by_team is None:
            idx: dict[str, list[NbaGame]] = {}
            for g in self.games():
                idx.setdefault(g.away_abbr, []).append(g)
                idx.setdefault(g.home_abbr, []).append(g)
            self._by_team = idx
        return self._by_team

    def team_games_before(self, team_abbr: str, as_of_date: str) -> list[NbaGame]:
        """Completed games for `team_abbr` strictly before `as_of_date`
        (YYYY-MM-DD), chronological -- the leak-safety primitive, mirrors
        data/nfl_data.py's identical method."""
        return [g for g in self._team_index().get(team_abbr, [])
                if g.date_str < as_of_date and g.state == "Final"]

    def find_game(self, date_str: str, team_a: str, team_b: str,
                  window_days: int = 2) -> Optional[NbaGame]:
        """Best-effort game lookup by approximate date + team pair (mirrors
        data/nfl_data.py's identical method; NBA games rarely shift a day but
        a small buffer is cheap insurance against timezone-boundary noise)."""
        import datetime as _dt
        want = {team_a, team_b}
        target = _dt.datetime.strptime(date_str, "%Y-%m-%d")
        best, best_dist = None, None
        for g in self._team_index().get(team_a, []):
            if g.teams() != want:
                continue
            dist = abs((_dt.datetime.strptime(g.date_str, "%Y-%m-%d") - target).days)
            if dist <= window_days and (best_dist is None or dist < best_dist):
                best, best_dist = g, dist
        return best

    def clear_cache(self) -> None:
        self._games = None
        self._by_team = None
