"""NHL schedule + results, from the NHL's own official API (api-web.nhle.com)
-- unlike data/nfl_data.py's/data/nba_data.py's static-file mirrors, this is a
live API client, the NHL analog of data/mlb_stats.py's pattern. Free, public,
no key required, confirmed reachable live 2026-09-10.

Fetches one season per team via `/v1/club-schedule-season/{team}/{season}`
(season format is an 8-digit start+end year, e.g. "20242025") -- one call
returns essentially that team's WHOLE season (preseason + regular + playoffs,
~85-100 games), so a full league-season needs 32 calls (one per team), deduped
by game id since every game appears in both teams' schedules. Each season is
cached to its own disk file.

Confirmed live 2026-09-10:
  - UNLIKE data/nba_data.py's third-party mirror, the NHL's own API already
    carries the FULL 2026-27 season schedule (preseason from Sep 20 through
    the regular season), even though the season hasn't started -- no "file
    doesn't exist yet" gap the way NBA has.
  - gameType: 1=preseason, 2=regular season, 3=playoffs (verified against a
    real team-season with games spanning all three, Sep 2024-Jun 2025).
    Preseason is filtered out here (this API doesn't exclude it upstream the
    way nflverse/the NBA mirror do) -- same "no game found => preseason"
    refusal signal the other two sports get for free.
  - Kalshi's 32 team codes differ from the NHL API's in FOUR places (more
    than NFL's two): Kalshi's `LA`/`NJ`/`SJ`/`TB` vs the API's
    `LAK`/`NJD`/`SJS`/`TBL` -- mapped explicitly below, same class of bug as
    NFL's `JAX`/`LA`. Utah (`UTA`) already matches on both sides despite its
    2024 relocation from Arizona.
  - A shootout's nominal winning goal IS baked into the official final score
    -- verified against a real game (2025-01-05, NYI 1 @ NSH 2 SO):
    play-by-play shows only 2 real goals total (1-1 through regulation), but
    the recorded final score is 1-2, the standard "shootout winner gets +1 in
    the box score" convention every broadcast/NHL.com uses. So the
    `total_goals` computed here already matches whatever "official final
    score" any settlement (Kalshi included) would reference -- no
    special-casing needed for shootout games.
  - UNLIKE nba_data.py, this source DOES carry a real start time
    (`startTimeUTC`) per game, not just a date -- see `puck_drop_utc` below.
    That's a genuine advantage over NBA: it gives an independent source to
    cross-check Kalshi's own `occurrence_datetime` against, the same
    protection NFL's `kickoff_utc` (data/nfl_data.py) already provides and
    NBA still lacks.
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import requests

from config.settings import PROJECT_ROOT

log = logging.getLogger("data.nhl")

NHL_API_BASE = "https://api-web.nhle.com/v1"
CACHE_DIR = PROJECT_ROOT / "data" / "nhl_games_cache"
# How many past-plus-current season files to load for Elo/scoring-rate history.
# Each season costs 32 HTTP calls (one per team) on a cold cache, unlike
# nba_data.py's one-file-per-season -- kept at the same depth as NBA's window
# for a comparable Elo burn-in.
NHL_HISTORY_SEASONS = 4
# A full season's worth of real games rarely changes; a long TTL avoids
# re-fetching 32 team-schedules every scan cycle.
CACHE_TTL_SECONDS = 12 * 3600

# Kalshi's ticker team codes vs the NHL API's -- confirmed live 2026-09-10 by
# diffing the full 32-team roster from Kalshi's KXNHL-27-<TEAM> championship
# futures against the NHL API's /v1/standings/now.
KALSHI_TO_NHL = {"LA": "LAK", "NJ": "NJD", "SJ": "SJS", "TB": "TBL"}
NHL_TO_KALSHI = {v: k for k, v in KALSHI_TO_NHL.items()}

# All 32 current Kalshi-style team codes (confirmed live via KXNHL-27-<TEAM>
# championship futures 2026-09-10).
TEAM_CODES = frozenset({
    "ANA", "BOS", "BUF", "CAR", "CBJ", "CGY", "CHI", "COL", "DAL", "DET", "EDM",
    "FLA", "LA", "MIN", "MTL", "NJ", "NSH", "NYI", "NYR", "OTT", "PHI", "PIT",
    "SEA", "SJ", "STL", "TB", "TOR", "UTA", "VAN", "VGK", "WPG", "WSH",
})

GAME_TYPE_PRESEASON, GAME_TYPE_REGULAR, GAME_TYPE_PLAYOFFS = 1, 2, 3


def _to_kalshi(code: str) -> str:
    return NHL_TO_KALSHI.get(code, code)


def _to_nhl(code: str) -> str:
    return KALSHI_TO_NHL.get(code, code)


def split_team_codes(blob: str) -> Optional[tuple[str, str]]:
    """Split a concatenated ticker team-blob into (first, second) using the
    fixed 32-code roster -- mirrors data/nfl_data.py's identical helper.
    Kalshi's NHL codes are 2-3 chars (LA/NJ/SJ/TB are 2; the rest are 3), the
    same length ambiguity class as NFL's, resolved the same way (no code's
    2-char prefix collides with another valid code, checked against the
    fixed roster)."""
    for i in (2, 3):
        a, b = blob[:i], blob[i:]
        if a in TEAM_CODES and b in TEAM_CODES:
            return a, b
    return None


def season_for_date(date_str: str) -> int:
    """NHL season year (its STARTING calendar year) for a calendar date --
    mirrors data/nba_data.py's identical function (NHL's season spans
    Sep/Oct-June, same as NBA's Oct-June, so the same August cutoff applies)."""
    year, month = int(date_str[:4]), int(date_str[5:7])
    return year if month >= 8 else year - 1


@dataclass
class NhlGame:
    game_id: int
    season: int
    game_type: int          # 1=preseason (filtered out), 2=regular, 3=playoffs
    date_str: str             # YYYY-MM-DD
    away_abbr: str
    home_abbr: str
    away_score: Optional[int]
    home_score: Optional[int]
    start_time_utc: str = ""  # raw ISO string from the API's startTimeUTC

    def teams(self) -> set[str]:
        return {self.away_abbr, self.home_abbr}

    @property
    def puck_drop_utc(self) -> Optional[datetime]:
        """Real scheduled start time as UTC -- the NHL analog of
        data/nfl_data.py's NflGame.kickoff_utc, used the same way (as an
        independent cross-check against Kalshi's own occurrence_datetime)."""
        if not self.start_time_utc:
            return None
        try:
            return datetime.fromisoformat(self.start_time_utc.replace("Z", "+00:00"))
        except ValueError:
            return None

    @property
    def state(self) -> str:
        return "Final" if self.away_score is not None and self.home_score is not None else "Preview"

    @property
    def winner_abbr(self) -> Optional[str]:
        if self.away_score is None or self.home_score is None or self.away_score == self.home_score:
            return None
        return self.home_abbr if self.home_score > self.away_score else self.away_abbr

    @property
    def total_goals(self) -> Optional[int]:
        if self.away_score is None or self.home_score is None:
            return None
        return self.away_score + self.home_score


class NhlDataClient:
    """Fetches + caches a rolling window of NHL seasons via per-team calls to
    the official NHL API, keyed by Kalshi-style team codes."""

    def __init__(self, cache_dir: Path = CACHE_DIR, ttl_seconds: int = CACHE_TTL_SECONDS,
                history_seasons: int = NHL_HISTORY_SEASONS):
        self.cache_dir = cache_dir
        self.ttl_seconds = ttl_seconds
        self.history_seasons = history_seasons
        self._games: Optional[list[NhlGame]] = None
        self._by_team: Optional[dict[str, list[NhlGame]]] = None

    def _fetch_team_season(self, nhl_code: str, season_str: str) -> Optional[list[dict]]:
        try:
            r = requests.get(f"{NHL_API_BASE}/club-schedule-season/{nhl_code}/{season_str}",
                             timeout=20)
            if r.status_code == 200:
                return r.json().get("games", [])
            if r.status_code != 404:
                log.warning("NHL schedule %s/%s -> HTTP %s", nhl_code, season_str, r.status_code)
        except requests.RequestException as e:
            log.warning("NHL schedule %s/%s fetch failed: %s", nhl_code, season_str, e)
        return None

    def _fetch_season_fresh(self, season: int) -> list[NhlGame]:
        season_str = f"{season}{season + 1}"
        by_id: dict[int, NhlGame] = {}
        for kalshi_code in TEAM_CODES:
            games = self._fetch_team_season(_to_nhl(kalshi_code), season_str)
            if not games:
                continue
            for g in games:
                if g.get("gameType") == GAME_TYPE_PRESEASON:
                    continue
                gid = g.get("id")
                if gid is None or gid in by_id:
                    continue
                by_id[gid] = NhlGame(
                    game_id=gid, season=season, game_type=g.get("gameType") or 0,
                    date_str=g.get("gameDate", ""),
                    away_abbr=_to_kalshi((g.get("awayTeam", {}).get("abbrev") or "").upper()),
                    home_abbr=_to_kalshi((g.get("homeTeam", {}).get("abbrev") or "").upper()),
                    away_score=g.get("awayTeam", {}).get("score"),
                    home_score=g.get("homeTeam", {}).get("score"),
                    start_time_utc=g.get("startTimeUTC", "") or "",
                )
        return sorted(by_id.values(), key=lambda g: g.date_str)

    def _load_season(self, season: int) -> list[NhlGame]:
        path = self.cache_dir / f"season_{season}.json"
        stale = True
        if path.exists():
            age = time.time() - path.stat().st_mtime
            stale = age > self.ttl_seconds
        if stale:
            games = self._fetch_season_fresh(season)
            if games:
                self.cache_dir.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps([asdict(g) for g in games]), encoding="utf-8")
                return games
            log.warning("NHL season %s fetch returned nothing; falling back to cache if any.", season)
        if path.exists():
            return [NhlGame(**g) for g in json.loads(path.read_text(encoding="utf-8"))]
        return []   # e.g. a season that hasn't started and has no cache yet -- not an error

    def games(self) -> list[NhlGame]:
        if self._games is not None:
            return self._games
        current_season = season_for_date(datetime.now().strftime("%Y-%m-%d"))
        out: list[NhlGame] = []
        for season in range(current_season - self.history_seasons + 1, current_season + 1):
            out.extend(self._load_season(season))
        out.sort(key=lambda g: g.date_str)
        self._games = out
        return out

    def _team_index(self) -> dict[str, list[NhlGame]]:
        if self._by_team is None:
            idx: dict[str, list[NhlGame]] = {}
            for g in self.games():
                idx.setdefault(g.away_abbr, []).append(g)
                idx.setdefault(g.home_abbr, []).append(g)
            self._by_team = idx
        return self._by_team

    def team_games_before(self, team_abbr: str, as_of_date: str) -> list[NhlGame]:
        """Completed games for `team_abbr` strictly before `as_of_date`
        (YYYY-MM-DD), chronological -- the leak-safety primitive, mirrors
        data/nba_data.py's identical method."""
        return [g for g in self._team_index().get(team_abbr, [])
                if g.date_str < as_of_date and g.state == "Final"]

    def find_game(self, date_str: str, team_a: str, team_b: str,
                  window_days: int = 2) -> Optional[NhlGame]:
        """Best-effort game lookup by approximate date + team pair -- mirrors
        data/nba_data.py's identical method."""
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
