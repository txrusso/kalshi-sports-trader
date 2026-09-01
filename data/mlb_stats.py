"""MLB Stats API (statsapi.mlb.com): schedule, team records, live win probability.

Open, reliable, no key required. This is the primary fair-value data source.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional

import requests

from config.settings import EASTERN

log = logging.getLogger("data.mlb")

BASE = "https://statsapi.mlb.com/api/v1"

# Recency window for team form (data/run_environment.py): last N regular-season
# games, not last N calendar days -- rosters/hot streaks matter more than the
# calendar. Fetches a wider day-window then trims to the most recent N games so
# off-days/rainouts don't shrink the sample.
RECENT_WINDOW_GAMES = 30
RECENT_LOOKBACK_DAYS = 55
RECENT_MIN_GAMES = 10   # below this, callers should fall back to season-to-date


@dataclass
class MlbGame:
    game_pk: int
    date_str: str                  # YYYY-MM-DD (official date)
    away_abbr: str
    home_abbr: str
    away_name: str
    home_name: str
    state: str                     # "Preview", "Live", "Final"
    detailed_state: str
    away_win_pct: Optional[float]  # season record pct
    home_win_pct: Optional[float]
    winner_abbr: Optional[str] = None   # set when Final
    total_runs: Optional[int] = None    # away_score + home_score when Final
    away_pitcher_id: Optional[int] = None   # probable starter (when announced)
    home_pitcher_id: Optional[int] = None
    game_datetime: Optional[datetime] = None   # authoritative first-pitch time (UTC)

    def teams(self) -> set[str]:
        return {self.away_abbr, self.home_abbr}


class MlbStatsClient:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "kalshi-agent research"})
        self._wp_cache: dict[int, Optional[float]] = {}
        self._pitcher_cache: dict[int, Optional[tuple]] = {}
        self._recent_stats_cache: dict[str, dict] = {}

    def _get(self, url: str, params: Optional[dict] = None) -> Optional[dict]:
        try:
            r = self.session.get(url, params=params, timeout=15)
            if r.status_code == 200:
                return r.json()
            log.warning("MLB %s -> HTTP %s", url, r.status_code)
        except requests.RequestException as e:
            log.warning("MLB request failed: %s", e)
        return None

    def schedule(self, date: Optional[datetime] = None) -> list[MlbGame]:
        d = (date or datetime.now(EASTERN)).strftime("%Y-%m-%d")
        data = self._get(f"{BASE}/schedule", {
            "sportId": 1, "date": d, "hydrate": "team,linescore,probablePitcher",
        })
        if not data:
            return []
        games: list[MlbGame] = []
        for date_entry in data.get("dates", []) or []:
            for g in date_entry.get("games", []) or []:
                try:
                    away = g["teams"]["away"]
                    home = g["teams"]["home"]
                    winner = None
                    if home.get("isWinner"):
                        winner = (home["team"].get("abbreviation") or "").upper()
                    elif away.get("isWinner"):
                        winner = (away["team"].get("abbreviation") or "").upper()
                    total_runs = None
                    if g["status"].get("abstractGameState") == "Final":
                        hs, as_ = home.get("score"), away.get("score")
                        if hs is not None and as_ is not None:
                            total_runs = int(hs) + int(as_)
                    games.append(MlbGame(
                        game_pk=g["gamePk"],
                        date_str=g.get("officialDate", d),
                        away_abbr=(away["team"].get("abbreviation") or "").upper(),
                        home_abbr=(home["team"].get("abbreviation") or "").upper(),
                        away_name=away["team"].get("name", ""),
                        home_name=home["team"].get("name", ""),
                        state=g["status"].get("abstractGameState", ""),
                        detailed_state=g["status"].get("detailedState", ""),
                        away_win_pct=_rec_pct(away.get("leagueRecord")),
                        home_win_pct=_rec_pct(home.get("leagueRecord")),
                        winner_abbr=winner,
                        total_runs=total_runs,
                        away_pitcher_id=(away.get("probablePitcher") or {}).get("id"),
                        home_pitcher_id=(home.get("probablePitcher") or {}).get("id"),
                        game_datetime=_parse_iso(g.get("gameDate")),
                    ))
                except (KeyError, TypeError):
                    continue
        return games

    def team_recent_stats(self, as_of: Optional[datetime] = None) -> dict[str, dict]:
        """{team_abbr: {win_pct, runs_scored_pg, runs_allowed_pg, n_games}} from each
        team's last RECENT_WINDOW_GAMES regular-season games strictly before `as_of`
        (default now). Recency-weighted alternative to the season-cumulative record/
        rates -- a team's April form shouldn't count the same as last week's after a
        trade or injury. Cached per as_of calendar day (ET, matching MLB's
        officialDate) so a live 30-min scan loop doesn't re-fetch the range every
        cycle, but a new day naturally busts the cache -- no staleness risk like
        team_run_rates()'s permanent-for-the-process-lifetime cache.

        Callers should fall back to season-to-date stats when a team's n_games is
        below RECENT_MIN_GAMES (early season, or a team with a short history here).
        """
        as_of_et = (as_of or datetime.now(EASTERN)).astimezone(EASTERN)
        cache_key = as_of_et.strftime("%Y-%m-%d")
        if cache_key in self._recent_stats_cache:
            return self._recent_stats_cache[cache_key]

        start = (as_of_et - timedelta(days=RECENT_LOOKBACK_DAYS)).strftime("%Y-%m-%d")
        end = as_of_et.strftime("%Y-%m-%d")
        data = self._get(f"{BASE}/schedule", {
            "sportId": 1, "startDate": start, "endDate": end, "hydrate": "team,linescore",
        })
        team_games: dict[str, list[tuple]] = {}
        if data:
            for de in data.get("dates", []) or []:
                for g in de.get("games", []) or []:
                    if g.get("gameType") != "R":
                        continue
                    if g["status"].get("abstractGameState") != "Final":
                        continue
                    date_str = g.get("officialDate")
                    away, home = g["teams"]["away"], g["teams"]["home"]
                    away_score, home_score = away.get("score"), home.get("score")
                    if not date_str or away_score is None or home_score is None:
                        continue
                    if date_str >= cache_key:   # strictly before as_of's calendar day
                        continue
                    away_abbr = (away["team"].get("abbreviation") or "").upper()
                    home_abbr = (home["team"].get("abbreviation") or "").upper()
                    team_games.setdefault(away_abbr, []).append((date_str, away_score > home_score, away_score, home_score))
                    team_games.setdefault(home_abbr, []).append((date_str, home_score > away_score, home_score, away_score))

        result: dict[str, dict] = {}
        for team, games in team_games.items():
            games.sort(key=lambda x: x[0])
            recent = games[-RECENT_WINDOW_GAMES:]
            n = len(recent)
            if n == 0:
                continue
            wins = sum(1 for g in recent if g[1])
            result[team] = {
                "win_pct": wins / n,
                "runs_scored_pg": sum(g[2] for g in recent) / n,
                "runs_allowed_pg": sum(g[3] for g in recent) / n,
                "n_games": n,
            }
        self._recent_stats_cache[cache_key] = result
        return result

    def home_win_probability(self, game_pk: int) -> Optional[float]:
        """Latest live home-team win probability (0..1). Cached per cycle."""
        if game_pk in self._wp_cache:
            return self._wp_cache[game_pk]
        data = self._get(f"{BASE}/game/{game_pk}/winProbability")
        result: Optional[float] = None
        if isinstance(data, list) and data:
            last = data[-1]
            pct = last.get("homeTeamWinProbability")
            if pct is not None:
                try:
                    result = float(pct) / 100.0
                except (TypeError, ValueError):
                    result = None
        self._wp_cache[game_pk] = result
        return result

    def clear_cache(self) -> None:
        self._wp_cache.clear()

    def team_run_rates(self, season: Optional[int] = None) -> dict[str, tuple[float, float]]:
        """{team_abbr: (runs_scored_per_game, runs_allowed_per_game)} from standings.

        Cached for the client's lifetime (season rates move slowly).
        """
        if getattr(self, "_run_rates", None):
            return self._run_rates
        season = season or datetime.now(EASTERN).year
        data = self._get(f"{BASE}/standings", {
            "leagueId": "103,104", "season": season,
            "standingsTypes": "regularSeason", "hydrate": "team",
        })
        rates: dict[str, tuple[float, float]] = {}
        if data:
            for block in data.get("records", []) or []:
                for tr in block.get("teamRecords", []) or []:
                    ab = (tr.get("team", {}).get("abbreviation") or "").upper()
                    gp = tr.get("gamesPlayed") or 0
                    rs, ra = tr.get("runsScored"), tr.get("runsAllowed")
                    if ab and gp and rs is not None and ra is not None:
                        rates[ab] = (rs / gp, ra / gp)
        self._run_rates = rates
        return rates

    def pitcher_ra9(self, pitcher_id: Optional[int], season: Optional[int] = None) -> Optional[tuple]:
        """(runs_allowed_per_9, innings_pitched, games_started) for a pitcher's season.

        Uses total runs (not just earned) since Kalshi totals count all runs. Cached.
        Returns None if unknown or no innings.
        """
        if not pitcher_id:
            return None
        if pitcher_id in self._pitcher_cache:
            return self._pitcher_cache[pitcher_id]
        season = season or datetime.now(EASTERN).year
        data = self._get(f"{BASE}/people/{pitcher_id}/stats",
                         {"stats": "season", "group": "pitching", "season": season})
        result = None
        try:
            st = data["stats"][0]["splits"][0]["stat"]
            ip = _ip_to_float(st.get("inningsPitched", "0"))
            runs = float(st.get("runs") or 0)
            if ip > 0:
                result = (runs * 9.0 / ip, ip, int(st.get("gamesStarted") or 0))
        except (KeyError, IndexError, TypeError):
            result = None
        self._pitcher_cache[pitcher_id] = result
        return result


def fetch_historical_games(start_date: str, end_date: Optional[str] = None,
                            session: Optional[requests.Session] = None) -> list[dict]:
    """Completed regular-season games in [start_date, end_date] (YYYY-MM-DD), chunked
    by calendar year (the schedule endpoint is queried one year at a time -- kept
    consistent across callers rather than risking a truncated response on a multi-year
    range) and sorted chronologically: {date_str, away_abbr, home_abbr, away_score,
    home_score}. Used to build MLB Elo history (signals/elo_mlb.py) -- shared by the
    live model (data/elo_mlb_history.py) and backtest/bankroll_sim.py's training cache
    so there's one fetch implementation, not two that could silently drift apart.
    """
    sess = session or requests.Session()
    end = end_date or datetime.now(EASTERN).strftime("%Y-%m-%d")
    start_year, end_year = int(start_date[:4]), int(end[:4])
    games: list[dict] = []
    for year in range(start_year, end_year + 1):
        y_start = start_date if year == start_year else f"{year}-01-01"
        y_end = end if year == end_year else f"{year}-12-31"
        data = None
        try:
            r = sess.get(f"{BASE}/schedule", params={
                "sportId": 1, "startDate": y_start, "endDate": y_end, "hydrate": "team,linescore",
            }, timeout=60)
            if r.status_code == 200:
                data = r.json()
            else:
                log.warning("MLB historical fetch %s -> HTTP %s", year, r.status_code)
        except requests.RequestException as e:
            log.warning("MLB historical fetch failed for %s: %s", year, e)
        if not data:
            continue
        for de in data.get("dates", []) or []:
            for g in de.get("games", []) or []:
                if g.get("gameType") != "R":
                    continue
                if g["status"].get("abstractGameState") != "Final":
                    continue
                away, home = g["teams"]["away"], g["teams"]["home"]
                away_score, home_score = away.get("score"), home.get("score")
                date_str = g.get("officialDate")
                if date_str is None or away_score is None or home_score is None:
                    continue
                games.append({
                    "date_str": date_str,
                    "away_abbr": (away["team"].get("abbreviation") or "").upper(),
                    "home_abbr": (home["team"].get("abbreviation") or "").upper(),
                    "away_score": int(away_score), "home_score": int(home_score),
                })
    games.sort(key=lambda g: g["date_str"])
    return games


def _parse_iso(s: Optional[str]) -> Optional[datetime]:
    if not s:
        return None
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except (ValueError, AttributeError):
        return None


def _ip_to_float(ip_str) -> float:
    """MLB innings-pitched '123.1' means 123 + 1/3 innings, '.2' = 2/3."""
    try:
        whole, _, frac = str(ip_str).partition(".")
        return int(whole) + (int(frac) / 3.0 if frac else 0.0)
    except (ValueError, TypeError):
        return 0.0


def _rec_pct(league_record: Optional[dict]) -> Optional[float]:
    if not league_record:
        return None
    pct = league_record.get("pct")
    try:
        return float(pct) if pct is not None else None
    except (TypeError, ValueError):
        return None
