"""Expected-points model for NBA totals -- the data/scoring_environment_nfl.py
analog. No roof/weather adjustment needed (NBA is played indoors year-round,
unlike NFL) -- otherwise the same shape: team offense/defense scoring rates
blend the current season-to-date rate with the PRIOR season's full-season
rate, weighted by games played so far this season (regression by sample size).

No per-player input for v1 (a documented gap, same treatment as MLB's missing
weather / NFL's missing QB-injury signal) -- team-level scoring/allowing rates
only. SEASON_BLEND_GAMES is an initial estimate to be validated/tuned by
backtest/nba_model_backtest.py, same as NFL's own constant was.
"""
from __future__ import annotations

from typing import Optional

from data.nba_data import NbaDataClient, NbaGame, season_for_date

# How many games of new-season data it takes to mostly trust the new season's
# own rate over last season's -- mirrors data/scoring_environment_nfl.py's
# identical constant. NBA's 82-game season reaches stability faster than NFL's
# 17, but early-October games still need a prior-season anchor.
SEASON_BLEND_GAMES = 8.0

DEFAULT_LEAGUE_TOTAL = 225.0   # fallback if no team has any rate history yet


def _rate(games: list[NbaGame], team: str) -> Optional[tuple[float, float]]:
    """(points_scored_pg, points_allowed_pg) from games this team played in."""
    if not games:
        return None
    scored = allowed = 0
    for g in games:
        if g.away_abbr == team:
            scored += g.away_score
            allowed += g.home_score
        else:
            scored += g.home_score
            allowed += g.away_score
    n = len(games)
    return scored / n, allowed / n


def team_scoring_rate(nba_data: NbaDataClient, team: str,
                      as_of_date: str) -> Optional[tuple[float, float]]:
    """(points_scored_pg, points_allowed_pg) entering as_of_date, blending this
    season-to-date with last season's full rate by sample size. Mirrors
    data/scoring_environment_nfl.py's identical function."""
    prior = nba_data.team_games_before(team, as_of_date)
    if not prior:
        return None
    season = season_for_date(as_of_date)
    this_season = [g for g in prior if g.season == season]
    last_season = [g for g in prior if g.season == season - 1]

    this_rate = _rate(this_season, team)
    last_rate = _rate(last_season, team)

    if this_rate is None:
        return last_rate                     # no games yet this season -- pure last-season prior
    if last_rate is None:
        return this_rate                     # no last-season history -- pure this-season

    n = len(this_season)
    w = n / (n + SEASON_BLEND_GAMES)
    return (w * this_rate[0] + (1 - w) * last_rate[0],
            w * this_rate[1] + (1 - w) * last_rate[1])


class ScoringEnvironmentModel:
    def __init__(self, nba_data: NbaDataClient):
        self.nba_data = nba_data
        self._league_avg_cache: dict[str, float] = {}

    def _league_avg_total(self, as_of_date: str) -> float:
        if as_of_date not in self._league_avg_cache:
            teams = {g.away_abbr for g in self.nba_data.games()} | \
                    {g.home_abbr for g in self.nba_data.games()}
            rates = [team_scoring_rate(self.nba_data, t, as_of_date) for t in teams]
            rates = [r for r in rates if r]
            self._league_avg_cache[as_of_date] = (
                (sum(s for s, _ in rates) + sum(a for _, a in rates)) / len(rates)
            ) if rates else DEFAULT_LEAGUE_TOTAL
        return self._league_avg_cache[as_of_date]

    def expected_points(self, away_abbr: str, home_abbr: str,
                        as_of_date: str) -> Optional[dict]:
        """Expected points for both teams (offense x opponent's allowed rate,
        vs league average). Mirrors
        data/scoring_environment_nfl.py::ScoringEnvironmentModel.expected_points()
        minus the roof factor (no indoor/outdoor distinction in the NBA)."""
        away_rate = team_scoring_rate(self.nba_data, away_abbr, as_of_date)
        home_rate = team_scoring_rate(self.nba_data, home_abbr, as_of_date)
        if not away_rate or not home_rate:
            return None
        lg_total = self._league_avg_total(as_of_date)
        lg_team = lg_total / 2.0
        e_away = away_rate[0] * (home_rate[1] / lg_team)
        e_home = home_rate[0] * (away_rate[1] / lg_team)
        lam = e_away + e_home
        return {"lam": lam, "e_away": e_away, "e_home": e_home}
