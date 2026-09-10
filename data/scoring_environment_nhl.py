"""Expected-goals model for NHL totals -- the data/scoring_environment_nba.py
analog. No roof/weather adjustment (NHL is played indoors) -- team
offense/defense scoring rates blend the current season-to-date rate with the
PRIOR season's full-season rate, weighted by games played so far this season
(regression by sample size), identical shape to NBA's and NFL's models.

No per-player input for v1 (a documented gap, same treatment as every other
sport's missing player-level signal) -- team-level scoring/allowing rates
only, and no goaltender-specific adjustment despite goaltending being an
unusually large swing factor in hockey (a real, known gap, worth revisiting).
SEASON_BLEND_GAMES is an initial estimate to be validated/tuned by
backtest/nhl_model_backtest.py, same as NBA's own constant was.
"""
from __future__ import annotations

from typing import Optional

from data.nhl_data import NhlDataClient, NhlGame, season_for_date

# Mirrors data/scoring_environment_nba.py's identical constant -- NHL's
# 82-game season is the same length as NBA's, so the same seed is used as a
# starting point pending its own sweep.
SEASON_BLEND_GAMES = 8.0

DEFAULT_LEAGUE_TOTAL = 6.0   # fallback if no team has any rate history yet (~3 goals/team)


def _rate(games: list[NhlGame], team: str) -> Optional[tuple[float, float]]:
    """(goals_scored_pg, goals_allowed_pg) from games this team played in."""
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


def team_scoring_rate(nhl_data: NhlDataClient, team: str,
                      as_of_date: str) -> Optional[tuple[float, float]]:
    """(goals_scored_pg, goals_allowed_pg) entering as_of_date, blending this
    season-to-date with last season's full rate by sample size. Mirrors
    data/scoring_environment_nba.py's identical function."""
    prior = nhl_data.team_games_before(team, as_of_date)
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
    def __init__(self, nhl_data: NhlDataClient):
        self.nhl_data = nhl_data
        self._league_avg_cache: dict[str, float] = {}

    def _league_avg_total(self, as_of_date: str) -> float:
        if as_of_date not in self._league_avg_cache:
            teams = {g.away_abbr for g in self.nhl_data.games()} | \
                    {g.home_abbr for g in self.nhl_data.games()}
            rates = [team_scoring_rate(self.nhl_data, t, as_of_date) for t in teams]
            rates = [r for r in rates if r]
            self._league_avg_cache[as_of_date] = (
                (sum(s for s, _ in rates) + sum(a for _, a in rates)) / len(rates)
            ) if rates else DEFAULT_LEAGUE_TOTAL
        return self._league_avg_cache[as_of_date]

    def expected_goals(self, away_abbr: str, home_abbr: str,
                       as_of_date: str) -> Optional[dict]:
        """Expected goals for both teams (offense x opponent's allowed rate,
        vs league average). Mirrors
        data/scoring_environment_nba.py::ScoringEnvironmentModel.expected_points()."""
        away_rate = team_scoring_rate(self.nhl_data, away_abbr, as_of_date)
        home_rate = team_scoring_rate(self.nhl_data, home_abbr, as_of_date)
        if not away_rate or not home_rate:
            return None
        lg_total = self._league_avg_total(as_of_date)
        lg_team = lg_total / 2.0
        e_away = away_rate[0] * (home_rate[1] / lg_team)
        e_home = home_rate[0] * (away_rate[1] / lg_team)
        lam = e_away + e_home
        return {"lam": lam, "e_away": e_away, "e_home": e_home}
