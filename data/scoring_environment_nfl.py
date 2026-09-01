"""Expected-points model for NFL totals -- the data/run_environment.py analog.

Key NFL-specific difference from MLB: a 17-game season doesn't reach a stable
season-to-date rate for months, so team offense/defense scoring rates blend the
current season-to-date rate with the PRIOR season's full-season rate, weighted
by how many games of the new season have been played so far (regression by
sample size -- the same shape as MLB's starter-innings regression in
data/run_environment.py, just applied to a team-season instead of a pitcher).

No pitcher/QB-level per-player input for v1 (a documented gap, same treatment
as MLB's missing-weather gap) -- team-level scoring/allowing rates only.
SEASON_BLEND_GAMES and ROOF_FACTOR are initial estimates to be validated/tuned
by backtest/nfl_model_backtest.py, same as MLB's TOTALS_PHI/LEAGUE_SHRINK were.
"""
from __future__ import annotations

from typing import Optional

from data.nfl_data import NflDataClient, NflGame, season_for_date

# How many games of new-season data it takes to mostly trust the new season's
# own rate over last season's, e.g. at 4 games played this season, weight on
# this season's rate is 4/(4+SEASON_BLEND_GAMES).
SEASON_BLEND_GAMES = 6.0

# Roof-based scoring bump vs open-air stadiums (domes/retractable roofs score a
# bit higher on average: no wind, controlled climate). Coarser than MLB's
# per-park factors -- no weather data folded in yet (documented gap).
ROOF_FACTOR = {"dome": 1.03, "closed": 1.02, "outdoors": 1.00, "open": 1.00}

DEFAULT_LEAGUE_TOTAL = 44.0   # fallback if no team has any rate history yet


def roof_factor(roof: str) -> float:
    return ROOF_FACTOR.get((roof or "").lower(), 1.00)


def _rate(games: list[NflGame], team: str) -> Optional[tuple[float, float]]:
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


def team_scoring_rate(nfl_data: NflDataClient, team: str,
                      as_of_date: str) -> Optional[tuple[float, float]]:
    """(points_scored_pg, points_allowed_pg) entering as_of_date, blending this
    season-to-date with last season's full rate by sample size."""
    prior = nfl_data.team_games_before(team, as_of_date)
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
    def __init__(self, nfl_data: NflDataClient):
        self.nfl_data = nfl_data
        self._league_avg_cache: dict[str, float] = {}

    def _league_avg_total(self, as_of_date: str) -> float:
        if as_of_date not in self._league_avg_cache:
            teams = {g.away_abbr for g in self.nfl_data.games()} | \
                    {g.home_abbr for g in self.nfl_data.games()}
            rates = [team_scoring_rate(self.nfl_data, t, as_of_date) for t in teams]
            rates = [r for r in rates if r]
            self._league_avg_cache[as_of_date] = (
                (sum(s for s, _ in rates) + sum(a for _, a in rates)) / len(rates)
            ) if rates else DEFAULT_LEAGUE_TOTAL
        return self._league_avg_cache[as_of_date]

    def expected_points(self, away_abbr: str, home_abbr: str, as_of_date: str,
                        roof: str = "") -> Optional[dict]:
        """Expected points for both teams (offense x opponent's allowed rate,
        vs league average), roof-adjusted. Mirrors
        data/run_environment.py::RunEnvironmentModel.expected_runs()."""
        away_rate = team_scoring_rate(self.nfl_data, away_abbr, as_of_date)
        home_rate = team_scoring_rate(self.nfl_data, home_abbr, as_of_date)
        if not away_rate or not home_rate:
            return None
        lg_total = self._league_avg_total(as_of_date)
        lg_team = lg_total / 2.0
        e_away = away_rate[0] * (home_rate[1] / lg_team)
        e_home = home_rate[0] * (away_rate[1] / lg_team)
        rf = roof_factor(roof)
        lam = (e_away + e_home) * rf
        return {"lam": lam, "e_away": e_away, "e_home": e_home, "roof_factor": rf}
