"""Chronological MLB Elo ratings, mirroring signals/elo_nfl.py's structure but tuned
for baseball. Built 2026-09-01 to test a specific hypothesis: MLB's log5-on-season-
record already has 162 games/season to work with (vs NFL's 17), so Elo's expected edge
here is narrower than it was for NFL -- mainly smoothing EARLY-season noise (an 8-2
April record is taken at face value by log5 today, while Elo stays anchored near last
season's rating and updates gradually) rather than replacing a fundamentally noisy
in-season signal the way it did for NFL.

NOT wired into production fair value (data/fair_value.py) -- this is a standalone
signal for backtest/bankroll_sim.py's `elo_weight` sweep (blended against log5) to
determine whether it clears the project's usual both-splits walk-forward bar before
shipping. See that module for the sweep and data/fair_value.py's WINNER_REGRESS comment
for the standard this project holds every winner-model change to.

Home-field constant derived from the same 54% MLB home-win-rate assumption already
used by data/fair_value.py's log5 home-field step (HOME_FIELD_ODDS_MULT = 0.54/0.46),
so an Elo-vs-log5 comparison isn't confounded by two independently-fit home-field
numbers -- it isolates the win-probability *input* (record vs rating), not home field.

K_FACTOR is seeded low relative to NFL's 20: MLB teams play ~10x as many games per
season, so each individual game should move a team's rating much less. Not yet
validated against this project's own data (same caveat elo_nfl.py carries for its own
constants) -- bankroll_sim.py's --sweep can test it directly.
"""
from __future__ import annotations

import math
from collections import defaultdict

INITIAL_RATING = 1500.0
K_FACTOR = 6.0
HOME_FIELD_ELO = 400.0 * math.log10(0.54 / 0.46)   # ~28.0
SEASON_REGRESSION = 1.0 / 3.0

# The Athletics' MLB Stats API abbreviation changed from OAK to ATH between the 2023
# and 2026 seasons (confirmed live 2026-09-01: 2023-04-10 schedule returns "OAK",
# 2026-04-10 returns "ATH", both same franchise). Without normalizing, the franchise's
# built-up rating would silently strand under "OAK" and "ATH" would restart at 1500
# mid-history -- same class of bug as the NFL JAX/LA mapping in data/nfl_data.py.
_TEAM_ALIASES = {"OAK": "ATH"}


def _normalize(abbr: str) -> str:
    return _TEAM_ALIASES.get(abbr, abbr)


def win_prob(rating_a: float, rating_b: float, a_is_home: bool) -> float:
    """P(A wins), folding home field into the pre-logistic Elo diff."""
    diff = rating_a - rating_b
    diff += HOME_FIELD_ELO if a_is_home else -HOME_FIELD_ELO
    return 1.0 / (1.0 + 10 ** (-diff / 400.0))


def _mov_multiplier(margin: int, elo_diff_winner: float) -> float:
    """538-style margin-of-victory multiplier (run differential here, not points):
    blowouts move ratings more, damped when the winner was already a big favorite."""
    if margin == 0:
        return 0.0
    return math.log(abs(margin) + 1.0) * (2.2 / (0.001 * elo_diff_winner + 2.2))


class EloRatings:
    """Builds once from a chronological list of completed games:
    [{"date_str": "YYYY-MM-DD", "away_abbr", "home_abbr", "away_score", "home_score"}, ...]
    (order doesn't matter -- sorted internally). Leak-safe by construction:
    rating_before() only uses games strictly before the query date.
    """

    def __init__(self, games: list[dict]):
        self._games = sorted(games, key=lambda g: g["date_str"])
        self._history: dict[str, list[tuple]] = {}
        self._built = False

    def _build(self) -> None:
        if self._built:
            return
        ratings: dict[str, float] = {}
        season_of: dict[str, int] = {}
        hist: dict[str, list[tuple]] = defaultdict(list)

        def entering(team: str, season: int) -> float:
            if team not in ratings:
                ratings[team] = INITIAL_RATING
                season_of[team] = season
                return ratings[team]
            if season_of[team] != season:
                ratings[team] = INITIAL_RATING + (1 - SEASON_REGRESSION) * (ratings[team] - INITIAL_RATING)
                season_of[team] = season
            return ratings[team]

        for g in self._games:
            date_str = g["date_str"]
            season = int(date_str[:4])
            away, home = _normalize(g["away_abbr"]), _normalize(g["home_abbr"])
            r_away = entering(away, season)
            r_home = entering(home, season)
            margin = g["home_score"] - g["away_score"]
            p_home = win_prob(r_home, r_away, a_is_home=True)
            actual_home = 1.0 if margin > 0 else (0.0 if margin < 0 else 0.5)
            elo_diff_winner = ((r_home - r_away + HOME_FIELD_ELO) if margin >= 0
                              else (r_away - r_home - HOME_FIELD_ELO))
            delta = K_FACTOR * _mov_multiplier(margin, elo_diff_winner) * (actual_home - p_home)
            r_home_after, r_away_after = r_home + delta, r_away - delta
            hist[home].append((date_str, season, r_home, r_home_after))
            hist[away].append((date_str, season, r_away, r_away_after))
            ratings[home], ratings[away] = r_home_after, r_away_after

        self._history = dict(hist)
        self._built = True

    def rating_before(self, team: str, date_str: str) -> float:
        """Team's rating entering a game on `date_str` (YYYY-MM-DD): the rating after
        their most recent completed game strictly before this date, with season-
        boundary regression applied for each season crossed since. Falls back to
        INITIAL_RATING for a team with no history before this date."""
        self._build()
        team = _normalize(team)
        prior = [h for h in self._history.get(team, []) if h[0] < date_str]
        if not prior:
            return INITIAL_RATING
        _, last_season, _before, last_after = prior[-1]
        rating = last_after
        query_season = int(date_str[:4])
        for _ in range(max(0, query_season - last_season)):
            rating = INITIAL_RATING + (1 - SEASON_REGRESSION) * (rating - INITIAL_RATING)
        return rating
