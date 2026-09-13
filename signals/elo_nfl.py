"""Walk-forward Elo ratings for NFL teams, built once from nflverse's full game
history (data/nfl_data.py). This is the NFL winner model's core signal, standing
in for MLB's log5-on-season-record (17 games/season is too few for a stable
in-season win-pct signal -- see docs/research-log.md's rationale).

Leak-safe by construction: rating_before() only ever uses games strictly before
the query date, and each team's rating carries over season-to-season with
regression toward the mean -- "train on last year" falls out for free (a team's
final 2025 rating regresses into its 2026 Week 1 prior), no separate training
step, no "as of date" field from an external API to accidentally trust (the
leakage class MLB's season_backtest.py found and fixed for the MLB Stats API).

Standard 538-style NFL Elo (fivethirtyeight.com/methodology/how-our-nfl-predictions-work):
  - initial rating 1500
  - home field folded into the pre-logistic Elo diff (not a separate
    probability adjustment the way MLB's log5+home-field-odds step works)
  - margin-of-victory multiplier, damped by the pre-game rating gap so an
    already-expected blowout doesn't move ratings as much as a mild upset
  - 1/3 regression toward 1500 at each season boundary

Constants (K_FACTOR, HOME_FIELD_ELO, SEASON_REGRESSION) are seeded from 538's
published values, not yet validated against this project's own held-out data --
that validation is backtest/nfl_model_backtest.py's job (with a --sweep mode),
the same empirical-tuning step MLB's TOTALS_PHI/LEAGUE_SHRINK went through.
"""
from __future__ import annotations

import math
from collections import defaultdict
from typing import Optional

from data.nfl_data import NflDataClient, season_for_date

INITIAL_RATING = 1500.0
K_FACTOR = 20.0
HOME_FIELD_ELO = 48.0
SEASON_REGRESSION = 1.0 / 3.0     # fraction pulled back toward the mean each new season


def win_prob(rating_a: float, rating_b: float, a_is_home: bool) -> float:
    """P(A wins), folding home field into the pre-logistic Elo diff."""
    diff = rating_a - rating_b
    diff += HOME_FIELD_ELO if a_is_home else -HOME_FIELD_ELO
    return 1.0 / (1.0 + 10 ** (-diff / 400.0))


def _mov_multiplier(margin: int, elo_diff_winner: float) -> float:
    """538's margin-of-victory multiplier: blowouts move ratings more, damped
    when the winner was already a big favorite by pre-game rating."""
    if margin == 0:
        return 0.0
    return math.log(abs(margin) + 1.0) * (2.2 / (0.001 * elo_diff_winner + 2.2))


class EloRatings:
    def __init__(self, nfl_data: Optional[NflDataClient] = None):
        self.nfl_data = nfl_data or NflDataClient()
        # team -> [(date_str, season, rating_before_this_game, rating_after_this_game)]
        self._history: dict[str, list[tuple]] = {}
        self._built = False
        # (r_home_before, r_away_before, actual_margin) for every completed game,
        # collected during _build() -- lets the spread model (data/fair_value_nfl_spread.py)
        # regress margin-of-victory on pre-game Elo diff against real history without
        # re-deriving ratings itself. See margin_samples().
        self._margin_samples: list[tuple[float, float, int]] = []

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

        for g in self.nfl_data.games():
            if g.state != "Final":
                continue
            r_away = entering(g.away_abbr, g.season)
            r_home = entering(g.home_abbr, g.season)
            margin = g.home_score - g.away_score       # positive = home won
            p_home = win_prob(r_home, r_away, a_is_home=True)
            actual_home = 1.0 if margin > 0 else (0.0 if margin < 0 else 0.5)
            elo_diff_winner = ((r_home - r_away + HOME_FIELD_ELO) if margin >= 0
                              else (r_away - r_home - HOME_FIELD_ELO))
            delta = K_FACTOR * _mov_multiplier(margin, elo_diff_winner) * (actual_home - p_home)
            r_home_after, r_away_after = r_home + delta, r_away - delta
            hist[g.home_abbr].append((g.date_str, g.season, r_home, r_home_after))
            hist[g.away_abbr].append((g.date_str, g.season, r_away, r_away_after))
            ratings[g.home_abbr], ratings[g.away_abbr] = r_home_after, r_away_after
            self._margin_samples.append((r_home, r_away, margin))

        self._history = dict(hist)
        self._built = True

    def margin_samples(self) -> list[tuple[float, float, int]]:
        """(r_home_before, r_away_before, actual_margin=home_score-away_score) for
        every completed game -- the training set for the spread model's Elo-diff ->
        margin regression (data/fair_value_nfl_spread.py)."""
        self._build()
        return self._margin_samples

    def rating_before(self, team: str, date_str: str) -> float:
        """Team's rating entering a game on `date_str` (YYYY-MM-DD): the rating
        after their most recent completed game strictly before this date, with
        season-boundary regression applied for each season crossed since (the
        live production case -- 'now' is always after the last completed game
        in the data). Falls back to INITIAL_RATING for a team with no history."""
        self._build()
        prior = [h for h in self._history.get(team, []) if h[0] < date_str]
        if not prior:
            return INITIAL_RATING
        _, last_season, _before, last_after = prior[-1]
        rating = last_after
        for _ in range(max(0, season_for_date(date_str) - last_season)):
            rating = INITIAL_RATING + (1 - SEASON_REGRESSION) * (rating - INITIAL_RATING)
        return rating
