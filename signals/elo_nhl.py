"""Walk-forward Elo ratings for NHL teams, built once from a rolling window of
data/nhl_data.py's per-season game logs. Plays the same role
signals/elo_nba.py and signals/elo_nfl.py play for their sports.

Leak-safe by construction: rating_before() only ever uses games strictly
before the query date, mirrors elo_nba.py's identical guarantee.

UNLIKE the NFL/NBA models, this is NOT seeded from a specific published
methodology -- 538 never publicly shipped an NHL Elo model the way it did for
NFL/NBA/MLB, so there's no external "standard value" to start from. Built as
a standard Elo instead:
  - initial rating 1500 (shared convention with every other sport here)
  - home ice folded into the pre-logistic Elo diff, same shape as every other
    sport
  - margin-of-victory multiplier uses NFL's log-based form
    (log(|margin|+1), damped by the pre-game rating gap), not NBA's
    power-law form -- hockey's real goal margins are small integers (usually
    1-3), much closer in scale to NFL's than to NBA's 538-published formula,
    which was deliberately shaped for NBA's much larger point margins. A
    shootout-decided game is definitionally a 1-goal final margin, so it
    naturally gets treated as a low-information result without any special
    casing -- the correct treatment for what is effectively a coin flip.
  - HOME_ICE_ELO seeded at 30 -- NHL's real home-ice win rate historically
    runs ~54-55% (well below NBA's ~60% or even NFL's ~57-58%), and a Elo gap
    of ~28 points alone reproduces a 54% win probability, which happens to
    land close to MLB's own already-validated HOME_FIELD_ELO=27.85.
  - K_FACTOR seeded at 20, matching NBA's initial (pre-correction) seed,
    since both sports play 82-game seasons.

K_FACTOR was swept immediately after this first build (2026-09-10,
backtest/nhl_model_backtest.py --sweep k_factor) against two independent real
completed seasons (2023-24 and 2024-25) and corrected: the initial seed of 20
(borrowed from NBA, both being 82-game seasons) was too high. Brier improved
monotonically as K dropped from 20 down toward single digits on BOTH seasons,
but the two seasons' individual optima didn't quite line up (2023-24 bottomed
near K=12, 2024-25 near K=6) -- so, per this project's standing discipline
against picking either season's individual peak, shipped at **K=8**, the
point where both seasons sit on their own near-flat plateau (2023-24: 0.2393
vs its own best of 0.2389; 2024-25: 0.2378 vs its own best of 0.2377) rather
than at either one's single-point maximum. HOME_ICE_ELO was then re-checked
at this corrected K and held up as originally seeded -- 30 was within a
point or two of optimal on both seasons jointly (2023-24 best at 30 exactly;
2024-25 best at 45, but 30 nearly tied at 0.2378 vs 0.2373) -- so it shipped
unchanged. SEASON_REGRESSION has NOT yet been swept.

Even at these corrected constants, NHL's real Brier (~0.237-0.239) sits much
closer to a coin flip than NBA's (~0.21) or NFL's -- a real, expected
property of the sport (single-game hockey outcomes are famously high-variance
even between mismatched teams: low scoring means small sample sizes of
"scoring events" per game, and empty-net/shootout dynamics add noise on top),
not a modeling defect to chase further without more evidence.
"""
from __future__ import annotations

import math
from collections import defaultdict
from typing import Optional

from data.nhl_data import NhlDataClient, season_for_date

INITIAL_RATING = 1500.0
K_FACTOR = 8.0
HOME_ICE_ELO = 30.0
SEASON_REGRESSION = 0.25     # fraction pulled back toward the mean each new season


def win_prob(rating_a: float, rating_b: float, a_is_home: bool) -> float:
    """P(A wins), folding home ice into the pre-logistic Elo diff."""
    diff = rating_a - rating_b
    diff += HOME_ICE_ELO if a_is_home else -HOME_ICE_ELO
    return 1.0 / (1.0 + 10 ** (-diff / 400.0))


def _mov_multiplier(margin: int, elo_diff_winner: float) -> float:
    """Log-based margin-of-victory multiplier (NFL's form, not NBA's -- see
    module docstring for why): blowouts move ratings a bit more, damped when
    the winner was already a big favorite by pre-game rating. A 1-goal
    (including shootout) margin gets the smallest possible bump."""
    if margin == 0:
        return 0.0
    return math.log(abs(margin) + 1.0) * (2.2 / (0.001 * elo_diff_winner + 2.2))


class EloRatings:
    def __init__(self, nhl_data: Optional[NhlDataClient] = None):
        self.nhl_data = nhl_data or NhlDataClient()
        # team -> [(date_str, season, rating_before_this_game, rating_after_this_game)]
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

        for g in sorted(self.nhl_data.games(), key=lambda g: (g.date_str, g.game_id)):
            if g.state != "Final":
                continue
            r_away = entering(g.away_abbr, g.season)
            r_home = entering(g.home_abbr, g.season)
            margin = g.home_score - g.away_score       # positive = home won
            p_home = win_prob(r_home, r_away, a_is_home=True)
            actual_home = 1.0 if margin > 0 else (0.0 if margin < 0 else 0.5)
            elo_diff_winner = ((r_home - r_away + HOME_ICE_ELO) if margin >= 0
                              else (r_away - r_home - HOME_ICE_ELO))
            delta = K_FACTOR * _mov_multiplier(margin, elo_diff_winner) * (actual_home - p_home)
            r_home_after, r_away_after = r_home + delta, r_away - delta
            hist[g.home_abbr].append((g.date_str, g.season, r_home, r_home_after))
            hist[g.away_abbr].append((g.date_str, g.season, r_away, r_away_after))
            ratings[g.home_abbr], ratings[g.away_abbr] = r_home_after, r_away_after

        self._history = dict(hist)
        self._built = True

    def rating_before(self, team: str, date_str: str) -> float:
        """Team's rating entering a game on `date_str` (YYYY-MM-DD) -- mirrors
        signals/elo_nba.py's identical method. Falls back to INITIAL_RATING
        for a team with no history."""
        self._build()
        prior = [h for h in self._history.get(team, []) if h[0] < date_str]
        if not prior:
            return INITIAL_RATING
        _, last_season, _before, last_after = prior[-1]
        rating = last_after
        for _ in range(max(0, season_for_date(date_str) - last_season)):
            rating = INITIAL_RATING + (1 - SEASON_REGRESSION) * (rating - INITIAL_RATING)
        return rating
