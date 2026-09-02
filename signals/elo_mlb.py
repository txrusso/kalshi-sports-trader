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

K_FACTOR=3.5, walk-forward validated 2026-09-02 (`backtest/bankroll_sim.py --sweep
elo_k_factor ...`, train <=2026-07-15 / validate after; EloRatings rebuilt per swept
value since K_FACTOR shapes the whole chronological rating build, not a per-row
parameter). Originally seeded at 6.0 by analogy to elo_nfl.py's 20 (MLB teams play
~10x as many games/season, so each individual game should move a rating less) but
never actually tested against this project's data until this sweep. The sweep (2.0
to 20.0, then fine-swept 3.0-5.5) found a flat plateau at 3.2-3.8 that clearly beats
6.0 on BOTH splits and every metric: TRAIN winner ROI -0.154->-0.099, TRAIN winner
win_rate 0.470->0.494, TRAIN max drawdown 42.2%->32.6%; VALID winner ROI 0.0995->
0.127 (now essentially matching totals' VALID ROI of ~0.119), VALID winner win_rate
0.589->0.597, VALID max drawdown roughly flat (23.8%->23.9%). Landed at 3.5 (plateau
midpoint, not the single-point peak of 3.4) per this project's standing discipline
against chasing knife-edge sweep peaks. Same one-partial-season caveat as every other
constant here -- re-sweep as more settled markets accumulate.
"""
from __future__ import annotations

import math
from collections import defaultdict

INITIAL_RATING = 1500.0
K_FACTOR = 3.5
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


def win_prob(rating_a: float, rating_b: float, a_is_home: bool,
             home_field_elo: float = None) -> float:
    """P(A wins), folding home field into the pre-logistic Elo diff."""
    hf = HOME_FIELD_ELO if home_field_elo is None else home_field_elo
    diff = rating_a - rating_b
    diff += hf if a_is_home else -hf
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

    def __init__(self, games: list[dict], k_factor: float = None, home_field_elo: float = None):
        self._games = sorted(games, key=lambda g: g["date_str"])
        self._history: dict[str, list[tuple]] = {}
        self._built = False
        # Overridable so backtest/bankroll_sim.py can sweep K_FACTOR (rebuilding ratings
        # per value) -- this is how the module constant above was walk-forward validated
        # to 3.5; see this module's docstring for the sweep.
        self._k_factor = K_FACTOR if k_factor is None else k_factor
        # Same idea for the home-field boost: HOME_FIELD_ELO was originally DERIVED to
        # match log5's fixed 0.54/0.46 home-win assumption purely so an early Elo-vs-log5
        # comparison wasn't confounded by two independently-fit home-field numbers (see
        # module docstring) -- back when Elo wasn't wired into production at all. Now that
        # it's blended into fair value at real weight, that constraint no longer has to
        # hold; exposed here so it can be swept on its own, same pattern as k_factor.
        self.home_field_elo = HOME_FIELD_ELO if home_field_elo is None else home_field_elo

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
            p_home = win_prob(r_home, r_away, a_is_home=True, home_field_elo=self.home_field_elo)
            actual_home = 1.0 if margin > 0 else (0.0 if margin < 0 else 0.5)
            elo_diff_winner = ((r_home - r_away + self.home_field_elo) if margin >= 0
                              else (r_away - r_home - self.home_field_elo))
            delta = self._k_factor * _mov_multiplier(margin, elo_diff_winner) * (actual_home - p_home)
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
