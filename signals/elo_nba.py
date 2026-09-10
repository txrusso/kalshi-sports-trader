"""Walk-forward Elo ratings for NBA teams, built once from a rolling window of
data/nba_data.py's per-season game logs. This is the NBA winner model's core
signal -- the same role signals/elo_nfl.py plays for NFL (an 82-game season is
plenty for a log5-on-record signal the way MLB uses, in principle, but Elo was
picked instead so ratings carry over season-to-season the same way NFL's do,
rather than needing a from-scratch in-season record to build up each October).

Leak-safe by construction: rating_before() only ever uses games strictly
before the query date, mirrors elo_nfl.py's identical guarantee.

Standard 538-style NBA Elo (fivethirtyeight.com/methodology/how-our-nba-predictions-work):
  - initial rating 1500 (kept at NFL's convention rather than 538's own 1300 --
    only the RATING GAP matters for win_prob(), not the absolute scale, and a
    shared convention across sports is one less thing to remember)
  - home court folded into the pre-logistic Elo diff, same shape as NFL
  - margin-of-victory multiplier using 538's own published NBA form (NOT the
    same formula as NFL's -- NBA margins run much larger in raw points, so
    538 uses a different exponent/shape here, not just a rescaled log)
  - fractional regression toward 1500 at each season boundary

Constants (K_FACTOR, SEASON_REGRESSION) are seeded from 538's published
values, exactly like elo_nfl.py's were at ship time -- NOT yet validated
against this project's own held-out data. That validation is
backtest/nba_model_backtest.py's job (with a --sweep mode), the same empirical
tuning step NFL's constants went through in nfl_model_backtest.py.

HOME_FIELD_ELO is the one exception, already corrected 2026-09-10: 538's
published ~100-point value was tested against two independent real completed
seasons (2023-24 and 2024-25, backtest/nba_model_backtest.py --sweep
home_field_elo) and came back clearly WORSE than lower values on BOTH seasons
independently -- monotonically so from 100 up to 150 (Brier 0.218/0.217 ->
0.230/0.229), and a finer sweep (20-70) found a flat plateau at 30-40 beating
100 on both seasons too (2023-24: 0.2120 vs 0.2181; 2024-25: 0.2105 vs
0.2167). Shipped at 40, the plateau's upper-middle (not the single-point
2023-24 peak of 30, matching this project's standing discipline against
picking a sample-specific peak over a well-supported band -- see
config/settings.py's min_edge_cents history for the same pattern). Plausible
real-world reason, not just a data artifact: NBA home-court advantage has
famously shrunk league-wide in recent years (charter travel, rule changes),
so 538's older-era-calibrated ~100 may simply be stale for the current
league. NOT yet jointly re-swept against K_FACTOR (see MLB's elo_weight x
K_FACTOR joint-grid precedent in docs/research-log.md) -- a natural next step
once more seasons of data or real Kalshi settlement exist.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Optional

from data.nba_data import NbaDataClient, season_for_date

INITIAL_RATING = 1500.0
K_FACTOR = 20.0
HOME_FIELD_ELO = 40.0
SEASON_REGRESSION = 0.25     # fraction pulled back toward the mean each new season


def win_prob(rating_a: float, rating_b: float, a_is_home: bool) -> float:
    """P(A wins), folding home court into the pre-logistic Elo diff."""
    diff = rating_a - rating_b
    diff += HOME_FIELD_ELO if a_is_home else -HOME_FIELD_ELO
    return 1.0 / (1.0 + 10 ** (-diff / 400.0))


def _mov_multiplier(margin: int, elo_diff_winner: float) -> float:
    """538's NBA margin-of-victory multiplier (distinct from NFL's -- NBA
    margins run much larger in raw points, so 538 publishes a different
    shape here than the log-based NFL/NHL form): ((margin+3)^0.8) /
    (7.5 + 0.006 * elo_diff_winner), damped when the winner was already a
    big favorite by pre-game rating."""
    if margin == 0:
        return 0.0
    return ((abs(margin) + 3.0) ** 0.8) / (7.5 + 0.006 * elo_diff_winner)


class EloRatings:
    def __init__(self, nba_data: Optional[NbaDataClient] = None):
        self.nba_data = nba_data or NbaDataClient()
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

        for g in sorted(self.nba_data.games(), key=lambda g: (g.date_str, g.game_id)):
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

        self._history = dict(hist)
        self._built = True

    def rating_before(self, team: str, date_str: str) -> float:
        """Team's rating entering a game on `date_str` (YYYY-MM-DD) -- mirrors
        signals/elo_nfl.py's identical method. Falls back to INITIAL_RATING
        for a team with no history (e.g. outside the loaded season window)."""
        self._build()
        prior = [h for h in self._history.get(team, []) if h[0] < date_str]
        if not prior:
            return INITIAL_RATING
        _, last_season, _before, last_after = prior[-1]
        rating = last_after
        for _ in range(max(0, season_for_date(date_str) - last_season)):
            rating = INITIAL_RATING + (1 - SEASON_REGRESSION) * (rating - INITIAL_RATING)
        return rating
