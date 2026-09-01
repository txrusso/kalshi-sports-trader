"""Shared pregame run-environment estimate: expected runs for both teams in a
matchup, folding in team offense/defense form and the starting pitchers.

Extracted from data/fair_value_totals.py so data/fair_value.py (the winner
model) can use the same run-environment picture as a pitching-matchup signal
via a Pythagorean nudge -- previously the winner model ignored who was
pitching entirely, while the totals model already leaned on it.
"""
from __future__ import annotations

from typing import Optional

from config.settings import Settings, DEFAULTS
from data.mlb_stats import MlbGame, MlbStatsClient

SP_WEIGHT = 0.62       # share of the game the starting pitcher is responsible for (~5.6 IP)
PITCHER_REG_IP = 40.0  # regress a starter's RA9 toward league avg by sample size (innings)
# Was 0.15 (blend team estimate toward league average). Set to 0 (no shrink)
# 2026-08-13 after backtest/bankroll_sim.py's walk-forward sweep (train Jun 7-Jul 15,
# validate Jul 16-Aug 13) showed 0.0 beat 0.15 on both splits -- consistent with the
# earlier finding that shrinkage was pulling estimates toward the league mean for
# exactly the high-scoring games where the raw team-rate signal needed more room, not
# less (see data/fair_value_totals.py's TOTALS_PHI comment for the related diagnosis).
LEAGUE_SHRINK = 0.0

# Run park factors (multiplier vs a neutral park, ~1.0), keyed by HOME team abbrev.
# Approximate, stable values -- the extremes (COL high; SF/SD/SEA low) matter most.
PARK_FACTORS = {
    "COL": 1.15, "CIN": 1.06, "BOS": 1.05, "KC": 1.04, "AZ": 1.03, "ARI": 1.03,
    "PHI": 1.02, "BAL": 1.02, "TEX": 1.02, "NYY": 1.02, "TOR": 1.01, "CHC": 1.01,
    "MIN": 1.01, "ATH": 1.02, "WSH": 1.00, "LAA": 1.00, "ATL": 1.00, "HOU": 1.00,
    "CWS": 1.00, "STL": 0.99, "MIL": 0.99, "PIT": 0.98, "CLE": 0.98, "MIA": 0.97,
    "DET": 0.97, "TB": 0.97, "LAD": 0.97, "NYM": 0.96, "SD": 0.94, "SF": 0.93, "SEA": 0.93,
}


def park_factor(home_abbr: str) -> float:
    return PARK_FACTORS.get(home_abbr.upper(), 1.0)

# NOTE (2026-08-31): a game-time WEATHER factor (wind out/in x mph, temperature) was
# built and walk-forward tested here to attack the totals model's documented low-
# probability miscalibration (season_backtest 0-10% bucket: predicts ~5.6% over, hits
# ~36%, n=630). It was REVERTED, same as the Pythagorean/recency overfits, because the
# data disproved the hypothesis: a bounded wind factor moves lam only ~9-12% for windy
# games, nowhere near enough to close a 5.6%->36% far-tail gap (the 0-10% bucket barely
# moved: pred 0.056 -> 0.055). On the bankroll-sim money guardrail it made things WORSE
# out-of-sample -- validate totals Brier rose monotonically with the wind coefficient and
# validate ROI/drawdown degraded with temperature. Conclusion: the low-bucket defect is a
# right-TAIL-SHAPE problem (the NB right tail is too thin at high lines), not a weather
# problem -- see backtest/baseline_20260831.txt for the full sweep. The MLB Stats API
# `weather` hydrate is available cheaply (0 extra calls) if a future asymmetric-tail model
# wants it.


def team_offense_defense(mlb: MlbStatsClient, team_abbr: str,
                          as_of=None) -> Optional[tuple[float, float]]:
    """(runs_scored_pg, runs_allowed_pg), season-to-date.

    A last-30-games recency version was tried 2026-08-13 and reverted the same day:
    backtest/season_backtest.py's ablation showed it made the WINNER model's win_pct
    input actively worse (too noisy at n=30 for a binary win/loss sample) and had no
    measurable effect either way on totals' run rates -- not worth the added API load
    and complexity for zero validated benefit. MlbStatsClient.team_recent_stats()
    still exists if a smarter blend (recent nudge on top of season, not a hard
    replacement) is worth trying later.
    """
    return mlb.team_run_rates().get(team_abbr)


class RunEnvironmentModel:
    def __init__(self, mlb: MlbStatsClient, settings: Settings = DEFAULTS):
        self.mlb = mlb
        self.settings = settings
        self._league_total: Optional[float] = None

    def _league_avg_total(self, as_of=None) -> float:
        if self._league_total is None:
            rates = list(self.mlb.team_run_rates().values())
            self._league_total = (
                (sum(rs for rs, _ in rates) + sum(ra for _, ra in rates)) / len(rates)) if rates else 9.0
        return self._league_total

    def _effective_ra9(self, team_ra_pg: float, pitcher_id, lg_team: float) -> tuple[float, bool]:
        """A team's run-prevention rate (per 9) for tonight, folding in the starter.

        = SP_WEIGHT * (starter RA9, regressed toward league) + rest * team rate.
        Falls back to the team rate when no starter is announced.
        """
        stat = self.mlb.pitcher_ra9(pitcher_id) if pitcher_id else None
        if not stat:
            return team_ra_pg, False
        sp_ra9, ip, _gs = stat
        reliab = ip / (ip + PITCHER_REG_IP)               # small samples -> toward league
        sp_reg = reliab * sp_ra9 + (1 - reliab) * lg_team
        eff = SP_WEIGHT * sp_reg + (1 - SP_WEIGHT) * team_ra_pg
        return eff, True

    def expected_runs(self, game: MlbGame) -> Optional[dict]:
        """Expected runs for both teams tonight (offense x opponent's effective
        run-prevention, park-adjusted). Used for totals lines directly, and as a
        pitching-matchup signal (via Pythagorean win prob) by the winner model."""
        as_of = game.game_datetime
        a = team_offense_defense(self.mlb, game.away_abbr, as_of)
        h = team_offense_defense(self.mlb, game.home_abbr, as_of)
        if not a or not h:
            return None
        lg_total = self._league_avg_total(as_of)          # ~9 runs/game
        lg_team = lg_total / 2.0                           # ~4.5 runs/team/game

        home_eff, hk = self._effective_ra9(h[1], game.home_pitcher_id, lg_team)
        away_eff, ak = self._effective_ra9(a[1], game.away_pitcher_id, lg_team)
        # Multiplicative runs model: offense scaled by opponent's pitching vs league.
        e_away = a[0] * (home_eff / lg_team)
        e_home = h[0] * (away_eff / lg_team)
        base = (1 - LEAGUE_SHRINK) * (e_away + e_home) + LEAGUE_SHRINK * lg_total
        pf = park_factor(game.home_abbr)                  # ballpark run environment
        lam = base * pf
        return {"lam": lam, "e_away": e_away, "e_home": e_home, "n_sp": int(hk) + int(ak),
                "park_factor": pf, "home_eff_ra9": round(home_eff, 2), "away_eff_ra9": round(away_eff, 2)}
