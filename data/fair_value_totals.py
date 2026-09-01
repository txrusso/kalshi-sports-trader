"""Fair value for Kalshi MLB total-runs (over/under) markets.

Each KXMLBTOTAL market is "Over X.5 runs scored" (YES = over, NO = under). We
estimate the game's expected total runs from team scoring/allowing rates, model
the total as a negative-binomial (baseball run totals are overdispersed vs
Poisson), and read off P(total > line).

Pre-game only (needs season rates + a scheduled game).
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

from data.distributions import nb_survival
from data.fair_value import FairValue, _MONTHS
from data.mlb_stats import MlbGame, MlbStatsClient
from data.run_environment import (
    RunEnvironmentModel, PARK_FACTORS, SP_WEIGHT, PITCHER_REG_IP, LEAGUE_SHRINK, park_factor,
)

# KXMLBTOTAL-<YY><MON><DD><HHMM><TEAMS>-<K>   (line = K - 0.5, YES = over)
_TOTAL_RE = re.compile(r"KXMLBTOTAL-(\d{2})([A-Z]{3})(\d{2})(\d{2})(\d{2})([A-Z]+)-(\d+)$")

# Overdispersion: Var(total) ≈ PHI * mean. Raw variance measured 2026-08-13 against
# 837 settled games gave var/mean ~= 2.37 (vs the original assumed 2.0), but a
# walk-forward ROI sweep (train Jun 7-Jul 15, validate Jul 16-Aug 13; see
# backtest/bankroll_sim.py) showed 2.2 generalizing better than either 2.0 or the
# raw-variance-implied 2.4 -- train and validate pulled in opposite directions across
# the range, so 2.2 is a deliberately moderate pick, not the single best grid point on
# either split alone. This does NOT fix the separate low-end mean-bias issue found in
# the same investigation -- lam is biased low for a subset of shootout games (several
# tied to hitter-friendly/wind-affected parks like Wrigley), which really wants a
# weather signal the model doesn't have (a known gap -- see CLAUDE.md's totals gaps).
TOTALS_PHI = 2.2
# The low-P(over)/high-line miscalibration (0-10% bucket predicts ~5.6%, hits ~36%) is a
# too-thin NB right tail, BUT a heavier-right-tail blowout mixture was tested and reverted
# 2026-09-01: it improved held-out calibration yet LOST money on the deep-tail lines it
# targets (the market prices those high-scoring games correctly). See the NOTE in
# data/distributions.py. The high-line miscalibration is real but not exploitable.


@dataclass
class ParsedTotal:
    event_ticker: str
    date: datetime
    teams: str          # concatenated abbrevs, e.g. "WSHPHI"
    line: float         # e.g. 8.5


def parse_total_ticker(ticker: str) -> Optional[ParsedTotal]:
    m = _TOTAL_RE.match(ticker)
    if not m:
        return None
    yy, mon, dd, hh, mm, teams, k = m.groups()
    month = _MONTHS.get(mon)
    if not month:
        return None
    try:
        date = datetime(2000 + int(yy), month, int(dd), int(hh), int(mm), tzinfo=timezone.utc)
    except ValueError:
        return None
    return ParsedTotal(event_ticker=ticker.rsplit("-", 1)[0], date=date,
                       teams=teams, line=int(k) - 0.5)


def _nb_sf(k: int, mean: float, phi: float = TOTALS_PHI) -> float:
    """P(N > k) for a negative-binomial with given mean and variance = phi*mean.

    Thin wrapper over data/distributions.py's shared implementation (also used
    by data/fair_value_nfl_totals.py) -- kept as a private name here since
    backtest/season_backtest.py already imports `_nb_sf` from this module.
    """
    return nb_survival(k, mean, phi)


class TotalsFairValueModel:
    def __init__(self, mlb: MlbStatsClient, settings=None):
        from config.settings import DEFAULTS
        self.mlb = mlb
        self.settings = settings or DEFAULTS
        self._sched_cache: dict[str, list[MlbGame]] = {}
        self._env = RunEnvironmentModel(mlb, self.settings)

    def _schedule(self, date: datetime) -> list[MlbGame]:
        key = date.strftime("%Y-%m-%d")
        if key not in self._sched_cache:
            self._sched_cache[key] = self.mlb.schedule(date)
        return self._sched_cache[key]

    def _match_game(self, pt: ParsedTotal) -> Optional[MlbGame]:
        for g in self._schedule(pt.date):
            if pt.teams in (g.away_abbr + g.home_abbr, g.home_abbr + g.away_abbr):
                return g
        return None

    def expected_total(self, game: MlbGame) -> Optional[dict]:
        """Thin wrapper kept for backward compatibility; the actual computation
        lives in data/run_environment.py so the winner model can share it."""
        return self._env.expected_runs(game)

    def estimate(self, ticker: str, quote=None) -> FairValue:
        # `quote` accepted-but-unused -- see FairValueModel.estimate's identical note.
        pt = parse_total_ticker(ticker)
        if not pt:
            return FairValue(None, "none", 0.0, {"reason": "unparseable total ticker"})
        game = self._match_game(pt)
        if not game:
            return FairValue(None, "none", 0.0, {"reason": "no MLB game match"})
        # See data/fair_value.py's identical check for why: MLB's status can flip
        # away from "Preview" a few minutes before the actual scheduled first
        # pitch, which silently killed real recommendations on 2026-08-10/11.
        truly_started = game.game_datetime is not None and datetime.now(timezone.utc) >= game.game_datetime
        effective_pregame = game.state == "Preview" or not truly_started
        if self.settings.pregame_only and not effective_pregame:
            return FairValue(None, "skip_non_pregame", 0.0,
                             {"reason": f"state={game.state}"}, game_state=game.state)

        et = self._env.expected_runs(game)
        if et is None:
            return FairValue(None, "none", 0.0, {"reason": "missing team run rates"},
                             game_state=game.state)
        lam = et["lam"]
        # P(total > line);  line = X.5  ->  P(N >= X+1) = P(N > X)
        prob_over = _nb_sf(int(pt.line), lam)
        # More confidence when the starters are known (pitching drives totals).
        conf = 0.40 + 0.05 * et["n_sp"]        # 0.40 / 0.45 / 0.50
        return FairValue(round(prob_over, 4), "pregame_totals", round(conf, 3),
                         {"line": pt.line, "exp_total": round(lam, 2),
                          "matchup": f"{game.away_abbr}@{game.home_abbr}",
                          "away_name": game.away_name, "home_name": game.home_name,
                          "starters_known": et["n_sp"], "park_factor": et["park_factor"],
                          "home_eff_ra9": et["home_eff_ra9"], "away_eff_ra9": et["away_eff_ra9"]},
                         game_state=game.state)
