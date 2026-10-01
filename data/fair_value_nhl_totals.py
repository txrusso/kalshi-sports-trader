"""Fair value for Kalshi NHL total-goals (over/under) markets -- the
data/fair_value_nba_totals.py analog.

Each KXNHLTOTAL market is "Over X.5 goals scored" (YES = over, NO = under),
same contract shape as every other sport's totals market. Estimates expected
total goals from team scoring/allowing rates
(data/scoring_environment_nhl.py), models the final total with
data/distributions.py::overtime_total_survival -- which accounts for hockey's
tie-breaking overtime/shootout goal, the dominant shape effect in NHL totals
-- and reads off P(total > line). Pre-game only. The shootout-goal question this
needed answering first (does a shootout-decided game's official final score
include the deciding goal?) was resolved empirically against the real NHL API
before this was built -- see data/nhl_data.py's module docstring: yes, it
does, the same convention every broadcast/NHL.com uses, so no special-casing
is needed here.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

from config.settings import Settings, DEFAULTS
from data.distributions import overtime_total_survival
from data.fair_value import FairValue
from data.fair_value_nhl import matchup_from_rules
from data.nhl_data import NhlDataClient, split_team_codes
from data.scoring_environment_nhl import ScoringEnvironmentModel
from kalshi.normalize import MarketQuote

# KXNHLTOTAL-<YY><MON><DD><TEAMS>-<K>   (line = K - 0.5, YES = over). No HHMM
# segment, same as KXNHLGAME (confirmed live via web search 2026-09-10, e.g.
# a real settled market "kxnhltotal-25dec28nyicbj" -- not yet directly
# confirmed against the live trade API since no KXNHLTOTAL market is open
# this far from the season).
_TOTAL_RE = re.compile(r"KXNHLTOTAL-(\d{2})([A-Z]{3})(\d{2})([A-Z]+)-(\d+)$")
_MONTHS = {m: i for i, m in enumerate(
    ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"], start=1)}

# Shape of the goal-total distribution.
#
# SUPERSEDED 2026-09-30 (was NHL_TOTALS_PHI = 0.863, a negative binomial that
# degenerated to pure Poisson). That model was measurably the wrong SHAPE, not
# merely the wrong width: a regulation tie in hockey is broken by exactly one
# more goal, so 68.1% of real regular-season games end on an ODD total against
# Poisson's 50.0%, and a 0-0 final cannot happen at all (0 in 3,936 games, vs
# the 8 Poisson expects). At real Kalshi lines the old model was off by up to
# 4.5 points with ALTERNATING sign -- under-stating P(over) at the 4.5 and 6.5
# lines, over-stating it at 5.5 and 7.5 -- which is precisely the shape that
# manufactures edges out of nothing. data/distributions.py::
# overtime_total_survival models the tie-breaking goal explicitly instead;
# the measured under-dispersion it was built to express (var/mean 0.863) now
# falls out of that construction plus the contraction term below.
#
# NHL_TOTALS_DISPERSION contracts the distribution around its mean to match
# hockey's measured under-dispersion, which independent Poissons cannot
# produce on their own. Swept jointly with
# data/scoring_environment_nhl.py::TEAM_SIGNAL_SHRINK on a walk-forward split
# (train 2024-25, validate 2025-26, real lines 3.5-7.5): every parity-aware
# setting beat the old Poisson on BOTH splits. Validate is flat across
# 0.70-0.85 while train improves monotonically toward 0.95, so 0.85 is the
# both-splits plateau rather than validate's single-point peak -- and it is
# independently corroborated, being the value whose implied var/mean (0.846)
# and full pmf (L1 0.036 over 3,936 games) fit the measured distribution best.
# Held out on 2025-26 the result is within 2 standard errors at every real
# line, where the old model was outside it at four of five.
NHL_TOTALS_DISPERSION = 0.85


@dataclass
class ParsedNhlTotal:
    event_ticker: str
    date_str: str
    away_abbr: str
    home_abbr: str
    line: float          # e.g. 5.5


def parse_total_ticker(ticker: str) -> Optional[ParsedNhlTotal]:
    m = _TOTAL_RE.match(ticker)
    if not m:
        return None
    yy, mon, dd, teams, k = m.groups()
    month = _MONTHS.get(mon)
    if not month:
        return None
    try:
        date_str = datetime(2000 + int(yy), month, int(dd)).strftime("%Y-%m-%d")
    except ValueError:
        return None
    split = split_team_codes(teams)
    if not split:
        return None
    away, home = split          # ticker blob order is AWAY-then-HOME (assumed, see fair_value_nhl.py)
    return ParsedNhlTotal(event_ticker=ticker.rsplit("-", 1)[0], date_str=date_str,
                          away_abbr=away, home_abbr=home, line=int(k) - 0.5)


class NhlTotalsFairValueModel:
    def __init__(self, nhl_data: NhlDataClient, settings: Settings = DEFAULTS):
        self.nhl_data = nhl_data
        self.settings = settings
        self._env = ScoringEnvironmentModel(nhl_data)

    def estimate(self, ticker: str, quote: Optional[MarketQuote] = None) -> FairValue:
        pt = parse_total_ticker(ticker)
        if not pt:
            return FairValue(None, "none", 0.0, {"reason": "unparseable total ticker"})

        g = self.nhl_data.find_game(pt.date_str, pt.away_abbr, pt.home_abbr)
        if g is None:
            return FairValue(None, "none", 0.0,
                             {"reason": "no game found (likely preseason)"})

        occurrence = quote.occurrence_datetime if quote else None
        if occurrence is None:
            return FairValue(None, "none", 0.0, {"reason": "no occurrence_datetime on quote"})

        started = datetime.now(timezone.utc) >= occurrence
        game_state = "Live" if started else "Preview"
        if self.settings.pregame_only and started:
            return FairValue(None, "skip_non_pregame", 0.0,
                             {"reason": "game has started"}, game_state=game_state)
        if started:
            return FairValue(None, "none", 0.0, {"reason": "no live NHL model yet"},
                             game_state=game_state)

        et = self._env.expected_goals(pt.away_abbr, pt.home_abbr, pt.date_str)
        if et is None:
            return FairValue(None, "none", 0.0, {"reason": "missing team scoring rates"},
                             game_state=game_state)
        lam = et["lam"]
        prob_over = overtime_total_survival(int(pt.line), et["e_away"], et["e_home"],
                                            NHL_TOTALS_DISPERSION)

        away_name, home_name = pt.away_abbr, pt.home_abbr
        matchup = matchup_from_rules(quote.rules_primary) if quote else None
        if matchup:
            away_name, home_name = matchup

        return FairValue(round(prob_over, 4), "pregame_nhl_totals", 0.40,
                         {"line": pt.line, "exp_total": round(lam, 2),
                          "e_away": round(et["e_away"], 4), "e_home": round(et["e_home"], 4),
                          "raw_exp_total": round(et.get("raw_lam", lam), 2),
                          "matchup": f"{pt.away_abbr}@{pt.home_abbr}",
                          "away_name": away_name, "home_name": home_name},
                         game_state=game_state)
