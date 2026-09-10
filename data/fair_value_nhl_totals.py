"""Fair value for Kalshi NHL total-goals (over/under) markets -- the
data/fair_value_nba_totals.py analog.

Each KXNHLTOTAL market is "Over X.5 goals scored" (YES = over, NO = under),
same contract shape as every other sport's totals market. Estimates expected
total goals from team scoring/allowing rates
(data/scoring_environment_nhl.py), models the total as a negative-binomial,
and reads off P(total > line). Pre-game only. The shootout-goal question this
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
from data.distributions import nb_survival
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

# Overdispersion: Var(total) ~= PHI * mean. Raw variance measured 2026-09-10
# against 3,936 real regular-season games (data/nhl_data.py, 4-season window)
# gave var/mean ~= 0.863 -- UNDER 1.0, i.e. actual goal totals are slightly
# LESS variable than a pure Poisson process (unlike MLB/NFL/NBA, which are
# all overdispersed, phi > 1). data/distributions.py::nb_survival() already
# treats any phi <= 1.0 as pure Poisson (its own documented degenerate case),
# so this ships as the true measured value rather than being clamped to 1.0 --
# functionally identical to Poisson either way. Plausible mechanism, not
# just noise: the shootout convention (data/nhl_data.py's module docstring)
# adds exactly +1 to the total in a capped, deterministic way rather than a
# genuine extra scoring event with its own variance, and shootouts
# specifically occur in already-close/low-scoring games -- structurally
# dampening the right tail a little versus a clean Poisson process.
NHL_TOTALS_PHI = 0.863


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
        prob_over = nb_survival(int(pt.line), lam, NHL_TOTALS_PHI)

        away_name, home_name = pt.away_abbr, pt.home_abbr
        matchup = matchup_from_rules(quote.rules_primary) if quote else None
        if matchup:
            away_name, home_name = matchup

        return FairValue(round(prob_over, 4), "pregame_nhl_totals", 0.40,
                         {"line": pt.line, "exp_total": round(lam, 2),
                          "matchup": f"{pt.away_abbr}@{pt.home_abbr}",
                          "away_name": away_name, "home_name": home_name},
                         game_state=game_state)
