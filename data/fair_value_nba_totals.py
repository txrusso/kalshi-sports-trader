"""Fair value for Kalshi NBA total-points (over/under) markets -- the
data/fair_value_nfl_totals.py analog.

Each KXNBATOTAL market is "Over X.5 points scored" (YES = over, NO = under),
same contract shape as MLB/NFL totals. Estimates expected total points from
team scoring/allowing rates (data/scoring_environment_nba.py), models the
total as a negative-binomial, and reads off P(total > line). Pre-game only.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

from config.settings import Settings, DEFAULTS
from data.distributions import nb_survival
from data.fair_value import FairValue
from data.fair_value_nba import matchup_from_rules
from data.nba_data import NbaDataClient, split_team_codes
from data.scoring_environment_nba import ScoringEnvironmentModel
from kalshi.normalize import MarketQuote

# KXNBATOTAL-<YY><MON><DD><TEAMS>-<K>   (line = K - 0.5, YES = over). No HHMM
# segment, same as KXNBAGAME (confirmed live via web search 2026-09-10, e.g.
# a real settled market "kxnbatotal-26jun10sasnyk" -- not yet directly
# confirmed against the live trade API since no KXNBATOTAL market is open
# this far from the season; worth a live re-check once one lists).
_TOTAL_RE = re.compile(r"KXNBATOTAL-(\d{2})([A-Z]{3})(\d{2})([A-Z]+)-(\d+)$")
_MONTHS = {m: i for i, m in enumerate(
    ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"], start=1)}

# Overdispersion: Var(total) ~= PHI * mean. Raw variance measured 2026-09-10
# against 6,140 real regular-season games (data/nba_data.py, 6-season window)
# gave var/mean ~= 1.79 -- seeded here as-is, exactly like NFL_TOTALS_PHI was
# at ship time (NOT yet through a walk-forward validate/tune pass -- that's
# backtest/nba_model_backtest.py --sweep's job once real Kalshi settlement
# data exists). Much lower than MLB's 2.2 or NFL's 4.22: an NBA final score is
# the sum of ~90-100 scoring plays per team (near-Poisson), vs NFL's ~6-9 or
# MLB's handful of runs -- the more scoring events summed, the closer to
# Poisson (var=mean, phi=1) raw scoring naturally gets, before team-strength
# heterogeneity pushes it back up a bit.
NBA_TOTALS_PHI = 1.79


@dataclass
class ParsedNbaTotal:
    event_ticker: str
    date_str: str
    away_abbr: str
    home_abbr: str
    line: float          # e.g. 224.5


def parse_total_ticker(ticker: str) -> Optional[ParsedNbaTotal]:
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
    away, home = split          # ticker blob order is AWAY-then-HOME (confirmed live)
    return ParsedNbaTotal(event_ticker=ticker.rsplit("-", 1)[0], date_str=date_str,
                          away_abbr=away, home_abbr=home, line=int(k) - 0.5)


class NbaTotalsFairValueModel:
    def __init__(self, nba_data: NbaDataClient, settings: Settings = DEFAULTS):
        self.nba_data = nba_data
        self.settings = settings
        self._env = ScoringEnvironmentModel(nba_data)

    def estimate(self, ticker: str, quote: Optional[MarketQuote] = None) -> FairValue:
        pt = parse_total_ticker(ticker)
        if not pt:
            return FairValue(None, "none", 0.0, {"reason": "unparseable total ticker"})

        # See data/fair_value_nba.py's identical check for why "no game found"
        # is treated as unsafe-to-estimate rather than an error.
        g = self.nba_data.find_game(pt.date_str, pt.away_abbr, pt.home_abbr)
        if g is None:
            return FairValue(None, "none", 0.0,
                             {"reason": "no game found (preseason, or season data not yet published)"})

        occurrence = quote.occurrence_datetime if quote else None
        if occurrence is None:
            return FairValue(None, "none", 0.0, {"reason": "no occurrence_datetime on quote"})

        started = datetime.now(timezone.utc) >= occurrence
        game_state = "Live" if started else "Preview"
        if self.settings.pregame_only and started:
            return FairValue(None, "skip_non_pregame", 0.0,
                             {"reason": "game has started"}, game_state=game_state)
        if started:
            return FairValue(None, "none", 0.0, {"reason": "no live NBA model yet"},
                             game_state=game_state)

        et = self._env.expected_points(pt.away_abbr, pt.home_abbr, pt.date_str)
        if et is None:
            return FairValue(None, "none", 0.0, {"reason": "missing team scoring rates"},
                             game_state=game_state)
        lam = et["lam"]
        prob_over = nb_survival(int(pt.line), lam, NBA_TOTALS_PHI)

        away_name, home_name = pt.away_abbr, pt.home_abbr
        matchup = matchup_from_rules(quote.rules_primary) if quote else None
        if matchup:
            away_name, home_name = matchup

        return FairValue(round(prob_over, 4), "pregame_nba_totals", 0.40,
                         {"line": pt.line, "exp_total": round(lam, 2),
                          "matchup": f"{pt.away_abbr}@{pt.home_abbr}",
                          "away_name": away_name, "home_name": home_name},
                         game_state=game_state)
