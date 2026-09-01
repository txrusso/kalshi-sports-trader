"""Fair value for Kalshi NFL total-points (over/under) markets.

Each KXNFLTOTAL market is "Over X.5 points scored" (YES = over, NO = under) --
same contract shape as MLB's KXMLBTOTAL. Estimates expected total points from
team scoring/allowing rates (data/scoring_environment_nfl.py), models the total
as a negative-binomial, and reads off P(total > line). Pre-game only.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

from config.settings import Settings, DEFAULTS
from data.distributions import nb_survival
from data.fair_value import FairValue
from data.fair_value_nfl import matchup_from_rules
from data.nfl_data import NflDataClient, split_team_codes
from data.scoring_environment_nfl import ScoringEnvironmentModel
from kalshi.normalize import MarketQuote

# KXNFLTOTAL-<YY><MON><DD><TEAMS>-<K>   (line = K - 0.5, YES = over). No HHMM
# segment, same as KXNFLGAME.
_TOTAL_RE = re.compile(r"KXNFLTOTAL-(\d{2})([A-Z]{3})(\d{2})([A-Z]+)-(\d+)$")
_MONTHS = {m: i for i, m in enumerate(
    ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"], start=1)}

# Overdispersion: Var(total) ~= PHI * mean. Raw variance measured 2026-08-29
# against 3,028 settled games (2015-2025 seasons) gave var/mean ~= 4.22 --
# seeded here as-is (unlike MLB's TOTALS_PHI, this has NOT yet been through a
# walk-forward validate/tune pass -- backtest/nfl_model_backtest.py --sweep is
# where that happens, once real Kalshi settlement data exists to validate ROI
# against; MLB's own 2.37 raw measurement got adjusted to 2.2 by that process).
NFL_TOTALS_PHI = 4.22


@dataclass
class ParsedNflTotal:
    event_ticker: str
    date_str: str
    away_abbr: str
    home_abbr: str
    line: float          # e.g. 58.5


def parse_total_ticker(ticker: str) -> Optional[ParsedNflTotal]:
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
    return ParsedNflTotal(event_ticker=ticker.rsplit("-", 1)[0], date_str=date_str,
                          away_abbr=away, home_abbr=home, line=int(k) - 0.5)


class NflTotalsFairValueModel:
    def __init__(self, nfl_data: NflDataClient, settings: Settings = DEFAULTS):
        self.nfl_data = nfl_data
        self.settings = settings
        self._env = ScoringEnvironmentModel(nfl_data)

    def estimate(self, ticker: str, quote: Optional[MarketQuote] = None) -> FairValue:
        pt = parse_total_ticker(ticker)
        if not pt:
            return FairValue(None, "none", 0.0, {"reason": "unparseable total ticker"})

        # See data/fair_value_nfl.py's identical check for why: nflverse excludes
        # preseason entirely, so "no game found" reliably flags a preseason
        # market -- confirmed 2026-08-29 via backtest/nfl_model_backtest.py
        # --kalshi-settled that this model loses money on real preseason markets
        # (-38.7% ROI on real trade prices). Refuse rather than mis-price.
        g = self.nfl_data.find_game(pt.date_str, pt.away_abbr, pt.home_abbr)
        if g is None:
            return FairValue(None, "none", 0.0,
                             {"reason": "no regular-season game found (likely preseason)"})

        occurrence = quote.occurrence_datetime if quote else None
        if occurrence is None:
            return FairValue(None, "none", 0.0, {"reason": "no occurrence_datetime on quote"})

        started = datetime.now(timezone.utc) >= occurrence
        game_state = "Live" if started else "Preview"
        if self.settings.pregame_only and started:
            return FairValue(None, "skip_non_pregame", 0.0,
                             {"reason": "game has started"}, game_state=game_state)
        if started:
            return FairValue(None, "none", 0.0, {"reason": "no live NFL model yet"},
                             game_state=game_state)

        et = self._env.expected_points(pt.away_abbr, pt.home_abbr, pt.date_str, roof=g.roof)
        if et is None:
            return FairValue(None, "none", 0.0, {"reason": "missing team scoring rates"},
                             game_state=game_state)
        lam = et["lam"]
        prob_over = nb_survival(int(pt.line), lam, NFL_TOTALS_PHI)

        # Full team names for headline_for() (signals/recommendation.py), same
        # detail-dict keys MLB's totals model populates.
        away_name, home_name = pt.away_abbr, pt.home_abbr
        matchup = matchup_from_rules(quote.rules_primary) if quote else None
        if matchup:
            away_name, home_name = matchup

        return FairValue(round(prob_over, 4), "pregame_nfl_totals", 0.40,
                         {"line": pt.line, "exp_total": round(lam, 2),
                          "matchup": f"{pt.away_abbr}@{pt.home_abbr}",
                          "away_name": away_name, "home_name": home_name,
                          "roof": g.roof, "roof_factor": et["roof_factor"]},
                         game_state=game_state)
