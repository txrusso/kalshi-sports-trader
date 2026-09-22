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

# Key-number bumps for totals (added 2026-09-21, mirroring the spread model's
# KEY_NUMBER_BUMP in data/fair_value_nfl_spread.py). NFL combined scores cluster
# at certain sums for the same reason spreads cluster at 3/7: both teams' scores
# are built from TD+XP=7 and FG=3 (plus occasional 2s/6s), so certain totals are
# reachable by far more (team_a, team_b) scoreline combinations than others.
# Online research (Action Network's 2015-2019 frequency table, covers.com's
# "key ranges" writeup) flagged 37, 40/41, 43-44, 47, and 51 as the recurring
# hot spots -- see:
#   https://www.actionnetwork.com/nfl/nfl-key-betting-numbers-over-unders-totals-line-value
#   https://www.covers.com/nfl/key-numbers
# Rather than trust those blog numbers directly, re-measured the same way the
# spread bump was: for every one of 7,287 completed nflverse games with a
# scoreable game environment, compared the ACTUAL rate of total==k to what the
# shipped NB(lam, NFL_TOTALS_PHI) predicts for that exact integer (pmf, not a
# continuity-corrected bin -- totals are already integer-valued, no correction
# needed). The two independent numbers-sources agree: every candidate the blogs
# named came back with a clear, non-noisy excess here too (all >=0.6pp on
# n=7,287, vs neighboring integers that are frequently negative/near-zero --
# e.g. 41: actual 3.82% vs model-implied 2.69%, +1.13pp; 44: 3.72% vs 2.60%,
# +1.12pp; 51: 3.77% vs 2.11%, +1.66pp -- full table in
# scratchpad measure_totals_key_numbers.py output, 2026-09-21 run). Unlike the
# spread bump there's no favorite/underdog side to average a +/- pair over (a
# total has no direction), so each key number's own measured excess is used
# directly. Every KXNFLTOTAL line is K-0.5 for a positive integer K (see
# parse_total_ticker); nb_survival(int(line), lam, phi) = P(total > line) =
# P(total >= K), so bumping the survival prob at line=K-0.5 by the excess mass
# measured at total==K is exact, same reasoning as the spread bump's boundary
# argument. UNVALIDATED against real Kalshi settlement/ROI, same caveat as
# NFL_TOTALS_PHI and the spread bump at their own ship dates -- re-check once
# real settled KXNFLTOTAL bets accumulate at these specific lines.
KEY_NUMBER_BUMP = {
    36.5: 0.0120,   # total == 37
    39.5: 0.0064,   # total == 40
    40.5: 0.0113,   # total == 41
    42.5: 0.0084,   # total == 43
    43.5: 0.0112,   # total == 44
    46.5: 0.0077,   # total == 47
    50.5: 0.0166,   # total == 51
}


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
        bump = KEY_NUMBER_BUMP.get(pt.line)
        if bump:
            prob_over = min(1.0, prob_over + bump)

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
