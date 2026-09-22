"""Fair value for Kalshi NFL point-spread markets via an Elo-implied margin.

Each KXNFLSPREAD market is "<Team> wins by over K.5 points?" (YES = that team
covers) -- unlike totals, which share ONE ladder per game, spread gives EACH
team its own full ladder of rungs (confirmed live 2026-09-13: a BUF@HOU event
carried 24 markets, 12 rungs x 2 teams). Models the actual (home - away) point
margin as Normal(mu, sigma): mu comes from NFL Elo's own rating gap (the same
signals/elo_nfl.py ratings the winner model uses), sigma from a real-data
regression of margin on Elo diff. Pre-game only, same policy as the winner/
totals NFL models.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

from config.settings import Settings, DEFAULTS
from data.distributions import normal_sf
from data.fair_value import FairValue
from data.fair_value_nfl import matchup_from_rules
from data.nfl_data import NflDataClient, split_team_codes
from kalshi.normalize import MarketQuote
from signals.elo_nfl import EloRatings, HOME_FIELD_ELO

# KXNFLSPREAD-<YY><MON><DD><TEAMS>-<TEAM><K>, e.g. KXNFLSPREAD-26SEP13BUFHOU-HOU8
# = "Houston wins by over 7.5 points?" (K-0.5 = line). No HHMM segment, same as
# KXNFLGAME/KXNFLTOTAL. Unlike totals' plain numeric suffix, the market suffix
# here is TEAM+NUMBER concatenated (which team's ladder this rung belongs to).
_SPREAD_RE = re.compile(r"KXNFLSPREAD-(\d{2})([A-Z]{3})(\d{2})([A-Z]+)-([A-Z]{2,3})(\d+)$")
_MONTHS = {m: i for i, m in enumerate(
    ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"], start=1)}

# Elo-diff -> point-margin regression, fit 2026-09-13 against 7,278 real completed
# games from signals/elo_nfl.py's own training history (EloRatings.margin_samples()):
#   margin ~= MARGIN_PER_ELO * (yes_elo - opp_elo + home-field adjustment)
# Ordinary least squares of actual (home-away) margin on (elo_diff + HOME_FIELD_ELO)
# gave slope 0.0414 pts/elo-pt (~1 point per 24 Elo, close to the commonly-cited
# "25 Elo per point" heuristic), intercept ~0.3 (dropped as negligible), and a
# residual std of 13.52 vs a raw (no-model) margin std of 14.59 -- the model
# explains real but modest variance, consistent with the winner model's own
# near-coin-flip Brier (this is the same signal, just read out as a margin
# instead of a win probability).
#
# UNVALIDATED against real Kalshi settlement/ROI -- unlike every other constant
# in this project, there is no backtest/nfl_spread_backtest.py walk-forward pass
# yet (no real settled KXNFLSPREAD markets existed until this model was built).
# Shipped live 2026-09-13 at the user's explicit request for same-day NFL spread
# coverage; treat MARGIN_PER_ELO/MARGIN_SIGMA the same as NFL_TOTALS_PHI was
# treated at ship time (seeded from a raw measurement, re-sweep once real
# settled spread markets accumulate).
MARGIN_PER_ELO = 0.0414
MARGIN_SIGMA = 13.52

# Key-number bumps (added 2026-09-17, at the user's request after reading
# https://www.oddsshopper.com/articles/betting-101/nfl-key-numbers). NFL margins
# cluster hard at exactly 3 and 7 (FG / TD+XP), which a smooth Normal cannot
# represent. Measured directly against the same 7,292-game margin_samples()
# history used for the OLS fit above: for each game, compared the ACTUAL rate
# of margin==3/7 to what the shipped Normal(mu, MARGIN_SIGMA) predicts for that
# integer bin (continuity-corrected: P(margin in [k-0.5, k+0.5])). Result was a
# large, unambiguous excess at both numbers, roughly symmetric between the
# favorite and underdog side (margin=+3: actual 8.13% vs model-implied 2.73%,
# +5.41pp; margin=-3: 6.97% vs 2.58%, +4.39pp; margin=+7: 4.83% vs 2.58%,
# +2.25pp; margin=-7: 4.20% vs 2.26%, +1.94pp) -- shipped as the average of the
# +/- pair for each number rather than two separate constants, since the model
# has no mechanism to predict which side of a game the asymmetry favors and the
# gap is small relative to the bump itself.
#
# Every KXNFLSPREAD line is a positive half-integer K-0.5 (see parse_spread_ticker;
# the underdog's own "+K.5" bet is the NO side of the FAVORITE's ticker, per
# headline_for()'s reframing -- there is no negative-K ticker to handle). Covering
# line=K-0.5 requires margin >= K, so the exact-margin excess at k=3/7 only needs
# to be added to the survival probability at line=k-0.5 (2.5, 6.5): P(margin>=k)
# gains the full excess mass at k, while P(margin>=k+1) (line=k+0.5, e.g. 3.5/7.5)
# is untouched, since that boundary already excludes margin=k under the
# continuous model too -- consistent by construction (sf(k-0.5) - sf(k+0.5) =
# P(margin=k) exactly, so bumping only the lower boundary reproduces the measured
# excess without needing to know where the compensating deficit sits elsewhere in
# the distribution). This is also exactly the "-2.5 beats -3.5, +3.5 beats +2.5"
# rule the user described -- it falls out of the bumped probabilities on its own
# (line=2.5 gets a higher P(cover), and the NO side of line=3.5, i.e. "+3.5", is
# left at the unbumped baseline while NO-of-2.5, i.e. "+2.5", is reduced by the
# same bump) rather than needing a separate hardcoded preference rule.
KEY_NUMBER_BUMP = {2.5: 0.049, 6.5: 0.021}


@dataclass
class ParsedNflSpread:
    event_ticker: str
    date_str: str
    yes_team: str
    opponent: str
    yes_is_home: bool
    line: float          # e.g. 3.5 -- yes_team must win by MORE than this to cover


def parse_spread_ticker(ticker: str) -> Optional[ParsedNflSpread]:
    m = _SPREAD_RE.match(ticker)
    if not m:
        return None
    yy, mon, dd, teams, yes_team, num = m.groups()
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
    away, home = split
    if yes_team not in (away, home):
        return None
    opponent = home if yes_team == away else away
    return ParsedNflSpread(event_ticker=ticker.rsplit("-", 1)[0], date_str=date_str,
                           yes_team=yes_team, opponent=opponent,
                           yes_is_home=(yes_team == home), line=int(num) - 0.5)


class NflSpreadFairValueModel:
    def __init__(self, nfl_data: NflDataClient, elo: Optional[EloRatings] = None,
                settings: Settings = DEFAULTS):
        self.nfl_data = nfl_data
        self.elo = elo or EloRatings(nfl_data)
        self.settings = settings

    def estimate(self, ticker: str, quote: Optional[MarketQuote] = None) -> FairValue:
        pt = parse_spread_ticker(ticker)
        if not pt:
            return FairValue(None, "none", 0.0, {"reason": "unparseable spread ticker"})

        # Same preseason-refusal signal as the winner/totals NFL models: nflverse
        # excludes preseason entirely, so "no game found" reliably flags it.
        away = pt.opponent if pt.yes_is_home else pt.yes_team
        home = pt.yes_team if pt.yes_is_home else pt.opponent
        if self.nfl_data.find_game(pt.date_str, away, home) is None:
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

        r_yes = self.elo.rating_before(pt.yes_team, pt.date_str)
        r_opp = self.elo.rating_before(pt.opponent, pt.date_str)
        diff = (r_yes - r_opp) + (HOME_FIELD_ELO if pt.yes_is_home else -HOME_FIELD_ELO)
        mu = MARGIN_PER_ELO * diff
        prob_cover = normal_sf(pt.line, mu, MARGIN_SIGMA)
        bump = KEY_NUMBER_BUMP.get(pt.line)
        if bump:
            prob_cover = min(1.0, prob_cover + bump)
        # Deliberately more conservative than the winner model's confidence shape
        # (0.35 base / 0.25 cap) -- this model is unvalidated, so its confidence
        # ceiling is capped lower until a real walk-forward pass exists.
        conf = round(0.30 + min(abs(r_yes - r_opp) / 800.0, 0.20), 3)

        # Full team names for headline_for() (signals/recommendation.py), same
        # detail-dict keys the winner model populates -- sourced from rules_primary
        # since NFL's own market `title` has no opponent name in it.
        yes_name = opp_name = ""
        matchup = matchup_from_rules(quote.rules_primary) if quote else None
        if matchup:
            away_name, home_name = matchup   # rules_primary order is away-then-home
            yes_name, opp_name = (home_name, away_name) if pt.yes_is_home else (away_name, home_name)

        return FairValue(round(prob_cover, 4), "pregame_nfl_spread", conf,
                         {"line": pt.line, "exp_margin": round(mu, 2),
                          "yes_is_home": pt.yes_is_home,
                          "yes_name": yes_name, "opp_name": opp_name}, game_state=game_state)
