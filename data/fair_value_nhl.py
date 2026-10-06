"""Fair value for Kalshi NHL winner markets via Elo win probability -- the
data/fair_value_nba.py analog.

Pre-game only for v1 -- no live in-game model yet (documented gap, same
treatment as NFL's/NBA's). Kickoff/puck-drop time comes from the market's own
`occurrence_datetime` (kalshi/normalize.py) since NHL tickers, like NFL's and
NBA's, carry no time-of-day segment (format confirmed via web search against
real historical tickers, e.g. "KXNHLGAME-26MAY26COLVGK-COL"; NOT yet
confirmed against the live trade API since no KXNHLGAME market is open this
far from the season -- worth a live re-check once one lists). UNLIKE NBA,
data/nhl_data.py DOES carry an independent start-time source
(NhlGame.puck_drop_utc) -- see engine/paper.py for how that's used to guard
against the same occurrence_datetime skew NFL found.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

from config.settings import Settings, DEFAULTS
from data.fair_value import FairValue, has_started
from data.nhl_data import NhlDataClient
from kalshi.normalize import MarketQuote
from signals.elo_nhl import EloRatings, win_prob

# Kalshi event ticker: KXNHLGAME-<YY><MON><DD><TEAMS>, market adds -<YESTEAM>.
# No HHMM segment, same shape as KXNFLGAME/KXNBAGAME (format confirmed via web
# search against real historical tickers, see module docstring).
_TICKER_RE = re.compile(r"KXNHLGAME-(\d{2})([A-Z]{3})(\d{2})([A-Z]+)$")
_MONTHS = {m: i for i, m in enumerate(
    ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"], start=1)}

# CONFIRMED LIVE 2026-09-29, once real markets listed for the season opener.
# The "NHL game" anchor guessed here was right (Kalshi's own URL/category
# naming, unlike MLB/NFL/NBA's "Professional X" pattern); "Pro/professional
# Hockey game" is kept as a defensive alternate. Real text, both kinds:
#   winner: "If San Jose wins the San Jose vs Dallas NHL game originally ..."
#   totals: "If the teams collectively score more than 5.5 goals in the
#            Edmonton vs Vancouver NHL game originally ..."
#
# That totals wording is why the team capture must refuse to span the word
# "the". `re.search` matches at the LEFTMOST start, so a plain `the (.+?) vs`
# anchored on "the teams collectively ..." and group(1) swallowed the whole
# clause -- real headlines read "teams collectively score more than 5.5 goals
# in the Florida vs Carolina Over 5.5" (seen live on the dashboard). Blocking
# `the` inside the capture kills that start position, so the match falls
# through to "the Edmonton vs Vancouver NHL game" and group(1) is the team.
# No NHL/NFL/NBA club name contains the standalone word "the", so this only
# ever narrows an over-capture. Display-only either way: the model's own
# inputs come from the parsed TICKER (pt.away_abbr/pt.home_abbr), never from
# this text, so a miss degrades to a generic headline and never a bad price.
# NOTE: data/fair_value_nfl.py and data/fair_value_nba.py still use the
# unhardened pattern. NFL is safe on its CONFIRMED live text ("If ATL Falcons
# and NO Saints collectively score ..." has no earlier "the"); NBA's is still
# unverified, and would break exactly like this if it uses NHL's wording.
_TEAM = r"(?:(?!\bthe\b).)+?"
_MATCHUP_RE = re.compile(
    rf"the ({_TEAM}) vs ({_TEAM}) (?:NHL|Pro Hockey|professional hockey) game",
    re.IGNORECASE)


def matchup_from_rules(rules_primary: str) -> Optional[tuple[str, str]]:
    """('Colorado', 'Vegas') from Kalshi's rules_primary text -- mirrors
    data/fair_value_nba.py's identical helper. See the UNCONFIRMED note above."""
    m = _MATCHUP_RE.search(rules_primary or "")
    return (m.group(1), m.group(2)) if m else None


@dataclass
class ParsedNhlTicker:
    event_ticker: str
    date_str: str          # YYYY-MM-DD -- the ticker has no time-of-day
    yes_team: str
    opponent: str
    yes_is_home: bool       # ticker team-blob order is AWAY-then-HOME (assumed
                             # from the NFL/NBA/MLB convention; not yet
                             # independently confirmed for NHL, see module note)


def parse_ticker(market_ticker: str) -> Optional[ParsedNhlTicker]:
    parts = market_ticker.rsplit("-", 1)
    if len(parts) != 2:
        return None
    event_ticker, yes_team = parts
    m = _TICKER_RE.match(event_ticker)
    if not m:
        return None
    yy, mon, dd, teams = m.groups()
    month = _MONTHS.get(mon)
    if not month:
        return None
    try:
        date_str = datetime(2000 + int(yy), month, int(dd)).strftime("%Y-%m-%d")
    except ValueError:
        return None
    if teams.startswith(yes_team):
        opponent, yes_is_home = teams[len(yes_team):], False
    elif teams.endswith(yes_team):
        opponent, yes_is_home = teams[: -len(yes_team)], True
    else:
        return None
    if not opponent:
        return None
    return ParsedNhlTicker(event_ticker=event_ticker, date_str=date_str,
                           yes_team=yes_team, opponent=opponent, yes_is_home=yes_is_home)


class NhlFairValueModel:
    def __init__(self, nhl_data: NhlDataClient, elo: Optional[EloRatings] = None,
                settings: Settings = DEFAULTS):
        self.nhl_data = nhl_data
        self.elo = elo or EloRatings(nhl_data)
        self.settings = settings

    def estimate(self, market_ticker: str, quote: Optional[MarketQuote] = None) -> FairValue:
        pt = parse_ticker(market_ticker)
        if not pt:
            return FairValue(None, "none", 0.0, {"reason": "unparseable ticker"})

        # data/nhl_data.py filters preseason itself (gameType==1), so "no
        # matching game found" reliably means preseason -- same refusal
        # signal NFL/NBA get, "unverifiable => unsafe" treatment.
        away = pt.opponent if pt.yes_is_home else pt.yes_team
        home = pt.yes_team if pt.yes_is_home else pt.opponent
        g = self.nhl_data.find_game(pt.date_str, away, home)
        if g is None:
            return FairValue(None, "none", 0.0,
                             {"reason": "no game found (likely preseason)"})

        occurrence = quote.occurrence_datetime if quote else None
        if occurrence is None:
            return FairValue(None, "none", 0.0, {"reason": "no occurrence_datetime on quote"})

        # Judge 'started' against the SCHEDULE's real start, not Kalshi's
        # occurrence_datetime, which runs ~3h late here. See
        # data/fair_value.py::has_started.
        started = has_started(occurrence, g.puck_drop_utc)
        game_state = "Live" if started else "Preview"
        if self.settings.pregame_only and started:
            return FairValue(None, "skip_non_pregame", 0.0,
                             {"reason": "game has started"}, game_state=game_state)
        if started:
            return FairValue(None, "none", 0.0, {"reason": "no live NHL model yet"},
                             game_state=game_state)

        r_yes = self.elo.rating_before(pt.yes_team, pt.date_str)
        r_opp = self.elo.rating_before(pt.opponent, pt.date_str)
        prob = win_prob(r_yes, r_opp, a_is_home=pt.yes_is_home)
        # More rating separation => more confidence, capped (mirrors NFL's/
        # NBA's identical confidence shape).
        conf = round(0.35 + min(abs(r_yes - r_opp) / 800.0, 0.25), 3)

        yes_name = opp_name = ""
        matchup = matchup_from_rules(quote.rules_primary) if quote else None
        if matchup:
            away_name, home_name = matchup   # rules_primary order is away-then-home
            yes_name, opp_name = (home_name, away_name) if pt.yes_is_home else (away_name, home_name)

        return FairValue(round(prob, 4), "pregame_elo", conf,
                         {"yes_elo": round(r_yes, 1), "opp_elo": round(r_opp, 1),
                          "yes_is_home": pt.yes_is_home,
                          "yes_name": yes_name, "opp_name": opp_name}, game_state=game_state)
