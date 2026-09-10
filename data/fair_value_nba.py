"""Fair value for Kalshi NBA winner markets via Elo win probability -- the
data/fair_value_nfl.py analog.

Pre-game only for v1 -- no live in-game model yet (documented gap, same
treatment as NFL's). Kickoff/tip-off time comes from the market's own
`occurrence_datetime` (kalshi/normalize.py) since NBA tickers, like NFL's,
carry no time-of-day segment (confirmed live 2026-09-10). UNLIKE NFL, there is
no independent tip-off-time source to cross-check occurrence_datetime against
(data/nba_data.py's game logs carry no time-of-day at all) -- so the systematic
+3h skew that NFL's model caught and corrected for (see engine/paper.py's
module note) cannot be detected or fixed here yet. Recommend a manual spot
check of occurrence_datetime against a real broadcast tip-off time before this
model's estimates are trusted for live paper-trading.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

from config.settings import Settings, DEFAULTS
from data.fair_value import FairValue
from data.nba_data import NbaDataClient
from kalshi.normalize import MarketQuote
from signals.elo_nba import EloRatings, win_prob

# Kalshi event ticker: KXNBAGAME-<YY><MON><DD><TEAMS>, market adds -<YESTEAM>.
# No HHMM segment, same shape as KXNFLGAME (confirmed live 2026-09-10).
_TICKER_RE = re.compile(r"KXNBAGAME-(\d{2})([A-Z]{3})(\d{2})([A-Z]+)$")
_MONTHS = {m: i for i, m in enumerate(
    ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"], start=1)}

# Confirmed live 2026-09-10 (6 markets, 3 games): "the X vs Y Pro Basketball
# game". The "professional basketball" alternative is NOT yet observed live --
# added defensively, mirroring NFL's confirmed Pro Football/professional
# football split, since both are plausible Kalshi copy variants and an
# open-ended word-class match already silently truncated NFL's "LA Rams" once
# (see data/fair_value_nfl.py's identical note).
_MATCHUP_RE = re.compile(r"the (.+?) vs (.+?) (?:Pro Basketball|professional basketball) game")


def matchup_from_rules(rules_primary: str) -> Optional[tuple[str, str]]:
    """('Boston', 'Detroit') from Kalshi's rules_primary text -- mirrors
    data/fair_value_nfl.py's identical helper. NBA's own market `title` field
    is just '<Team> wins' (no opponent name), confirmed live."""
    m = _MATCHUP_RE.search(rules_primary or "")
    return (m.group(1), m.group(2)) if m else None


@dataclass
class ParsedNbaTicker:
    event_ticker: str
    date_str: str          # YYYY-MM-DD -- the ticker has no time-of-day
    yes_team: str
    opponent: str
    yes_is_home: bool       # ticker team-blob order is AWAY-then-HOME (confirmed live)


def parse_ticker(market_ticker: str) -> Optional[ParsedNbaTicker]:
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
    return ParsedNbaTicker(event_ticker=event_ticker, date_str=date_str,
                           yes_team=yes_team, opponent=opponent, yes_is_home=yes_is_home)


class NbaFairValueModel:
    def __init__(self, nba_data: NbaDataClient, elo: Optional[EloRatings] = None,
                settings: Settings = DEFAULTS):
        self.nba_data = nba_data
        self.elo = elo or EloRatings(nba_data)
        self.settings = settings

    def estimate(self, market_ticker: str, quote: Optional[MarketQuote] = None) -> FairValue:
        pt = parse_ticker(market_ticker)
        if not pt:
            return FairValue(None, "none", 0.0, {"reason": "unparseable ticker"})

        # This data source excludes preseason entirely (see data/nba_data.py),
        # so "no matching game found" reliably means either a preseason
        # market or a real 2026-27 game whose season file hasn't been
        # published upstream yet (see data/nba_data.py's module docstring for
        # both cases). Refuse to estimate rather than guess -- game_state
        # stays None, same "unverifiable => unsafe" treatment MLB/NFL's own
        # "no game match" case gets.
        away = pt.opponent if pt.yes_is_home else pt.yes_team
        home = pt.yes_team if pt.yes_is_home else pt.opponent
        if self.nba_data.find_game(pt.date_str, away, home) is None:
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

        r_yes = self.elo.rating_before(pt.yes_team, pt.date_str)
        r_opp = self.elo.rating_before(pt.opponent, pt.date_str)
        prob = win_prob(r_yes, r_opp, a_is_home=pt.yes_is_home)
        # More rating separation => more confidence, capped (mirrors NFL's
        # identical confidence shape).
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
