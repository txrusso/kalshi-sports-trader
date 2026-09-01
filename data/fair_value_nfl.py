"""Fair value for Kalshi NFL winner markets via Elo win probability.

Pre-game only for v1 -- no live in-game model yet (a documented gap; MLB has
one, NFL doesn't, same treatment as MLB's own documented gaps like missing
weather). Kickoff time comes from the market's own `occurrence_datetime`
(kalshi/normalize.py) since NFL tickers, unlike MLB's, don't encode a
time-of-day (confirmed live 2026-08-29 -- see CLAUDE.md).
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

from config.settings import Settings, DEFAULTS
from data.fair_value import FairValue
from data.nfl_data import NflDataClient
from kalshi.normalize import MarketQuote
from signals.elo_nfl import EloRatings, win_prob

# Kalshi event ticker: KXNFLGAME-<YY><MON><DD><TEAMS>, market adds -<YESTEAM>.
# No HHMM segment (unlike MLB's ticker, which has one) -- confirmed live.
_TICKER_RE = re.compile(r"KXNFLGAME-(\d{2})([A-Z]{3})(\d{2})([A-Z]+)$")
_MONTHS = {m: i for i, m in enumerate(
    ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"], start=1)}

# Confirmed live 2026-08-29 that the competition-type phrase between the teams
# and "game" varies ("Pro Football" for some events, "professional football" for
# others) -- both alternatives anchored explicitly rather than an open-ended
# word-class match, which was tried first and silently absorbed part of the
# second team's name (e.g. "LA Rams" -> group 2 came back as just "LA").
_MATCHUP_RE = re.compile(r"the (.+?) vs (.+?) (?:Pro Football|professional football) game")


def matchup_from_rules(rules_primary: str) -> Optional[tuple[str, str]]:
    """('NY Giants', 'LA Rams') from Kalshi's rules_primary text -- the NFL
    equivalent of MLB's shared 'A vs B Winner?' title. NFL's own market `title`
    field is just '<Team> wins' (no opponent name), so headline_for() can't use
    MLB's title-parsing trick; rules_primary spells the matchup out in full and
    is already present on every market in the bulk list response (confirmed
    live) -- zero extra API calls, same as MLB's zero-extra-call title parse."""
    m = _MATCHUP_RE.search(rules_primary or "")
    return (m.group(1), m.group(2)) if m else None


@dataclass
class ParsedNflTicker:
    event_ticker: str
    date_str: str          # YYYY-MM-DD -- the ticker has no time-of-day
    yes_team: str
    opponent: str
    yes_is_home: bool       # ticker team-blob order is AWAY-then-HOME (confirmed
                             # against nflverse's game_id convention)


def parse_ticker(market_ticker: str) -> Optional[ParsedNflTicker]:
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
    return ParsedNflTicker(event_ticker=event_ticker, date_str=date_str,
                           yes_team=yes_team, opponent=opponent, yes_is_home=yes_is_home)


class NflFairValueModel:
    def __init__(self, nfl_data: NflDataClient, elo: Optional[EloRatings] = None,
                settings: Settings = DEFAULTS):
        self.nfl_data = nfl_data
        self.elo = elo or EloRatings(nfl_data)
        self.settings = settings

    def estimate(self, market_ticker: str, quote: Optional[MarketQuote] = None) -> FairValue:
        pt = parse_ticker(market_ticker)
        if not pt:
            return FairValue(None, "none", 0.0, {"reason": "unparseable ticker"})

        # nflverse's games.csv excludes preseason entirely (see data/nfl_data.py),
        # so "no matching game found" reliably means this is a preseason/exhibition
        # market -- confirmed 2026-08-29 via backtest/nfl_model_backtest.py
        # --kalshi-settled against real settled preseason markets: applying each
        # team's REGULAR-season Elo to preseason games (backup-heavy rosters) was
        # worse than a coin flip (Brier 0.284 vs 0.25) and lost money at the
        # production edge gate (16% win rate on real trade prices). Refuse to
        # estimate rather than confidently mis-price these -- game_state stays
        # None, which build_recommendation() already treats as unverified/unsafe
        # under pregame_only (same as MLB's "no game match" case).
        away = pt.opponent if pt.yes_is_home else pt.yes_team
        home = pt.yes_team if pt.yes_is_home else pt.opponent
        if self.nfl_data.find_game(pt.date_str, away, home) is None:
            return FairValue(None, "none", 0.0,
                             {"reason": "no regular-season game found (likely preseason)"})

        occurrence = quote.occurrence_datetime if quote else None
        if occurrence is None:
            # Can't verify the game hasn't started -- unverifiable, treated as
            # unsafe (mirrors MLB's "no game match" leaving game_state=None).
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
        prob = win_prob(r_yes, r_opp, a_is_home=pt.yes_is_home)
        # More rating separation => more confidence, capped (mirrors MLB's
        # log5 confidence shape in data/fair_value.py).
        conf = round(0.35 + min(abs(r_yes - r_opp) / 800.0, 0.25), 3)

        # Full team names for headline_for() (signals/recommendation.py), same
        # detail-dict keys MLB's model populates -- sourced from rules_primary
        # since NFL's own market `title` has no opponent name in it.
        yes_name = opp_name = ""
        matchup = matchup_from_rules(quote.rules_primary) if quote else None
        if matchup:
            away_name, home_name = matchup   # rules_primary order is away-then-home
            yes_name, opp_name = (home_name, away_name) if pt.yes_is_home else (away_name, home_name)

        return FairValue(round(prob, 4), "pregame_elo", conf,
                         {"yes_elo": round(r_yes, 1), "opp_elo": round(r_opp, 1),
                          "yes_is_home": pt.yes_is_home,
                          "yes_name": yes_name, "opp_name": opp_name}, game_state=game_state)
