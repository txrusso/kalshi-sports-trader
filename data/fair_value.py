"""Estimate a fair YES probability for a Kalshi MLB market.

Live games   -> MLB Stats API live win probability (high confidence).
Pre-game     -> log5 on season records + home-field adjustment (moderate).
Otherwise    -> no estimate (the money-flow signal stands alone).
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

from config.settings import Settings, DEFAULTS
from data.elo_mlb_history import get_elo_ratings
from data.mlb_stats import MlbGame, MlbStatsClient
from signals.elo_mlb import win_prob as elo_win_prob

log = logging.getLogger("data.fair_value")

# Kalshi event ticker: KXMLBGAME-<YY><MON><DD><HHMM><TEAMS>, market adds -<YESTEAM>
_TICKER_RE = re.compile(r"KXMLBGAME-(\d{2})([A-Z]{3})(\d{2})(\d{2})(\d{2})([A-Z]+)$")
_MONTHS = {m: i for i, m in enumerate(
    ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"], start=1)}

HOME_FIELD_ODDS_MULT = 0.54 / 0.46  # MLB home teams win ~54%

# Regress each team's season win% toward .500 before feeding log5. p' = 0.5 + (1-r)(p-0.5);
# 0.0 = identity (current behavior). Idea: raw records overstate true-talent spread, so
# log5-on-records is overconfident -> phantom favorite edges. BUILT AND SWEEP-TESTED
# 2026-08-31, left at 0.0 (NOT enabled) -- the evidence didn't clear the walk-forward bar:
#   * The premise was small-sample noise. A 21-bet diagnostic showed favorites at 42%
#     actual vs 63% predicted, but on ALL 864 settled winner markets the favorite gap is
#     only 2.7pts (58.0% pred / 55.3% actual) -- the model is only MILDLY overconfident.
#   * Full-sample Brier improves a trivial 0.0008 (0.2471->0.2463 at r=0.30) and directional
#     ACCURACY drops (0.572->0.557) -- compressing toward .500 flips correct favorite picks.
#   * bankroll_sim splits DISAGREE: validate winner ROI rises (0.078->0.152) but TRAIN
#     winner ROI falls (-0.15->-0.28). kelly_fraction won on both halves; this doesn't.
# Kept as an inert, sweepable tunable (r in bankroll_sim's DEFAULT_PARAMS as `winner_regress`)
# so it can be re-tested cheaply once the sample is larger; do not ship a non-zero default
# without a clean both-splits win. A mild r~0.15 did cut validate max-drawdown (32%->23%),
# which is the one result worth revisiting with more data.
WINNER_REGRESS = 0.0


def _regress_win_pct(p: float, regress: float = None) -> float:
    r = WINNER_REGRESS if regress is None else regress
    return 0.5 + (1.0 - r) * (p - 0.5)

# Blend weight for signals/elo_mlb.py's Elo rating against log5-on-record, built and
# walk-forward-swept 2026-09-01 (backtest/bankroll_sim.py's `elo_weight` sweep, train
# <=2026-07-15 / validate after). A moderate blend (0.15-0.25) beat pure log5 (weight=0)
# on BOTH splits -- validate ROI 9.8%->11.0%, and drawdown dropped meaningfully on both
# train (43%->32%) and validate (32%->24%). Brier barely moved (0.2455-0.2457 across the
# whole sweep) -- Elo isn't making probabilities more *accurate*, it's changing which
# bets clear the edge gate and how they're sized, and that shift reduced drawdown and
# lifted ROI on this sample. Pushing weight past ~0.5 overfits (train keeps improving to
# weight=1.0, validate degrades) -- same shape as the reverted Pythagorean/weather
# experiments, which is why this shipped at a moderate 0.2, not higher. Caveat: one
# partial season (n~200-330 bets/split) -- directional, not conclusive; re-sweep as more
# settled markets accumulate. Set to 0.0 to fall back to pure log5.
ELO_WEIGHT = 0.2


# A Pythagorean pitching-matchup blend (log5 nudged by a run-environment estimate
# folding in starters) was tried 2026-08-13 and reverted the same day. It looked
# like a real improvement on a single full-sample backtest (Brier 0.2500 -> 0.2475)
# but failed a proper walk-forward test: tuning on Jun 7-Jul 15 and validating on
# Jul 16-Aug 13 showed validation performance getting monotonically WORSE as the
# blend weight increased (best at weight=0, i.e. no blend at all) -- a textbook
# overfit to the single-sample test. See backtest/bankroll_sim.py for the sweep.


@dataclass
class FairValue:
    prob: Optional[float]      # fair P(YES team wins), 0..1
    source: str                # "live", "pregame_log5", "skip_non_pregame", "none"
    confidence: float          # 0..1
    detail: dict
    game_state: Optional[str] = None   # matched MLB game state: Preview/Live/Final


@dataclass
class ParsedTicker:
    event_ticker: str
    date: datetime
    yes_team: str
    opponent: str


def parse_ticker(market_ticker: str) -> Optional[ParsedTicker]:
    parts = market_ticker.rsplit("-", 1)
    if len(parts) != 2:
        return None
    event_ticker, yes_team = parts
    m = _TICKER_RE.match(event_ticker)
    if not m:
        return None
    yy, mon, dd, hh, mm, teams = m.groups()
    month = _MONTHS.get(mon)
    if not month:
        return None
    try:
        date = datetime(2000 + int(yy), month, int(dd), int(hh), int(mm), tzinfo=timezone.utc)
    except ValueError:
        return None
    # Opponent = the team segment with the YES team removed.
    opp = None
    if teams.startswith(yes_team):
        opp = teams[len(yes_team):]
    elif teams.endswith(yes_team):
        opp = teams[: -len(yes_team)]
    return ParsedTicker(event_ticker=event_ticker, date=date, yes_team=yes_team, opponent=opp or "")


def log5(p_a: float, p_b: float) -> float:
    """P(A beats B) on a neutral site from season win pcts."""
    denom = p_a + p_b - 2 * p_a * p_b
    if denom <= 0:
        return 0.5
    return (p_a - p_a * p_b) / denom


def _apply_home_field(p: float, yes_is_home: bool) -> float:
    odds = p / (1 - p) if p < 1 else 999.0
    odds = odds * HOME_FIELD_ODDS_MULT if yes_is_home else odds / HOME_FIELD_ODDS_MULT
    return odds / (1 + odds)


class FairValueModel:
    def __init__(self, mlb: MlbStatsClient, settings: Settings = DEFAULTS):
        self.mlb = mlb
        self.settings = settings
        self._sched_cache: dict[str, list[MlbGame]] = {}

    def _schedule(self, date: datetime) -> list[MlbGame]:
        key = date.strftime("%Y-%m-%d")
        if key not in self._sched_cache:
            self._sched_cache[key] = self.mlb.schedule(date)
        return self._sched_cache[key]

    def _match_game(self, pt: ParsedTicker) -> Optional[MlbGame]:
        want = {pt.yes_team}
        if pt.opponent:
            want.add(pt.opponent)
        for date in (pt.date, pt.date.astimezone(timezone.utc)):
            for g in self._schedule(date):
                if pt.yes_team in g.teams() and (not pt.opponent or pt.opponent in g.teams()):
                    return g
        return None

    def estimate(self, market_ticker: str, quote=None) -> FairValue:
        # `quote` is accepted-but-unused here so FairValueRouter can call every
        # sport's model uniformly -- MLB does its own schedule lookup and
        # doesn't need the market quote the way NFL's model does (no
        # occurrence_datetime on its ticker; see data/fair_value_nfl.py).
        pt = parse_ticker(market_ticker)
        if not pt:
            return FairValue(None, "none", 0.0, {"reason": "unparseable ticker"})
        game = self._match_game(pt)
        if not game:
            return FairValue(None, "none", 0.0, {"reason": "no MLB game match", "yes_team": pt.yes_team})

        # MLB's status can flip away from "Preview" a few minutes before the
        # actual scheduled first pitch (confirmed 2026-08-10/11: this silently
        # killed real recommendations -- Tampa Bay and Pittsburgh/Miami both
        # lost their fair-value estimate ~5-10 min before their listed first
        # pitch, right on the last scan cycle before the paper-bet trigger).
        # Trust the schedule over the status string for "has this actually
        # started yet" -- a premature status flip still gets the pregame
        # estimate below instead of nothing.
        truly_started = game.game_datetime is not None and datetime.now(timezone.utc) >= game.game_datetime
        effective_pregame = game.state == "Preview" or not truly_started

        # Pre-game-only policy: don't estimate (or spend an API call) on live/final games.
        if self.settings.pregame_only and not effective_pregame:
            return FairValue(None, "skip_non_pregame", 0.0,
                             {"reason": f"pregame_only, state={game.state}", "game_pk": game.game_pk},
                             game_state=game.state)

        yes_is_home = (pt.yes_team == game.home_abbr)

        if game.state == "Live" and not effective_pregame:
            hwp = self.mlb.home_win_probability(game.game_pk)
            if hwp is not None:
                prob = hwp if yes_is_home else 1.0 - hwp
                yes_name = game.home_name if yes_is_home else game.away_name
                opp_name = game.away_name if yes_is_home else game.home_name
                return FairValue(round(prob, 4), "live", 0.80,
                                 {"game_pk": game.game_pk, "state": game.detailed_state,
                                  "home_win_prob": round(hwp, 4), "yes_is_home": yes_is_home,
                                  "yes_name": yes_name, "opp_name": opp_name},
                                 game_state=game.state)

        if effective_pregame:
            # Season-to-date record (game.home_win_pct/away_win_pct, from MLB's
            # leagueRecord). A last-30-games recency window was also tried and
            # reverted 2026-08-13 -- see backtest/bankroll_sim.py's sweep: too
            # noisy at n=30 for a binary win/loss sample to feed into log5.
            pa = game.home_win_pct if yes_is_home else game.away_win_pct
            pb = game.away_win_pct if yes_is_home else game.home_win_pct
            if pa is not None and pb is not None:
                neutral = log5(_regress_win_pct(pa), _regress_win_pct(pb))
                log5_prob = _apply_home_field(neutral, yes_is_home)
                elo_prob, elo_yes_rating, elo_opp_rating = None, None, None
                if ELO_WEIGHT > 0:
                    try:
                        ratings = get_elo_ratings()
                        date_str = pt.date.strftime("%Y-%m-%d")
                        elo_yes_rating = ratings.rating_before(pt.yes_team, date_str)
                        elo_opp_rating = ratings.rating_before(pt.opponent, date_str)
                        elo_prob = elo_win_prob(elo_yes_rating, elo_opp_rating, a_is_home=yes_is_home)
                    except Exception:
                        log.exception("Elo estimate failed for %s; falling back to pure log5", market_ticker)
                prob = ((1 - ELO_WEIGHT) * log5_prob + ELO_WEIGHT * elo_prob
                        if elo_prob is not None else log5_prob)
                # More record separation => more confidence, capped.
                conf = 0.35 + min(abs(pa - pb) * 1.5, 0.25)
                yes_name = game.home_name if yes_is_home else game.away_name
                opp_name = game.away_name if yes_is_home else game.home_name
                return FairValue(round(prob, 4), "pregame_log5", round(conf, 3),
                                 {"game_pk": game.game_pk, "yes_win_pct": pa, "opp_win_pct": pb,
                                  "neutral": round(neutral, 4), "log5_prob": round(log5_prob, 4),
                                  "elo_prob": round(elo_prob, 4) if elo_prob is not None else None,
                                  "elo_yes_rating": round(elo_yes_rating, 1) if elo_yes_rating is not None else None,
                                  "elo_opp_rating": round(elo_opp_rating, 1) if elo_opp_rating is not None else None,
                                  "yes_is_home": yes_is_home,
                                  "yes_name": yes_name, "opp_name": opp_name},
                                 game_state=game.state)

        return FairValue(None, "none", 0.0,
                         {"reason": f"state={game.state}", "game_pk": game.game_pk},
                         game_state=game.state)


class FairValueRouter:
    """Dispatches fair-value estimation by (sport, market kind) so
    engine/scanner.py and cli.py don't need their own is_total()-branching
    logic for which model to call -- one router replaces the four models
    threaded around individually."""

    def __init__(self, mlb_winner, mlb_totals, nfl_winner, nfl_totals,
                nba_winner=None, nba_totals=None, nhl_winner=None, nhl_totals=None):
        self._models = {
            ("mlb", "winner"): mlb_winner,
            ("mlb", "total"): mlb_totals,
            ("nfl", "winner"): nfl_winner,
            ("nfl", "total"): nfl_totals,
            ("nba", "winner"): nba_winner,
            ("nba", "total"): nba_totals,
            ("nhl", "winner"): nhl_winner,
            ("nhl", "total"): nhl_totals,
        }

    def estimate(self, ticker: str, quote=None) -> FairValue:
        from config.sports import market_kind, sport_of
        model = self._models.get((sport_of(ticker), market_kind(ticker)))
        if model is None:
            return FairValue(None, "none", 0.0, {"reason": "unknown sport/market"})
        return model.estimate(ticker, quote)

    def clear_caches(self) -> None:
        """Per-cycle cache reset (fresh schedule/win-prob each scan). Only MLB's
        model needs this -- NFL's data changes slowly enough that its own TTL
        cache (data/nfl_data.py) doesn't need a per-cycle bust."""
        mlb_winner = self._models.get(("mlb", "winner"))
        if mlb_winner is not None:
            mlb_winner._sched_cache.clear()
