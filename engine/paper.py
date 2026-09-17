"""Paper-trade ledger + the T-minus-first-pitch trigger.

Strategy: while the loop runs through the day, each game is triggered on the
LAST cycle before its first pitch — i.e. when it starts within the trigger
window and hasn't been bet yet — provided a recommendation clears the usual
thresholds at that moment. That captures each bet at its real decision point
(pitchers set, money has flowed, OI-momentum populated).

`iter_trigger_candidates` is the shared selection logic; this module's
`run_paper_trigger` just records candidates to a no-real-money ledger.
engine/live.py's `run_live_trigger` uses the same selection to submit REAL
orders instead — see that module.

`settle` grades the paper ledger against realized outcomes after games finish.
"""
from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from config.settings import OUTPUT_DIR, EASTERN
from config.sports import market_kind, sport_of
from data.games import Game, match_game
from signals.recommendation import Recommendation

log = logging.getLogger("engine.paper")

_MONTHS = {m: i for i, m in enumerate(
    ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"], start=1)}
# MLB winner/total tickers embed the ET first pitch as <YY><MON><DD><HH><MM> right after
# the series prefix. That is Kalshi-authoritative and reliable; occurrence_datetime is NOT
# -- on 2026-09-01 NYM@TB and SEA@BOS were stamped 01:40Z/01:45Z (9:40/9:45pm ET) for games
# whose real first pitch was 22:40Z/22:45Z (6:40/6:45pm ET, MLB Stats API confirmed), a whole
# 3-hour skew that pushed both games out of the trigger window and silently killed their
# paper bets.
#
# NFL tickers carry no time-of-day segment, so there is no ticker time to prefer -- and the
# same +3h skew is present on NFL markets too (confirmed 2026-09-08 against nflverse's own
# gametime on five Week 1/2 markets: Kalshi said 11:20pm ET for a 8:20pm ET opener, 4:00pm
# for the 1:00pm slot, etc). NFL therefore resolves kickoff from the nflverse schedule
# (NflGame.kickoff_utc) instead. This path had never fired in production -- preseason
# fair-value refusal meant no NFL rec ever reached the trigger, and the ledger held 0 NFL
# rows -- so the skew would have silently killed every NFL bet from Week 1 onward.
#
# NBA tickers (added 2026-09-10) also carry no time-of-day segment, but UNLIKE NFL there is
# no independent tip-off-time source to check for the same skew against -- data/nba_data.py's
# game logs carry no time-of-day at all, only game_date. So NBA has no choice but to trust
# occurrence_datetime as-is (falls through to `fp = sched_fp or r.game_datetime` below with
# sched_fp always None for NBA). If the same +3h-style skew exists here, it would silently
# misfire the NBA trigger window the same way it did for NFL until that was caught -- worth a
# manual spot check against a real broadcast tip-off time before trusting this in production.
#
# NHL tickers (added 2026-09-10) also carry no time-of-day segment, but data/nhl_data.py DOES
# carry a real per-game start time (NhlGame.puck_drop_utc, from the NHL's own official API) --
# same protection class as NFL's, so NHL joins NFL in the schedule-lookup branch below instead
# of blindly trusting occurrence_datetime the way NBA has to. Not yet confirmed whether the
# same +3h-style skew exists on NHL markets specifically (no NHL market has been live yet to
# check against) -- but unlike NBA, this path is already wired to catch and log it the moment
# one is.
_MLB_TICKER_TIME_RE = re.compile(r"^KXMLB(?:GAME|TOTAL)-(\d{2})([A-Z]{3})(\d{2})(\d{2})(\d{2})")


def first_pitch_from_ticker(ticker: str) -> Optional[datetime]:
    """UTC first pitch parsed from an MLB ticker's embedded ET time, or None when the
    ticker carries no time-of-day segment (e.g. NFL) or can't be parsed."""
    m = _MLB_TICKER_TIME_RE.match(ticker or "")
    if not m:
        return None
    yy, mon, dd, hh, mm = m.groups()
    month = _MONTHS.get(mon)
    if not month:
        return None
    try:
        naive_et = datetime(2000 + int(yy), month, int(dd), int(hh), int(mm))
    except ValueError:
        return None
    return naive_et.replace(tzinfo=EASTERN).astimezone(timezone.utc)


def _lookup_game(ticker: str, clients: dict, sched_cache: dict) -> Optional[Game]:
    """Best-effort schedule lookup. A flaky MLB Stats API / nflverse fetch must never
    stall or crash the trigger, so every failure degrades to None."""
    try:
        return match_game(ticker, clients, sched_cache)
    except Exception:
        log.warning("paper-trigger: match_game failed for %s.", ticker, exc_info=True)
        return None


class PaperLedger:
    def __init__(self, path: Optional[Path] = None):
        self.path = Path(path or (OUTPUT_DIR / "paper_ledger.jsonl"))
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._bet_events: set[str] = {b["event_key"] for b in self.load()}

    def load(self) -> list[dict]:
        if not self.path.exists():
            return []
        out = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                out.append(json.loads(line))
        return out

    def already_bet(self, event_key: str) -> bool:
        return event_key in self._bet_events

    def record(self, bet: dict) -> None:
        self._bet_events.add(bet["event_key"])
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(bet, default=str) + "\n")


def iter_trigger_candidates(recs: list[Recommendation], clients: dict, ledger: PaperLedger,
                            window_minutes: float, now: Optional[datetime] = None):
    """Yield (rec, bet_dict, game) for every recommendation whose game just
    entered the T-minus-start trigger window and hasn't been bet yet (per
    `ledger`). Does NOT record anything to `ledger` -- that's left to the
    caller, so a live order that fails to submit (engine/live.py) is never
    marked as bet and gets reconsidered next cycle, while a paper bet
    (run_paper_trigger, below) can record unconditionally.

    `clients` is {"mlb": MlbStatsClient(), "nfl": NflDataClient(), "nba": NbaDataClient(),
    "nhl": NhlDataClient()}, used both to label the row with a "matchup" string and --
    for NFL/NHL -- to resolve the real kickoff/puck-drop (data/games.py::match_game). Kalshi's
    own `occurrence_datetime` (`r.game_datetime`) is only a last-resort fallback for
    MLB/NFL/NHL, and the ONLY option for NBA (no independent kickoff source exists yet); see
    the module note above."""
    now = now or datetime.now(timezone.utc)
    sched_cache: dict = {}
    for r in recs:
        event_key = r.ticker.rsplit("-", 1)[0]
        # Start time, in preference order (occurrence_datetime last -- it carries a
        # systematic +3h skew on MLB/NFL; see the module note above):
        #   MLB -> the ET first pitch embedded in the ticker.
        #   NFL/NHL -> the schedule lookup (nflverse's gameday+gametime / the NHL API's
        #              startTimeUTC).
        #   NBA -> occurrence_datetime directly (no independent source to prefer yet).
        g: Optional[Game] = None
        sched_fp = first_pitch_from_ticker(r.ticker)
        if sched_fp is None and sport_of(r.ticker) in ("nfl", "nhl"):
            g = _lookup_game(r.ticker, clients, sched_cache)
            sched_fp = g.game_datetime if g else None
        fp = sched_fp or r.game_datetime
        if sched_fp is not None and r.game_datetime is not None:
            skew_min = abs((sched_fp - r.game_datetime).total_seconds()) / 60.0
            if skew_min > 30:
                log.warning("paper-trigger: %s occurrence_datetime %s disagrees with "
                            "scheduled start %s by %.0f min; trusting the schedule.",
                            r.ticker, r.game_datetime.isoformat(), sched_fp.isoformat(), skew_min)
        minutes = (fp - now).total_seconds() / 60.0 if fp is not None else None
        # "In the trigger window" = game starts within window_minutes (and hasn't yet).
        in_window = minutes is not None and 0 < minutes <= window_minutes
        # Log skip reasons only for games actually in the window, so a near-miss on a
        # bet we expected to fire is visible instead of silently dropped.
        if ledger.already_bet(event_key):
            if in_window:
                log.info("paper-trigger: %s in window but already bet this event; skip.", r.ticker)
            continue
        if r.suggested_contracts <= 0:
            if in_window:
                log.info("paper-trigger: %s in window but Kelly-sized to 0 contracts; skip.", r.ticker)
            continue                      # Kelly-sized to nothing -- not an actionable bet
        if fp is None:
            log.warning("paper-trigger: %s has no resolvable start time (no ticker time, no "
                        "schedule match, no occurrence_datetime); cannot time-trigger.", r.ticker)
            continue
        if not in_window:
            continue                          # too early (or already started) -- normal, no log
        # Matchup label is best-effort; on any failure fall back to the headline -- the bet
        # still records correctly, only the display label degrades. NFL/NHL already looked
        # the game up above for their timing, so this only costs a lookup for MLB/NBA.
        if g is None:
            g = _lookup_game(r.ticker, clients, sched_cache)
        matchup = f"{g.away_abbr} vs {g.home_abbr}" if g else (r.headline or r.yes_team)
        bet = {
            "ts": now.isoformat(),
            "first_pitch": fp.isoformat(),
            "minutes_before": round(minutes, 1),
            "ticker": r.ticker,
            "event_key": event_key,
            "matchup": matchup,
            "market": market_kind(r.ticker),
            "side": r.side,
            "label": r.headline or r.yes_team,
            "entry_price": r.entry_price,
            "contracts": r.suggested_contracts,
            "wager_usd": round(r.entry_price * r.suggested_contracts, 2),
            "stake_usd": r.suggested_stake_usd,
            "fair_prob": r.fair_prob,
            "edge_cents": r.edge_cents,
            "confidence": r.confidence,
            "money_flow": r.money_flow_score,
        }
        yield r, bet, g


def run_paper_trigger(recs: list[Recommendation], clients: dict, ledger: PaperLedger,
                      window_minutes: float, now: Optional[datetime] = None) -> list[dict]:
    """Paper-bet any recommended game that starts within `window_minutes` and
    hasn't been bet yet. Returns the newly placed paper bets."""
    placed: list[dict] = []
    for r, bet, _g in iter_trigger_candidates(recs, clients, ledger, window_minutes, now):
        ledger.record(bet)
        placed.append(bet)
        log.info("PAPER BET (%.0f min before): %s %s @ %.2f  edge %s  conf %.2f",
                 bet["minutes_before"], r.side, bet["label"], r.entry_price, r.edge_cents, r.confidence)
    return placed
