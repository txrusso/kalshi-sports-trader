"""Paper-trade ledger + the T-minus-first-pitch trigger.

Strategy (recommend-only, no real orders): while the loop runs through the day,
each game is "paper-bet" on the LAST cycle before its first pitch — i.e. when it
starts within the trigger window and hasn't been bet yet — provided a
recommendation clears the usual thresholds at that moment. That captures each bet
at its real decision point (pitchers set, money has flowed, OI-momentum populated).

`settle` grades the ledger against realized outcomes after games finish.
"""
from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from config.settings import OUTPUT_DIR, EASTERN
from config.sports import market_kind
from data.games import match_game
from signals.recommendation import Recommendation

log = logging.getLogger("engine.paper")

_MONTHS = {m: i for i, m in enumerate(
    ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"], start=1)}
# MLB winner/total tickers embed the ET first pitch as <YY><MON><DD><HH><MM> right after
# the series prefix. That is Kalshi-authoritative and reliable; occurrence_datetime is NOT
# -- on 2026-09-01 NYM@TB and SEA@BOS were stamped 01:40Z/01:45Z (9:40/9:45pm ET) for games
# whose real first pitch was 22:40Z/22:45Z (6:40/6:45pm ET, MLB Stats API confirmed), a whole
# 3-hour skew that pushed both games out of the trigger window and silently killed their
# paper bets. NFL tickers carry no time-of-day segment, so they still fall back to
# occurrence_datetime (there is no ticker time to prefer).
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


def run_paper_trigger(recs: list[Recommendation], clients: dict, ledger: PaperLedger,
                      window_minutes: float, now: Optional[datetime] = None) -> list[dict]:
    """Paper-bet any recommended game that starts within `window_minutes` and
    hasn't been bet yet. Returns the newly placed paper bets.

    `clients` is {"mlb": MlbStatsClient(), "nfl": NflDataClient()} -- only used
    to label the ledger row with a "matchup" string (data/games.py::match_game);
    the trigger's own timing comes from `r.game_datetime` (the market's own
    occurrence_datetime, sport-agnostic, confirmed present on both MLB and NFL
    markets), not a schedule lookup."""
    now = now or datetime.now(timezone.utc)
    sched_cache: dict = {}
    placed: list[dict] = []
    for r in recs:
        event_key = r.ticker.rsplit("-", 1)[0]
        # First pitch: prefer the MLB ticker's embedded ET time (Kalshi-authoritative)
        # over occurrence_datetime, which has been observed mis-stamped by whole hours and
        # would silently push a game out of the trigger window. NFL tickers have no embedded
        # time, so first_pitch_from_ticker() returns None and we fall back to occurrence.
        ticker_fp = first_pitch_from_ticker(r.ticker)
        fp = ticker_fp or r.game_datetime
        if ticker_fp is not None and r.game_datetime is not None:
            skew_min = abs((ticker_fp - r.game_datetime).total_seconds()) / 60.0
            if skew_min > 30:
                log.warning("paper-trigger: %s occurrence_datetime %s disagrees with ticker "
                            "first pitch %s by %.0f min; trusting the ticker.",
                            r.ticker, r.game_datetime.isoformat(), ticker_fp.isoformat(), skew_min)
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
            log.warning("paper-trigger: %s has no game_datetime (occurrence_datetime missing); "
                        "cannot time-trigger.", r.ticker)
            continue
        if not in_window:
            continue                          # too early (or already started) -- normal, no log
        # Matchup label is best-effort: a flaky MLB Stats API must never stall or crash
        # the trigger. On any failure, fall back to the headline -- the bet still records
        # correctly; only the display label degrades.
        try:
            g = match_game(r.ticker, clients, sched_cache)
        except Exception:
            log.warning("paper-trigger: match_game failed for %s; using fallback label.",
                        r.ticker, exc_info=True)
            g = None
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
        ledger.record(bet)
        placed.append(bet)
        log.info("PAPER BET (%.0f min before): %s %s @ %.2f  edge %s  conf %.2f",
                 minutes, r.side, bet["label"], r.entry_price, r.edge_cents, r.confidence)
    return placed
