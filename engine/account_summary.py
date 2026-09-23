"""One-line-per-cycle summary of the real Kalshi account: balance plus
settled record/P&L for today and all-time.

"Real account" here means the combined paper + live ledgers -- both represent
real money on the real Kalshi account. Paper-ledger bets were historically
placed manually by the user off the loop's notifications (see CLAUDE.md's
"Live operational setup" section); live-ledger bets are the ones `loop --live`
auto-submits itself. Grading reuses the same settled-outcome logic `cli.py
settle` uses (data.games.resolve_outcomes), and both load bets through
engine.real_bets.load_real_bets -- which drops live orders that never filled --
so this can never disagree with what `settle` reports for the same bets.

Best-effort by design: a balance-fetch or outcome-resolution failure must
never crash the loop's console output, so every failure degrades to
"unavailable" rather than raising.
"""
from __future__ import annotations

import logging
from datetime import datetime

from backtest.evaluate import _game_date
from config.settings import EASTERN
from data.games import resolve_outcomes
from engine.real_bets import load_real_bets
from kalshi.client import KalshiClient

log = logging.getLogger("engine.account_summary")


def _grade(bets: list[dict], outcomes: dict[str, bool]) -> tuple[int, int, float]:
    """(wins, losses, net_usd) over whichever of `bets` have a settled outcome."""
    wins = losses = 0
    net = 0.0
    for b in bets:
        yes_won = outcomes.get(b["ticker"])
        if yes_won is None:
            continue
        won = yes_won if b["side"] == "YES" else (not yes_won)
        net += ((1 - b["entry_price"]) if won else -b["entry_price"]) * (b.get("contracts") or 0)
        wins += int(won)
        losses += int(not won)
    return wins, losses, net


def _fmt(label: str, wins: int, losses: int, net: float) -> str:
    n = wins + losses
    if n == 0:
        return f"{label} no settled bets"
    return f"{label} {wins}-{losses} ({wins / n:.0%})  net {net:+.2f}"


def render(client: KalshiClient, clients: dict) -> str:
    try:
        bal = client.balance()
        cents = bal.get("balance")
        balance = float(cents) / 100.0 if cents is not None else float(bal.get("balance_dollars") or 0)
        bal_str = f"${balance:.2f}"
    except Exception:
        log.warning("Account summary: balance fetch failed.", exc_info=True)
        bal_str = "unavailable"

    try:
        bets = load_real_bets(client)
        today = datetime.now(EASTERN).strftime("%Y-%m-%d")
        outcomes = resolve_outcomes({b["ticker"] for b in bets}, clients)
        today_bets = [b for b in bets if _game_date(b["ticker"]) == today]
        tw, tl, tnet = _grade(today_bets, outcomes)
        ow, ol, onet = _grade(bets, outcomes)
        record_str = f"{_fmt('today', tw, tl, tnet)}   |   {_fmt('overall', ow, ol, onet)}"
    except Exception:
        log.warning("Account summary: ledger grading failed.", exc_info=True)
        record_str = "today/overall record unavailable"

    return f"  ACCOUNT   balance {bal_str}   |   {record_str}"
