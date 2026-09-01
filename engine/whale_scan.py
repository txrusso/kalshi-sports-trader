"""Live 'whale tape' — surface the large prints on Kalshi's sports markets right now.

Kalshi is a centralized, CFTC-regulated exchange: trades are ANONYMOUS. There is no
account id on a trade, no per-trader P&L, and no public leaderboard (unlike on-chain
venues like Polymarket). So "track the top 30 accounts and follow them" is not possible
on Kalshi -- the identities simply aren't in the data.

What IS possible, and what this does: detect large prints from the anonymized tape. A
single print that is large relative to a market's own median trade size is very likely a
sophisticated / sized account acting on conviction, even though we can't say WHOSE. This
is the practical proxy for 'follow the whales', and it feeds the same
`signals.money_flow.large_print_threshold` detector the live trade-flow signal now uses,
so what you see here is exactly what the model up-weights.

    py -3 cli.py whales                 # today's slate, ranked by whale activity
    py -3 cli.py whales --top 40         # more markets
    py -3 cli.py whales --min-size 100   # only prints of >= 100 contracts

Read-only. Nothing here places, previews, or suggests an order.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from config.settings import EASTERN, Settings, DEFAULTS
from engine.scanner import _candidate_markets
from kalshi.client import KalshiClient
from kalshi.normalize import MarketQuote, Trade, parse_trades
from signals.money_flow import is_large_print, large_print_threshold


@dataclass
class LargePrint:
    ticker: str
    title: str
    ts: Optional[datetime]
    taker_side: str          # "yes" / "no" -- the aggressor's side
    size: float
    yes_price: float
    is_block: bool

    @property
    def notional(self) -> float:
        # Actual dollars the aggressor paid: YES cost = yes_price, NO cost = 1 - yes_price.
        px = self.yes_price if self.taker_side == "yes" else (1.0 - self.yes_price)
        return self.size * px


@dataclass
class MarketWhales:
    q: MarketQuote
    threshold: float
    prints: list[LargePrint]

    @property
    def whale_yes_d(self) -> float:
        return sum(p.notional for p in self.prints if p.taker_side == "yes")

    @property
    def whale_no_d(self) -> float:
        return sum(p.notional for p in self.prints if p.taker_side == "no")

    @property
    def net_lean(self) -> float:
        tot = self.whale_yes_d + self.whale_no_d
        return (self.whale_yes_d - self.whale_no_d) / tot if tot > 0 else 0.0

    @property
    def whale_dollars(self) -> float:
        return self.whale_yes_d + self.whale_no_d


def scan_whales(client: KalshiClient, settings: Settings = DEFAULTS,
                max_markets: int = 40, min_size: float = 0.0) -> list[MarketWhales]:
    """Deep-scan the richest games' markets and collect their large prints.

    Ranks events by liquidity (same basis the real scanner uses), deep-scans up to
    `max_markets` markets to bound API load, and returns one MarketWhales per market
    that had at least one qualifying large print.
    """
    quotes = [q for q in _candidate_markets(client, settings) if q.has_two_sided_quote]

    # Rank whole events (games) by liquidity, take the richest until the market budget.
    events: dict[str, list[MarketQuote]] = {}
    for q in quotes:
        events.setdefault(q.event_ticker, []).append(q)
    ranked = sorted(events.values(),
                    key=lambda ms: sum(q.volume + q.open_interest for q in ms), reverse=True)
    deep: list[MarketQuote] = []
    for ms in ranked:
        if len(deep) + len(ms) > max_markets:
            break
        deep.extend(ms)

    out: list[MarketWhales] = []
    for q in deep:
        try:
            trades: list[Trade] = parse_trades(
                client.get_trades(q.ticker, limit=settings.trade_flow_lookback))
        except Exception:
            continue
        thr = large_print_threshold(trades, settings)
        prints = [
            LargePrint(ticker=q.ticker, title=q.title, ts=t.ts, taker_side=t.taker_side,
                       size=t.size, yes_price=t.yes_price, is_block=t.is_block)
            for t in trades
            if t.taker_side in ("yes", "no") and is_large_print(t, thr) and t.size >= min_size
        ]
        if prints:
            out.append(MarketWhales(q=q, threshold=thr, prints=prints))
    out.sort(key=lambda m: m.whale_dollars, reverse=True)
    return out


def _fmt_ts(ts: Optional[datetime]) -> str:
    if not ts:
        return "  --  "
    return ts.astimezone(EASTERN).strftime("%m/%d %I:%M%p")


def render(scanned: list[MarketWhales], top_prints: int = 25) -> str:
    lines: list[str] = []
    if not scanned:
        return ("No large prints on the current slate.\n"
                "(No qualifying whale-sized trades vs each market's median trade size. "
                "Note: Kalshi trades are anonymous — this is size-based inference, not account tracking.)")

    lines.append(f"\n=== WHALE TAPE — {len(scanned)} markets with large prints ===")
    lines.append("Anonymous exchange: these are size-inferred whales, not tracked accounts.\n")

    # View 1: markets ranked by whale dollars, with the net whale lean.
    lines.append(f"{'MARKET':<34}{'PRINTS':>7}{'WHALE$':>9}{'LEAN':>7}{'THRESH':>8}  DIRECTION")
    lines.append("-" * 92)
    for m in scanned:
        lean = m.net_lean
        arrow = "→ YES" if lean > 0.08 else ("→ NO" if lean < -0.08 else "  mixed")
        thr = "--" if m.threshold == float("inf") else f"{m.threshold:.0f}ct"
        lines.append(f"{m.q.ticker:<34}{len(m.prints):>7}{m.whale_dollars:>9.0f}"
                     f"{lean:>+7.2f}{thr:>8}  {arrow}")

    # View 2: the biggest individual prints across the slate.
    all_prints = sorted((p for m in scanned for p in m.prints),
                        key=lambda p: p.notional, reverse=True)[:top_prints]
    lines.append(f"\n--- Biggest individual prints (top {len(all_prints)}) ---")
    lines.append(f"{'WHEN (ET)':<14}{'MARKET':<32}{'AGGRESSOR':>18}{'SIZE':>7}{'$':>8}")
    lines.append("-" * 91)
    for p in all_prints:
        block = " [block]" if p.is_block else ""
        act = f"bought {p.taker_side.upper()} @ {p.yes_price:.2f}" if p.taker_side == "yes" \
            else f"bought NO @ {1 - p.yes_price:.2f}"
        lines.append(f"{_fmt_ts(p.ts):<14}{p.ticker[:31]:<32}{act:>18}{p.size:>7.0f}"
                     f"{p.notional:>8.0f}{block}")

    lines.append("\nLEAN = whale $ tilt within the market (+YES / −NO). This is the same large-print "
                 "signal now folded into trade flow; `inspect <TICKER>` shows a market's blended read.")
    return "\n".join(lines)


def run(settings: Settings = DEFAULTS, max_markets: int = 40, min_size: float = 0.0) -> str:
    client = KalshiClient(settings)
    scanned = scan_whales(client, settings, max_markets=max_markets, min_size=min_size)
    return render(scanned)
