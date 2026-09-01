"""Parse Kalshi's fixed-point (`_dollars` / `_fp`) payloads into clean structures.

All prices are dollars in [0, 1] (probability-like). All sizes are contract
quantities (floats; Kalshi now reports fractional sizes).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional


def _f(v, default: float = 0.0) -> float:
    if v is None:
        return default
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def position_size(p: dict) -> float:
    """Signed contract count from a /portfolio/positions market_positions entry.

    Kalshi reports this as 'position_fp' (fixed-point size), not 'position'.
    """
    return _f(p.get("position_fp"))


def _parse_ts(s: Optional[str]) -> Optional[datetime]:
    if not s:
        return None
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except ValueError:
        return None


@dataclass
class MarketQuote:
    ticker: str
    event_ticker: str
    title: str
    yes_sub_title: str          # the team this YES market is for
    status: str
    yes_bid: float
    yes_ask: float
    no_bid: float
    no_ask: float
    last_price: float
    volume: float
    volume_24h: float
    open_interest: float
    liquidity: float
    close_time: Optional[datetime]
    rules_primary: str = ""                        # full-text rules, e.g. spells out
                                                     # the matchup by name (NFL needs this --
                                                     # its own title field doesn't have it)
    occurrence_datetime: Optional[datetime] = None  # authoritative event start time
                                                     # (NFL tickers don't encode kickoff time
                                                     # the way MLB's do)

    @property
    def mid(self) -> float:
        if self.yes_bid > 0 and self.yes_ask > 0:
            return (self.yes_bid + self.yes_ask) / 2.0
        return self.last_price or self.yes_ask or self.yes_bid

    @property
    def spread_cents(self) -> float:
        if self.yes_bid > 0 and self.yes_ask > 0:
            return round((self.yes_ask - self.yes_bid) * 100, 2)
        return 100.0  # no two-sided market => treat as untradeable

    @property
    def has_two_sided_quote(self) -> bool:
        return self.yes_bid > 0 and self.yes_ask > 0 and self.yes_ask > self.yes_bid


def parse_market(m: dict) -> MarketQuote:
    return MarketQuote(
        ticker=m.get("ticker", ""),
        event_ticker=m.get("event_ticker", ""),
        title=m.get("title", ""),
        yes_sub_title=m.get("yes_sub_title", ""),
        status=m.get("status", ""),
        yes_bid=_f(m.get("yes_bid_dollars")),
        yes_ask=_f(m.get("yes_ask_dollars")),
        no_bid=_f(m.get("no_bid_dollars")),
        no_ask=_f(m.get("no_ask_dollars")),
        last_price=_f(m.get("last_price_dollars")),
        volume=_f(m.get("volume_fp")),
        volume_24h=_f(m.get("volume_24h_fp")),
        open_interest=_f(m.get("open_interest_fp")),
        liquidity=_f(m.get("liquidity_dollars")),
        close_time=_parse_ts(m.get("close_time")),
        rules_primary=m.get("rules_primary", "") or "",
        occurrence_datetime=_parse_ts(m.get("occurrence_datetime")),
    )


@dataclass
class OrderBook:
    """yes_levels / no_levels are (price, size), sorted ascending by price.

    A 'yes' level is a resting bid to BUY YES at that price.
    A 'no' level is a resting bid to BUY NO at that price.
    """
    yes_levels: list[tuple[float, float]] = field(default_factory=list)
    no_levels: list[tuple[float, float]] = field(default_factory=list)

    @classmethod
    def from_raw(cls, raw: dict) -> "OrderBook":
        ob = raw.get("orderbook_fp") or raw.get("orderbook") or {}
        yes = ob.get("yes_dollars") or ob.get("yes") or []
        no = ob.get("no_dollars") or ob.get("no") or []
        def conv(levels):
            out = []
            for lvl in levels:
                if not lvl:
                    continue
                out.append((_f(lvl[0]), _f(lvl[1])))
            out.sort(key=lambda x: x[0])
            return out
        return cls(yes_levels=conv(yes), no_levels=conv(no))

    @property
    def best_yes_bid(self) -> float:
        return self.yes_levels[-1][0] if self.yes_levels else 0.0

    @property
    def best_no_bid(self) -> float:
        return self.no_levels[-1][0] if self.no_levels else 0.0

    @property
    def best_yes_ask(self) -> float:
        nb = self.best_no_bid
        return round(1.0 - nb, 4) if nb else 0.0

    @property
    def mid(self) -> float:
        yb, ya = self.best_yes_bid, self.best_yes_ask
        if yb and ya:
            return (yb + ya) / 2.0
        return yb or ya

    def dollar_depth(self, side: str) -> float:
        """Total capital resting on a side = sum(price * size)."""
        levels = self.yes_levels if side == "yes" else self.no_levels
        return sum(p * s for p, s in levels)


@dataclass
class Trade:
    ts: Optional[datetime]
    taker_side: str        # "yes" or "no" (the aggressor's outcome side)
    yes_price: float
    size: float
    is_block: bool

    @property
    def signed_yes_size(self) -> float:
        """+size if the aggressor bought YES, -size if bought NO."""
        return self.size if self.taker_side == "yes" else -self.size


def parse_trades(raw: list[dict]) -> list[Trade]:
    out = []
    for t in raw:
        out.append(Trade(
            ts=_parse_ts(t.get("created_time")),
            taker_side=(t.get("taker_side") or t.get("taker_outcome_side") or "").lower(),
            yes_price=_f(t.get("yes_price_dollars")),
            size=_f(t.get("count_fp")),
            is_block=bool(t.get("is_block_trade")),
        ))
    return out
