"""The 'follow the money' signal.

Combines three Kalshi-native measures of where capital is leaning on a market:

  1. book_imbalance  -- dollar-weighted resting depth, decayed by distance from
                        the touch so far-off market-maker walls don't dominate.
  2. trade_flow      -- net aggressor (taker) dollar volume, recency-weighted;
                        LARGE PRINTS ("whales") get extra weight (big money). A print
                        is large if its size is a big multiple of the market's own
                        median trade size, or Kalshi flags it as a formal block trade.
  3. oi_momentum     -- change in open interest in the direction of price change
                        (new conviction money vs. churn). Needs a prior snapshot.

Each component and the composite are in [-1, +1]:
  +1 => money strongly leaning YES,  -1 => strongly leaning NO.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

from config.settings import Settings, DEFAULTS
from kalshi.normalize import MarketQuote, OrderBook, Trade


@dataclass
class MoneyFlow:
    score: float                       # composite, [-1, +1]
    book_imbalance: float
    trade_flow: float
    oi_momentum: float
    strength: float                    # 0..1 magnitude / conviction
    components: dict = field(default_factory=dict)

    @property
    def direction(self) -> str:
        if self.score > 0.05:
            return "YES"
        if self.score < -0.05:
            return "NO"
        return "FLAT"


def _clamp(x: float, lo: float = -1.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


def book_weights(ob: OrderBook, settings: Settings, decay_cents: float = 3.0) -> tuple[float, float]:
    """Near-touch, distance-decayed resting capital on each side (yes-support, no-support)."""
    mid = ob.mid
    if not mid:
        return 0.0, 0.0
    levels = settings.book_depth_levels

    def weighted(side_levels, to_yes_price):
        total = 0.0
        for p, s in side_levels[-levels:]:               # `levels` closest-to-touch entries
            yes_equiv = to_yes_price(p)
            dist = abs(yes_equiv - mid) * 100.0          # cents from mid
            total += p * s * math.exp(-dist / decay_cents)  # capital * proximity
        return total

    yes_wt = weighted(ob.yes_levels, lambda p: p)         # yes bid price is a yes price
    no_wt = weighted(ob.no_levels, lambda p: 1.0 - p)     # no bid -> yes-equivalent
    return yes_wt, no_wt


def book_imbalance(yes_wt: float, no_wt: float,
                   sibling_yes_wt: Optional[float] = None) -> tuple[float, dict]:
    """Resting-depth imbalance for this market's YES side, in [-1, 1].

    Kalshi books are structurally NO-heavy (market makers rest large NO walls),
    so a raw within-market imbalance skews negative and is weakly directional.
    The primary measure is CROSS-MARKET: this team's near-touch YES support vs the
    opposing team's market's YES support. Within-market imbalance is a secondary tilt.
    """
    within_tot = yes_wt + no_wt
    within = (yes_wt - no_wt) / within_tot if within_tot > 0 else 0.0

    if sibling_yes_wt is None:
        return _clamp(within), {"mode": "within", "yes_wt": round(yes_wt, 1),
                                "no_wt": round(no_wt, 1), "within": round(within, 3)}

    cross_tot = yes_wt + sibling_yes_wt
    cross = (yes_wt - sibling_yes_wt) / cross_tot if cross_tot > 0 else 0.0
    score = _clamp(0.6 * cross + 0.4 * within)
    return score, {"mode": "cross", "yes_wt": round(yes_wt, 1),
                   "sibling_yes_wt": round(sibling_yes_wt, 1),
                   "cross": round(cross, 3), "within": round(within, 3)}


def _median(vals: list[float]) -> float:
    s = sorted(vals)
    n = len(s)
    if n == 0:
        return 0.0
    mid = n // 2
    return s[mid] if n % 2 else (s[mid - 1] + s[mid]) / 2.0


def large_print_threshold(trades: list[Trade], settings: Settings = DEFAULTS) -> float:
    """Contract-size cutoff above which a single print is treated as a "whale".

    Kalshi trades are anonymous (no account id), so a whale can only be inferred from
    a print that is large RELATIVE TO THIS MARKET'S OWN flow: threshold = the larger of
    (median trade size x mult) and an absolute contract floor. The floor stops a market
    whose median size is ~1 contract from flagging routine 5-lots as whales.
    """
    sizes = [t.size for t in trades if t.size > 0]
    if not sizes:
        return float("inf")
    return max(settings.large_print_min_contracts, _median(sizes) * settings.large_print_mult)


def is_large_print(t: Trade, threshold: float) -> bool:
    """A print is a whale if Kalshi tagged it a formal block, or it clears the
    size-relative threshold from `large_print_threshold`."""
    return bool(t.is_block) or t.size >= threshold


def aggressive_dollars(trades: list[Trade],
                       settings: Settings = DEFAULTS) -> tuple[float, float, dict]:
    """Recency- and whale-weighted aggressive (taker) dollars, split yes/no.

    Trades come newest-first from the API, so index 0 is the most recent. Large prints
    ("whales", see `large_print_threshold`) get `settings.large_print_weight`x weight so a
    single conviction block outweighs a crowd of retail clicks. Returns (yes_d, no_d, lp)
    where `lp` summarizes the large prints for display/recording (UNWEIGHTED notional, so
    it reads as real dollars, not the weighted signal input).
    """
    thr = large_print_threshold(trades, settings)
    w_large = settings.large_print_weight
    yes_d = no_d = 0.0
    lp = {"n": 0, "yes_d": 0.0, "no_d": 0.0, "max_size": 0.0,
          "threshold": None if thr == float("inf") else round(thr, 1)}
    n = len(trades)
    for i, t in enumerate(trades):
        recency = math.exp(-i / max(n / 2.0, 1.0))        # newest ~1.0, decays
        large = is_large_print(t, thr)
        w = w_large if large else 1.0
        yes_price = t.yes_price or 0.5
        # Kalshi always reports the trade price in YES terms. A NO-side taker's
        # actual per-contract cost is the complement (1 - yes_price), not yes_price.
        if t.taker_side == "yes":
            yes_d += t.size * yes_price * recency * w
        elif t.taker_side == "no":
            no_d += t.size * (1.0 - yes_price) * recency * w
        else:
            continue
        if large:
            lp["n"] += 1
            lp["max_size"] = max(lp["max_size"], t.size)
            if t.taker_side == "yes":
                lp["yes_d"] += t.size * yes_price
            else:
                lp["no_d"] += t.size * (1.0 - yes_price)
    lp["yes_d"] = round(lp["yes_d"], 1)
    lp["no_d"] = round(lp["no_d"], 1)
    lp["net"] = round((lp["yes_d"] - lp["no_d"]) / (lp["yes_d"] + lp["no_d"]), 3) \
        if (lp["yes_d"] + lp["no_d"]) > 0 else 0.0
    return yes_d, no_d, lp


def trade_flow(self_yes_d: float, self_no_d: float,
               sibling_yes_d: Optional[float] = None) -> tuple[float, dict]:
    """Directional aggressive-money signal for this market's YES side, in [-1, 1].

    Kalshi team markets are structurally YES-heavy (retail lifts offers), so a raw
    per-market net is almost always positive and non-directional. The primary
    measure is therefore CROSS-MARKET: this team's aggressive YES dollars vs the
    opposing team's market's aggressive YES dollars -- i.e. which side of the game
    the money is actually piling into. The within-market net is a secondary tilt.
    """
    within_tot = self_yes_d + self_no_d
    within = (self_yes_d - self_no_d) / within_tot if within_tot > 0 else 0.0

    if sibling_yes_d is None:
        return _clamp(within), {"mode": "within", "self_yes_d": round(self_yes_d, 1),
                                "self_no_d": round(self_no_d, 1), "within": round(within, 3)}

    cross_tot = self_yes_d + sibling_yes_d
    cross = (self_yes_d - sibling_yes_d) / cross_tot if cross_tot > 0 else 0.0
    score = _clamp(0.65 * cross + 0.35 * within)
    return score, {"mode": "cross", "self_yes_d": round(self_yes_d, 1),
                   "sibling_yes_d": round(sibling_yes_d, 1),
                   "cross": round(cross, 3), "within": round(within, 3)}


def oi_momentum(oi_now: float, mid_now: float,
                oi_prev: Optional[float], mid_prev: Optional[float]) -> tuple[float, dict]:
    if oi_prev is None or mid_prev is None or oi_now <= 0:
        return 0.0, {"available": False}
    d_oi = oi_now - oi_prev
    d_price = mid_now - mid_prev
    if abs(d_price) < 1e-6 or d_oi <= 0:
        # OI flat/falling, or no price move => no fresh directional conviction.
        return 0.0, {"available": True, "d_oi": round(d_oi, 1), "d_price": round(d_price, 4)}
    frac_new = _clamp(d_oi / oi_now, 0.0, 1.0)            # share of book that is new money
    score = _clamp(math.copysign(frac_new, d_price))     # new money in price direction
    return score, {"available": True, "d_oi": round(d_oi, 1), "d_price": round(d_price, 4),
                   "frac_new": round(frac_new, 3)}


def compute_money_flow(quote: MarketQuote, ob: OrderBook, trades: list[Trade],
                       prev: Optional[dict] = None, sibling_yes_d: Optional[float] = None,
                       sibling_yes_wt: Optional[float] = None,
                       settings: Settings = DEFAULTS) -> MoneyFlow:
    yes_wt, no_wt = book_weights(ob, settings)
    bi, bi_dbg = book_imbalance(yes_wt, no_wt, sibling_yes_wt)
    self_yes_d, self_no_d, lp = aggressive_dollars(trades, settings)
    tf, tf_dbg = trade_flow(self_yes_d, self_no_d, sibling_yes_d)
    oi_prev = prev.get("open_interest") if prev else None
    mid_prev = prev.get("mid") if prev else None
    oim, oi_dbg = oi_momentum(quote.open_interest, quote.mid, oi_prev, mid_prev)

    w1, w2, w3 = settings.w_book_imbalance, settings.w_trade_flow, settings.w_oi_momentum
    # If OI momentum is unavailable, renormalize over the two available components.
    if not oi_dbg.get("available"):
        wsum = w1 + w2
        score = (w1 * bi + w2 * tf) / wsum if wsum else 0.0
    else:
        wsum = w1 + w2 + w3
        score = (w1 * bi + w2 * tf + w3 * oim) / wsum if wsum else 0.0
    score = _clamp(score)

    # Strength: agreement across components + trade participation.
    comps = [bi, tf] + ([oim] if oi_dbg.get("available") else [])
    agreement = 0.0
    if comps:
        same_sign = all(c >= 0 for c in comps) or all(c <= 0 for c in comps)
        agreement = (1.0 if same_sign else 0.4) * (sum(abs(c) for c in comps) / len(comps))
    strength = _clamp(agreement, 0.0, 1.0)

    return MoneyFlow(
        score=round(score, 4),
        book_imbalance=round(bi, 4),
        trade_flow=round(tf, 4),
        oi_momentum=round(oim, 4),
        strength=round(strength, 4),
        components={"book": bi_dbg, "trades": tf_dbg, "oi": oi_dbg, "large_prints": lp},
    )
