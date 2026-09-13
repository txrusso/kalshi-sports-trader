"""One full scan cycle: markets -> money flow + fair value -> ranked recommendations."""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Optional

from config.settings import Settings, DEFAULTS
from config.sports import is_spread, is_total, market_kind, sport_of
from data.fair_value import FairValueRouter
from kalshi.client import KalshiClient
from kalshi.normalize import MarketQuote, OrderBook, parse_market, parse_trades
from signals.calibration import Calibration
from signals.money_flow import aggressive_dollars, book_weights, compute_money_flow
from signals.recommendation import Recommendation, build_recommendation

log = logging.getLogger("engine.scanner")


class ScanResult:
    def __init__(self):
        self.recommendations: list[Recommendation] = []
        self.snapshot_rows: list[dict] = []
        self.new_state: dict[str, dict] = {}
        self.scanned = 0
        self.deep_scanned = 0


def _candidate_markets(client: KalshiClient, settings: Settings) -> list[MarketQuote]:
    seen: dict[str, MarketQuote] = {}
    for prefix in settings.sport_series_prefixes:
        if is_total(prefix) and not settings.include_totals:
            continue
        if is_spread(prefix) and not settings.include_spreads:
            continue
        try:
            for m in client.get_markets(series_ticker=prefix, status=settings.market_status):
                q = parse_market(m)
                if not q.ticker:
                    continue
                # Totals: keep only near-the-money O/U lines (skip deep ITM/OTM rungs).
                if is_total(q.ticker):
                    if not (settings.totals_min_mid <= q.mid <= settings.totals_max_mid):
                        continue
                # Spread: same idea, but more important -- each event has up to
                # ~24 rungs (12 lines x 2 teams), far more than totals' 8-10, so
                # skipping the deep ITM/OTM rungs matters even more here for the
                # deep-scan budget (settings.max_deep_markets).
                if is_spread(q.ticker):
                    if not (settings.spread_min_mid <= q.mid <= settings.spread_max_mid):
                        continue
                seen[q.ticker] = q
        except Exception as e:  # one bad series shouldn't kill the cycle
            log.warning("get_markets(%s) failed: %s", prefix, e)
    return list(seen.values())


def run_scan(client: KalshiClient, fair_router: FairValueRouter,
             prev_state: dict[str, dict], settings: Settings = DEFAULTS,
             calibration: Optional[Calibration] = None) -> ScanResult:
    result = ScanResult()
    fair_router.clear_caches()

    quotes = _candidate_markets(client, settings)
    result.scanned = len(quotes)

    # Select deep-scan targets by EVENT (game), not by individual market, so both
    # sides of a game are always scanned together -- required for the cross-market
    # trade-flow signal. Rank events by total liquidity, take the richest games.
    tradeable = [q for q in quotes if q.has_two_sided_quote]
    events: dict[str, list[MarketQuote]] = {}
    for q in tradeable:
        events.setdefault(q.event_ticker, []).append(q)
    ranked_events = sorted(
        events.values(),
        key=lambda ms: sum(q.volume + q.open_interest for q in ms),
        reverse=True,
    )
    deep: list[MarketQuote] = []
    for ms in ranked_events:
        if len(deep) + len(ms) > settings.max_deep_markets:
            break
        deep.extend(ms)
    result.deep_scanned = len(deep)
    log.info("Scanned %d markets; %d tradeable; deep-scanning %d markets across %d games.",
             result.scanned, len(tradeable), len(deep),
             len({q.event_ticker for q in deep}))

    # Pass 1: fetch book + trades for every deep market and compute this market's
    # aggressive (taker) dollar split. Needed before money flow so we can pair the
    # two sides of each game (event) for the cross-market trade-flow signal.
    fetched: dict[str, dict] = {}
    for q in deep:
        try:
            ob = OrderBook.from_raw(client._get(f"/markets/{q.ticker}/orderbook",
                                                {"depth": settings.book_depth_levels}))
            trades = parse_trades(client.get_trades(q.ticker, limit=settings.trade_flow_lookback))
        except Exception as e:
            log.warning("Deep scan failed for %s: %s", q.ticker, e)
            continue
        yes_d, _no_d, _lp = aggressive_dollars(trades, settings)
        yes_wt, _no_wt = book_weights(ob, settings)
        fetched[q.ticker] = {"q": q, "ob": ob, "trades": trades,
                             "yes_d": yes_d, "yes_wt": yes_wt}

    # Sibling YES-dollars per market: total YES-dollars of the OTHER markets in
    # the same event (the opposing team), for the cross-market comparison.
    by_event: dict[str, list[str]] = {}
    for tk, d in fetched.items():
        by_event.setdefault(d["q"].event_ticker, []).append(tk)

    # Pass 2: money flow + fair value + recommendation.
    for tk, d in fetched.items():
        q, ob, trades = d["q"], d["ob"], d["trades"]
        # Cross-market pairing applies ONLY to winner markets (team A vs team B,
        # a clean 2-sided complementary pair). Totals AND spread lines share an
        # event across many rungs (spread: up to ~24, one team+line per market)
        # that aren't simple opposites of each other, so no sibling pairing for
        # either — the within-market flow already captures the two sides of that
        # one line/rung.
        sibling_yes_d = sibling_yes_wt = None
        if market_kind(tk) == "winner":
            siblings = [t for t in by_event.get(q.event_ticker, []) if t != tk and market_kind(t) == "winner"]
            if siblings:
                sibling_yes_d = sum(fetched[s]["yes_d"] for s in siblings)
                sibling_yes_wt = sum(fetched[s]["yes_wt"] for s in siblings)

        prev = prev_state.get(q.ticker)
        try:
            mf = compute_money_flow(q, ob, trades, prev=prev, sibling_yes_d=sibling_yes_d,
                                    sibling_yes_wt=sibling_yes_wt, settings=settings)
            fv = fair_router.estimate(q.ticker, q)
            rec = build_recommendation(q, mf, fv, settings, calibration)
        except Exception:
            # One bad market (e.g. a model bug on a single ticker) must never take
            # down the whole cycle -- confirmed 2026-08-30: an NFL-totals NameError
            # silently killed ~15 consecutive cycles (~7.5h) overnight because this
            # loop had no per-ticker isolation. Skip just this ticker and keep going.
            log.exception("Skipping %s this cycle -- money flow/fair value/recommendation failed.", tk)
            continue

        result.new_state[q.ticker] = {"mid": q.mid, "open_interest": q.open_interest,
                                      "ts": datetime.now(timezone.utc).isoformat()}
        result.snapshot_rows.append({
            "ticker": q.ticker, "title": q.title, "yes_team": q.yes_sub_title,
            "sport": sport_of(q.ticker),
            "mid": q.mid, "yes_bid": q.yes_bid, "yes_ask": q.yes_ask,
            "volume": q.volume, "open_interest": q.open_interest, "spread_cents": q.spread_cents,
            "mf_score": mf.score, "mf_book": mf.book_imbalance, "mf_trades": mf.trade_flow,
            "mf_oi": mf.oi_momentum, "mf_strength": mf.strength,
            # Large-print ("whale") record -- accumulates the raw per-market whale data the
            # detector can be walk-forward-validated on later (see config.settings note).
            "lp_n": (mf.components.get("large_prints") or {}).get("n", 0),
            "lp_net": (mf.components.get("large_prints") or {}).get("net", 0.0),
            "lp_max": (mf.components.get("large_prints") or {}).get("max_size", 0.0),
            "lp_thr": (mf.components.get("large_prints") or {}).get("threshold"),
            "fair_prob": fv.prob, "fair_source": fv.source, "fair_conf": fv.confidence,
            "rec_side": rec.side if rec else None,
            "rec_edge_cents": rec.edge_cents if rec else None,
            "rec_confidence": rec.confidence if rec else None,
            "rec_headline": rec.headline if rec else None,
        })
        if rec:
            result.recommendations.append(rec)

    # Dedupe to one recommendation per game: NO-on-A and YES-on-B are the same
    # economic bet in a no-tie market, so keep only the higher-confidence side.
    best_per_event: dict[str, Recommendation] = {}
    for r in result.recommendations:
        ev = r.ticker.rsplit("-", 1)[0]
        cur = best_per_event.get(ev)
        if cur is None or r.rank_key() > cur.rank_key():
            best_per_event[ev] = r
    deduped = list(best_per_event.values())

    deduped.sort(key=lambda r: r.rank_key(), reverse=True)
    result.recommendations = deduped[: settings.top_n_recommendations]
    return result
