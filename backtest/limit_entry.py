"""Does resting a LIMIT order (vs paying the ask) actually improve realized ROI?

Today the model/paper ledger assume a TAKER fill: you pay `yes_ask` (or `no_ask`) at the
last pregame cycle. ROI is brutally entry-price-sensitive -- a 62% win-prob bet is +1.2c
edge at 0.61 but +4.2c at 0.58 -- so shaving cents off entry could turn no-trades into
trades. The catch is ADVERSE SELECTION: a resting buy fills most easily exactly when the
price is moving AGAINST you (you fill on the losers, miss the winners). Cheaper entries
(good) vs adverse selection (bad) -- only a backtest settles which wins.

What this does: for every game the model would bet, it reconstructs the pregame time
series of the entry-side ask from `snapshots/*.jsonl` and simulates resting a limit
`offset` cents inside the ask, placed at the FIRST pregame snapshot (max lead time). A
fill is recorded if the entry-side ask ever trades down to the limit at a later snapshot.
Then it grades three policies against real settled outcomes:
  * taker      -- pay the ask at the last pregame snapshot (today's behavior; always fills)
  * limit_skip -- rest the limit; if it never fills, place NO bet
  * limit_fb   -- rest the limit; if it never fills, fall back to the taker ask
...and reports fill rate, the adverse-selection gap (filled win% vs unfilled win%), and
the average entry-price savings on fills.

Fill proxy & assumptions (stated plainly, since they bound the result):
  * Snapshots are ~20-30 min apart, so fills are checked only at those discrete points --
    this UNDERSTATES fills (intra-cycle dips are invisible). A real resting order fills
    more often than this shows; read fill rates as a conservative floor.
  * Placement = first pregame snapshot (most generous lead time). A later placement would
    fill less. The side bet is the model's own pick at the LAST pregame snapshot, so all
    three policies bet the identical games -- an apples-to-apples entry-method comparison.
  * Only games with >=2 pregame snapshots are compared (need a window to observe a fill);
    the excluded short-window count is reported.

Same data limit as backtest/replay_moneyflow.py: this can only run on the forward
snapshot history (Kalshi doesn't expose point-in-time books historically), so it's
in-sample on one partial 2026 season -- read RELATIVE policy differences, not absolute ROI.

Usage:
    py -3 -m backtest.limit_entry                       # offsets 0.5/1/2/3c
    py -3 -m backtest.limit_entry --offsets 1 2 3 4
    py -3 -m backtest.limit_entry --since 2026-08-15
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass

from config.sports import is_total
from backtest.evaluate import load_rows, resolve_outcomes, build_clients
from backtest.replay_moneyflow import PROD_WEIGHTS, replay_rec, _PREGAME_SOURCES


def _entry_ask(r: dict, side: str) -> float:
    """Cost to BUY the recommended side at this snapshot (the entry-side ask).
    NO side isn't stored; use the standard no-tie convention (1 - best YES bid),
    identical to backtest/evaluate.py::_entry and replay_moneyflow's quote rebuild."""
    return r["yes_ask"] if side == "YES" else round(1 - r["yes_bid"], 4)


@dataclass
class Bet:
    ticker: str
    side: str
    conf: float
    taker_entry: float      # entry-side ask at the LAST pregame snapshot (today's fill)
    series: list[float]     # chronological entry-side ask, pregame only, valid quotes
    won: bool               # did the BET side win (already resolved from outcome + side)
    is_total: bool


def build_bets(rows: list[dict], outcomes: dict[str, bool], weights=PROD_WEIGHTS) -> tuple[list[Bet], int]:
    """One Bet per game the model would place, carrying its pregame ask series.
    Returns (bets, n_excluded_short_window)."""
    by_ticker: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        if r.get("fair_source") in _PREGAME_SOURCES and r["ticker"] in outcomes:
            by_ticker[r["ticker"]].append(r)

    candidates: list[Bet] = []
    short = 0
    for tk, rs in by_ticker.items():
        rs.sort(key=lambda r: r.get("cycle_ts") or "")
        rec = replay_rec(rs[-1], weights)          # model's bet at the last pregame snapshot
        if rec is None:
            continue
        if not (0 < rec.entry_price < 1):
            continue
        series = [_entry_ask(r, rec.side) for r in rs
                  if (r.get("yes_bid") or 0) > 0 and (r.get("yes_ask") or 0) > 0]
        if len(series) < 2:
            short += 1
            continue
        won = outcomes[tk] if rec.side == "YES" else (not outcomes[tk])
        candidates.append(Bet(tk, rec.side, rec.confidence, rec.entry_price, series, won, is_total(tk)))

    # Dedupe to one bet per game (event) by confidence -- mirrors the scanner's per-event
    # dedupe so NO-on-A / YES-on-B don't double-count.
    best: dict[str, Bet] = {}
    for b in candidates:
        ev = b.ticker.rsplit("-", 1)[0]
        if ev not in best or b.conf > best[ev].conf:
            best[ev] = b
    return list(best.values()), short


def _simulate_fill(bet: Bet, offset: float, placement: str) -> tuple[bool, float]:
    """Rest a limit `offset` dollars inside the ask; fill if a later snapshot's entry-side
    ask reaches it. The taker baseline always pays the LAST pregame ask (bet.taker_entry),
    so `placement` sets only where the limit is anchored and how long it can rest:
      * "first"       -- anchor at the first pregame ask, observe the whole window (max lead
                         time, but the limit is pegged to a possibly-stale early price).
      * "penultimate" -- anchor at the second-to-last ask, observe only the final cycle
                         (a realistic 'peg a limit one cycle before you'd take it' test;
                         apples-to-apples against paying the last-cycle ask)."""
    if placement == "penultimate":
        place_ask, window = bet.series[-2], bet.series[-1:]
    else:
        place_ask, window = bet.series[0], bet.series[1:]
    limit = max(0.01, round(place_ask - offset, 4))
    filled = any(a <= limit for a in window)
    return filled, limit


def _acc(s: dict, entry: float, won: bool) -> None:
    s["n"] += 1
    s["wins"] += int(won)
    s["pnl"] += (1 - entry) if won else -entry
    s["staked"] += entry


def _roi(s: dict):
    return round(s["pnl"] / s["staked"], 4) if s["staked"] else None


def _wr(s: dict):
    return round(s["wins"] / s["n"], 3) if s["n"] else None


def grade(bets: list[Bet], offset: float, placement: str) -> dict:
    pol = {p: {"n": 0, "wins": 0, "pnl": 0.0, "staked": 0.0}
           for p in ("taker", "limit_skip", "limit_fb")}
    by_kind = {p: {k: {"n": 0, "wins": 0, "pnl": 0.0, "staked": 0.0}
                   for k in ("winner", "total")} for p in ("taker", "limit_fb")}
    filled_n = filled_wins = 0
    unfilled_n = unfilled_wins = 0
    savings = 0.0

    for b in bets:
        filled, limit = _simulate_fill(b, offset, placement)
        kind = "total" if b.is_total else "winner"
        _acc(pol["taker"], b.taker_entry, b.won)
        _acc(by_kind["taker"][kind], b.taker_entry, b.won)
        if filled:
            _acc(pol["limit_skip"], limit, b.won)
            _acc(pol["limit_fb"], limit, b.won)
            _acc(by_kind["limit_fb"][kind], limit, b.won)
            filled_n += 1
            filled_wins += int(b.won)
            savings += b.taker_entry - limit
        else:
            _acc(pol["limit_fb"], b.taker_entry, b.won)
            _acc(by_kind["limit_fb"][kind], b.taker_entry, b.won)
            unfilled_n += 1
            unfilled_wins += int(b.won)

    n = len(bets)
    return {
        "offset_c": round(offset * 100, 1),
        "n": n,
        "fill_rate": round(filled_n / n, 3) if n else None,
        "avg_savings_c": round(savings / filled_n * 100, 2) if filled_n else None,
        "filled_wr": round(filled_wins / filled_n, 3) if filled_n else None,
        "unfilled_wr": round(unfilled_wins / unfilled_n, 3) if unfilled_n else None,
        "taker": {"roi": _roi(pol["taker"]), "wr": _wr(pol["taker"]), "n": pol["taker"]["n"]},
        "limit_skip": {"roi": _roi(pol["limit_skip"]), "wr": _wr(pol["limit_skip"]), "n": pol["limit_skip"]["n"]},
        "limit_fb": {"roi": _roi(pol["limit_fb"]), "wr": _wr(pol["limit_fb"]), "n": pol["limit_fb"]["n"]},
        "by_kind": {p: {k: {"n": by_kind[p][k]["n"], "roi": _roi(by_kind[p][k])}
                        for k in ("winner", "total")} for p in ("taker", "limit_fb")},
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Limit-entry fill-rate backtest")
    ap.add_argument("--since", help="only snapshots on/after YYYY-MM-DD")
    ap.add_argument("--offsets", nargs="+", type=float, default=[0.5, 1.0, 2.0, 3.0],
                    help="limit offsets inside the ask, in CENTS (default 0.5 1 2 3)")
    args = ap.parse_args()

    rows = load_rows(args.since)
    if not rows:
        print("No snapshots found yet.")
        return
    print(f"Loaded {len(rows)} snapshot rows.")
    clients = build_clients()
    tickers = {r["ticker"] for r in rows}
    print(f"Resolving outcomes for {len(tickers)} markets (one network pass)...")
    outcomes = resolve_outcomes(tickers, clients)
    print(f"Resolved {len(outcomes)} settled market-sides.")

    bets, short = build_bets(rows, outcomes, PROD_WEIGHTS)
    if not bets:
        print("No settled, model-recommended games with a fill window yet.")
        return
    print(f"{len(bets)} model bets with a >=2-snapshot fill window "
          f"({short} excluded for a single-snapshot window).\n")

    base = grade(bets, args.offsets[0] / 100.0, "first")
    print(f"TAKER baseline (pay the last-cycle ask, today's behavior): "
          f"n={base['taker']['n']}  win%={base['taker']['wr']}  ROI={base['taker']['roi']}")
    tk = base["by_kind"]["taker"]
    print(f"   by kind -> winner {tk['winner']['n']}/{tk['winner']['roi']}   "
          f"total {tk['total']['n']}/{tk['total']['roi']}")

    for placement, blurb in [("penultimate", "peg a limit ONE CYCLE before you'd take it, "
                                             "vs paying the last-cycle ask (the realistic test)"),
                             ("first", "commit early: anchor at the FIRST pregame ask, rest the "
                                       "whole window (max lead time)")]:
        print(f"\n=== placement = {placement} - {blurb} ===")
        hdr = (f"{'offset':>7}{'fill%':>7}{'save_c':>7}{'fillWR':>7}{'missWR':>7}"
               f"{'skipROI':>9}{'fbROI':>8}{'fb_win':>8}{'fb_tot':>8}")
        print(hdr)
        print("-" * len(hdr))
        for oc in args.offsets:
            m = grade(bets, oc / 100.0, placement)
            fbk = m["by_kind"]["limit_fb"]
            print(f"{m['offset_c']:>6}c{str(m['fill_rate']):>7}{str(m['avg_savings_c']):>7}"
                  f"{str(m['filled_wr']):>7}{str(m['unfilled_wr']):>7}"
                  f"{str(m['limit_skip']['roi']):>9}{str(m['limit_fb']['roi']):>8}"
                  f"{str(fbk['winner']['roi']):>8}{str(fbk['total']['roi']):>8}")

    print("\nfill%   = share of bets whose limit filled (conservative: only checked at ~20-30min snapshots)")
    print("save_c  = avg cents saved on entry vs the taker ask, on filled bets")
    print("fillWR/missWR = win rate of filled vs never-filled bets -- if fillWR << missWR, "
          "limits are adversely selected (filling on losers)")
    print("skipROI = ROI resting the limit and skipping if unfilled;  fbROI = ROI falling back to "
          "the taker ask if unfilled (fb_win/fb_tot = fbROI split by market kind)")
    print("Compare fbROI vs the taker baseline ROI above: that's the net effect of using limits. "
          "In-sample, one partial season -- directional, not significant.")


if __name__ == "__main__":
    main()
