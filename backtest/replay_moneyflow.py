"""ROI replay harness for the money-flow weights.

The gap this closes: `backtest/bankroll_sim.py` and `season_backtest.py` pick sides
purely from fair value and bet every edge>=min_edge row -- they apply NEITHER money flow
NOR the confidence gate that production actually uses. So the "follow-the-money" signal
(the design's PRIMARY driver) and the 40/35/25 weights had no ROI validation at all;
`backtest/evaluate.py` only scores an in-sample IC on the blended score, which partly
re-reads the market price and can't tell you whether acting on the signal makes money.

This harness replays the recorded live snapshots through the REAL
`signals.recommendation.build_recommendation` -- the exact selection/gating/sizing logic
that trades -- under an arbitrary money-flow weighting, and grades the realized P&L of the
bets it would have placed, after paying entry prices. So the strategy MEASURED equals the
strategy TRADED, and swapping the weights answers "does trimming OI actually help the
money, not just the IC?".

Why it can only run on snapshots (not the settled-market cache the other backtests use):
money flow needs point-in-time order books + trade flow, which Kalshi does not expose
historically. The only money-flow-bearing record is the forward `snapshots/*.jsonl`
history, which already stores each component (mf_book/mf_trades/mf_oi) and mf_strength.

Method (consistent across weightings, so relative ROI comparisons are valid):
  1. Reconstruct the money-flow blend from stored components under the given weights,
     recovering the per-row "OI available?" renormalization from the stored mf_score
     (strength is weight-independent, so it's reused as-is).
  2. Rebuild the real MoneyFlow/FairValue/MarketQuote and call build_recommendation with
     current production settings (calibration OFF, so the comparison isolates the weights;
     calibration is confidence-only and can't flip a pick anyway).
  3. Take the latest pre-game snapshot per market (closest to the live paper-trigger),
     dedupe to one bet per game by confidence (mirrors the scanner's per-event dedupe),
     and grade flat $1/contract against settled outcomes.

Usage:
    py -3 -m backtest.replay_moneyflow                    # compare candidate weightings
    py -3 -m backtest.replay_moneyflow --weights 0.55,0.45,0   # one custom weighting
    py -3 -m backtest.replay_moneyflow --since 2026-08-01
"""
from __future__ import annotations

import argparse

from config.settings import DEFAULTS
from config.sports import is_total
from data.fair_value import FairValue
from kalshi.normalize import MarketQuote
from signals.money_flow import MoneyFlow, _clamp
from signals.recommendation import build_recommendation
from backtest.evaluate import load_rows, resolve_outcomes, build_clients

PROD_WEIGHTS = (DEFAULTS.w_book_imbalance, DEFAULTS.w_trade_flow, DEFAULTS.w_oi_momentum)
# real pregame sources whose game_state was non-None at scan time (so they pass the
# pregame gate); "none"/"skip_non_pregame" have null prob and never produced a rec, and
# "live" is excluded to reflect the current pregame_only=True policy.
_PREGAME_SOURCES = {"pregame_log5", "pregame_totals", "pregame_elo", "pregame_nfl_totals"}


def _recover_oi_available(bi: float, tf: float, oim: float, stored_score: float) -> bool:
    """Recover the per-row 'OI momentum available?' flag (which drives the money-flow
    renormalization) by matching the stored mf_score against both formulas under the
    production weights. A stored mf_oi of 0 is ambiguous between 'unavailable' and
    'available but flat', so we can't read it directly -- but the two formulas give
    different scores, and only one matches what was actually recorded."""
    wb, wt, wo = PROD_WEIGHTS
    avail = _clamp((wb * bi + wt * tf + wo * oim) / (wb + wt + wo))
    unavail = _clamp((wb * bi + wt * tf) / (wb + wt))
    # stored score is rounded to 4dp; pick the closer formula.
    return abs(avail - stored_score) <= abs(unavail - stored_score)


def _blend(bi: float, tf: float, oim: float, weights, available: bool) -> float:
    wb, wt, wo = weights
    if available:
        wsum = wb + wt + wo
        return _clamp((wb * bi + wt * tf + wo * oim) / wsum) if wsum else 0.0
    wsum = wb + wt
    return _clamp((wb * bi + wt * tf) / wsum) if wsum else 0.0


def _quote_from_row(r: dict) -> MarketQuote:
    yb, ya = r.get("yes_bid") or 0.0, r.get("yes_ask") or 0.0
    # NO side isn't stored; use the standard no-tie convention (same as evaluate.py):
    # buying NO at ask = 1 - best YES bid.
    return MarketQuote(
        ticker=r["ticker"], event_ticker=r["ticker"].rsplit("-", 1)[0],
        title=r.get("title", ""), yes_sub_title=r.get("yes_team", ""), status="active",
        yes_bid=yb, yes_ask=ya,
        no_bid=round(1 - ya, 4) if ya else 0.0, no_ask=round(1 - yb, 4) if yb else 0.0,
        last_price=r.get("mid") or 0.0, volume=r.get("volume") or 0.0, volume_24h=0.0,
        open_interest=r.get("open_interest") or 0.0, liquidity=0.0, close_time=None,
    )


def replay_rec(r: dict, weights):
    """Rebuild the real objects for one snapshot row and run the real recommendation
    logic under `weights`. Returns a Recommendation or None."""
    bi, tf, oim = r.get("mf_book"), r.get("mf_trades"), r.get("mf_oi")
    if bi is None or tf is None or oim is None or r.get("mf_score") is None:
        return None
    available = _recover_oi_available(bi, tf, oim, r["mf_score"])
    score = _blend(bi, tf, oim, weights, available)
    mf = MoneyFlow(score=round(score, 4), book_imbalance=round(bi, 4),
                   trade_flow=round(tf, 4), oi_momentum=round(oim, 4),
                   strength=r.get("mf_strength") or 0.0, components={})
    gs = "Preview" if r.get("fair_source") in _PREGAME_SOURCES else None
    fv = FairValue(prob=r.get("fair_prob"), source=r.get("fair_source") or "none",
                   confidence=r.get("fair_conf") or 0.0, detail={}, game_state=gs)
    q = _quote_from_row(r)
    # calibration OFF: isolates the weight effect; confidence-only, can't flip a pick.
    return build_recommendation(q, mf, fv, DEFAULTS, calibration=None)


def grade(rows: list[dict], outcomes: dict[str, bool], weights) -> dict:
    # latest snapshot per market (closest to the live paper-trigger cutoff)
    latest: dict[str, dict] = {}
    for r in rows:
        k = r["ticker"]
        if k not in latest or (r.get("cycle_ts") or "") > (latest[k].get("cycle_ts") or ""):
            latest[k] = r

    # replay -> keep settled markets that yield a rec -> dedupe one bet/game by confidence
    best: dict[str, tuple] = {}
    for tk, r in latest.items():
        if tk not in outcomes:
            continue
        rec = replay_rec(r, weights)
        if rec is None:
            continue
        entry = rec.entry_price
        if not (0 < entry < 1):
            continue
        ev = tk.rsplit("-", 1)[0]
        if ev not in best or rec.confidence > best[ev][0].confidence:
            best[ev] = (rec, tk)

    n = wins = 0
    pnl = staked = 0.0
    by_kind = {"winner": {"n": 0, "wins": 0, "pnl": 0.0, "staked": 0.0},
               "total": {"n": 0, "wins": 0, "pnl": 0.0, "staked": 0.0}}
    for rec, tk in best.values():
        yes_won = outcomes[tk]
        won = yes_won if rec.side == "YES" else (not yes_won)
        p = (1 - rec.entry_price) if won else -rec.entry_price
        n += 1
        wins += int(won)
        pnl += p
        staked += rec.entry_price
        b = by_kind["total" if is_total(tk) else "winner"]
        b["n"] += 1; b["wins"] += int(won); b["pnl"] += p; b["staked"] += rec.entry_price
    return {
        "n_bets": n, "wins": wins,
        "win_rate": round(wins / n, 3) if n else None,
        "pnl": round(pnl, 2),
        "roi": round(pnl / staked, 4) if staked else None,
        "by_kind": {k: {"n": v["n"], "win_rate": round(v["wins"] / v["n"], 3) if v["n"] else None,
                        "roi": round(v["pnl"] / v["staked"], 4) if v["staked"] else None}
                    for k, v in by_kind.items()},
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="ROI replay harness for money-flow weights")
    ap.add_argument("--since", help="only snapshots on/after this YYYY-MM-DD")
    ap.add_argument("--weights", help="one weighting 'book,trades,oi' (default: compare a set)")
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
    print(f"Resolved {len(outcomes)} settled market-sides.\n")

    if args.weights:
        parts = [float(x) for x in args.weights.split(",")]
        cands = {f"custom {tuple(parts)}": tuple(parts)}
    else:
        cands = {
            "production 0.40/0.35/0.25": (0.40, 0.35, 0.25),
            "drop OI    0.55/0.45/0.00": (0.55, 0.45, 0.00),
            "book-heavy 0.60/0.30/0.10": (0.60, 0.30, 0.10),
            "book-only  1.00/0.00/0.00": (1.00, 0.00, 0.00),
        }
    print(f"{'weighting':<28} {'n':>4} {'win%':>6} {'roi':>8} {'pnl':>8}   by_kind (n/roi)")
    print("-" * 92)
    for name, w in cands.items():
        m = grade(rows, outcomes, w)
        wk = m["by_kind"]
        bk = f"W:{wk['winner']['n']}/{wk['winner']['roi']}  T:{wk['total']['n']}/{wk['total']['roi']}"
        print(f"{name:<28} {m['n_bets']:>4} {str(m['win_rate']):>6} {str(m['roi']):>8} "
              f"{str(m['pnl']):>8}   {bk}")
    print("\nFlat $1/contract on the latest pregame snapshot per game, real settled outcomes, "
          "current production gates (edge 5.5c, min_conf 0.35), calibration off. "
          "In-sample (one partial 2026 season); read relative differences, not absolute ROI.")


if __name__ == "__main__":
    main()
