"""Descriptive stats on the REAL paper ledger -- the slices `cli.py settle` doesn't cut.

`settle` answers "what's the record?"; this answers "where does the money actually come
from?" Same rows, same grading (a bet wins if its side matches the settled outcome), but
sliced by entry price, model edge, confidence, money-flow strength, and week, plus the
real-dollar equity curve (drawdown, streaks) and a predicted-vs-actual probability check
against the bets we really placed.

Every number here is the FORWARD test -- decisions made live near first pitch, graded
after -- so it's the honest read, and also the smallest sample we have (tens of bets, not
thousands). Read the slices as hypotheses to watch, not as findings: at n~10-25 per
bucket one standard error is huge, and the buckets are correlated with each other (bigger
edges get sized bigger, chalk clusters at high prices). Nothing here should change a
production constant on its own -- that's what backtest/bankroll_sim.py's walk-forward
splits are for.

ROI is on stake (pnl per $1 risked, price-weighted), matching `settle`'s ROI column, so
the two agree on the overall line; `net $` is real dollars at the sizes actually bet.

Usage:
    py -3 -m backtest.ledger_stats
    py -3 -m backtest.ledger_stats --sport mlb
    py -3 -m backtest.ledger_stats --sport nba
    py -3 -m backtest.ledger_stats --sport nhl
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from typing import Callable, Optional

from config.sports import market_kind, sport_of
from data.games import build_clients, resolve_outcomes
from engine.real_bets import load_real_bets


def _graded_rows(sport: Optional[str] = None) -> list[dict]:
    """Settled ledger bets, each annotated with its realized result and economics.

    Both ledgers, live orders that never filled dropped (engine/real_bets.py)."""
    bets = load_real_bets()
    if sport:
        bets = [b for b in bets if sport_of(b["ticker"]) == sport.lower()]
    if not bets:
        return []
    outcomes = resolve_outcomes({b["ticker"] for b in bets}, build_clients())
    rows = []
    for b in sorted((x for x in bets if x["ticker"] in outcomes), key=lambda x: x["first_pitch"]):
        yes_won = outcomes[b["ticker"]]
        won = yes_won if b["side"] == "YES" else (not yes_won)
        ct = b.get("contracts") or 0
        price = b["entry_price"]
        rows.append({
            "bet": b,
            "won": won,
            "contracts": ct,
            "price": price,
            "per_contract": (1 - price) if won else -price,
            "net": ((1 - price) if won else -price) * ct,
            "wager": price * ct,
            # Model's probability for the side we actually took (ledger stores P(YES)).
            "fair_side": (b["fair_prob"] if b["side"] == "YES" else 1 - b["fair_prob"])
                         if b.get("fair_prob") is not None else None,
            "kind": market_kind(b["ticker"]),
            "date": b["first_pitch"][:10],
        })
    return rows


def _dollars(rows: list[dict]) -> None:
    wagered = sum(r["wager"] for r in rows)
    net = sum(r["net"] for r in rows)
    print("=== DOLLARS (at the sizes actually bet) ===")
    print(f"  settled bets        : {len(rows)}")
    print(f"  total wagered       : ${wagered:.2f}")
    print(f"  net P&L             : ${net:+.2f}")
    if wagered:
        print(f"  return on $ wagered : {100 * net / wagered:+.1f}%")
    print(f"  avg wager           : ${wagered / len(rows):.2f}   "
          f"avg contracts {sum(r['contracts'] for r in rows) / len(rows):.1f}")
    print(f"  biggest win / loss  : ${max(r['net'] for r in rows):+.2f} / "
          f"${min(r['net'] for r in rows):+.2f}")

    # Real-dollar equity curve: drawdown in the order bets actually settled.
    equity = peak = drawdown = 0.0
    for r in rows:
        equity += r["net"]
        peak = max(peak, equity)
        drawdown = max(drawdown, peak - equity)
    print(f"  max drawdown ($)    : ${drawdown:.2f}  (peak equity ${peak:.2f})")

    best = worst = win_run = loss_run = 0
    for r in rows:
        win_run, loss_run = (win_run + 1, 0) if r["won"] else (0, loss_run + 1)
        best, worst = max(best, win_run), max(worst, loss_run)
    print(f"  longest win streak  : {best}   longest losing streak: {worst}")
    tail = rows[-10:]
    print(f"  last 10 bets        : {sum(r['won'] for r in tail)}-"
          f"{len(tail) - sum(r['won'] for r in tail)}  "
          f"${sum(r['net'] for r in tail):+.2f}")


def _block(title: str, rows: list[dict], key: Callable[[dict], str]) -> None:
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        groups[key(r)].append(r)
    print(f"\n=== {title} ===")
    for name in sorted(groups):
        rs = groups[name]
        wins = sum(r["won"] for r in rs)
        stake = sum(r["price"] for r in rs)
        pnl = sum(r["per_contract"] for r in rs)
        roi = f"{100 * pnl / stake:+6.1f}%" if stake else "   n/a"
        print(f"  {name:<18}n={len(rs):<4}{wins:>3}-{len(rs) - wins:<3}({100 * wins / len(rs):>3.0f}%)"
              f"  ROI {roi}  net ${sum(r['net'] for r in rs):+7.2f}")


def _price_bucket(r: dict) -> str:
    p = r["price"]
    if p < 0.45:
        return "1. <45c (dog)"
    if p < 0.55:
        return "2. 45-55c"
    if p < 0.65:
        return "3. 55-65c"
    if p < 0.75:
        return "4. 65-75c"
    return "5. 75c+ (chalk)"


def _edge_bucket(r: dict) -> str:
    e = r["bet"].get("edge_cents")
    if e is None:
        return "9. unknown"
    if e < 5:
        return "1. 3-5c"
    if e < 8:
        return "2. 5-8c"
    if e < 12:
        return "3. 8-12c"
    return "4. 12c+"


def _confidence_bucket(r: dict) -> str:
    c = r["bet"].get("confidence")
    if c is None:
        return "9. unknown"
    if c < 0.40:
        return "1. <0.40"
    if c < 0.45:
        return "2. 0.40-0.45"
    if c < 0.50:
        return "3. 0.45-0.50"
    return "4. 0.50+"


def _flow_bucket(r: dict) -> str:
    """How strongly money flow leaned, regardless of direction (it picks the side)."""
    mf = r["bet"].get("money_flow")
    if mf is None:
        return "9. unknown"
    if abs(mf) < 0.10:
        return "1. flat (<0.10)"
    if abs(mf) < 0.25:
        return "2. mild"
    return "3. strong (0.25+)"


def _week_bucket(r: dict) -> str:
    day = int(r["date"][8:10])
    return f"wk of {r['date'][:8]}{1 + 7 * ((day - 1) // 7):02d}"


def _probability_check(rows: list[dict]) -> None:
    """Did the bets we placed win as often as the model said they would?

    This is NOT the same as signals/calibration.py's multiplier (which pools snapshot
    history + ledger and shrinks toward the model's expectation) -- it's the raw,
    unshrunk read on placed bets only, so it's noisy by construction.
    """
    print("\n=== MODEL PROBABILITY vs REALITY (placed bets only) ===")
    graded = [r for r in rows if r["fair_side"] is not None]
    if not graded:
        print("  (no bets carry a fair_prob)")
        return
    for lo, hi in [(0.0, 0.5), (0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 1.01)]:
        rs = [r for r in graded if lo <= r["fair_side"] < hi]
        if not rs:
            continue
        predicted = sum(r["fair_side"] for r in rs) / len(rs)
        actual = sum(r["won"] for r in rs) / len(rs)
        print(f"  model {100 * lo:>3.0f}-{min(100 * hi, 100):>3.0f}%   n={len(rs):<3}  "
              f"predicted {100 * predicted:>3.0f}%   actual {100 * actual:>3.0f}%")
    brier = sum((r["fair_side"] - int(r["won"])) ** 2 for r in graded) / len(graded)
    print(f"  brier (placed bets) : {brier:.4f}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--sport", help="limit to one sport (mlb|nfl)")
    args = ap.parse_args()

    rows = _graded_rows(args.sport)
    if not rows:
        print("No settled bets in the paper ledger"
              + (f" for {args.sport.upper()}." if args.sport else "."))
        return

    _dollars(rows)
    _block("BY MARKET TYPE", rows, lambda r: r["kind"])
    _block("BY SPORT", rows, lambda r: sport_of(r["bet"]["ticker"]).upper())
    _block("BY ENTRY PRICE", rows, _price_bucket)
    _block("BY MODEL EDGE", rows, _edge_bucket)
    _block("BY CONFIDENCE", rows, _confidence_bucket)
    _block("BY MONEY-FLOW STRENGTH", rows, _flow_bucket)
    _block("BY WEEK", rows, _week_bucket)
    _probability_check(rows)
    print("\nSlices are tens of bets each -- treat them as hypotheses to watch, not findings. "
          "Production constants change only on backtest/bankroll_sim.py's walk-forward splits.")


if __name__ == "__main__":
    main()
