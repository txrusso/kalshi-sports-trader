"""Validate the signal against realized outcomes.

Replays recorded snapshot rows, resolves each game's actual winner from the MLB
Stats API, and reports:
  * recommendation P&L / ROI (did acting on recs make money?),
  * signal information coefficient (does money-flow score predict YES winning?),
  * fair-value calibration (are modeled probabilities accurate?).

This needs accumulated snapshots (from live cycles) plus settled games, so it is
most useful after the agent has been running for a while.

Usage:
    py -3 -m backtest.evaluate                 # all snapshots
    py -3 -m backtest.evaluate --since 2026-08-05
"""
from __future__ import annotations

import argparse
import glob
import json
from datetime import datetime, timezone

from config.settings import SNAPSHOTS_DIR
from config.sports import is_total, sport_of
from data.fair_value import parse_ticker
from data.fair_value_nfl import parse_ticker as parse_nfl_ticker
from data.fair_value_nfl_totals import parse_total_ticker as parse_nfl_total_ticker
from data.fair_value_totals import parse_total_ticker
from data.games import build_clients, resolve_outcomes
from signals.recommendation import headline_for

__all__ = ["load_rows", "resolve_outcomes", "graded_bets", "evaluate", "_game_date", "build_clients"]


def load_rows(since: str | None) -> list[dict]:
    rows: list[dict] = []
    for path in sorted(glob.glob(str(SNAPSHOTS_DIR / "*.jsonl"))):
        day = path.split("/")[-1].split("\\")[-1].replace(".jsonl", "")
        if since and day < since:
            continue
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    return rows


def _win_prob(r: dict) -> float | None:
    fp = r.get("fair_prob")
    if fp is None:
        return None
    return fp if r.get("rec_side") == "YES" else 1 - fp


def _entry(r: dict) -> float:
    return r["yes_ask"] if r.get("rec_side") == "YES" else round(1 - r["yes_bid"], 4)


def _game_key(r: dict) -> str:
    return r["ticker"].rsplit("-", 1)[0]


def _bet_rank(r: dict) -> tuple:
    # Prefer the WIN side (model's favored side, P>50%) to resolve a game's
    # double-counted / two-sided snapshots, then higher confidence, then latest.
    is_win_side = 1 if (_win_prob(r) or 0) > 0.50 else 0
    return (is_win_side, r.get("rec_confidence") or 0.0, r.get("cycle_ts") or "")


def _game_date(ticker: str) -> str | None:
    if sport_of(ticker) == "nfl":
        p = parse_nfl_total_ticker(ticker) if is_total(ticker) else parse_nfl_ticker(ticker)
        return p.date_str if p else None
    p = parse_total_ticker(ticker) if is_total(ticker) else parse_ticker(ticker)
    return p.date.strftime("%Y-%m-%d") if p else None


def _label_for(r: dict) -> str:
    """Display-only headline via the canonical formatter (team-wins-vs-team /
    game Over-Under phrasing). Prefers rec_headline if the row already has it
    (current scanner writes it); falls back to deriving it from the row's own
    title text for older history recorded before that field existed."""
    return r.get("rec_headline") or headline_for(
        r["ticker"], r["rec_side"], r.get("yes_team") or "", r.get("title") or "")


def graded_bets(rows: list[dict], outcomes: dict[str, bool]) -> list[dict]:
    """One graded bet per game (fixes double-counting), each tagged with its result.

    A game has two market tickers and is snapshotted every cycle, so the same bet
    appears many times. Collapse to a single representative per game (preferring the
    win-side rec), then attach the realized outcome.
    """
    rec_rows = [r for r in rows if r.get("rec_side") and r["ticker"] in outcomes and 0 < _entry(r) < 1]
    best_bet: dict[str, dict] = {}
    for r in rec_rows:
        k = _game_key(r)
        if k not in best_bet or _bet_rank(r) > _bet_rank(best_bet[k]):
            best_bet[k] = r

    out: list[dict] = []
    for r in best_bet.values():
        yes_won = outcomes[r["ticker"]]
        won = yes_won if r["rec_side"] == "YES" else (not yes_won)
        out.append({"ticker": r["ticker"], "side": r["rec_side"], "entry": _entry(r),
                    "won": won, "win_prob": _win_prob(r), "confidence": r.get("rec_confidence"),
                    "fair_prob": r.get("fair_prob"), "yes_team": r.get("yes_team"),
                    "label": _label_for(r),
                    "is_total": is_total(r["ticker"]),
                    "game_date": _game_date(r["ticker"])})
    return out


def _blended_score(r: dict, weights: tuple[float, float, float] | None) -> float | None:
    """Recompute the money-flow blend from stored components under `weights`
    (w_book, w_trades, w_oi). Returns the row's own `mf_score` when no override is
    given. Point-biserial IC is scale-invariant, so this fixed-weight linear blend
    is a faithful *signal-quality* diagnostic for weight sweeps; it does NOT
    reproduce production's per-row renormalization when OI is unavailable (that
    only rescales, and can't be recovered here since a stored mf_oi of 0 is
    ambiguous between 'flat' and 'no prior snapshot'), so read it as signal
    quality, not as the exact live blend."""
    if weights is None:
        return r.get("mf_score")
    wb, wt, wo = weights
    b, t, o = r.get("mf_book"), r.get("mf_trades"), r.get("mf_oi")
    if b is None or t is None or o is None:
        return None
    return wb * b + wt * t + wo * o


def evaluate(rows: list[dict], outcomes: dict[str, bool],
             weights: tuple[float, float, float] | None = None) -> dict:
    bets = graded_bets(rows, outcomes)

    pnl = staked = 0.0
    wins = 0
    for b in bets:
        pnl += (1 - b["entry"]) if b["won"] else -b["entry"]
        staked += b["entry"]
        wins += int(b["won"])
    n = len(bets)

    # --- signal IC / Brier: ONE snapshot per market (latest cycle), not per cycle ---
    latest: dict[str, dict] = {}
    for r in rows:
        if r["ticker"] in outcomes and r.get("mf_score") is not None:
            k = r["ticker"]
            if k not in latest or (r.get("cycle_ts") or "") > (latest[k].get("cycle_ts") or ""):
                latest[k] = r
    graded = list(latest.values())
    yes_won = [1 if outcomes[r["ticker"]] else 0 for r in graded]
    blended = [(_blended_score(r, weights), y) for r, y in zip(graded, yes_won)]
    blended = [(s, y) for s, y in blended if s is not None]
    ic = _point_biserial([s for s, _ in blended], [y for _, y in blended])

    # Per-component IC: does each money-flow component individually predict a YES
    # win, or is only the blend informative? The composite `mf_score` alone can
    # hide a component that contributes noise (or the wrong sign). Aligned the same
    # way as the blend: a positive component = YES-favorable flow, so a positive IC
    # means the component points the right way. `oi_momentum` is 0 on any market's
    # first observation (no prior snapshot), so we also report how many graded
    # markets had a non-zero OI reading -- a low count means its 25% weight was
    # mostly inert over this sample, which the blended IC can't reveal.
    component_ic: dict[str, float | None] = {}
    for field in ("mf_book", "mf_trades", "mf_oi"):
        vals = [r.get(field) for r in graded]
        pairs = [(v, y) for v, y in zip(vals, yes_won) if v is not None]
        component_ic[field] = (
            _point_biserial([v for v, _ in pairs], [y for _, y in pairs])
            if len(pairs) >= 3 else None)
    oi_active = sum(1 for r in graded if (r.get("mf_oi") or 0) != 0)

    fv_rows = [r for r in graded if r.get("fair_prob") is not None]
    brier = None
    if fv_rows:
        brier = sum((r["fair_prob"] - (1 if outcomes[r["ticker"]] else 0)) ** 2
                    for r in fv_rows) / len(fv_rows)

    settled_games = len({t.rsplit("-", 1)[0] for t in outcomes})
    return {
        "settled_games": settled_games,              # distinct games (not market-sides)
        "settled_market_sides": len(outcomes),       # 2 per game when both sides resolved
        "unique_markets_graded": len(graded),
        "bets_placed": n,                       # one per game, win-side only
        "bet_wins": wins,
        "bet_win_rate": round(wins / n, 3) if n else None,
        "pnl_per_contract": round(pnl, 3),
        "roi": round(pnl / staked, 3) if staked else None,
        "signal_ic": round(ic, 3) if ic is not None else None,
        "ic_book": round(component_ic["mf_book"], 3) if component_ic["mf_book"] is not None else None,
        "ic_trades": round(component_ic["mf_trades"], 3) if component_ic["mf_trades"] is not None else None,
        "ic_oi": round(component_ic["mf_oi"], 3) if component_ic["mf_oi"] is not None else None,
        "oi_active_markets": f"{oi_active}/{len(graded)}",
        "fair_value_brier": round(brier, 4) if brier is not None else None,
    }


def _point_biserial(xs: list[float], ys: list[int]) -> float | None:
    n = len(xs)
    if n < 3:
        return None
    mx = sum(xs) / n
    my = sum(ys) / n
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / n
    vx = sum((x - mx) ** 2 for x in xs) / n
    vy = sum((y - my) ** 2 for y in ys) / n
    if vx <= 0 or vy <= 0:
        return None
    return cov / (vx ** 0.5 * vy ** 0.5)


def main() -> None:
    ap = argparse.ArgumentParser(description="Backtest the money-flow signal")
    ap.add_argument("--since", help="only snapshots on/after this YYYY-MM-DD")
    ap.add_argument("--weights", help="override money-flow weights as 'book,trades,oi' "
                    "(e.g. '0.40,0.35,0.25' = production, or '0.55,0.45,0' to drop OI) "
                    "and report the resulting blended signal_ic")
    args = ap.parse_args()

    weights = None
    if args.weights:
        parts = [float(x) for x in args.weights.split(",")]
        if len(parts) != 3:
            ap.error("--weights needs exactly three comma-separated numbers: book,trades,oi")
        weights = (parts[0], parts[1], parts[2])

    rows = load_rows(args.since)
    if not rows:
        print("No snapshots found yet. Run the live agent first to accumulate data.")
        return
    print(f"Loaded {len(rows)} snapshot rows.")

    clients = build_clients()
    tickers = {r["ticker"] for r in rows}
    print(f"Resolving outcomes for {len(tickers)} markets...")
    outcomes = resolve_outcomes(tickers, clients)
    settled_games = len({t.rsplit("-", 1)[0] for t in outcomes})
    print(f"Resolved {len(outcomes)} settled market-sides across {settled_games} games.")

    report = evaluate(rows, outcomes, weights)
    if weights:
        print(f"\n(signal_ic computed under override weights book/trades/oi = {weights})")
    print("\n=== BACKTEST REPORT ===")
    for k, v in report.items():
        print(f"  {k:26}: {v}")
    print("\nGuide: signal_ic > 0 means the blended money-flow score predicts YES wins; "
          "ic_book/ic_trades/ic_oi are the per-component versions (a near-zero or "
          "negative one is a component pulling its weight the wrong way or not at all); "
          "oi_active_markets shows how often OI momentum was even non-zero; "
          "rec_roi > 0 means recs were profitable; lower Brier = better calibration.")


if __name__ == "__main__":
    main()
