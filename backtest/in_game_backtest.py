"""Does betting MLB games ALREADY IN PROGRESS make money?

In-game trading shipped 2026-09-22 with no backtest behind it, because only 18
of ~269,000 snapshot rows carried fair_source="live" -- `pregame_only=True` had
suppressed the model's in-game output all along. That made it look unbacktestable.

It isn't. The missing piece was never the prices, it was the probabilities:
  * PRICES: 7,789 snapshot rows are `skip_non_pregame` -- markets scanned WHILE a
    game was in progress. They carry real point-in-time yes_bid/yes_ask, money
    flow, and a cycle_ts. The scanner recorded them all along; only the fair-value
    half was refused.
  * PROBABILITIES: MLB's own `/game/{pk}/winProbability` returns the FULL
    play-by-play win-probability history for a completed game -- one entry per
    plate appearance, each stamped with a real UTC `about.endTime`.
    data/mlb_stats.py::home_win_probability only ever reads the LAST entry (the
    live value); the rest of that list is exactly the point-in-time history this
    backtest needs.

So for any snapshot taken at time T during a game, we can recover what the live
model WOULD have said at T -- the last play completed at or before T -- and run
the real production recommendation logic against the real price at T.

LEAK SAFETY. A play is only used if `about.endTime <= cycle_ts`, so the win
probability never sees the future. Prices are whatever was on the book at T.
Outcomes are real Kalshi settlement. Nothing here is fit on the test data.

WHAT IT MIRRORS. Production bets at most once per event and fires on the FIRST
qualifying cycle after a game starts (the ledger's already_bet dedupe), staying
eligible for `in_game_max_minutes`. This takes the same first qualifying snapshot
per event, through the same `build_recommendation` -- same edge gate, same
confidence floor, same Kelly sizing, same 8c in-game bar.

WINNER MARKETS ONLY. There is no live MLB totals model: `expected_runs()` is a
full-game estimate that ignores runs already scored, and data/fair_value_totals.py
now refuses in-game outright (that refusal was added as a direct result of
building this -- it was a live money bug). So KXMLBTOTAL is out of scope here by
construction, not by omission.

Usage:
    py -3 -m backtest.in_game_backtest
    py -3 -m backtest.in_game_backtest --since 2026-08-15
    py -3 -m backtest.in_game_backtest --max-minutes 60 --min-edge 10
"""
from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from typing import Optional

from backtest.evaluate import load_rows
from config.settings import DEFAULTS, PROJECT_ROOT
from data.fair_value import FairValue, parse_ticker
from data.games import build_clients, resolve_outcomes
from data.mlb_stats import BASE, MlbStatsClient
from signals.money_flow import MoneyFlow
from signals.recommendation import build_recommendation
from backtest.replay_moneyflow import _quote_from_row

WP_CACHE = PROJECT_ROOT / "backtest" / "in_game_wp_cache.json"


# --------------------------------------------------------------------------
# Win-probability history (the point-in-time model input)
# --------------------------------------------------------------------------
def _load_cache() -> dict:
    if WP_CACHE.exists():
        try:
            return json.loads(WP_CACHE.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def _save_cache(cache: dict) -> None:
    WP_CACHE.write_text(json.dumps(cache), encoding="utf-8")


def win_prob_timeline(mlb: MlbStatsClient, game_pk: int, cache: dict) -> list[tuple[str, float]]:
    """[(play_end_time_iso, home_win_prob)] for a completed game, chronological.

    This is the whole reason an in-game backtest is possible: every play carries
    its own UTC timestamp, so a snapshot at time T can be matched to the model's
    state at T without leaking the rest of the game."""
    key = str(game_pk)
    if key in cache:
        return [(t, p) for t, p in cache[key]]
    try:
        raw = mlb._get(f"{BASE}/game/{game_pk}/winProbability")
    except Exception:
        raw = None
    out: list[tuple[str, float]] = []
    if isinstance(raw, list):
        for e in raw:
            about = e.get("about") or {}
            end = about.get("endTime") or about.get("startTime")
            pct = e.get("homeTeamWinProbability")
            if not end or pct is None:
                continue
            try:
                out.append((end, float(pct) / 100.0))
            except (TypeError, ValueError):
                continue
    out.sort(key=lambda x: x[0])
    cache[key] = out
    return out


def prob_at(timeline: list[tuple[str, float]], when: datetime) -> Optional[float]:
    """Home win probability as of `when` -- the last play that had FINISHED by
    then. None if the game hadn't produced a play yet (leak-safe by design)."""
    best = None
    for end, p in timeline:
        dt = _parse_play_ts(end)
        if dt is None:
            continue
        if dt <= when:
            best = p
        else:
            break
    return best


def _parse_play_ts(s: str):
    """MLB stamps plays as e.g. '2026-08-05T19:37:39.610Z'. Parsed to a real
    datetime rather than compared as strings -- ISO strings with differing
    sub-second precision compare correctly only by accident."""
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except (TypeError, ValueError, AttributeError):
        return None


# --------------------------------------------------------------------------
# Replay
# --------------------------------------------------------------------------
def _mf_from_row(r: dict) -> Optional[MoneyFlow]:
    bi, tf, oim = r.get("mf_book"), r.get("mf_trades"), r.get("mf_oi")
    if bi is None or tf is None or oim is None or r.get("mf_score") is None:
        return None
    return MoneyFlow(score=r["mf_score"], book_imbalance=bi, trade_flow=tf,
                     oi_momentum=oim, strength=r.get("mf_strength") or 0.0, components={})


def _parse_ts(s: str) -> Optional[datetime]:
    try:
        d = datetime.fromisoformat(s)
    except (TypeError, ValueError):
        return None
    return d if d.tzinfo else d.replace(tzinfo=timezone.utc)


def build_in_game_bets(rows: list[dict], outcomes: dict[str, bool], settings,
                       verbose: bool = True) -> tuple[list[dict], dict]:
    """One bet per event: the first in-game snapshot that clears the production
    gate. Returns (bets, diagnostics)."""
    mlb = MlbStatsClient()
    cache = _load_cache()
    stats = defaultdict(int)

    by_ticker: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        t = r.get("ticker", "")
        if t.startswith("KXMLBGAME") and t in outcomes:
            by_ticker[t].append(r)
    stats["tickers_settled"] = len(by_ticker)

    # Resolve each ticker to its MLB game once (schedule lookups are cached in-client).
    from data.fair_value import FairValueModel
    model = FairValueModel(mlb, settings)

    best: dict[str, dict] = {}          # event_key -> bet (first qualifying)
    for tk, rs in sorted(by_ticker.items()):
        pt = parse_ticker(tk)
        if not pt:
            stats["unparseable"] += 1
            continue
        game = model._match_game(pt)
        if not game or not game.game_pk or not game.game_datetime:
            stats["no_game_match"] += 1
            continue
        timeline = win_prob_timeline(mlb, game.game_pk, cache)
        if not timeline:
            stats["no_wp_history"] += 1
            continue
        stats["games_with_wp"] += 1

        yes_is_home = (pt.yes_team == game.home_abbr)
        rs.sort(key=lambda r: r.get("cycle_ts") or "")
        event_key = tk.rsplit("-", 1)[0]

        for r in rs:
            ts = _parse_ts(r.get("cycle_ts") or "")
            if ts is None:
                continue
            mins_in = (ts - game.game_datetime).total_seconds() / 60.0
            if mins_in <= 0:
                continue                                  # pregame snapshot
            stats["in_game_snapshots"] += 1
            if mins_in > settings.in_game_max_minutes:
                stats["past_max_minutes"] += 1
                continue
            hp = prob_at(timeline, ts)
            if hp is None:
                stats["no_play_yet"] += 1
                continue
            stats["priced"] += 1

            prob_yes = hp if yes_is_home else 1.0 - hp
            fv = FairValue(round(prob_yes, 4), "live", 0.80,
                           {"yes_is_home": yes_is_home,
                            "yes_name": game.home_name if yes_is_home else game.away_name,
                            "opp_name": game.away_name if yes_is_home else game.home_name},
                           game_state="Live")
            mf = _mf_from_row(r)
            if mf is None:
                stats["no_money_flow"] += 1
                continue
            rec = build_in_game_rec(r, mf, fv, settings)
            if rec is None:
                stats["gated_out"] += 1
                continue
            if event_key in best:
                stats["event_already_bet"] += 1
                break
            won = outcomes[tk] if rec.side == "YES" else (not outcomes[tk])
            best[event_key] = {
                "ticker": tk, "event": event_key, "side": rec.side,
                "entry": rec.entry_price, "edge_c": rec.edge_cents,
                "conf": rec.confidence, "fair": rec.fair_prob,
                "minutes_in": round(mins_in, 1), "won": won,
                "contracts": rec.suggested_contracts,
                "date": ts.date().isoformat(),
            }
            stats["bets"] += 1
            break

    _save_cache(cache)
    return list(best.values()), dict(stats)


def build_in_game_rec(r: dict, mf: MoneyFlow, fv: FairValue, settings):
    """Run the REAL production recommendation logic (calibration off, matching
    replay_moneyflow's convention -- confidence-only, can't flip a pick)."""
    q = _quote_from_row(r)
    return build_recommendation(q, mf, fv, settings, calibration=None)


# --------------------------------------------------------------------------
# Grading
# --------------------------------------------------------------------------
def grade(bets: list[dict]) -> dict:
    from config.fees import fee_for
    n = len(bets)
    if not n:
        return {"n": 0}
    wins = sum(1 for b in bets if b["won"])
    staked = sum(b["entry"] for b in bets)
    pnl = sum((1 - b["entry"]) if b["won"] else -b["entry"] for b in bets)
    fees = sum(fee_for(1.0, b["entry"], b["ticker"], True) for b in bets)
    return {
        "n": n, "wins": wins, "win_rate": round(wins / n, 3),
        "roi": round(pnl / staked, 4) if staked else None,
        "roi_net": round((pnl - fees) / staked, 4) if staked else None,
        "pnl": round(pnl, 3), "fees": round(fees, 4),
        "avg_entry": round(staked / n, 4),
        "avg_edge_c": round(sum(b["edge_c"] or 0 for b in bets) / n, 2),
        "avg_minutes_in": round(sum(b["minutes_in"] for b in bets) / n, 1),
    }


def _bucket(bets: list[dict], key, edges, label) -> list[str]:
    out = []
    for lo, hi in zip(edges, edges[1:]):
        sel = [b for b in bets if lo <= key(b) < hi]
        if not sel:
            continue
        g = grade(sel)
        out.append(f"    {label} {lo:>5g}-{hi:<5g}  n={g['n']:<4} win%={g['win_rate']:<6} "
                   f"roi={str(g['roi']):>8}  net={str(g['roi_net']):>8}")
    return out


def pregame_bets_for(rows: list[dict], outcomes: dict[str, bool], events: set) -> list[dict]:
    """The pregame bet the model WOULD have made on the same games, as a baseline.

    Production bets once per event, so an in-game bet only ever happens on a game
    the pregame trigger did NOT already take. This answers the decision-relevant
    question: on these specific games, was entering mid-game better than entering
    pregame would have been? Uses replay_moneyflow's real replay, so it's the
    identical recommendation logic, just on the pregame snapshots."""
    from backtest.replay_moneyflow import PROD_WEIGHTS, replay_rec, _PREGAME_SOURCES
    by_ticker = defaultdict(list)
    for r in rows:
        tk = r.get("ticker", "")
        if (tk.rsplit("-", 1)[0] in events and tk in outcomes
                and r.get("fair_source") in _PREGAME_SOURCES):
            by_ticker[tk].append(r)
    out: dict[str, dict] = {}
    for tk, rs in by_ticker.items():
        rs.sort(key=lambda r: r.get("cycle_ts") or "")
        rec = replay_rec(rs[-1], PROD_WEIGHTS)     # last pregame cycle = the real trigger point
        if rec is None or not (0 < rec.entry_price < 1):
            continue
        ev = tk.rsplit("-", 1)[0]
        if ev in out:
            continue
        won = outcomes[tk] if rec.side == "YES" else (not outcomes[tk])
        out[ev] = {"ticker": tk, "event": ev, "side": rec.side, "entry": rec.entry_price,
                   "edge_c": rec.edge_cents, "conf": rec.confidence, "fair": rec.fair_prob,
                   "minutes_in": 0.0, "won": won, "contracts": rec.suggested_contracts,
                   "date": (rs[-1].get("cycle_ts") or "")[:10]}
    return list(out.values())


def walk_forward(bets: list[dict]) -> tuple[list[dict], list[dict], str]:
    """Split by date at the median bet, so train and validate hold roughly equal
    counts. This project's standing bar is a win on BOTH splits, not the pooled
    number -- every production constant here was chosen that way."""
    dated = sorted([b for b in bets if b.get("date")], key=lambda b: b["date"])
    if len(dated) < 4:
        return dated, [], ""
    cut = dated[len(dated) // 2]["date"]
    train = [b for b in dated if b["date"] < cut]
    validate = [b for b in dated if b["date"] >= cut]
    return train, validate, cut


def _row(label: str, g: dict) -> str:
    if not g.get("n"):
        return f"  {label:<22} (none)"
    return (f"  {label:<22} n={g['n']:<4} win%={g['win_rate']:<6} "
            f"ROI={str(g['roi']):>8}  net={str(g['roi_net']):>8}")


def main() -> None:
    ap = argparse.ArgumentParser(description="In-game (live) MLB entry backtest")
    ap.add_argument("--since", help="only snapshots on/after YYYY-MM-DD")
    ap.add_argument("--max-minutes", type=float, default=None,
                    help="override in_game_max_minutes (default: production's 90)")
    ap.add_argument("--min-edge", type=float, default=None,
                    help="override in_game_min_edge_cents (default: production's 8.0)")
    args = ap.parse_args()

    settings = replace(DEFAULTS, pregame_only=False, in_game_trade=True)
    if args.max_minutes is not None:
        settings = replace(settings, in_game_max_minutes=args.max_minutes)
    if args.min_edge is not None:
        settings = replace(settings, in_game_min_edge_cents=args.min_edge)

    rows = load_rows(args.since)
    if not rows:
        print("No snapshots found.")
        return
    print(f"Loaded {len(rows)} snapshot rows.")
    mlb_rows = [r for r in rows if (r.get("ticker") or "").startswith("KXMLBGAME")]
    print(f"{len(mlb_rows)} are MLB winner-market rows.")

    clients = build_clients()
    tickers = {r["ticker"] for r in mlb_rows}
    print(f"Resolving outcomes for {len(tickers)} markets (one network pass)...")
    outcomes = resolve_outcomes(tickers, clients)
    print(f"Resolved {len(outcomes)} settled market-sides.\n")

    print(f"Replaying in-game entries  (max_minutes={settings.in_game_max_minutes:g}, "
          f"min_edge={settings.in_game_min_edge_cents:g}c)")
    print("Fetching per-play win-probability history (cached after the first run)...")
    bets, stats = build_in_game_bets(mlb_rows, outcomes, settings)

    print("\nPIPELINE")
    for k in ("tickers_settled", "games_with_wp", "no_game_match", "no_wp_history",
              "in_game_snapshots", "past_max_minutes", "no_play_yet", "priced",
              "no_money_flow", "gated_out", "bets"):
        if k in stats:
            print(f"  {k:<22} {stats[k]}")

    g = grade(bets)
    print("\n" + "=" * 68)
    print("IN-GAME ENTRIES (MLB winner markets, one bet per game)")
    print("=" * 68)
    if not g.get("n"):
        print("  No qualifying in-game bets -- nothing to grade.")
        return
    print(f"  n={g['n']}  win%={g['win_rate']}  ROI={g['roi']}  ROI net of fees={g['roi_net']}")
    print(f"  avg entry={g['avg_entry']}  avg edge={g['avg_edge_c']}c  "
          f"avg minutes into game={g['avg_minutes_in']}")
    print(f"  flat-$1 P&L={g['pnl']:+.2f}  fees=${g['fees']:.2f}")

    print("\n  by minutes into the game:")
    for line in _bucket(bets, lambda b: b["minutes_in"], [0, 15, 30, 45, 60, 90, 999], "min"):
        print(line)
    print("\n  by modeled edge:")
    for line in _bucket(bets, lambda b: b["edge_c"] or 0, [0, 10, 15, 20, 30, 999], "edge"):
        print(line)
    print("\n  by entry price:")
    for line in _bucket(bets, lambda b: b["entry"], [0, 0.2, 0.4, 0.6, 0.8, 1.01], "px"):
        print(line)

    # ---- the bar this project actually holds things to ----
    train, validate, cut = walk_forward(bets)
    gt, gv = grade(train), grade(validate)
    print("\n" + "=" * 68)
    print(f"WALK-FORWARD SPLIT (cut at {cut})")
    print("=" * 68)
    print(_row("in-game TRAIN", gt))
    print(_row("in-game VALIDATE", gv))
    both = (gt.get("roi_net") or -9) > 0 and (gv.get("roi_net") or -9) > 0
    print(f"\n  profitable on BOTH splits net of fees: {'YES' if both else 'NO'}")

    # ---- same games, entered pregame instead ----
    events = {b["event"] for b in bets}
    pre = pregame_bets_for(mlb_rows, outcomes, events)
    print("\n" + "=" * 68)
    print("SAME GAMES, ENTERED PREGAME INSTEAD")
    print("=" * 68)
    print(f"  {len(pre)} of these {len(events)} games also produced a pregame bet.")
    print(_row("pregame (same games)", grade(pre)))
    print(_row("in-game (same games)", grade(bets)))
    matched = {b["event"] for b in pre}
    head = [b for b in bets if b["event"] in matched]
    if head:
        print("\n  head-to-head on the overlap only:")
        print(_row("  pregame", grade([b for b in pre if b["event"] in matched])))
        print(_row("  in-game", grade(head)))
        print("\n  NOTE: production bets once per event, so these two COMPETE --")
        print("  a game taken pregame is never re-taken in-game.")

    print("\nNOTE: in-sample on one partial season, and every bucket is small -- read")
    print("direction, not significance. The win-probability model is MLB's own, so a")
    print("positive result means the MARKET was slow to it, not that our model is good.")


if __name__ == "__main__":
    main()
