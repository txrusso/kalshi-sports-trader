"""Would a liquidity floor (min volume / min open interest) have helped?

Built 2026-10-06, after two live NHL totals recommendations were found sitting
on markets with ZERO traded volume and zero open interest -- and after
discovering that `min_market_volume` / `min_open_interest` had been defined in
config/settings.py for months and read by nothing at all.

Method: load every real bet the way every report loads them
(engine/real_bets.load_real_bets, i.e. repriced from actual fills), then join
each one to the snapshot row for its OWN ticker at the scan cycle it was placed
on, which carries that market's volume and open interest at that moment. Grade
the blocked set against the kept set across a range of thresholds.

Result: the floor LOSES money at every threshold tested -- at the configured
50/50 it blocks 7 bets that went 6-1 for +$1.28, which is 19% of all profit to
date. NOT SHIPPED; the two dead settings were deleted instead. Full writeup in
backtest/liquidity_floor_20261006.txt, summary in config/settings.py.

Run: .venv/Scripts/python.exe -m backtest.liquidity_floor
"""
from __future__ import annotations

import sys, json, pathlib
from datetime import datetime, timedelta, timezone

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from backtest.evaluate import load_rows
from engine.real_bets import load_real_bets
from data.games import build_clients, resolve_outcomes
from config.sports import sport_of, market_kind
from kalshi.client import KalshiClient

def ts(s):
    return datetime.fromisoformat(s) if s else None

print("loading ledgers (repricing from real fills)...", file=sys.stderr)
try:
    client = KalshiClient()
except Exception as e:
    print("  no client:", e, file=sys.stderr)
    client = None
bets = load_real_bets(client)
print(f"  {len(bets)} graded-eligible bets", file=sys.stderr)

print("loading snapshots...", file=sys.stderr)
rows = load_rows(None)
print(f"  {len(rows)} snapshot rows", file=sys.stderr)

# index: ticker -> [(cycle_ts, volume, oi)] sorted
from collections import defaultdict
idx = defaultdict(list)
for r in rows:
    t = r.get("ticker")
    if not t:
        continue
    idx[t].append((r["cycle_ts"], r.get("volume"), r.get("open_interest")))
for t in idx:
    idx[t].sort()
del rows

WINDOW = timedelta(minutes=35)   # one scan cycle of slack either side

matched = unmatched = 0
for b in bets:
    bt = ts(b["ts"])
    cand = idx.get(b["ticker"])
    b["_vol"] = b["_oi"] = None
    if not cand:
        unmatched += 1
        continue
    # the latest cycle at or before the bet; else the earliest after, within the window
    best = None
    for cts, v, oi in cand:
        c = ts(cts)
        if c <= bt:
            best = (c, v, oi)
        else:
            if best is None and c - bt <= WINDOW:
                best = (c, v, oi)
            break
    if best is None or abs(best[0] - bt) > WINDOW:
        unmatched += 1
        continue
    matched += 1
    b["_vol"], b["_oi"] = best[1], best[2]

print(f"  matched {matched}, unmatched {unmatched}", file=sys.stderr)

print("resolving outcomes...", file=sys.stderr)
clients = build_clients()
outcomes = resolve_outcomes({b["ticker"] for b in bets}, clients)
print(f"  {len(outcomes)} settled", file=sys.stderr)

def grade(bs):
    w = l = 0; net = 0.0; wager = 0.0; fees = 0.0
    for b in bs:
        yw = outcomes.get(b["ticker"])
        if yw is None:
            continue
        won = yw if b["side"] == "YES" else (not yw)
        c = b.get("contracts") or 0
        net += ((1 - b["entry_price"]) if won else -b["entry_price"]) * c
        wager += b["entry_price"] * c
        fees += b.get("fees_usd") or 0.0
        w += int(won); l += int(not won)
    return w, l, net, net - fees, wager

def line(label, bs):
    w, l, net, nf, wager = grade(bs)
    n = w + l
    if n == 0:
        return f"{label:<28} n=0"
    return (f"{label:<28} {w:>3}-{l:<3} ({w/n:>4.0%})  "
            f"wager ${wager:>7.2f}  gross {net:>+7.2f}  net of fees {nf:>+7.2f}  "
            f"ROI {nf/wager:>+7.1%}")

have = [b for b in bets if b["_vol"] is not None]
print()
print("=" * 108)
print(f"ALL BETS WITH A LIQUIDITY READ AT BET TIME  (n={len(have)} of {len(bets)})")
print("=" * 108)
print(line("all", have))
print()
print("A FLOOR ON *VOLUME* (contracts traded in the market at the moment we bet)")
print("-" * 108)
for thr in (1, 10, 25, 50, 100, 250, 500, 1000, 5000):
    blocked = [b for b in have if (b["_vol"] or 0) < thr]
    kept = [b for b in have if (b["_vol"] or 0) >= thr]
    bw, bl, bn, bnf, bwag = grade(blocked)
    print(f"\n  min_volume = {thr}:  blocks {len(blocked)}/{len(have)} bets ({len(blocked)/len(have):.0%})")
    print("   ", line("BLOCKED (what you'd lose)", blocked))
    print("   ", line("KEPT (what you'd keep)", kept))

print()
print("A FLOOR ON *OPEN INTEREST*")
print("-" * 108)
for thr in (1, 10, 25, 50, 100, 250, 500, 1000, 5000):
    blocked = [b for b in have if (b["_oi"] or 0) < thr]
    kept = [b for b in have if (b["_oi"] or 0) >= thr]
    print(f"\n  min_oi = {thr}:  blocks {len(blocked)}/{len(have)} bets ({len(blocked)/len(have):.0%})")
    print("   ", line("BLOCKED", blocked))
    print("   ", line("KEPT", kept))

print()
print("BOTH, AT THE CONFIGURED (DEAD) VALUES  volume>=50 AND oi>=50")
print("-" * 108)
blocked = [b for b in have if (b["_vol"] or 0) < 50 or (b["_oi"] or 0) < 50]
kept = [b for b in have if (b["_vol"] or 0) >= 50 and (b["_oi"] or 0) >= 50]
print(line("BLOCKED", blocked))
print(line("KEPT", kept))

print()
print("DISTRIBUTION OF LIQUIDITY AT BET TIME")
print("-" * 108)
vols = sorted((b["_vol"] or 0) for b in have)
ois = sorted((b["_oi"] or 0) for b in have)
def pct(a, p):
    return a[min(len(a) - 1, int(len(a) * p))]
for name, a in (("volume", vols), ("open interest", ois)):
    print(f"  {name:<14} min {a[0]:>10.0f}   p10 {pct(a,.10):>10.0f}   p25 {pct(a,.25):>10.0f}   "
          f"median {pct(a,.50):>10.0f}   p75 {pct(a,.75):>10.0f}   max {a[-1]:>12.0f}")

print()
print("BY SPORT / MARKET KIND: how many bets sat below 50 contracts of volume")
print("-" * 108)
buck = defaultdict(lambda: [0, 0])
for b in have:
    k = f"{sport_of(b['ticker'])}:{market_kind(b['ticker'])}"
    buck[k][0] += 1
    buck[k][1] += int((b["_vol"] or 0) < 50)
for k in sorted(buck):
    n, lo = buck[k]
    print(f"  {k:<16} {n:>4} bets, {lo:>3} below 50 volume ({lo/n:.0%})")

print()
print("THE LEAST-LIQUID BETS WE ACTUALLY PLACED (by volume at bet time)")
print("-" * 108)
for b in sorted(have, key=lambda x: x["_vol"] or 0)[:15]:
    yw = outcomes.get(b["ticker"])
    res = "pending" if yw is None else ("WON " if (yw if b["side"] == "YES" else not yw) else "LOST")
    print(f"  vol {b['_vol']:>9.0f}  oi {b['_oi']:>9.0f}  {b['ts'][:16]}  {b['ticker']:<34} "
          f"{b['side']:<3} @{b['entry_price']:.2f} x{b.get('contracts',0):>5.2f}  {res}")
