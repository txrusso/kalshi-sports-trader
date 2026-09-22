"""Kalshi MLB agent — terminal CLI.

Command-driven interface over the money-flow engine. Read/analysis commands run
end-to-end; `place` and `loop --live` submit real orders (see below).

    py -3 cli.py search              # scan markets, show money-flow reads
    py -3 cli.py rank                # ranked, actionable recommendations
    py -3 cli.py inspect <ticker>    # deep dive on one market
    py -3 cli.py status              # account + exchange + positions/orders
    py -3 cli.py positions           # open positions
    py -3 cli.py orders              # resting orders
    py -3 cli.py results [YYYY-MM-DD]# game outcomes for a date
    py -3 cli.py backtest [--since D]# validate the signal vs outcomes
    py -3 cli.py loop                # continuous autonomous loop (recommend-only by default)
    py -3 cli.py loop --live         # LIVE: auto-submit real orders, no confirmation step
    py -3 cli.py place ...           # PREVIEW an order, or submit one with --confirm

Every command accepts --allow-live to include in-progress games (default: pregame only).
"""
from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import replace
from datetime import datetime

from dateutil import parser as dateparser

from config.settings import DEFAULTS, EASTERN, Settings
from config.sports import is_total
from data.fair_value import FairValueModel, FairValueRouter
from data.fair_value_nfl import NflFairValueModel
from data.fair_value_nfl_totals import NflTotalsFairValueModel
from data.fair_value_nfl_spread import NflSpreadFairValueModel
from data.fair_value_nba import NbaFairValueModel
from data.fair_value_nba_totals import NbaTotalsFairValueModel
from data.fair_value_nhl import NhlFairValueModel
from data.fair_value_nhl_totals import NhlTotalsFairValueModel
from data.fair_value_totals import TotalsFairValueModel
from data.games import build_clients
from data.mlb_stats import MlbStatsClient
from data.nfl_data import NflDataClient
from data.nba_data import NbaDataClient
from data.nhl_data import NhlDataClient
from engine.loop import build_context, run_loop, run_once
from engine.scanner import run_scan
from kalshi.client import KalshiClient, KalshiError
from kalshi.normalize import OrderBook, parse_market, parse_trades, position_size
from signals.money_flow import compute_money_flow

log = logging.getLogger("cli")


def _setup(verbose: bool = False) -> None:
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError):
            pass
    logging.basicConfig(level=logging.DEBUG if verbose else logging.WARNING,
                        format="%(levelname)s %(name)s: %(message)s")


def _settings_from(args) -> Settings:
    ov = {}
    if getattr(args, "allow_live", False):
        ov["pregame_only"] = False
    if getattr(args, "bankroll", None):
        ov["bankroll_usd"] = args.bankroll
        ov["dynamic_bankroll"] = False        # explicit --bankroll overrides the live pull
    if getattr(args, "demo", False):
        ov["use_demo"] = True
    if getattr(args, "max_stake_pct", None) is not None:
        ov["max_stake_pct"] = args.max_stake_pct
    return replace(DEFAULTS, **ov) if ov else DEFAULTS


def _dollars(cents_str) -> str:
    try:
        return f"${float(cents_str):,.2f}"
    except (TypeError, ValueError):
        return str(cents_str)


# --------------------------------------------------------------------------- #
# search / rank
# --------------------------------------------------------------------------- #
def cmd_search(args) -> None:
    settings = _settings_from(args)
    ctx = build_context(settings)
    _client, _mlb, _nfl, _nba, _nhl, fair_router, store, calibration = ctx
    res = run_scan(_client, fair_router, store.load_prev_state(), settings, calibration)

    rows = [r for r in res.snapshot_rows if r.get("fair_prob") is not None or abs(r.get("mf_score") or 0) > 0.1]
    rows.sort(key=lambda r: abs(r.get("mf_score") or 0), reverse=True)
    print(f"\nMoney-flow reads — {len(rows)} markets (scanned {res.scanned}, deep {res.deep_scanned})")
    print(f"{'TICKER':<34}{'MID':>6}{'FLOW':>7}{'BOOK':>7}{'TRADE':>7}{'FAIR':>7}{'SRC':>13}")
    print("-" * 88)
    for r in rows[: args.top]:
        fair = f"{r['fair_prob']:.0%}" if r.get("fair_prob") is not None else "--"
        print(f"{r['ticker']:<34}{r['mid']:>6.2f}{r['mf_score']:>+7.2f}{r['mf_book']:>+7.2f}"
              f"{r['mf_trades']:>+7.2f}{fair:>7}{str(r.get('fair_source')):>13}")
    print("\nTip: `rank` applies edge/confidence filters + sizing to produce actionable trades.")


def cmd_rank(args) -> None:
    settings = _settings_from(args)
    if args.top:
        settings = replace(settings, top_n_recommendations=args.top)
    run_once(settings)


# --------------------------------------------------------------------------- #
# inspect
# --------------------------------------------------------------------------- #
def cmd_inspect(args) -> None:
    settings = _settings_from(args)
    client = KalshiClient(settings)
    mlb = MlbStatsClient()
    nfl = NflDataClient()
    nba = NbaDataClient()
    nhl = NhlDataClient()
    fair_router = FairValueRouter(
        mlb_winner=FairValueModel(mlb, settings), mlb_totals=TotalsFairValueModel(mlb, settings),
        nfl_winner=NflFairValueModel(nfl, settings=settings),
        nfl_totals=NflTotalsFairValueModel(nfl, settings),
        nfl_spread=NflSpreadFairValueModel(nfl, settings=settings),
        nba_winner=NbaFairValueModel(nba, settings=settings),
        nba_totals=NbaTotalsFairValueModel(nba, settings),
        nhl_winner=NhlFairValueModel(nhl, settings=settings),
        nhl_totals=NhlTotalsFairValueModel(nhl, settings),
    )

    m = client.get_market(args.ticker)
    if not m:
        print(f"Market not found: {args.ticker}")
        return
    q = parse_market(m)
    ob = OrderBook.from_raw(client._get(f"/markets/{q.ticker}/orderbook", {"depth": 10}))
    trades = parse_trades(client.get_trades(q.ticker, limit=settings.trade_flow_lookback))
    mf = compute_money_flow(q, ob, trades, settings=settings)
    fv = fair_router.estimate(q.ticker, q)

    print(f"\n=== {q.ticker} ===")
    print(f"  {q.title}  (YES = {q.yes_sub_title})   status={q.status}")
    print(f"  quote: yes_bid {q.yes_bid:.2f} / yes_ask {q.yes_ask:.2f}  mid {q.mid:.2f}  "
          f"spread {q.spread_cents:.1f}c")
    print(f"  volume {q.volume:.0f}  open_interest {q.open_interest:.0f}  "
          f"liquidity ${q.liquidity:,.0f}")
    print(f"\n  ORDER BOOK (top):  best YES bid {ob.best_yes_bid:.2f}  best YES ask {ob.best_yes_ask:.2f}")
    print(f"     YES levels (price x size, near touch): "
          f"{[(round(p,2), round(s)) for p, s in ob.yes_levels[-4:]]}")
    print(f"     NO  levels (price x size, near touch): "
          f"{[(round(p,2), round(s)) for p, s in ob.no_levels[-4:]]}")
    print(f"\n  MONEY FLOW: score {mf.score:+.2f}  ({mf.direction})  strength {mf.strength:.2f}")
    print(f"     book {mf.book_imbalance:+.2f}   trades {mf.trade_flow:+.2f}   oi {mf.oi_momentum:+.2f}")
    lp = mf.components.get("large_prints") or {}
    if lp.get("n"):
        print(f"     LARGE PRINTS (whales): {lp['n']} print(s) >= {lp.get('threshold')}ct  "
              f"net lean {lp.get('net', 0):+.2f}  (yes ${lp.get('yes_d',0):.0f} / no ${lp.get('no_d',0):.0f}, "
              f"biggest {lp.get('max_size',0):.0f}ct)")
    else:
        print(f"     LARGE PRINTS (whales): none >= {lp.get('threshold')}ct this window")
    print(f"     detail: {mf.components}")
    if fv.prob is not None:
        print(f"\n  FAIR VALUE: P(YES)={fv.prob:.1%} via {fv.source} (conf {fv.confidence:.2f})")
        print(f"     edge on YES @ ask: {(fv.prob - q.yes_ask)*100:+.1f}c")
        print(f"     detail: {fv.detail}")
    else:
        print(f"\n  FAIR VALUE: none ({fv.source}) — {fv.detail}")


# --------------------------------------------------------------------------- #
# account: status / positions / orders
# --------------------------------------------------------------------------- #
def cmd_status(args) -> None:
    settings = _settings_from(args)
    client = KalshiClient(settings)
    try:
        ex = client.exchange_status()
        bal = client.balance()
        pos = client.get_positions()
        orders = client.get_orders(status="resting")
    except KalshiError as e:
        print(f"API error: {e}")
        return
    mkt_pos = pos.get("market_positions", []) or []
    print("\n=== ACCOUNT STATUS ===")
    print(f"  exchange: active={ex.get('exchange_active')} trading={ex.get('trading_active')}")
    print(f"  balance:  {_dollars(bal.get('balance_dollars', bal.get('balance')))}")
    print(f"  open positions: {len([p for p in mkt_pos if position_size(p) != 0])}")
    print(f"  resting orders: {len(orders)}")
    if float(str(bal.get('balance') or 0)) == 0:
        print("  note: account is unfunded — recommendations are advisory until you deposit.")


def cmd_positions(args) -> None:
    settings = _settings_from(args)
    client = KalshiClient(settings)
    pos = client.get_positions().get("market_positions", []) or []
    live = [p for p in pos if position_size(p) != 0]
    if not live:
        print("No open positions.")
        return
    print(f"\n{'TICKER':<36}{'POS':>6}{'EXPOSURE':>12}{'REAL_PNL':>12}")
    print("-" * 66)
    for p in live:
        print(f"{p.get('ticker',''):<36}{round(position_size(p)):>6}"
              f"{_dollars(p.get('market_exposure_dollars', p.get('market_exposure'))):>12}"
              f"{_dollars(p.get('realized_pnl_dollars', p.get('realized_pnl'))):>12}")


def cmd_orders(args) -> None:
    settings = _settings_from(args)
    client = KalshiClient(settings)
    orders = client.get_orders(status="resting")
    if not orders:
        print("No resting orders.")
        return
    print(f"\n{'ORDER_ID':<40}{'TICKER':<30}{'SIDE':>5}{'ACT':>5}{'CT':>6}{'PRICE':>7}")
    print("-" * 93)
    for o in orders:
        price = o.get("yes_price") or o.get("no_price") or o.get("price")
        print(f"{str(o.get('order_id',''))[:38]:<40}{o.get('ticker',''):<30}"
              f"{o.get('side',''):>5}{o.get('action',''):>5}{o.get('remaining_count', o.get('count','')):>6}"
              f"{str(price):>7}")


# --------------------------------------------------------------------------- #
# results
# --------------------------------------------------------------------------- #
def cmd_results(args) -> None:
    mlb = MlbStatsClient()
    when = dateparser.parse(args.date) if args.date else datetime.now(EASTERN)
    day = when.strftime("%Y-%m-%d")
    games = mlb.schedule(when)
    print(f"\nMLB games {day}:")
    if not games:
        print("  (none)")
    for g in sorted(games, key=lambda g: g.state):
        score = f"  winner: {g.winner_abbr}" if g.winner_abbr else ""
        print(f"  {g.away_abbr:>4} @ {g.home_abbr:<4}  {g.detailed_state:<14}{score}")

    nfl = NflDataClient()
    nfl_games = [g for g in nfl.games() if g.date_str == day]
    if nfl_games:
        print(f"\nNFL games {day}:")
        for g in sorted(nfl_games, key=lambda g: g.state):
            score = f"  winner: {g.winner_abbr}" if g.winner_abbr else ""
            print(f"  {g.away_abbr:>4} @ {g.home_abbr:<4}  {g.state:<14}{score}")

    nba = NbaDataClient()
    nba_games = [g for g in nba.games() if g.date_str == day]
    if nba_games:
        print(f"\nNBA games {day}:")
        for g in sorted(nba_games, key=lambda g: g.state):
            score = f"  winner: {g.winner_abbr}" if g.winner_abbr else ""
            print(f"  {g.away_abbr:>4} @ {g.home_abbr:<4}  {g.state:<14}{score}")

    nhl = NhlDataClient()
    nhl_games = [g for g in nhl.games() if g.date_str == day]
    if nhl_games:
        print(f"\nNHL games {day}:")
        for g in sorted(nhl_games, key=lambda g: g.state):
            score = f"  winner: {g.winner_abbr}" if g.winner_abbr else ""
            print(f"  {g.away_abbr:>4} @ {g.home_abbr:<4}  {g.state:<14}{score}")


# --------------------------------------------------------------------------- #
# backtest / loop
# --------------------------------------------------------------------------- #
def cmd_backtest(args) -> None:
    from backtest.evaluate import load_rows, evaluate
    from data.games import resolve_outcomes
    rows = load_rows(args.since)
    if not rows:
        print("No snapshots yet. Run `rank`/`loop` first to accumulate data.")
        return
    outcomes = resolve_outcomes({r["ticker"] for r in rows}, build_clients())
    report = evaluate(rows, outcomes)
    settled_games = len({t.rsplit("-", 1)[0] for t in outcomes})
    print(f"\n=== BACKTEST ({len(rows)} snapshots, {settled_games} settled games) ===")
    for k, v in report.items():
        print(f"  {k:26}: {v}")


def cmd_outcome(args) -> None:
    from backtest.evaluate import load_rows, graded_bets
    from data.games import resolve_outcomes
    day = args.date or datetime.now(EASTERN).strftime("%Y-%m-%d")
    rows = load_rows(None)
    if not rows:
        print("No snapshots recorded yet.")
        return
    outcomes = resolve_outcomes({r["ticker"] for r in rows}, build_clients())
    bets = [b for b in graded_bets(rows, outcomes) if b["game_date"] == day]
    if not bets:
        print(f"No settled bets for games on {day} yet.")
        return

    bets.sort(key=lambda b: (b["is_total"], b["ticker"]))
    print(f"\n=== BET OUTCOMES for {day} ===")
    print(f"{'RESULT':<7}{'BET':<34}{'ENTRY':>6}{'CONF':>6}   MARKET")
    print("-" * 78)
    pnl = staked = wins = 0.0
    for b in bets:
        res = "WIN " if b["won"] else "LOSS"
        pnl += (1 - b["entry"]) if b["won"] else -b["entry"]
        staked += b["entry"]
        wins += int(b["won"])
        print(f"{res:<7}{b['label']:<34}{b['entry']:>6.2f}"
              f"{(b['confidence'] or 0):>6.2f}   {b['ticker']}")
    n = len(bets)
    win = [b for b in bets if not b["is_total"]]
    tot = [b for b in bets if b["is_total"]]
    print("-" * 78)
    print(f"Record: {int(wins)}-{n - int(wins)} ({wins/n:.0%})   "
          f"P&L {pnl:+.2f}/contract   ROI {pnl/staked:+.0%}"
          if staked else "no staked amount")
    print(f"   winners: {sum(b['won'] for b in win)}/{len(win)}   "
          f"totals(O/U): {sum(b['won'] for b in tot)}/{len(tot)}")


def cmd_whales(args) -> None:
    from engine.whale_scan import run as run_whales
    settings = _settings_from(args)
    print(run_whales(settings, max_markets=args.max_markets, min_size=args.min_size))


def cmd_fees(args) -> None:
    """Audit config/fees.py against what Kalshi actually charged.

    Two jobs. First, prove the fee model still matches reality -- Kalshi has
    changed sports fee multipliers before, and a stale model would quietly
    corrupt every maker-vs-taker comparison. Second, surface the MAKER rate on
    `quadratic_with_maker_fees` series, which is the one number in the model
    that is taken from published docs rather than measured, because no maker
    fill has ever landed on such a series on this account."""
    from config.fees import (audit_fills, fee_cents_per_contract, maker_saving_cents,
                             series_fee_params, MAKER_RATE, TAKER_RATE)
    settings = _settings_from(args)
    client = KalshiClient(settings)
    fills = client.get_fills(limit=1000)
    a = audit_fills(fills, client)

    print("FEE MODEL AUDIT")
    print("=" * 62)
    print(f"  fills examined : {a['n']}  ({a['n_taker']} taker, {a['n_maker']} maker)")
    print(f"  total fees paid: ${a['total_fee']:.4f} on ${a['total_notional']:.2f} notional "
          f"({100 * a['total_fee'] / a['total_notional']:.2f}%)" if a["total_notional"]
          else "  no notional")
    matched = a["n"] - len(a["mismatches"])
    print(f"  model matches  : {matched}/{a['n']} exactly")
    if a["mismatches"]:
        print(f"\n{len(a['mismatches'])} fill(s) the model did not reproduce:")
        for m in a["mismatches"][:10]:
            ratio = (m["actual"] / m["predicted"]) if m["predicted"] else 0
            print(f"    {m['ticker']:<36} {'taker' if m['is_taker'] else 'MAKER'} "
                  f"C={m['count']:<5} P={m['price']:<5} "
                  f"pred=${m['predicted']:.4f} actual=${m['actual']:.4f} ({ratio:.2f}x)")
        if any(not m["is_taker"] for m in a["mismatches"]):
            print("\nA MAKER mismatch is the important one: it means the real maker rate")
            print("    differs from the assumed MAKER_RATE. Update config/fees.py.")

    print("\nPER-SERIES FEE PARAMS + MAKER SAVING (at a 50c contract, per contract)")
    print("-" * 62)
    print(f"  {'series':<13} {'fee_type':<28} {'mult':>5} {'taker':>7} {'maker':>7} {'save':>7}")
    for srs in settings.sport_series_prefixes:
        tk = f"{srs}-PROBE"
        ft, mult = series_fee_params(tk, client)
        t = fee_cents_per_contract(0.50, tk, True, client)
        mk = fee_cents_per_contract(0.50, tk, False, client)
        flag = "  <- maker FREE" if mk == 0 else ""
        print(f"  {srs:<13} {ft:<28} {mult:>5} {t:>6.3f}c {mk:>6.3f}c "
              f"{maker_saving_cents(0.50, tk, client):>6.3f}c{flag}")
    print(f"\ntaker rate {TAKER_RATE}, maker rate {MAKER_RATE} "
          f"(maker rate is UNVERIFIED -- see config/fees.py)")



def cmd_loop(args) -> None:
    # The loop is long-running and unattended, so it needs INFO-level visibility
    # (cycle completion, paper-trigger decisions, notification sends) -- the shared
    # _setup() defaults to WARNING, which silently dropped all of that. Raise the
    # root logger to INFO for the loop only; keep noisy HTTP libraries quiet.
    logging.getLogger().setLevel(logging.INFO)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    settings = _settings_from(args)
    if args.interval:
        settings = replace(settings, scan_interval_seconds=args.interval)
    if args.paper:
        settings = replace(settings, paper_trade=True)
    if getattr(args, "maker", False) and not args.live:
        print("--maker only applies with --live (it manages REAL resting orders); ignoring.")
    if getattr(args, "in_game", False) and not args.live:
        print("--in-game only applies with --live (it gates REAL in-game orders); ignoring.")
    if args.live:
        in_game = bool(getattr(args, "in_game", False))
        settings = replace(settings, live_trade=True,
                           maker_mode=bool(getattr(args, "maker", False)),
                           in_game_trade=in_game,
                           # In-game fair value is only produced when pregame_only is off;
                           # both switches are required, so --in-game sets both.
                           pregame_only=(settings.pregame_only and not in_game))
        print("\n" + "=" * 70)
        print("LIVE TRADING ENABLED — this will submit REAL orders with REAL money.")
        print(f"Environment: {'DEMO' if settings.use_demo else 'PRODUCTION'}")
        print(f"Per-order caps: <= {settings.max_order_contracts} contracts, "
             f"<= ${settings.max_order_cost_usd:,.0f} cost. No other limit.")
        if settings.in_game_trade:
            print("")
            print(f"IN-GAME TRADING ON: games that have ALREADY STARTED stay eligible for "
                  f"{settings.in_game_max_minutes:.0f} min after first pitch,")
            print(f"  at a stiffer edge bar ({settings.in_game_min_edge_cents:.1f}c vs "
                  f"{settings.min_edge_cents:.1f}c pregame).")
            print("  MLB WINNER MARKETS ONLY -- NFL/NBA/NHL have no live model, and MLB")
            print("  totals' expected_runs() is a full-game estimate; all are refused mid-game.")
            print("  Backtested (backtest/in_game_backtest.py): n=50, TRAIN +0.246 / "
                  "VALIDATE +0.268 net of fees.")
        if settings.maker_mode:
            print("")
            print(f"MAKER MODE ON: orders REST on the book at best_bid+"
                  f"{settings.maker_improve_cents:.0f}c (post-only), re-pegged every "
                  f"{settings.maker_poll_seconds:.0f}s,")
            print(f"  crossing to a taker order at T-{settings.maker_taker_fallback_min:.0f} "
                  f"min if still unfilled. Ledger rows are written ON FILL.")
            print("  NOTE: backtest says resting costs more entry price than it saves in "
                  "fees (taker +0.0419 vs limit -0.0114).")
        print("=" * 70 + "\n")
    run_loop(settings)


def cmd_settle(args) -> None:
    from engine.paper import PaperLedger
    from backtest.evaluate import _game_date
    from data.games import resolve_outcomes
    from config.sports import sport_of, market_kind
    bets = PaperLedger().load()
    if args.date:
        bets = [b for b in bets if _game_date(b["ticker"]) == args.date]
    if getattr(args, "sport", None):
        want = args.sport.lower()
        bets = [b for b in bets if sport_of(b["ticker"]) == want]
        if not bets:
            print(f"No {want.upper()} bets in the paper ledger"
                  f"{' for ' + args.date if args.date else ''}.")
            return
    if not bets:
        print("Paper ledger is empty (run `loop --paper` to accumulate bets).")
        return
    outcomes = resolve_outcomes({b["ticker"] for b in bets}, build_clients())
    settled = [b for b in bets if b["ticker"] in outcomes]
    pending = [b for b in bets if b["ticker"] not in outcomes]

    print(f"\n=== PAPER LEDGER: {len(bets)} bets ({len(settled)} settled, {len(pending)} pending) ===")
    if settled:
        print(f"{'RESULT':<7}{'BET':<34}{'ENTRY':>6}{'CT':>5}{'NET$':>9}  MARKET")
        print("-" * 82)
        pnl = staked = wins = dollar = 0.0
        # Per-sport / per-market-kind tallies. MLB and NFL share one account, ledger and
        # dashboard, but they are two independently-validated models (NFL's has no real
        # regular-season track record yet -- see docs/research-log.md), so a single blended P&L line
        # hides which one is actually working. Keyed off config/sports.py, so a third
        # sport would show up here automatically.
        tally: dict[str, dict[str, dict]] = {}

        def _bucket(sport: str, kind: str) -> dict:
            return tally.setdefault(sport, {}).setdefault(
                kind, {"n": 0, "w": 0, "pnl": 0.0, "staked": 0.0, "dollar": 0.0})

        for b in sorted(settled, key=lambda x: x["first_pitch"]):
            yes_won = outcomes[b["ticker"]]
            won = yes_won if b["side"] == "YES" else (not yes_won)
            per = (1 - b["entry_price"]) if won else -b["entry_price"]
            ct = b.get("contracts") or 0
            dollar += per * ct
            pnl += per
            staked += b["entry_price"]
            wins += int(won)
            st = _bucket(sport_of(b["ticker"]), market_kind(b["ticker"]))
            st["n"] += 1
            st["w"] += int(won)
            st["pnl"] += per
            st["staked"] += b["entry_price"]
            st["dollar"] += per * ct
            print(f"{'WIN ' if won else 'LOSS':<7}{b['label']:<34}{b['entry_price']:>6.2f}"
                  f"{ct:>5}{per*ct:>9.2f}  {b['ticker']}")
        n = len(settled)
        print("-" * 82)
        print(f"Record {int(wins)}-{n-int(wins)} ({wins/n:.0%})   P&L {pnl:+.2f}/contract   "
              f"ROI {pnl/staked:+.0%}   net ${dollar:+.2f}")

        def _line(label: str, st: dict, indent: str = "") -> str:
            roi = f"{st['pnl'] / st['staked']:+.0%}" if st["staked"] else "  n/a"
            return (f"{indent}{label:<14}{st['w']:>3}-{st['n'] - st['w']:<3} "
                    f"({st['w'] / st['n']:>3.0%})   ROI {roi:>5}   net ${st['dollar']:+.2f}")

        if len(tally) > 1 or any(len(k) > 1 for k in tally.values()):
            print("\nBY SPORT")
            for sport in sorted(tally):
                kinds = tally[sport]
                roll = {"n": 0, "w": 0, "pnl": 0.0, "staked": 0.0, "dollar": 0.0}
                for st in kinds.values():
                    for k in roll:
                        roll[k] += st[k]
                print(_line(sport.upper(), roll, "  "))
                for kind in sorted(kinds):
                    print(_line(kind, kinds[kind], "      "))

        sport_bits = []
        for sport in sorted(tally):
            roll_n = sum(st["n"] for st in tally[sport].values())
            roll_w = sum(st["w"] for st in tally[sport].values())
            roll_d = sum(st["dollar"] for st in tally[sport].values())
            sport_bits.append(f"{sport.upper()} {roll_w}-{roll_n - roll_w} ${roll_d:+.2f}")
        summary = (f"KALSHI settle: {int(wins)}-{n-int(wins)} ({wins/n:.0%}), "
                   f"net ${dollar:+.2f}, ROI {pnl/staked:+.0%}"
                   + (f" [{' | '.join(sport_bits)}]" if len(tally) > 1 else "")
                   + f". Pending: {len(pending)}.")
    else:
        summary = f"KALSHI settle: 0 bets settled, {len(pending)} pending."
    if pending:
        print(f"\nPending ({len(pending)} not final yet):")
        for b in sorted(pending, key=lambda x: x["first_pitch"]):
            fp_et = dateparser.parse(b["first_pitch"]).astimezone(EASTERN)
            print(f"   {b['label']:<34} @ {b['entry_price']:.2f}  "
                  f"(first pitch {fp_et:%Y-%m-%d %I:%M %p} ET)")

    if getattr(args, "notify", False):
        from engine.notify import SmsNotifier
        SmsNotifier().send(summary)
        print(f"\n[texted summary: {summary}]")


def cmd_calibration(args) -> None:
    from signals.calibration import build_calibration, MIN_BUCKET_N, PRIOR_STRENGTH
    cal = build_calibration()
    if not cal.bucket_stats:
        print("No settled bets yet (backtest history + paper ledger are both empty).")
        return
    print(f"\n=== CONFIDENCE CALIBRATION (min sample {MIN_BUCKET_N}, prior strength {PRIOR_STRENGTH}) ===")
    for key, s in cal.bucket_stats.items():
        note = s.get("note", "")
        print(f"  {key:<8} n={s['n']:<4} win_rate={s['win_rate']:.0%}  "
              f"expected={s['expected_wr']:.0%}  multiplier={s['multiplier']:.2f}x  {note}")
    print("\nMultiplier scales Recommendation.confidence for that market type; 1.00x = neutral "
          "(either below min sample, or realized performance matches what the model expected).")


def cmd_dash(args) -> None:
    from output.dashboard import build_dashboard
    settings = _settings_from(args)
    path = build_dashboard(settings, open_after=not args.no_open)
    print(f"Dashboard written to {path}")


# --------------------------------------------------------------------------- #
# place (guarded)
# --------------------------------------------------------------------------- #
def cmd_place(args) -> None:
    from engine.execution import preview_order, submit_order, OrderRequest, ExecutionDisabled
    settings = _settings_from(args)
    req = OrderRequest(ticker=args.ticker, side=args.side, action=args.action,
                       count=args.count, limit_price=args.price)
    client = KalshiClient(settings)
    preview = preview_order(client, req, settings)
    print(preview.render())
    if not preview.ok:
        print("\nBlocked by a safety check above — not sending.")
        return
    if not args.confirm:
        print("\nDRY RUN. This did NOT place an order.")
        print(f"To place it yourself, re-run with --confirm:\n"
              f"  py -3 cli.py place {args.ticker} --side {args.side} "
              f"--count {args.count} --price {args.price} --confirm")
        return
    try:
        result = submit_order(client, req, settings)
        print(f"\nORDER SUBMITTED: {result}")
    except ExecutionDisabled as e:
        print(f"\nLive execution is disabled: {e}")


def cmd_cancel(args) -> None:
    settings = _settings_from(args)
    client = KalshiClient(settings)
    result = client.cancel_order(args.order_id)
    print(f"CANCELED {args.order_id}: {result}")


# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="kalshi", description="Kalshi MLB money-flow agent")
    p.add_argument("--allow-live", action="store_true", help="include in-progress games")
    p.add_argument("--demo", action="store_true", help="use Kalshi demo API")
    p.add_argument("--bankroll", type=float, help="bankroll for advisory sizing")
    p.add_argument("--max-stake-pct", type=float, dest="max_stake_pct",
                   help="cap on suggested stake as a fraction of bankroll (default 0.05 = 5%%)")
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("search", help="scan markets, show money-flow reads")
    s.add_argument("--top", type=int, default=25)
    s.set_defaults(func=cmd_search)

    s = sub.add_parser("rank", help="ranked actionable recommendations")
    s.add_argument("--top", type=int, default=None)
    s.set_defaults(func=cmd_rank)

    s = sub.add_parser("inspect", help="deep dive on one market ticker")
    s.add_argument("ticker")
    s.set_defaults(func=cmd_inspect)

    for name, fn, helptext in [("status", cmd_status, "account + exchange summary"),
                               ("positions", cmd_positions, "open positions"),
                               ("orders", cmd_orders, "resting orders")]:
        s = sub.add_parser(name, help=helptext)
        s.set_defaults(func=fn)

    s = sub.add_parser("results", help="MLB game outcomes for a date")
    s.add_argument("date", nargs="?", default=None, help="YYYY-MM-DD (default: today)")
    s.set_defaults(func=cmd_results)

    s = sub.add_parser("outcome", help="how the agent's bets did on a given day")
    s.add_argument("date", nargs="?", default=None, help="YYYY-MM-DD (default: today)")
    s.set_defaults(func=cmd_outcome)

    s = sub.add_parser("backtest", help="validate signal vs realized outcomes")
    s.add_argument("--since", default=None, help="only snapshots on/after YYYY-MM-DD")
    s.set_defaults(func=cmd_backtest)

    s = sub.add_parser("whales", help="live large-print ('whale') tape from the anonymized feed")
    s.add_argument("--max-markets", type=int, default=40, help="deep-scan budget (bounds API load)")
    s.add_argument("--min-size", type=float, default=0.0, help="only show prints of >= N contracts")
    s.set_defaults(func=cmd_whales)

    s = sub.add_parser("fees", help="audit the fee model against real fills; show maker savings")
    s.set_defaults(func=cmd_fees)

    s = sub.add_parser("loop", help="continuous autonomous loop")
    s.add_argument("--interval", type=int, default=None, help="seconds between scans")
    s.add_argument("--paper", action="store_true",
                   help="paper-bet each game on the last cycle before first pitch")
    s.add_argument("--live", action="store_true",
                   help="LIVE: automatically submit real orders (real money, no confirmation) "
                        "on the last cycle before each game starts")
    s.add_argument("--in-game", action="store_true",
                   help="with --live: ALSO bet games that have already started (MLB only -- "
                        "the other sports have no live model and are refused). Sets both "
                        "pregame_only=False and in_game_trade=True. UNVALIDATED: there is no "
                        "backtest for in-game entries and almost no history to build one from")
    s.add_argument("--maker", action="store_true",
                   help="with --live: REST orders on the book (post-only, best_bid+1c) and "
                        "re-peg them instead of crossing the spread, falling back to a taker "
                        "order shortly before each game starts. Saves the taker fee; costs "
                        "entry price. See config/settings.py's maker_mode for the backtest "
                        "that says this is NOT yet a win overall")
    s.set_defaults(func=cmd_loop)

    s = sub.add_parser("settle", help="grade the paper-trade ledger vs outcomes")
    s.add_argument("date", nargs="?", default=None, help="filter to games on YYYY-MM-DD")
    s.add_argument("--notify", action="store_true", help="text a summary of the results")
    s.add_argument("--sport", default=None, choices=["mlb", "nfl", "nba", "nhl"],
                   help="grade only one sport's bets (default: all, broken out by sport)")
    s.set_defaults(func=cmd_settle)

    s = sub.add_parser("calibration", help="show adaptive confidence calibration by market type")
    s.set_defaults(func=cmd_calibration)

    s = sub.add_parser("dash", help="build + open a static PNG dashboard")
    s.add_argument("--no-open", action="store_true", help="write the PNG without opening it")
    s.set_defaults(func=cmd_dash)

    s = sub.add_parser("place", help="PREVIEW/place an order (dry-run unless --confirm)")
    s.add_argument("ticker")
    s.add_argument("--side", choices=["yes", "no"], required=True)
    s.add_argument("--action", choices=["buy", "sell"], default="buy")
    s.add_argument("--count", type=float, required=True, help="number of contracts (fractional OK)")
    s.add_argument("--price", type=float, required=True, help="limit price in dollars (0-1)")
    s.add_argument("--confirm", action="store_true", help="actually submit (you run this)")
    s.set_defaults(func=cmd_place)

    s = sub.add_parser("cancel", help="cancel a resting order by id")
    s.add_argument("order_id")
    s.set_defaults(func=cmd_cancel)
    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    _setup(getattr(args, "verbose", False))
    args.func(args)


if __name__ == "__main__":
    main()
