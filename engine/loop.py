"""The agent loop: run scan cycles on a cadence, emit + persist recommendations."""
from __future__ import annotations

import ctypes
import logging
import sys
import time
from dataclasses import replace
from datetime import datetime, timezone

from config.settings import EASTERN, Settings, DEFAULTS
from data.fair_value import FairValueModel, FairValueRouter
from data.fair_value_nfl import NflFairValueModel
from data.fair_value_nfl_totals import NflTotalsFairValueModel
from data.fair_value_nfl_spread import NflSpreadFairValueModel
from data.fair_value_nba import NbaFairValueModel
from data.fair_value_nba_totals import NbaTotalsFairValueModel
from data.fair_value_nhl import NhlFairValueModel
from data.fair_value_nhl_totals import NhlTotalsFairValueModel
from data.fair_value_totals import TotalsFairValueModel
from data.mlb_stats import MlbStatsClient
from data.nfl_data import NflDataClient
from data.nba_data import NbaDataClient
from data.nhl_data import NhlDataClient
from engine.notify import (SmsNotifier, PushNotifier, format_bet_sms, format_bet_push,
                          format_order_placed_sms, format_order_placed_push)
from engine.live import LiveLedger, run_live_trigger
from engine.paper import PaperLedger, run_paper_trigger
from engine.scanner import run_scan
from engine.snapshot import SnapshotStore
from kalshi.client import KalshiClient
from output.reporter import render_console, write_outputs
from signals.calibration import build_calibration
from signals.elo_nfl import EloRatings
from signals.elo_nba import EloRatings as NbaEloRatings
from signals.elo_nhl import EloRatings as NhlEloRatings

log = logging.getLogger("engine.loop")


def build_context(settings: Settings = DEFAULTS):
    client = KalshiClient(settings)
    mlb = MlbStatsClient()
    nfl = NflDataClient()
    nba = NbaDataClient()
    nhl = NhlDataClient()
    elo = EloRatings(nfl)
    nba_elo = NbaEloRatings(nba)
    nhl_elo = NhlEloRatings(nhl)
    fair_router = FairValueRouter(
        mlb_winner=FairValueModel(mlb, settings),
        mlb_totals=TotalsFairValueModel(mlb, settings),
        nfl_winner=NflFairValueModel(nfl, elo, settings),
        nfl_totals=NflTotalsFairValueModel(nfl, settings),
        nfl_spread=NflSpreadFairValueModel(nfl, elo, settings),
        nba_winner=NbaFairValueModel(nba, nba_elo, settings),
        nba_totals=NbaTotalsFairValueModel(nba, settings),
        nhl_winner=NhlFairValueModel(nhl, nhl_elo, settings),
        nhl_totals=NhlTotalsFairValueModel(nhl, settings),
    )
    store = SnapshotStore()
    try:
        calibration = build_calibration()
        adjusted = {k: v for k, v in calibration.multipliers.items() if v != 1.0}
        if adjusted:
            log.info("Calibration active: %s", adjusted)
    except Exception as e:
        log.warning("Calibration build failed (%s); proceeding with neutral confidence.", e)
        calibration = None
    return client, mlb, nfl, nba, nhl, fair_router, store, calibration


def resolve_bankroll(client: KalshiClient, settings: Settings) -> Settings:
    """Set the sizing bankroll to the live account balance (pulled once per run).

    Falls back to the configured bankroll_usd if disabled or the fetch fails.
    """
    if not settings.dynamic_bankroll:
        return settings
    try:
        bal = client.balance()
        cents = bal.get("balance")
        usd = float(cents) / 100.0 if cents is not None else float(bal.get("balance_dollars") or 0)
    except Exception as e:
        log.warning("Balance fetch failed (%s); using fallback bankroll $%.2f",
                    e, settings.bankroll_usd)
        return settings
    if usd <= 0:
        log.warning("Live balance is $0.00 — suggested sizes will be $0 until funded.")
    print(f"Bankroll for this run: ${usd:.2f} (live account balance)")
    return replace(settings, bankroll_usd=usd)


def run_once(settings: Settings = DEFAULTS, ctx=None) -> list:
    standalone = ctx is None
    client, mlb, nfl, nba, nhl, fair_router, store, calibration = ctx or build_context(settings)
    if standalone:
        settings = resolve_bankroll(client, settings)

    prev_state = store.load_prev_state()
    result = run_scan(client, fair_router, prev_state, settings, calibration)

    store.write_state(result.new_state)
    store.append_rows(result.snapshot_rows)
    write_outputs(result.recommendations, result.scanned, result.deep_scanned)

    print(render_console(result.recommendations, result.scanned, result.deep_scanned))

    # Both triggers fire on the same last-cycle-before-first-pitch window.
    window = settings.scan_interval_seconds / 60.0 + settings.paper_trigger_buffer_min

    # Live trigger: submit REAL orders through the Kalshi API. No confirmation
    # step -- see engine/live.py for exactly what is and isn't guarded.
    if settings.live_trade:
        live_ledger = LiveLedger()
        live_placed = run_live_trigger(result.recommendations,
                                       {"mlb": mlb, "nfl": nfl, "nba": nba, "nhl": nhl},
                                       live_ledger, client, settings, window)
        if live_placed:
            try:
                print(f"\n>> LIVE ORDER {len(live_placed)} placed "
                     f"(recorded to output/live_ledger.jsonl):")
                for b in live_placed:
                    print(f"   {b['side']:>3} {b['label']:<22} @ {b['entry_price']:.2f}  "
                         f"order {b.get('order_id')} ({b.get('order_status')})")
            except Exception:
                log.exception("Failed to print live-order announcement (orders are still recorded).")
            notifier = SmsNotifier()
            pusher = PushNotifier()
            for b in live_placed:
                try:
                    pusher.send(format_order_placed_push(b))
                except Exception:
                    log.exception("Push alert crashed for live order %r (order still placed).",
                                 b.get("label"))
                try:
                    sent = notifier.send(format_order_placed_sms(b))
                    if not sent:
                        log.warning("SMS not sent for live order %r -- check notify_config.txt.",
                                   b.get("label"))
                except Exception:
                    log.exception("Text alert crashed for live order %r (order still placed).",
                                 b.get("label"))

    # Paper-trade trigger: bet each game on the last cycle before its first pitch.
    if settings.paper_trade:
        ledger = PaperLedger()
        placed = run_paper_trigger(result.recommendations,
                                   {"mlb": mlb, "nfl": nfl, "nba": nba, "nhl": nhl},
                                   ledger, window)
        if placed:
            # The bets are already recorded to the ledger above -- announcing/texting them
            # is best-effort. A formatting hiccup here must never look like "nothing happened";
            # log it loudly and still attempt the text with whatever we can format safely.
            try:
                print(f"\n>> PAPER-BET {len(placed)} game(s) starting within {window:.0f} min "
                      f"(recorded to output/paper_ledger.jsonl):")
                for b in placed:
                    mb = b.get("minutes_before")
                    mb_str = f"{mb:.0f}" if isinstance(mb, (int, float)) else "?"
                    print(f"   {b['side']:>3} {b['label']:<22} @ {b['entry_price']:.2f}  "
                          f"edge {b['edge_cents']}c  ({mb_str} min before first pitch)")
            except Exception:
                log.exception("Failed to print paper-bet announcement (bets are still recorded).")
            # Text the user so they can place the bets on the Kalshi app. One SMS per
            # bet (not bundled) -- a bundled multi-bet message can exceed the carrier
            # gateway's silent truncation limit and get cut off mid-word.
            # Phone-first: an ntfy push with a full order slip + tap-through to the
            # Kalshi market (PushNotifier). SMS is kept as a backup delivery channel.
            # Both are recommend-only -- they tell the user what to place; neither
            # ever submits an order. Each is independent and best-effort.
            notifier = SmsNotifier()
            pusher = PushNotifier()
            for b in placed:
                try:
                    pusher.send(format_bet_push(b))
                except Exception:
                    log.exception("Push alert crashed for bet %r (bet is still recorded).", b.get("label"))
                try:
                    sent = notifier.send(format_bet_sms(b))
                    if not sent:
                        log.warning("SMS not sent for bet %r -- check notify_config.txt.", b.get("label"))
                except Exception:
                    log.exception("Text alert crashed for bet %r (bet is still recorded).", b.get("label"))

    log.info("Cycle complete: %d recommendations.", len(result.recommendations))
    return result.recommendations


def _keep_awake() -> None:
    """Ask Windows not to idle-sleep while the loop runs (like a media player).

    ES_SYSTEM_REQUIRED alone blocks classic sleep but NOT the display from
    turning off. On this machine (Modern Standby / S0 Low Power Idle), a dark
    display previously triggered background-process throttling regardless of
    ES_SYSTEM_REQUIRED (confirmed incident, 2026-08-07/08), which is why
    ES_DISPLAY_REQUIRED was added to force the display to also stay on.
    Changed 2026-09-01 to drop ES_DISPLAY_REQUIRED (system-sleep-only, screen
    allowed to turn off) at the user's request, to be watched for a cycle or
    two -- if scans start getting delayed/missed again, re-add
    ES_DISPLAY_REQUIRED (0x00000002) to the flags below. Belt-and-suspenders
    with the AC power-plan timeouts (see docs/research-log.md); does not override a
    manual sleep, lid-close, or shutdown. Windows only.
    """
    if sys.platform != "win32":
        return
    ES_CONTINUOUS = 0x80000000
    ES_SYSTEM_REQUIRED = 0x00000001
    try:
        ctypes.windll.kernel32.SetThreadExecutionState(
            ES_CONTINUOUS | ES_SYSTEM_REQUIRED)
        log.info("Requested keep-awake (system will not idle-sleep; display may turn off).")
    except Exception as e:
        log.warning("Could not set keep-awake: %s", e)


def run_loop(settings: Settings = DEFAULTS) -> None:
    _keep_awake()
    ctx = build_context(settings)
    settings = resolve_bankroll(ctx[0], settings)   # pull balance once, before the day's bets
    log.info("Starting loop; interval=%ds, demo=%s", settings.scan_interval_seconds, settings.use_demo)
    cycle = 0
    while True:
        cycle += 1
        started = time.time()
        try:
            run_once(settings, ctx=ctx)
        except KeyboardInterrupt:
            log.info("Interrupted; shutting down.")
            break
        except Exception:
            log.exception("Cycle %d failed; continuing to next cycle.", cycle)
        elapsed = time.time() - started
        sleep_for = max(5, settings.scan_interval_seconds - int(elapsed))
        next_at = datetime.now(timezone.utc).timestamp() + sleep_for
        log.info("Cycle %d done in %.1fs; next in %ds (~%s).", cycle, elapsed, sleep_for,
                 datetime.fromtimestamp(next_at, EASTERN).strftime("%H:%M:%S ET"))
        try:
            time.sleep(sleep_for)
        except KeyboardInterrupt:
            log.info("Interrupted during sleep; shutting down.")
            break
