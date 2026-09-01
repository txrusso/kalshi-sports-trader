"""Kalshi MLB money-flow agent — entrypoint.

RECOMMEND-ONLY: scans markets, follows the money, and prints/persists ranked
trade recommendations. It never places, cancels, or modifies any order.

Examples:
    py -3 run.py --once            # single scan cycle
    py -3 run.py                   # continuous loop (~20 min cadence)
    py -3 run.py --interval 900    # loop every 15 min
    py -3 run.py --once --top 10
"""
from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import replace

from config.settings import DEFAULTS, LOGS_DIR
from engine.loop import run_loop, run_once


def _setup_logging(verbose: bool) -> None:
    # Windows consoles default to cp1252; force UTF-8 so symbols never crash output.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError):
            pass
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    handlers = [logging.StreamHandler(), logging.FileHandler(LOGS_DIR / "agent.log", encoding="utf-8")]
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        handlers=handlers,
    )
    # Quiet noisy libraries.
    logging.getLogger("urllib3").setLevel(logging.WARNING)


def main() -> None:
    p = argparse.ArgumentParser(description="Kalshi MLB money-flow agent (recommend-only)")
    p.add_argument("--once", action="store_true", help="run a single scan cycle and exit")
    p.add_argument("--interval", type=int, help="loop interval in seconds")
    p.add_argument("--top", type=int, help="max recommendations to show")
    p.add_argument("--bankroll", type=float, help="bankroll for advisory sizing (USD)")
    p.add_argument("--demo", action="store_true", help="use Kalshi demo API")
    p.add_argument("--allow-live", action="store_true",
                   help="also recommend in-progress games (default: pre-game only)")
    p.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    args = p.parse_args()

    _setup_logging(args.verbose)

    settings = DEFAULTS
    overrides = {}
    if args.interval is not None:
        overrides["scan_interval_seconds"] = args.interval
    if args.top is not None:
        overrides["top_n_recommendations"] = args.top
    if args.bankroll is not None:
        overrides["bankroll_usd"] = args.bankroll
        overrides["dynamic_bankroll"] = False
    if args.demo:
        overrides["use_demo"] = True
    if args.allow_live:
        overrides["pregame_only"] = False
    if overrides:
        settings = replace(settings, **overrides)

    if args.once:
        run_once(settings)
    else:
        run_loop(settings)


if __name__ == "__main__":
    main()
