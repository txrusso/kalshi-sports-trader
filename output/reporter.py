"""Render + persist recommendations. Console table + JSON + JSONL history."""
from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path

from config.settings import EASTERN, OUTPUT_DIR
from config.sports import is_total
from signals.recommendation import Recommendation

# The book/trade/oi breakdown and calibration multiplier live only inside the
# pre-built `rationale` string (see signals/recommendation.py::_rationale) --
# pulled back out here for the console's own layout rather than duplicated at
# the source, so the two can't drift out of sync.
_BOOK_TRADE_OI_RE = re.compile(
    r"book ([+-]?\d+\.\d+), trades ([+-]?\d+\.\d+), oi ([+-]?\d+\.\d+)")
_CALIBRATION_RE = re.compile(r"calibration (\d+\.\d+)x")


def bet_label(ticker: str, yes_team: str, side: str) -> str:
    """Human-readable description of a bet from its ticker + YES subtitle + side."""
    y = yes_team or ""
    if is_total(ticker):
        line = y.lower().replace("over", "").replace("runs scored", "").replace(
            "points scored", "").strip()
        return f"{'Over' if side == 'YES' else 'Under'} {line}"
    return y if side == "YES" else f"{y} loses"             # winner market


def _bet_label(r: Recommendation) -> str:
    return bet_label(r.ticker, r.yes_team, r.side)


def _fmt_rec_block(i: int, r: Recommendation) -> list[str]:
    """Three short lines per rec instead of one 130+ char row -- avoids the
    terminal-wrap garbling a single wide line used to produce, and drops info
    that was shown twice (the old rationale line repeated the headline, price,
    and fair% that the header line already carries)."""
    edge = f"{r.edge_cents:+.1f}c" if r.edge_cents is not None else "n/a"
    fair = f"{r.fair_prob:.0%}" if r.fair_prob is not None else "--"
    flag = "⚠ " if r.conflict else ""
    headline = r.headline or _bet_label(r)     # fallback for any pre-headline Recommendation

    head = f"{i:>2}. {flag}{headline}"
    stats = f"edge {edge:>6}   conf {r.confidence:.2f}   fair {fair:>4}"
    lines = [f"{head:<52} {stats}"]

    lines.append(f"      {r.ticker} @ {r.entry_price:.2f}   "
                 f"stake ${r.suggested_stake_usd:.2f} ({r.suggested_contracts:.2f}ct)")

    flow = f"flow {r.flow_direction} {r.money_flow_score:+.2f}"
    bto = _BOOK_TRADE_OI_RE.search(r.rationale or "")
    if bto:
        flow += f"  (book {float(bto.group(1)):+.2f} / trades {float(bto.group(2)):+.2f} / oi {float(bto.group(3)):+.2f})"

    tail = f"src {r.fair_source}" if r.fair_source else "flow-only (no fair-value model)"
    cal = _CALIBRATION_RE.search(r.rationale or "")
    if cal:
        tail += f"   calib {float(cal.group(1)):.2f}x"
    if r.conflict:
        tail += "   ⚠ flow conflicts with fair value"
    lines.append(f"      {flow}   |   {tail}")
    return lines


def render_console(recs: list[Recommendation], scanned: int, deep: int) -> str:
    ts = datetime.now(EASTERN).strftime("%Y-%m-%d %H:%M:%S ET")
    lines = [
        "=" * 100,
        f" KALSHI — money-flow recommendations   {ts}   (RECOMMEND-ONLY, no orders placed)",
        f" scanned {scanned} markets, deep-scanned {deep}, {len(recs)} recommendations",
        "-" * 100,
    ]
    if not recs:
        lines.append("  (no markets cleared the edge/confidence thresholds this cycle)")
    for i, r in enumerate(recs, 1):
        lines.extend(_fmt_rec_block(i, r))
        lines.append("")
    lines.append("=" * 100)
    return "\n".join(lines).rstrip("\n")


def write_outputs(recs: list[Recommendation], scanned: int, deep: int,
                  out_dir: Path = OUTPUT_DIR) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(EASTERN)
    payload = {
        "generated_at": ts.isoformat(),
        "mode": "recommend-only",
        "scanned": scanned,
        "deep_scanned": deep,
        "recommendations": [r.__dict__ for r in recs],
    }
    (out_dir / "recommendations_latest.json").write_text(
        json.dumps(payload, indent=2, default=str), encoding="utf-8")
    # No recommendations_history.jsonl any more (dropped 2026-09-23): every cycle
    # appended the full payload to it and NOTHING ever read it back -- not a
    # backtest, not the dashboard, not the CLI. It had reached 20 MB and grew
    # about 1 MB a day. Everything it held is already covered: the current cycle
    # by recommendations_latest.json, and the history by snapshots/*.jsonl, which
    # the backtests actually do read.
