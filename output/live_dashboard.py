"""Live terminal dashboard (TUI) for the Kalshi money-flow agent.

READ-ONLY VIEWER, SEPARATE PROCESS. This never places, cancels or modifies an
order, and never writes to the ledgers or the snapshot store. It is a window
onto files the trading loop already writes, plus a few read-only Kalshi GETs.

Why a separate process rather than a widget inside `cli.py loop`:
`run_paper_loop.bat` runs the loop under a Windows scheduled task and pipes its
stdout through PowerShell into `logs/paper_loop.log`. A TUI cannot render into a
pipe -- it needs a real terminal -- so wiring one into the loop would mean
giving up that log, and putting a rendering layer inside the process that moves
real money. The loop is therefore left exactly as it is, and this attaches to
what it already emits:

    loop  ->  output/recommendations_latest.json   (rewritten every cycle)
          ->  output/paper_ledger.jsonl / live_ledger.jsonl
    here  <-  watches those files and redraws when they change

Run it in Windows Terminal (any width; the tables scroll horizontally):

    .venv\\Scripts\\python.exe -m output.live_dashboard     (or scripts\\run_dashboard.bat)

Keys: q quit, r force refresh, p / s switch the Pending / Settled tabs.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional

from rich.text import Text

from config.settings import DEFAULTS, EASTERN, OUTPUT_DIR
from config.sports import sport_of
# The book/trades/oi component scores and the calibration multiplier exist in
# full ONLY inside the pre-built `rationale` string: signals/money_flow.py's
# debug dict keeps the raw inputs and, in cross-market mode, the un-blended
# `cross`/`within` halves -- not the blended per-component score the console
# shows. output/reporter.py already scrapes them back out for its own layout,
# so reuse ITS regexes rather than writing a second pair that could drift.
from output.reporter import _BOOK_TRADE_OI_RE, _CALIBRATION_RE

ROOT = Path(__file__).resolve().parents[1]
LATEST = OUTPUT_DIR / "recommendations_latest.json"

# Explicit hex, not the ANSI names ("green"/"red"/...): those are indexes into
# the terminal's own 16-colour palette, so a monochrome or heavily-restyled
# Windows Terminal scheme renders the whole dashboard in greys and the
# green/red profit signal disappears. Truecolor bypasses the palette.
GREEN = "bold #3fb950"
RED = "bold #f85149"
CYAN = "#58a6ff"
YELLOW = "bold #e3b341"
WHITE = "#e6edf3"
MUTED = "#8b949e"
DIM = "dim"


# --------------------------------------------------------------------------
# data loading (pure functions -- no UI, so --selftest can exercise them)
# --------------------------------------------------------------------------

@dataclass
class ScanData:
    """One scan cycle, as the loop wrote it to recommendations_latest.json."""
    generated_at: Optional[datetime]
    scanned: int
    deep_scanned: int
    recs: list[dict]
    mtime: float


@dataclass
class LoopInfo:
    """What the trading loop is actually doing, read off its command line.

    The dashboard is a separate process, so it cannot know the loop's flags by
    asking. The JSON payload's own "mode" field is hardcoded to "recommend-only"
    in output/reporter.py regardless of how the loop was started, so it cannot
    be trusted for this. The live command line is the only honest source;
    run_paper_loop.bat (what the scheduled task launches) is the fallback, and
    then the config defaults.
    """
    running: bool = False
    pid: Optional[int] = None
    interval: int = DEFAULTS.scan_interval_seconds
    mode: str = "unknown"
    source: str = "defaults"


@dataclass
class AccountData:
    """Everything that only changes when a bet settles or a scan completes."""
    balance: Optional[float] = None
    today: tuple[int, int, float] = (0, 0, 0.0)
    overall: tuple[int, int, float] = (0, 0, 0.0)
    today_by_sport: list[tuple[str, tuple[int, int, float]]] = field(default_factory=list)
    overall_by_sport: list[tuple[str, tuple[int, int, float]]] = field(default_factory=list)
    open_positions: Optional[int] = None
    pending: list[dict] = field(default_factory=list)
    settled: list[dict] = field(default_factory=list)
    unrealized: Optional[float] = None
    placed: dict[str, dict[str, datetime]] = field(default_factory=dict)
    error: str = ""


def load_scan(path: Path = LATEST) -> Optional[ScanData]:
    """Parse the loop's latest scan output. None if unreadable.

    A partial read is expected and normal: the loop rewrites this file while we
    may be reading it, so a JSONDecodeError here means "try again in a second",
    not an error worth surfacing.
    """
    try:
        mtime = path.stat().st_mtime
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    gen = payload.get("generated_at")
    try:
        generated_at = datetime.fromisoformat(gen) if gen else None
    except (TypeError, ValueError):
        generated_at = None
    return ScanData(
        generated_at=generated_at,
        scanned=int(payload.get("scanned") or 0),
        deep_scanned=int(payload.get("deep_scanned") or 0),
        recs=list(payload.get("recommendations") or []),
        mtime=mtime,
    )


def flow_components(rec: dict) -> tuple[Optional[float], Optional[float], Optional[float]]:
    """(book, trades, oi) component scores for one recommendation."""
    m = _BOOK_TRADE_OI_RE.search(rec.get("rationale") or "")
    if not m:
        return None, None, None
    return float(m.group(1)), float(m.group(2)), float(m.group(3))


def bet_side_prob(rec: dict) -> Optional[float]:
    """The model's probability that the BET wins, not P(YES).

    `fair_prob` is always P(YES), so on a NO bet (every Under, and a winner bet
    placed as NO on the opponent's ticker) it reads backwards next to the price
    actually paid. Flipping it here makes FAIR - PX equal EDGE on every row.
    """
    fair = rec.get("fair_prob")
    if fair is None:
        return None
    fair = float(fair)
    return 1.0 - fair if str(rec.get("side") or "").upper() == "NO" else fair


def calibration_mult(rec: dict) -> Optional[float]:
    """Confidence calibration multiplier: debug carries it at full precision,
    the rationale string is the fallback for older rows."""
    dbg = rec.get("debug") or {}
    if isinstance(dbg, dict) and dbg.get("calibration_mult") is not None:
        try:
            return float(dbg["calibration_mult"])
        except (TypeError, ValueError):
            pass
    m = _CALIBRATION_RE.search(rec.get("rationale") or "")
    return float(m.group(1)) if m else None


def short_source(source: str) -> str:
    """'pregame_nfl_totals' -> 'nfl_totals'. Pregame is the default everywhere;
    what matters at a glance is WHICH model, and that 'live' is not pregame."""
    return (source or "flow-only").replace("pregame_", "")


_INTERVAL_RE = re.compile(r"--interval\s+(\d+)")


def _mode_from_args(args: str) -> str:
    parts = []
    if "--live" in args:
        parts.append("LIVE")
        if "--in-game" in args:
            parts.append("IN-GAME")
        if "--maker" in args:
            parts.append("MAKER")
    elif "--paper" in args:
        parts.append("PAPER")
    else:
        parts.append("RECOMMEND-ONLY")
    return " + ".join(parts)


def probe_loop() -> LoopInfo:
    """Find the running loop and read its flags off its own command line.

    The loop always shows up as TWO python processes (a .venv launcher and its
    anaconda worker child -- see CLAUDE.md); both carry the same `cli.py loop`
    command line, so either answers the question. We only read; nothing here
    signals or touches those processes.
    """
    info = LoopInfo()
    try:
        out = subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | "
             "Select-Object ProcessId,CommandLine | ConvertTo-Json -Compress"],
            capture_output=True, text=True, timeout=25,
        ).stdout.strip()
        rows = json.loads(out) if out else []
        if isinstance(rows, dict):
            rows = [rows]
    except Exception:
        rows = []

    for row in rows:
        cmd = (row or {}).get("CommandLine") or ""
        if "cli.py" in cmd and " loop" in cmd:
            info.running = True
            info.pid = row.get("ProcessId")
            info.mode = _mode_from_args(cmd)
            info.source = "running process"
            m = _INTERVAL_RE.search(cmd)
            if m:
                info.interval = int(m.group(1))
            return info

    # Not running (or the probe failed): fall back to what the scheduled task
    # WOULD run, so the header still describes the configured mode.
    try:
        text = (ROOT / "scripts" / "run_paper_loop.bat").read_text(encoding="utf-8", errors="replace")
    except OSError:
        return info
    for line in text.splitlines():
        if "cli.py" in line and " loop" in line:
            info.mode = _mode_from_args(line)
            info.source = "run_paper_loop.bat"
            m = _INTERVAL_RE.search(line)
            if m:
                info.interval = int(m.group(1))
            break
    return info


def sport_records(bets: list[dict], outcomes: dict[str, bool],
                  grade) -> list[tuple[str, tuple[int, int, float]]]:
    """Per-sport (wins, losses, net), using the project's own ticker registry.

    MLB/NFL/NBA/NHL share one account, one ledger and one dashboard, but they
    are independently-validated models at very different stages of maturity, so
    a single blended record hides which one is actually working -- the same
    reason `cli.py settle` grew its BY SPORT block. Sports with nothing settled
    are left out rather than shown as 0-0.
    """
    by: dict[str, list[dict]] = {}
    for b in bets:
        by.setdefault(sport_of(b.get("ticker") or ""), []).append(b)
    out = []
    for sport in sorted(by):
        record = grade(by[sport], outcomes)
        if record[0] + record[1]:
            out.append((sport.upper(), record))
    return out


def load_account() -> AccountData:
    """Balance, graded record (total and per sport), open positions, and the
    still-pending bets marked to the current market.

    Deliberately reuses engine/account_summary's own grading helpers instead of
    reimplementing them, so this panel can never disagree with what the loop
    prints or what `cli.py settle` reports for the same bets. Slow (live GETs +
    outcome resolution), so callers run it off the UI thread.
    """
    data = AccountData()
    errs: list[str] = []

    from backtest.evaluate import _game_date
    from data.games import resolve_outcomes
    from engine.account_summary import _grade
    from engine.real_bets import load_real_bets
    from kalshi.client import KalshiClient
    from kalshi.normalize import position_size

    client = None
    try:
        client = KalshiClient(DEFAULTS)
        bal = client.balance()
        cents = bal.get("balance")
        data.balance = (float(cents) / 100.0 if cents is not None
                        else float(bal.get("balance_dollars") or 0))
    except Exception as e:
        errs.append(f"balance: {e}")

    positions: list[dict] = []
    if client is not None:
        try:
            positions = (client.get_positions() or {}).get("market_positions") or []
            data.open_positions = sum(1 for p in positions if position_size(p) != 0)
        except Exception as e:
            errs.append(f"positions: {e}")

    try:
        from data.mlb_stats import MlbStatsClient
        from data.nba_data import NbaDataClient
        from data.nfl_data import NflDataClient
        from data.nhl_data import NhlDataClient

        # Both ledgers, each row tagged live/paper, live rows corrected to what
        # actually filled (engine/real_bets.py) -- the same view `cli.py settle`
        # grades, so the cards can't disagree with it. Never-filled orders are
        # kept here only so Settled can list them as "no fill"; nothing that
        # grades or marks sees them.
        all_bets = load_real_bets(client, include_unfilled=True)
        bets = [b for b in all_bets if not b.get("_nofill")]

        clients = {"mlb": MlbStatsClient(), "nfl": NflDataClient(),
                   "nba": NbaDataClient(), "nhl": NhlDataClient()}
        outcomes = resolve_outcomes({b["ticker"] for b in all_bets}, clients)
        today = datetime.now(EASTERN).strftime("%Y-%m-%d")
        today_bets = [b for b in bets if _game_date(b["ticker"]) == today]
        data.today = _grade(today_bets, outcomes)
        data.overall = _grade(bets, outcomes)
        data.today_by_sport = sport_records(today_bets, outcomes, _grade)
        data.overall_by_sport = sport_records(bets, outcomes, _grade)
        data.pending = pending_bets(bets, outcomes)
        # The loop skips any event already in its ledger, filled or not, so the
        # BET column must see the unfilled rows too.
        data.placed = placed_events(all_bets)
        data.settled = settled_bets(all_bets, outcomes)
    except Exception as e:
        errs.append(f"ledger: {e}")

    if client is not None and data.pending:
        try:
            mark_pending(data, client, positions, position_size)
        except Exception as e:
            errs.append(f"mark: {e}")

    data.error = "; ".join(errs)
    return data


def mark_pending(data: AccountData, client, positions: list[dict], position_size) -> None:
    """Mark each pending bet to the current market and total the unrealized P&L.

    The cost basis comes from the POSITION (`market_exposure_dollars`), not the
    ledger row: the position is what the exchange actually holds, so a partial
    fill or a re-pegged maker order is reflected honestly rather than assumed to
    have filled at the requested size and price.

    The mark is the BID on the side we hold -- what the position could actually
    be liquidated at right now -- not the mid. Marking at the mid would
    systematically overstate unrealized P&L by half the spread on every row.

    A pending bet with no position is an order that has not filled yet (the
    ledger records a live order at submit time, and it may still be resting), so
    it is left unmarked rather than shown as a loss.
    """
    from kalshi.normalize import parse_market

    by_ticker = {p.get("ticker"): p for p in positions}
    total = 0.0
    marked = False
    for bet in data.pending:
        pos = by_ticker.get(bet.get("ticker"))
        if pos is None:
            continue
        size = position_size(pos)
        if size == 0:
            continue
        basis = float(pos.get("market_exposure_dollars") or 0.0)
        try:
            quote = parse_market(client.get_market(bet["ticker"]) or {})
        except Exception:
            continue
        # position_fp is signed: positive = long YES, negative = long NO.
        mark = quote.yes_bid if size > 0 else max(0.0, 1.0 - quote.yes_ask)
        bet["_mark"] = mark
        bet["_unrealized"] = abs(size) * mark - basis
        bet["_filled"] = abs(size)
        total += bet["_unrealized"]
        marked = True
    data.unrealized = total if marked else None


def pending_bets(bets: list[dict], outcomes: dict[str, bool]) -> list[dict]:
    """Placed bets still awaiting settlement, most recent start first.

    Cut off at 36 hours because some rows can never settle -- NFL preseason bets
    resolve through nflverse's games.csv, which excludes preseason, so they sit
    unsettled forever (documented in CLAUDE.md). Without the cutoff those would
    silently accumulate in a panel meant to answer "what is at risk right now".
    """
    now = datetime.now(timezone.utc)
    out = []
    for b in bets:
        if outcomes.get(b.get("ticker")) is not None:
            continue
        start = parse_utc(b.get("first_pitch"))
        if start is None or start < now - timedelta(hours=36):
            continue
        out.append({**b, "_start": start})
    out.sort(key=lambda b: b["_start"], reverse=True)
    return out


SETTLED_DAYS = 7


def bet_fee(bet: dict) -> float:
    """Kalshi's fee on this bet, in dollars.

    The fee the live ledger recorded when there is one (maker fills write the
    real `fees_usd`); otherwise config/fees.py's model -- the same formula
    `cli.py fees` audits against every real fill. Offline: the per-series
    multiplier comes from its measured fallback table, not a live GET.
    """
    if bet.get("fees_usd") is not None:
        return float(bet["fees_usd"])
    from config.fees import fee_for
    return fee_for(float(bet.get("contracts") or 0), float(bet.get("entry_price") or 0),
                   bet.get("ticker") or "", is_taker=bet.get("execution") != "maker")


def settled_bets(bets: list[dict], outcomes: dict[str, bool],
                 days: int = SETTLED_DAYS, now: Optional[datetime] = None) -> list[dict]:
    """Bets graded in the last `days`, newest game first, each with its result.

    Profit is computed exactly as engine/account_summary._grade does (so the
    rows sum to the same gross number the record cards show); the fee and net
    are layered on top rather than changing that shared definition.
    """
    now = now or datetime.now(timezone.utc)
    out = []
    for b in bets:
        yes_won = outcomes.get(b.get("ticker"))
        if yes_won is None:
            continue
        start = parse_utc(b.get("first_pitch"))
        if start is None or start < now - timedelta(days=days):
            continue
        won = yes_won if b.get("side") == "YES" else not yes_won
        price = float(b.get("entry_price") or 0)
        contracts = float(b.get("contracts") or 0)
        wager = float(b.get("wager_usd") or price * contracts)
        profit = ((1 - price) if won else -price) * contracts
        fee = bet_fee(b)
        row = {**b, "_start": start, "_won": won, "_profit": profit, "_fee": fee,
               "_net": profit - fee, "_wager": wager,
               "_roi": (profit - fee) / wager if wager else None}
        if b.get("_nofill"):
            # A canceled order that never filled: listed, but no money moved.
            row.update(_profit=0.0, _fee=0.0, _net=0.0, _wager=0.0, _roi=None)
        out.append(row)
    out.sort(key=lambda b: b["_start"], reverse=True)
    return out


def settled_summary(rows: list[dict]) -> tuple[int, int, float, float, float, Optional[float]]:
    """(wins, losses, gross, fees, net, net ROI) over settled rows that actually
    filled -- a never-filled order is neither a win nor a loss."""
    real = [r for r in rows if not r.get("_nofill")]
    wins = sum(1 for r in real if r["_won"])
    gross = sum(r["_profit"] for r in real)
    fees = sum(r["_fee"] for r in real)
    staked = sum(r["_wager"] for r in real)
    return (wins, len(real) - wins, gross, fees, gross - fees,
            (gross - fees) / staked if staked else None)


def book_label(src: str) -> str:
    """Ledger name for display. The paper ledger is NOT paper money: those
    bets (Aug 6 - Sep 16) were real orders placed by hand from the alert. Only
    the placer differs, so it reads 'manual' rather than implying they were fake."""
    return "manual" if src == "paper" else (src or "")


def parse_utc(value: Any) -> Optional[datetime]:
    if not isinstance(value, str):
        return None
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def placed_events(bets: list[dict]) -> dict[str, dict[str, datetime]]:
    """{"live"/"paper": {event_key: when it was placed}} from the ledger rows.

    Kept per ledger because the loop only checks its OWN ledger before betting
    (live mode reads live_ledger.jsonl, paper mode paper_ledger.jsonl), so a row
    in the other one does not stop a bet from firing.
    """
    out: dict[str, dict[str, datetime]] = {"live": {}, "paper": {}}
    for b in bets:
        key, ts = b.get("event_key"), parse_utc(b.get("ts"))
        if key and ts:
            src = "live" if b.get("order_id") else "paper"
            out[src][key] = min(ts, out[src].get(key, ts))
    return out


def resolve_schedule_starts(tickers: list[str]) -> dict[str, datetime]:
    """Real kickoff / puck-drop for NFL and NHL tickers, via the same schedule
    lookup the loop's trigger uses (data/games.py::match_game).

    Needed because those tickers carry no time of day, and the market's own
    `game_datetime` (Kalshi's occurrence_datetime) runs a documented +3h late
    on NFL -- sorting on it would be right, but the START column would lie.
    Slow on a cold cache, so the app calls this off the UI thread. Any failure
    just leaves that ticker out, and rec_start falls back.
    """
    from data.games import match_game
    from data.nfl_data import NflDataClient
    from data.nhl_data import NhlDataClient

    wanted = [t for t in tickers if sport_of(t) in ("nfl", "nhl")]
    if not wanted:
        return {}
    clients = {"nfl": NflDataClient(), "nhl": NhlDataClient()}
    cache: dict = {}
    out: dict[str, datetime] = {}
    for t in wanted:
        try:
            g = match_game(t, clients, cache)
        except Exception:
            continue
        if g is not None and g.game_datetime is not None:
            out[t] = g.game_datetime
    return out


def rec_start(rec: dict, sched: dict[str, datetime]) -> tuple[Optional[datetime], bool]:
    """(UTC start time, trusted?) for one recommendation.

    Same preference order as engine/paper.py::iter_trigger_candidates, so the
    time shown is the time the loop is actually counting down to: MLB's ticker
    time, then the NFL/NHL schedule, then Kalshi's occurrence_datetime. That
    last one is only trusted for NBA (which has no other source, and is what the
    loop uses too); for NFL/NHL it is the known-skewed fallback, flagged so the
    START column can mark it.
    """
    from engine.paper import first_pitch_from_ticker

    ticker = rec.get("ticker") or ""
    fp = first_pitch_from_ticker(ticker)
    if fp is not None:
        return fp, True
    if ticker in sched:
        return sched[ticker], True
    return parse_utc(rec.get("game_datetime")), sport_of(ticker) == "nba"


def game_key(ticker: str) -> str:
    """'KXNFLSPREAD-26SEP27MINTB-MIN3' -> '26SEP27MINTB': the game itself, shared
    by its winner, total and spread markets so they sort next to each other."""
    parts = (ticker or "").split("-")
    return parts[1] if len(parts) > 1 else ticker or ""


def group_recs(recs: list[dict], sched: dict[str, datetime]
               ) -> list[tuple[str, list[tuple[dict, Optional[datetime], bool]]]]:
    """Recommendations grouped by sport, soonest game first.

    Sports are ordered by their earliest game; within a sport, by start time,
    then by game (so a game's spread/total/winner stay together), then biggest
    edge first. A rec with no resolvable start sinks to the bottom of its sport.
    """
    far = datetime.max.replace(tzinfo=timezone.utc)
    by: dict[str, list[tuple[dict, Optional[datetime], bool]]] = {}
    for r in recs:
        start, trusted = rec_start(r, sched)
        by.setdefault(sport_of(r.get("ticker") or ""), []).append((r, start, trusted))
    for rows in by.values():
        rows.sort(key=lambda x: (x[1] or far, game_key(x[0].get("ticker") or ""),
                                 -float(x[0].get("edge_cents") or 0.0)))
    return sorted(((sport.upper(), rows) for sport, rows in by.items()),
                  key=lambda s: (s[1][0][1] or far, s[0]))


def bet_eta(rec: dict, start: Optional[datetime], now: datetime,
            next_scan: Optional[datetime], interval: int, mode: str,
            placed: dict[str, dict[str, datetime]],
            settings=DEFAULTS) -> tuple[str, Optional[datetime]]:
    """When the loop will place this bet: (kind, time).

    kind is "placed" (already in the ledger the loop checks), "eta" (projected),
    "now" (eligible on the very next scan), or "none" with a short reason in
    place of a time. The projection mirrors engine/loop.py + engine/paper.py:
    a pregame bet fires on the first scan at or after
    start - (interval/60 + paper_trigger_buffer_min) minutes, and scans land
    every `interval` seconds from the next one. It is only a projection -- the
    bet fires only if the recommendation still clears every gate on that scan.
    """
    event_key = (rec.get("ticker") or "").rsplit("-", 1)[0]
    if "LIVE" in mode:
        ledgers = [placed.get("live", {})]
    elif "PAPER" in mode:
        ledgers = [placed.get("paper", {})]
    elif "RECOMMEND" in mode:
        return "none", None                    # recommend-only never places
    else:
        ledgers = [placed.get("live", {}), placed.get("paper", {})]
    for ledger in ledgers:
        if event_key in ledger:
            return "placed", ledger[event_key]
    if float(rec.get("suggested_contracts") or 0) <= 0 or start is None:
        return "none", None
    window = timedelta(minutes=interval / 60.0 + settings.paper_trigger_buffer_min)
    if start <= now:
        # In-game: only MLB winners with a live model, only in --in-game mode,
        # only for in_game_max_minutes after first pitch.
        if ("IN-GAME" in mode and rec.get("fair_source") == "live"
                and now - start <= timedelta(minutes=settings.in_game_max_minutes)):
            return "now", next_scan
        return "none", None
    target = start - window
    if next_scan is None:
        return "eta", max(target, now)
    if target <= next_scan:
        return "eta", next_scan
    steps = -(-(target - next_scan).total_seconds() // interval)    # ceil
    return "eta", next_scan + timedelta(seconds=steps * interval)


# --------------------------------------------------------------------------
# formatting helpers
# --------------------------------------------------------------------------

def signed(value: Optional[float], fmt: str = "{:+.2f}", zero_dim: bool = True) -> Text:
    """Green when positive, red when negative -- the dashboard's core rule."""
    if value is None:
        return Text("--", style=DIM)
    if value > 0:
        return Text(fmt.format(value), style=GREEN)
    if value < 0:
        return Text(fmt.format(value), style=RED)
    return Text(fmt.format(value), style=DIM if zero_dim else WHITE)


def fmt_duration(seconds: float) -> str:
    seconds = int(max(0, seconds))
    if seconds >= 3600:
        return f"{seconds // 3600}h{(seconds % 3600) // 60:02d}m"
    return f"{seconds // 60}:{seconds % 60:02d}"


def fmt_countdown(target: Optional[datetime]) -> tuple[str, str]:
    """(text, style) for the time remaining until `target`."""
    if target is None:
        return "--", DIM
    delta = (target - datetime.now(timezone.utc)).total_seconds()
    if delta <= 0:
        return "scanning...", YELLOW
    return fmt_duration(delta), "bold " + WHITE if delta > 60 else YELLOW


def fmt_start_offset(start: Optional[datetime]) -> Text:
    """'in 42:00' before a game starts, 'live 12:00' once it has."""
    if start is None:
        return Text("--", style=DIM)
    delta = (start - datetime.now(timezone.utc)).total_seconds()
    if delta > 0:
        return Text(f"in {fmt_duration(delta)}", style=WHITE)
    return Text(f"live {fmt_duration(-delta)}", style=YELLOW)


def fmt_clock(dt: datetime, now: datetime, with_day: bool = True) -> str:
    """'7:40p' today (ET), 'Sun 1:00p' within the week, '10/4 1:00p' beyond it
    (two different Sundays must not read the same)."""
    local, today = dt.astimezone(EASTERN), now.astimezone(EASTERN).date()
    clock = f"{local.hour % 12 or 12}:{local.minute:02d}{'a' if local.hour < 12 else 'p'}"
    if not with_day or local.date() == today:
        return clock
    if (local.date() - today).days < 7:
        return f"{local:%a} {clock}"
    return f"{local.month}/{local.day} {clock}"


def start_text(start: Optional[datetime], trusted: bool, now: datetime) -> Text:
    """Start time in ET; 'live' once it has begun. A '?' marks a time that came
    from Kalshi's skewed occurrence_datetime rather than a real schedule."""
    if start is None:
        return Text("--", style=DIM)
    if start <= now:
        return Text("live", style=YELLOW)
    return Text(fmt_clock(start, now) + ("" if trusted else "?"),
                style=WHITE if trusted else MUTED)


def bet_text(kind: str, when: Optional[datetime], now: datetime) -> Text:
    if kind == "placed" and when is not None:
        return Text("✓ " + fmt_clock(when, now, with_day=False), style=GREEN)
    if kind == "now":
        return Text("next scan", style=YELLOW)
    if kind == "eta" and when is not None:
        return Text("~" + fmt_clock(when, now, with_day=False), style=CYAN)
    return Text("--", style=DIM)


def sport_header(sport: str, n: int) -> Text:
    t = Text(sport, style="bold " + YELLOW)
    t.append(f"  {n} rec{'s' if n != 1 else ''}", style=MUTED)
    return t


def record_text(record: tuple[int, int, float]) -> Text:
    wins, losses, net = record
    n = wins + losses
    if n == 0:
        return Text("no settled bets", style=DIM)
    t = Text(f"{wins}-{losses} ", style="bold " + WHITE)
    t.append(f"({wins / n:.0%})  ", style=CYAN)
    t.append(f"{net:+.2f}", style=GREEN if net > 0 else RED if net < 0 else DIM)
    return t


def record_card(record: tuple[int, int, float],
                by_sport: list[tuple[str, tuple[int, int, float]]]) -> Text:
    """The blended record on the first line, then one line per sport."""
    t = record_text(record)
    for name, (wins, losses, net) in by_sport:
        t.append(f"\n{name} {wins}-{losses} ", style=MUTED)
        t.append(f"{net:+.2f}", style=GREEN if net > 0 else RED if net < 0 else DIM)
    return t


def name_cell(name: str, ticker: str) -> Text:
    """Two-line cell: the bet in full on top, its ticker beneath.

    Stacking them is what lets both be shown untruncated -- side by side they
    need ~85 columns between them, which pushed the right-hand columns off the
    screen entirely.
    """
    t = Text(name or "", style="bold " + WHITE)
    t.append("\n" + (ticker or ""), style=CYAN)
    return t


def flow_text(direction: str, score: float) -> Text:
    style = GREEN if direction == "YES" else RED if direction == "NO" else MUTED
    return Text(f"{direction} {score:+.2f}", style=style)


def bto_text(book: Optional[float], trades: Optional[float],
             oi: Optional[float]) -> Text:
    """book/trades/oi in one cell, each independently colored."""
    t = Text()
    for i, value in enumerate((book, trades, oi)):
        if i:
            t.append(" ")
        t.append_text(signed(value))
    return t


def truncate(s: str, width: int) -> str:
    s = s or ""
    return s if len(s) <= width else s[: width - 1] + "…"


def name_column_width(names: list[str], tickers: list[str],
                      floor: int = 34, cap: int = 64) -> int:
    """Width that shows every name and ticker in full, within reason.

    Measured from the data rather than hardcoded, so a longer matchup than any
    seen so far still renders in full instead of being silently cut off.
    """
    longest = max([len(s or "") for s in names + tickers] + [0])
    return max(floor, min(cap, longest))


# --------------------------------------------------------------------------
# the app
# --------------------------------------------------------------------------

from textual.app import App, ComposeResult                       # noqa: E402
from textual.containers import Horizontal, Vertical              # noqa: E402
from textual.widgets import (DataTable, Footer, Header, Static,  # noqa: E402
                             TabbedContent, TabPane)

# (label, width) for every column except the name column, whose width is
# measured from the data at render time. Kept tight so the whole table fits a
# normal window without horizontal scrolling.
SIGNAL_TAIL = [
    ("START", 12), ("BET", 8), ("PX", 4), ("FAIR", 4), ("CONF", 4), ("EDGE", 7), ("STAKE / CT", 14),
    ("FLOW", 10), ("BOOK  TRD    OI", 17), ("MODEL / CAL", 17),
]
PENDING_TAIL = [
    ("SIDE", 4), ("PX", 4), ("CT", 5), ("WAGER", 6), ("MARK", 4),
    ("UNREAL", 7), ("EDGE", 7), ("CONF", 4), ("BOOK", 6), ("ORDER", 8),
]
# Settled swaps the live mark for the realized result. PROFIT is gross (it sums
# to the record cards), FEE is Kalshi's fee, NET = PROFIT - FEE, and ROI is NET
# over the dollars wagered.
SETTLED_TAIL = [
    ("SIDE", 4), ("PX", 4), ("CT", 5), ("WAGER", 6), ("W/L", 4),
    ("PROFIT", 7), ("FEE", 6), ("NET", 7), ("ROI", 6),
    ("EDGE", 7), ("CONF", 4), ("BOOK", 6), ("ORDER", 8),
]


class KpiCard(Static):
    """One labelled value in the top strip."""

    def __init__(self, label: str, card_id: str) -> None:
        super().__init__(id=card_id)
        self.border_title = label

    def update_value(self, value: Text | str) -> None:
        self.update(value)


class DashboardApp(App):
    TITLE = "Kalshi money-flow agent"
    CSS = """
    Screen { background: $surface; }
    #kpis { height: 7; padding: 0 1; }
    KpiCard {
        width: 1fr; height: 7; content-align: center middle;
        border: round $primary; padding: 0 1;
    }
    #tables { height: 1fr; }
    #pending_box { height: 1fr; min-height: 12; padding: 0 1; }
    #bets_tabs, #bets_tabs TabPane { height: 1fr; }
    #bets_tabs TabPane { padding: 0; }
    #signals_box { height: 2fr; padding: 0 1; }
    DataTable { height: 1fr; border: round $primary; }
    #status { height: 1; padding: 0 2; }
    """
    BINDINGS = [("q", "quit", "Quit"), ("r", "force_refresh", "Refresh now"),
                ("p", "show_tab('tab_pending')", "Pending"),
                ("s", "show_tab('tab_settled')", "Settled")]

    def __init__(self, latest: Path = LATEST, poll_seconds: float = 2.0) -> None:
        super().__init__()
        self._latest = latest
        self._poll = poll_seconds
        self._scan: Optional[ScanData] = None
        self._account = AccountData()
        self._loop_info = LoopInfo()
        self._account_busy = False
        self._last_mtime: float = -1.0
        # NFL/NHL real start times, by ticker. Filled off the UI thread; until a
        # ticker resolves, rec_start falls back to the market's own time.
        self._starts: dict[str, datetime] = {}
        self._starts_busy = False

    # -- layout ------------------------------------------------------------
    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Horizontal(id="kpis"):
            yield KpiCard("BALANCE", "kpi_balance")
            yield KpiCard("TODAY", "kpi_today")
            yield KpiCard("OVERALL", "kpi_overall")
            yield KpiCard("MODE", "kpi_mode")
            yield KpiCard("LAST SCAN", "kpi_scan")
            yield KpiCard("NEXT SCAN", "kpi_next")
            yield KpiCard("COVERAGE", "kpi_coverage")
        with Vertical(id="tables"):
            # Pending first: money already at risk outranks money we might risk.
            with Vertical(id="pending_box"):
                # Tabs: money at risk now, and what recently came back.
                with TabbedContent(id="bets_tabs", initial="tab_pending"):
                    with TabPane("Pending", id="tab_pending"):
                        yield DataTable(id="pending", cursor_type="row", zebra_stripes=True)
                    with TabPane("Settled", id="tab_settled"):
                        yield DataTable(id="settled", cursor_type="row", zebra_stripes=True)
            with Vertical(id="signals_box"):
                yield DataTable(id="signals", cursor_type="row", zebra_stripes=True)
        yield Static("", id="status")
        yield Footer()

    def on_mount(self) -> None:
        self.query_one("#pending", DataTable).border_title = (
            "PENDING — placed bets awaiting settlement")
        self.query_one("#signals", DataTable).border_title = (
            "SIGNALS — this cycle's recommendations")
        self.reload_scan(force=True)
        self.refresh_account()
        self.refresh_loop_info()
        # The 1s repaint keeps the countdown and clock alive between 10-minute
        # scans; the file poll is what actually notices new data.
        self.set_interval(1.0, self.tick)
        self.set_interval(self._poll, self.reload_scan)
        self.set_interval(120.0, self.refresh_loop_info)

    @staticmethod
    def _sync_columns(table: DataTable, spec: list[tuple[str, int]]) -> None:
        """Rebuild the table's columns when the measured layout changes.

        DataTable fixes a column's width when it is added, so the name column
        cannot simply grow -- the columns are rebuilt instead. Cheap: it only
        happens when the longest name actually changes length.
        """
        if getattr(table, "_dash_col_spec", None) == spec:
            return
        table.clear(columns=True)
        for label, width in spec:
            table.add_column(Text(label, style="bold"), width=width)
        table._dash_col_spec = spec

    # -- data refresh ------------------------------------------------------
    def reload_scan(self, force: bool = False) -> None:
        """Redraw the signals table only when the loop has written a new cycle."""
        try:
            mtime = self._latest.stat().st_mtime
        except OSError:
            return
        if not force and mtime == self._last_mtime:
            return
        scan = load_scan(self._latest)
        if scan is None:
            return                      # mid-write; the next poll will catch it
        self._last_mtime = scan.mtime
        self._scan = scan
        self.render_signals()
        self.render_kpis()
        self.refresh_starts()
        if not force:
            # A new cycle is the only thing that can have changed the account
            # state (a bet may have been placed, and every pending bet needs
            # re-marking), so this is the only time the expensive refresh is
            # warranted.
            self.refresh_account()

    def refresh_account(self) -> None:
        if self._account_busy:
            return
        self._account_busy = True
        self.run_worker(self._account_worker, thread=True, exit_on_error=False)

    def _account_worker(self) -> None:
        try:
            data = load_account()
        except Exception as e:                                   # pragma: no cover
            data = AccountData(error=f"{type(e).__name__}: {e}")
        self.call_from_thread(self._apply_account, data)

    def _apply_account(self, data: AccountData) -> None:
        self._account = data
        self._account_busy = False
        self.render_kpis()
        self.render_pending()
        self.render_settled()
        self.render_signals()      # the BET column reads the ledgers too

    def refresh_starts(self) -> None:
        """Look up real start times for any NFL/NHL ticker not already known."""
        if self._starts_busy or self._scan is None:
            return
        todo = [r.get("ticker") or "" for r in self._scan.recs
                if (r.get("ticker") or "") not in self._starts]
        if not any(sport_of(t) in ("nfl", "nhl") for t in todo):
            return
        self._starts_busy = True
        self.run_worker(lambda: self._starts_worker(todo), thread=True, exit_on_error=False)

    def _starts_worker(self, tickers: list[str]) -> None:
        try:
            found = resolve_schedule_starts(tickers)
        except Exception:
            found = {}
        self.call_from_thread(self._apply_starts, found)

    def _apply_starts(self, found: dict[str, datetime]) -> None:
        self._starts_busy = False
        before = len(self._starts)
        self._starts.update(found)
        if len(self._starts) != before:
            self.render_signals()
            self.render_kpis()

    def refresh_loop_info(self) -> None:
        self.run_worker(self._loop_worker, thread=True, exit_on_error=False)

    def _loop_worker(self) -> None:
        info = probe_loop()
        self.call_from_thread(self._apply_loop, info)

    def _apply_loop(self, info: LoopInfo) -> None:
        self._loop_info = info
        self.render_kpis()

    def action_force_refresh(self) -> None:
        self.reload_scan(force=True)
        self.refresh_account()
        self.refresh_loop_info()

    def tick(self) -> None:
        self._set("kpi_next", self._next_scan_text())
        self.render_status()

    # -- rendering ---------------------------------------------------------
    def _set(self, card_id: str, value: Text | str) -> None:
        try:
            self.query_one(f"#{card_id}", KpiCard).update_value(value)
        except Exception:
            pass

    def _next_scan_at(self) -> Optional[datetime]:
        if self._scan is None or self._scan.generated_at is None:
            return None
        return (self._scan.generated_at.astimezone(timezone.utc)
                + timedelta(seconds=self._loop_info.interval))

    def _next_scan_text(self) -> Text:
        if self._loop_info.source != "running process":
            return Text("loop stopped", style=RED)
        text, style = fmt_countdown(self._next_scan_at())
        return Text(text, style=style)

    def render_kpis(self) -> None:
        acct = self._account
        self._set("kpi_balance",
                  Text(f"${acct.balance:.2f}", style=GREEN) if acct.balance is not None
                  else Text("unavailable", style=DIM))
        self._set("kpi_today", record_card(acct.today, acct.today_by_sport))
        self._set("kpi_overall", record_card(acct.overall, acct.overall_by_sport))

        mode = Text(self._loop_info.mode,
                    style=YELLOW if "LIVE" in self._loop_info.mode else CYAN)
        if self._loop_info.running:
            mode.append(f"\npid {self._loop_info.pid}", style=DIM)
        else:
            mode.append("\nLOOP NOT RUNNING", style=RED)
        self._set("kpi_mode", mode)

        if self._scan and self._scan.generated_at:
            local = self._scan.generated_at.astimezone(EASTERN)
            age = (datetime.now(timezone.utc)
                   - self._scan.generated_at.astimezone(timezone.utc)).total_seconds()
            scan_text = Text(local.strftime("%H:%M:%S ET"), style="bold " + WHITE)
            scan_text.append(f"\n{fmt_duration(age)} ago", style=DIM)
        else:
            scan_text = Text("no scan yet", style=DIM)
        self._set("kpi_scan", scan_text)
        self._set("kpi_next", self._next_scan_text())

        if self._scan:
            cov = Text(f"{len(self._scan.recs)} recs", style="bold " + WHITE)
            groups = group_recs(self._scan.recs, self._starts)
            if groups:
                cov.append("\n" + " · ".join(f"{s} {len(rows)}" for s, rows in groups),
                           style=MUTED)
            cov.append(f"\n{self._scan.scanned} scanned\n{self._scan.deep_scanned} deep",
                       style=DIM)
        else:
            cov = Text("--", style=DIM)
        self._set("kpi_coverage", cov)

    def render_signals(self) -> None:
        """Signals grouped into one section per sport, soonest game first."""
        table = self.query_one("#signals", DataTable)
        recs = self._scan.recs if self._scan else []
        groups = group_recs(recs, self._starts)
        width = name_column_width([r.get("headline") or r.get("title") or "" for r in recs],
                                  [r.get("ticker") or "" for r in recs])
        self._sync_columns(table, [("#", 3), ("MATCHUP / TICKER", width)] + SIGNAL_TAIL)
        table.clear()
        now = datetime.now(timezone.utc)
        next_scan = self._next_scan_at()
        for sport, rows in groups:
            table.add_row(Text(""), sport_header(sport, len(rows)),
                          *[Text("") for _ in SIGNAL_TAIL], height=1)
            for i, (r, start, trusted) in enumerate(rows, 1):
                table.add_row(Text(str(i), style=DIM),
                              *self._signal_cells(r, start, trusted, now, next_scan),
                              height=2)

    def _signal_cells(self, r: dict, start: Optional[datetime], trusted: bool,
                      now: datetime, next_scan: Optional[datetime]) -> list[Text]:
        book, trades, oi = flow_components(r)
        cal = calibration_mult(r)
        fair = bet_side_prob(r)

        model = Text(short_source(r.get("fair_source")), style=WHITE)
        if cal is not None:
            model.append(f" {cal:.2f}x", style=GREEN if cal >= 1 else RED)

        headline = r.get("headline") or r.get("title") or r.get("ticker") or ""
        name = name_cell(headline, r.get("ticker", ""))
        # A conflict is the loop's own warning that money flow disagrees with
        # fair value; the console marks it, so keep it visible here too.
        if r.get("conflict"):
            name.stylize("bold #d2a8ff", 0, len(headline))

        kind, when = bet_eta(r, start, now, next_scan, self._loop_info.interval,
                             self._loop_info.mode, self._account.placed)
        return [
            name,
            start_text(start, trusted, now),
            bet_text(kind, when, now),
            Text(f"{float(r.get('entry_price') or 0):.2f}", style=WHITE),
            Text(f"{fair:.0%}" if fair is not None else "--",
                 style=WHITE if fair is not None else DIM),
            Text(f"{float(r.get('confidence') or 0):.2f}", style=YELLOW),
            signed(r.get("edge_cents"), "{:+.1f}c", zero_dim=False),
            Text(f"${float(r.get('suggested_stake_usd') or 0):.2f} / "
                 f"{float(r.get('suggested_contracts') or 0):.2f}ct", style=WHITE),
            flow_text(r.get("flow_direction", "?"),
                      float(r.get("money_flow_score") or 0.0)),
            bto_text(book, trades, oi),
            model,
        ]

    def render_pending(self) -> None:
        table = self.query_one("#pending", DataTable)
        pending = self._account.pending
        width = name_column_width([b.get("label") or "" for b in pending],
                                  [b.get("ticker") or "" for b in pending])
        self._sync_columns(table, [("WHEN", 10), ("BET / TICKER", width)] + PENDING_TAIL)
        table.clear()

        self._set_tab_label("tab_pending", f"Pending ({len(pending)})")
        title = f"PENDING — {len(pending)} placed bet(s) awaiting settlement"
        if self._account.unrealized is not None:
            title += f"   unrealized {self._account.unrealized:+.2f}"
        table.border_title = title

        for b in pending:
            unreal = b.get("_unrealized")
            mark = b.get("_mark")
            table.add_row(
                fmt_start_offset(b.get("_start")),
                name_cell(b.get("label") or "", b.get("ticker") or ""),
                Text(b.get("side", ""), style=GREEN if b.get("side") == "YES" else RED),
                Text(f"{float(b.get('entry_price') or 0):.2f}", style=WHITE),
                Text(f"{float(b.get('contracts') or 0):.2f}", style=WHITE),
                Text(f"${float(b.get('wager_usd') or 0):.2f}", style=WHITE),
                Text(f"{mark:.2f}" if mark is not None else "--",
                     style=WHITE if mark is not None else DIM),
                signed(unreal, "{:+.2f}", zero_dim=False) if unreal is not None
                else Text("--", style=DIM),
                signed(b.get("edge_cents"), "{:+.1f}c", zero_dim=False),
                Text(f"{float(b.get('confidence') or 0):.2f}", style=YELLOW),
                Text(book_label(b.get("_src", "")),
                     style=CYAN if b.get("_src") == "live" else DIM),
                Text(truncate(b.get("order_status") or "-", 8), style=DIM),
                height=2,
            )

    def _set_tab_label(self, pane_id: str, label: str) -> None:
        try:
            self.query_one("#bets_tabs", TabbedContent).get_tab(pane_id).label = label
        except Exception:
            pass

    def action_show_tab(self, pane_id: str) -> None:
        self.query_one("#bets_tabs", TabbedContent).active = pane_id

    def render_settled(self) -> None:
        table = self.query_one("#settled", DataTable)
        rows = self._account.settled
        width = name_column_width([b.get("label") or "" for b in rows],
                                  [b.get("ticker") or "" for b in rows])
        self._sync_columns(table, [("WHEN", 10), ("BET / TICKER", width)] + SETTLED_TAIL)
        table.clear()

        wins, losses, gross, fees, net, roi = settled_summary(rows)
        self._set_tab_label("tab_settled", f"Settled ({len(rows)})")
        if rows:
            title = (f"SETTLED — last {SETTLED_DAYS} days: {wins}-{losses} "
                     f"({wins / max(1, wins + losses):.0%})   profit {gross:+.2f}   fees {fees:.2f}   "
                     f"net {net:+.2f}")
            if roi is not None:
                title += f"   ROI {roi:+.1%}"
            nofill = sum(1 for r in rows if r.get("_nofill"))
            if nofill:
                title += f"   ({nofill} never filled, excluded)"
        else:
            title = f"SETTLED — no bets settled in the last {SETTLED_DAYS} days"
        table.border_title = title

        now = datetime.now(timezone.utc)
        for b in rows:
            roi_b = b.get("_roi")
            if b.get("_nofill"):
                result = Text("--", style=DIM)
            else:
                result = Text("W", style=GREEN) if b["_won"] else Text("L", style=RED)
            # CT is what actually filled; a partial fill reads e.g. "0.80/1.45".
            ct = f"{float(b.get('contracts') or 0):.2f}"
            if b.get("_nofill"):
                ct = "0"
            elif b.get("requested_contracts") is not None:
                ct = f"{ct}/{float(b['requested_contracts']):.2f}"
            status = b.get("_fill_status")
            order = ("filled" if status == "executed" else status) or b.get("order_status") or "-"
            if b.get("_nofill"):
                order = "no fill"
            table.add_row(
                Text(fmt_clock(b["_start"], now), style=DIM),
                name_cell(b.get("label") or "", b.get("ticker") or ""),
                Text(b.get("side", ""), style=GREEN if b.get("side") == "YES" else RED),
                Text(f"{float(b.get('entry_price') or 0):.2f}", style=WHITE),
                Text(ct, style=DIM if b.get("_nofill") else WHITE),
                Text(f"${b['_wager']:.2f}", style=WHITE),
                result,
                *((Text("--", style=DIM),) * 4 if b.get("_nofill") else (
                    signed(b["_profit"], "{:+.2f}", zero_dim=False),
                    Text(f"{b['_fee']:.2f}", style=MUTED),
                    signed(b["_net"], "{:+.2f}", zero_dim=False),
                    signed(roi_b * 100 if roi_b is not None else None, "{:+.0f}%",
                           zero_dim=False))),
                signed(b.get("edge_cents"), "{:+.1f}c", zero_dim=False),
                Text(f"{float(b.get('confidence') or 0):.2f}", style=YELLOW),
                Text(book_label(b.get("_src", "")),
                     style=CYAN if b.get("_src") == "live" else DIM),
                Text(truncate(order, 8), style=YELLOW if b.get("_nofill") else DIM),
                height=2,
            )

    def render_status(self) -> None:
        bits = [f"watching {self._latest.name}",
                f"interval {self._loop_info.interval}s ({self._loop_info.source})"]
        if self._account.open_positions is not None:
            bits.append(f"{self._account.open_positions} open position(s)")
        if self._account_busy:
            bits.append("refreshing account...")
        if self._account.error:
            bits.append(f"[!] {self._account.error}")
        try:
            self.query_one("#status", Static).update(
                Text("  |  ".join(bits), style=RED if self._account.error else DIM))
        except Exception:
            pass


def _selftest() -> int:
    """Load every data source and print it as plain text -- no UI.

    The fast check that the wiring is right, separate from the rendering itself.
    """
    scan = load_scan()
    print(f"scan file : {LATEST}")
    if scan is None:
        print("  !! unreadable or missing")
    else:
        print(f"  generated {scan.generated_at}  scanned={scan.scanned} "
              f"deep={scan.deep_scanned} recs={len(scan.recs)}")
        for i, r in enumerate(scan.recs[:5], 1):
            book, trades, oi = flow_components(r)
            print(f"  {i}. {r.get('headline')!r} {r.get('ticker')} @ "
                  f"{r.get('entry_price')} edge={r.get('edge_cents')} "
                  f"conf={r.get('confidence')} fair={r.get('fair_prob')} "
                  f"flow={r.get('flow_direction')} {r.get('money_flow_score')} "
                  f"book={book} trades={trades} oi={oi} calib={calibration_mult(r)}")
    info = probe_loop()
    print(f"loop      : running={info.running} pid={info.pid} mode={info.mode!r} "
          f"interval={info.interval} via {info.source}")
    acct = load_account()
    print(f"account   : balance={acct.balance} today={acct.today} "
          f"overall={acct.overall} open_positions={acct.open_positions} "
          f"pending={len(acct.pending)} unrealized={acct.unrealized}")
    print(f"  today by sport   : {acct.today_by_sport}")
    print(f"  overall by sport : {acct.overall_by_sport}")
    for b in acct.pending:
        print(f"  pending {b.get('label')!r} {b.get('side')} @{b.get('entry_price')} "
              f"x{b.get('contracts')} src={b.get('_src')} "
              f"mark={b.get('_mark')} unrealized={b.get('_unrealized')}")
    w, l, gross, fees, net, roi = settled_summary(acct.settled)
    print(f"settled   : last {SETTLED_DAYS}d {w}-{l} gross={gross:+.2f} fees={fees:.2f} "
          f"net={net:+.2f} roi={roi if roi is None else round(roi, 4)}")
    for b in acct.settled[:5]:
        print(f"  {'W' if b['_won'] else 'L'} {b.get('label')!r} {b.get('side')} "
              f"@{b.get('entry_price')} x{b.get('contracts')} profit={b['_profit']:+.2f} "
              f"fee={b['_fee']:.4f} net={b['_net']:+.2f} roi={b['_roi']} "
              f"book={book_label(b.get('_src', ''))}")
    if acct.error:
        print(f"  errors: {acct.error}")
    return 0


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Live TUI dashboard for the Kalshi agent.")
    ap.add_argument("--selftest", action="store_true",
                    help="load and print every data source, then exit (no UI)")
    ap.add_argument("--poll", type=float, default=2.0,
                    help="seconds between checks for a new scan file (default 2)")
    args = ap.parse_args(argv)
    if args.selftest:
        return _selftest()
    DashboardApp(poll_seconds=args.poll).run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
