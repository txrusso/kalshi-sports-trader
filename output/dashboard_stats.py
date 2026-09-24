"""Pure data functions behind the live dashboard's summary cards and tables.

Everything here is computation over data the dashboard already has (graded
ledger rows, the loop's log, notifications.jsonl, task/order probes). No UI,
no network -- the callers in output/live_dashboard.py fetch, these compute,
so each number can be unit-tested on its own.

Money conventions match the rest of the dashboard:
- PROFIT is gross of fees, graded exactly as engine/account_summary._grade,
  so it always agrees with the record cards and `cli.py settle`.
- FEE is Kalshi's fee: the recorded one where the live ledger has it,
  otherwise config/fees.py's model (audited against every real fill).
- EXPECTED is what the model thought each bet was worth when it was placed:
  edge x contracts, i.e. (P(bet wins) - price) per contract. Gross of fees,
  like PROFIT, so PROFIT - EXPECTED isolates how results ran vs the model.
"""
from __future__ import annotations

import ast
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Iterable, Optional

from config.settings import EASTERN
from config.sports import market_kind, sport_of

SPORT_ORDER = ["MLB", "NFL", "NBA", "NHL"]


def _utc(value) -> Optional[datetime]:
    if not isinstance(value, str):
        return None
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def bet_fee(bet: dict) -> float:
    """Kalshi's fee on one bet, in dollars: the recorded `fees_usd` when the live
    ledger has one (maker fills), else config/fees.py's model -- offline, using
    its measured per-series fallback table."""
    if bet.get("fees_usd") is not None:
        return float(bet["fees_usd"])
    from config.fees import fee_for
    return fee_for(float(bet.get("contracts") or 0), float(bet.get("entry_price") or 0),
                   bet.get("ticker") or "", is_taker=bet.get("execution") != "maker")


# ---------------------------------------------------------------------------
# graded rows: one record per settled real bet, everything the tables need
# ---------------------------------------------------------------------------

def graded_rows(bets: Iterable[dict], outcomes: dict[str, bool]) -> list[dict]:
    """Every settled bet with its sport, market kind, result and money.

    Bets flagged `_nofill` (orders that never filled) are skipped: no money
    moved, so they have no result.
    """
    out = []
    for b in bets:
        if b.get("_nofill"):
            continue
        yes_won = outcomes.get(b.get("ticker"))
        if yes_won is None:
            continue
        ticker = b.get("ticker") or ""
        won = yes_won if b.get("side") == "YES" else not yes_won
        price = float(b.get("entry_price") or 0)
        ct = float(b.get("contracts") or 0)
        fair = b.get("fair_prob")
        p_side = None
        if fair is not None:
            p_side = float(fair) if b.get("side") == "YES" else 1.0 - float(fair)
        edge = b.get("edge_cents")
        out.append({
            "ticker": ticker,
            "sport": sport_of(ticker).upper(),
            "kind": market_kind(ticker),
            "won": won,
            "price": price,
            "contracts": ct,
            "wager": price * ct,
            "profit": ((1 - price) if won else -price) * ct,
            "fee": bet_fee(b),
            "expected": float(edge) / 100.0 * ct if edge is not None else None,
            "p_side": p_side,
            "start": _utc(b.get("first_pitch")),
            "placed": _utc(b.get("ts")),
            # A bet placed after the game started (minutes_before < 0) was an in-game bet.
            "in_game": b.get("minutes_before") is not None and float(b["minutes_before"]) < 0,
        })
    return out


@dataclass
class Tally:
    """Record and money over a set of graded rows."""
    n: int = 0
    wins: int = 0
    staked: float = 0.0
    profit: float = 0.0
    fees: float = 0.0
    expected: float = 0.0
    n_expected: int = 0          # rows that carried an edge (all of them, in practice)

    @property
    def losses(self) -> int:
        return self.n - self.wins

    @property
    def net(self) -> float:
        return self.profit - self.fees

    @property
    def roi(self) -> Optional[float]:
        """Net ROI on dollars staked."""
        return self.net / self.staked if self.staked else None

    @property
    def win_rate(self) -> Optional[float]:
        return self.wins / self.n if self.n else None

    @property
    def vs_expected(self) -> Optional[float]:
        """Gross profit minus what the model expected. Positive = results ran
        ahead of the model (luck, or edges understated); negative = behind."""
        return self.profit - self.expected if self.n_expected else None

    def add(self, r: dict) -> None:
        self.n += 1
        self.wins += int(r["won"])
        self.staked += r["wager"]
        self.profit += r["profit"]
        self.fees += r["fee"]
        if r["expected"] is not None:
            self.expected += r["expected"]
            self.n_expected += 1


def tally(rows: Iterable[dict]) -> Tally:
    t = Tally()
    for r in rows:
        t.add(r)
    return t


def _sport_key(s: str) -> tuple:
    return (SPORT_ORDER.index(s) if s in SPORT_ORDER else len(SPORT_ORDER), s)


def rec_table(rows: list[dict]) -> list[tuple[str, str, Tally]]:
    """(level, label, tally) rows for the Rec Table tab: ALL first, then each
    sport's total followed by its market kinds. level is "all" / "sport" / "kind"."""
    out: list[tuple[str, str, Tally]] = [("all", "ALL SPORTS", tally(rows))]
    for sport in sorted({r["sport"] for r in rows}, key=_sport_key):
        srows = [r for r in rows if r["sport"] == sport]
        out.append(("sport", sport, tally(srows)))
        for kind in ("winner", "total", "spread"):
            krows = [r for r in srows if r["kind"] == kind]
            if krows:
                out.append(("kind", kind, tally(krows)))
        for kind in sorted({r["kind"] for r in srows} - {"winner", "total", "spread"}):
            out.append(("kind", kind, tally([r for r in srows if r["kind"] == kind])))
    return out


# ---------------------------------------------------------------------------
# header cards
# ---------------------------------------------------------------------------

@dataclass
class Drawdown:
    peak: float
    peak_at: Optional[datetime]
    now: float
    down: float                    # now - peak (<= 0)
    trough: float                  # lowest point since the peak
    at_peak: bool


def drawdown(curve: list[tuple[datetime, float]]) -> Optional[Drawdown]:
    """Peak-to-now drawdown of a cumulative-profit curve (which starts at 0
    before the first bet, so the peak is never below 0)."""
    if not curve:
        return None
    peak, peak_at = 0.0, None
    for t, v in curve:
        if v > peak:
            peak, peak_at = v, t
    now = curve[-1][1]
    after = [v for t, v in curve if peak_at is None or t >= peak_at]
    return Drawdown(peak=peak, peak_at=peak_at, now=now, down=now - peak,
                    trough=min(after) if after else now, at_peak=now >= peak - 1e-9)


@dataclass
class TodayRisk:
    staked: float = 0.0            # dollars bet on today's games
    open_risk: float = 0.0         # of that, still unsettled
    n_bets: int = 0
    n_open: int = 0


def today_risk(bets: Iterable[dict], outcomes: dict[str, bool], today: str, game_date
               ) -> TodayRisk:
    """Money on today's games (same "today" as the TODAY record: the game's
    ET date, via backtest.evaluate._game_date passed in as `game_date`)."""
    r = TodayRisk()
    for b in bets:
        if b.get("_nofill") or game_date(b.get("ticker") or "") != today:
            continue
        wager = float(b.get("wager_usd") or float(b.get("entry_price") or 0)
                      * float(b.get("contracts") or 0))
        r.staked += wager
        r.n_bets += 1
        if outcomes.get(b.get("ticker")) is None:
            r.open_risk += wager
            r.n_open += 1
    return r


# ---------------------------------------------------------------------------
# calibration
# ---------------------------------------------------------------------------

CAL_EDGES = [0.0, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0001]


def calibration_table(rows: list[dict]) -> list[dict]:
    """Placed bets bucketed by the model's P(bet wins): count, average predicted,
    actual win rate. Only buckets with bets are returned."""
    out = []
    for lo, hi in zip(CAL_EDGES, CAL_EDGES[1:]):
        b = [r for r in rows if r["p_side"] is not None and lo <= r["p_side"] < hi]
        if not b:
            continue
        pred = sum(r["p_side"] for r in b) / len(b)
        act = sum(r["won"] for r in b) / len(b)
        label = (f"<{hi:.0%}" if lo == 0.0 else
                 f"{lo:.0%}+" if hi > 1.0 else f"{lo:.0%}-{hi:.0%}")
        out.append({"bucket": label, "n": len(b), "predicted": pred, "actual": act,
                    "diff": act - pred})
    return out


_CAL_LINE_RE = re.compile(r"Calibration built: (\{.*\})\s*$")


def parse_calibration_log(text: str) -> dict[str, dict]:
    """Stakey's own calibration table, from the LAST `Calibration built:` line
    the loop logged (it rebuilds once per loop start). Keys are buckets like
    "mlb:total"; values carry n, win_rate, expected_wr, posterior_wr, multiplier.
    Empty if the current log has no such line."""
    found = None
    for line in text.splitlines():
        m = _CAL_LINE_RE.search(line)
        if m:
            found = m.group(1)
    if not found:
        return {}
    try:
        data = ast.literal_eval(found)
    except (ValueError, SyntaxError):
        return {}
    return data if isinstance(data, dict) else {}


# ---------------------------------------------------------------------------
# loop health (#8-#11)
# ---------------------------------------------------------------------------

_SUP_START_RE = re.compile(r"===== supervisor (\d{4}-\d\d-\d\d \d\d:\d\d:\d\d) start;")
_RESTART_RE = re.compile(r"===== supervisor (\d{4}-\d\d-\d\d \d\d:\d\d:\d\d) loop exited .*restart")
_CYCLE_RE = re.compile(r"Cycle (\d+) done in ([\d.]+)s")


@dataclass
class LogHealth:
    session_start: Optional[datetime] = None     # ET, from the supervisor marker
    restarts: int = 0                            # auto-restarts this session
    errors: int = 0
    warnings: int = 0
    rate_limits: int = 0                         # HTTP 429s
    last_cycle_secs: Optional[float] = None
    cycles: int = 0
    last_error: str = ""


def log_health(text: str) -> LogHealth:
    """Health counters for the CURRENT loop session: everything after the last
    supervisor `start;` marker (the loop log carries no per-line timestamps,
    and one file can span several days). Falls back to the whole text."""
    h = LogHealth()
    lines = text.splitlines()
    start_idx = 0
    for i, line in enumerate(lines):
        m = _SUP_START_RE.search(line)
        if m:
            start_idx = i
            try:
                h.session_start = datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S"
                                                    ).replace(tzinfo=EASTERN)
            except ValueError:
                pass
    for line in lines[start_idx:]:
        if _RESTART_RE.search(line):
            h.restarts += 1
        elif line.startswith("ERROR") or "Traceback (most recent call last)" in line:
            h.errors += 1
            h.last_error = line.strip()
        elif line.startswith("WARNING"):
            h.warnings += 1
        if "429" in line and ("HTTP" in line or "Too Many" in line or "rate" in line.lower()):
            h.rate_limits += 1
        m = _CYCLE_RE.search(line)
        if m:
            h.cycles += 1
            h.last_cycle_secs = float(m.group(2))
    return h


@dataclass
class ChannelStatus:
    last_at: Optional[datetime] = None
    last_ok: Optional[bool] = None
    failures_today: int = 0
    sent_today: int = 0


def alert_status(jsonl_text: str, today: str) -> dict[str, ChannelStatus]:
    """Per channel ("sms", "push"): last send attempt and today's counts, from
    logs/notifications.jsonl. `accepted` means handed to the transport (Gmail /
    ntfy) -- for SMS that is NOT proof of delivery to the phone."""
    out: dict[str, ChannelStatus] = {}
    for line in jsonl_text.splitlines():
        try:
            row = json.loads(line)
        except ValueError:
            continue
        ch = row.get("channel")
        if not ch:
            continue
        st = out.setdefault(ch, ChannelStatus())
        at = _utc(row.get("ts_utc"))
        ok = bool(row.get("accepted"))
        if at and (st.last_at is None or at >= st.last_at):
            st.last_at, st.last_ok = at, ok
        if (row.get("ts_local") or "")[:10] == today:
            st.sent_today += int(ok)
            st.failures_today += int(not ok)
    return out


# Task Scheduler result codes that are NOT failures.
TASK_OK_CODES = {0, 267009, 267011}   # success, currently running, has not run yet


@dataclass
class TaskStatus:
    name: str
    last_run: Optional[datetime]
    result: Optional[int]
    next_run: Optional[datetime]

    @property
    def ok(self) -> bool:
        return self.result is None or self.result in TASK_OK_CODES

    @property
    def never_ran(self) -> bool:
        return self.result == 267011


def parse_tasks(lines: Iterable[str]) -> list[TaskStatus]:
    """`name|last_run_iso|result|next_run_iso` lines (from the PowerShell probe).
    Task Scheduler reports 1999-11-30 for "never run"; that becomes None."""
    out = []
    for line in lines:
        parts = line.strip().split("|")
        if len(parts) != 4:
            continue
        name, last, res, nxt = parts

        def when(s):
            try:
                dt = datetime.fromisoformat(s)
            except ValueError:
                return None
            return None if dt.year < 2000 else dt

        try:
            code = int(res)
        except ValueError:
            code = None
        out.append(TaskStatus(name, when(last), code, when(nxt)))
    return out


@dataclass
class RestingOrders:
    count: int = 0
    oldest_age_min: Optional[float] = None


def resting_orders(orders: list[dict], now: Optional[datetime] = None) -> RestingOrders:
    """Orders sitting on the book unfilled, and how long the oldest has waited."""
    now = now or datetime.now(timezone.utc)
    ages = []
    for o in orders:
        t = _utc(o.get("created_time"))
        if t:
            ages.append((now - t).total_seconds() / 60.0)
    return RestingOrders(count=len(orders), oldest_age_min=max(ages) if ages else None)
