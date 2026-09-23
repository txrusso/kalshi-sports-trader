"""Alert delivery: ntfy push (primary) + carrier email-to-SMS gateway (backup).

Two families of alert live here:
  * format_bet_sms / format_bet_push — a still-to-place order slip, for the
    recommend-only paper-trade path (engine/paper.py). Tells the user what to
    place; this module never submits anything itself.
  * format_order_placed_sms / format_order_placed_push — a past-tense receipt
    for the live auto-execution path (engine/live.py). The order has ALREADY
    been submitted via the Kalshi API by the time this fires; these just
    report what happened (order id/status), they don't gate or confirm it.

Needs an SMTP sender account (a Gmail App Password works well). Config is read
from notify_config.txt (gitignored). If it's missing or incomplete, alerts are
silently skipped and the loop keeps running.

DELIVERY MODEL — why this is not just "call smtplib" (rewritten 2026-09-22)
--------------------------------------------------------------------------
The user reported texts arriving intermittently and sometimes ~12 hours late.
Measured against logs/paper_loop.log: 87 texts were sent, ZERO reported a
failure, and 47 of the 87 (54%) were fired inside a same-second burst of 2-4
messages — one SMTP connection per bet, back to back, whenever a scan cycle
triggered several games at once. That is exactly the shape a carrier
email-to-SMS gateway rate-limits. Verizon's vtext.com then either drops a
message outright (nothing arrives) or issues a temporary 4xx deferral, at
which point Gmail retries on its own schedule — which is where a text that
lands half a day later comes from.

The old code could not see any of this, because `smtp.send_message()`
returning cleanly only means Gmail accepted the message for delivery. It says
nothing about whether the carrier gateway accepted it, and nothing at all
about whether a handset ever rang. The old log line ("Text alert sent") was
therefore overstating what had happened.

So this module now:
  1. Serializes every alert through ONE background dispatcher thread, with a
     configurable minimum gap between consecutive messages per channel
     (`sms_min_gap_seconds`, default 15s). Bursts are the root cause; spacing
     them is the one lever available on this side of the gateway.
  2. Retries transient SMTP failures with exponential backoff instead of
     giving up after a single attempt.
  3. Writes every attempt to logs/notifications.jsonl with real timestamps, so
     "this text arrived 12 hours late" can be checked against when it actually
     left this machine (and attributed to the carrier rather than guessed at).
  4. Reports honestly: the log now says "handed to SMTP", not "sent".
  5. Accepts a comma-separated `recipient` list, so a second, more reliable
     address can be added alongside the SMS gateway (e.g. Verizon's MMS
     gateway <number>@vzwpix.com, or a plain email inbox) with no code change.

None of this can make a best-effort carrier gateway reliable. Email-to-SMS
offers no delivery receipt and no SLA, and Verizon has been winding vtext.com
down. ntfy push is the channel to trust (an HTTP POST with a real status code,
which this module checks); SMS is a backup. A genuinely reliable text needs a
real SMS API (e.g. Twilio), a paid dependency this project does not have.
"""
from __future__ import annotations

import atexit
import json
import logging
import queue
import smtplib
import ssl
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from email.message import EmailMessage
from pathlib import Path
from typing import Callable, Optional

import requests

log = logging.getLogger("engine.notify")

DEFAULT_CONFIG = Path(__file__).resolve().parent.parent / "notify_config.txt"
AUDIT_LOG = Path(__file__).resolve().parent.parent / "logs" / "notifications.jsonl"
_REQUIRED = ("smtp_host", "smtp_port", "sender_email", "sender_password", "recipient")

# Delivery tuning. Every one of these is overridable from notify_config.txt
# (same `key = value` syntax as the credentials).
_TUNING_DEFAULTS = {
    # Minimum seconds between two consecutive messages on the same channel.
    # The whole point of the rewrite: 4 texts in one second is what the carrier
    # gateway punishes. 15s spreads a 4-bet cycle over ~45s, still far inside
    # the ~600s scan interval and well before first pitch.
    "sms_min_gap_seconds": 15.0,
    # ntfy is a normal HTTP API and does not need this, but a token gap keeps
    # the dispatcher from hammering it either.
    "push_min_gap_seconds": 1.0,
    # Total attempts per message (1 = the old behaviour, no retry).
    "max_attempts": 4,
    # First retry delay; doubles each attempt (20s, 40s, 80s).
    "retry_backoff_seconds": 20.0,
    # SMS gateways truncate long messages silently. Stay under the 160-char
    # GSM limit with room for the gateway's own prefix, rather than letting it
    # cut mid-word.
    "sms_max_chars": 150,
    # How long to wait at process exit for queued alerts to drain.
    "shutdown_flush_seconds": 90.0,
}


_cfg_cache: dict[str, tuple[float, dict]] = {}
_CFG_TTL_SECONDS = 30.0


def _cached_config(path: Optional[Path] = None) -> dict:
    """`_load_config` with a short TTL. The dispatcher's scheduling loop asks
    for the pacing knobs several times a second; re-reading the file each time
    would be pointless disk traffic. A 30s TTL still picks up an edit to
    notify_config.txt without a restart.

    The path defaults to None rather than to DEFAULT_CONFIG directly: a default
    argument is bound once at import, which would pin this to whatever the
    constant was then and quietly ignore any later reassignment."""
    path = Path(path or DEFAULT_CONFIG)
    key = str(path)
    now = time.monotonic()
    hit = _cfg_cache.get(key)
    if hit is not None and now - hit[0] < _CFG_TTL_SECONDS:
        return hit[1]
    cfg = _load_config(path)
    _cfg_cache[key] = (now, cfg)
    return cfg


def _load_config(path: Path) -> dict:
    """Parse the `key = value` config. Shared by both notifiers and by the
    dispatcher, which needs the tuning knobs without caring about creds."""
    cfg: dict = {}
    if not path.exists():
        return cfg
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        cfg[k.strip().lower()] = v.strip()
    return cfg


def _tunable(cfg: dict, key: str):
    """Read a delivery-tuning knob, falling back to the default. A malformed
    value must never take the notifier down — fall back and warn."""
    default = _TUNING_DEFAULTS[key]
    raw = cfg.get(key)
    if raw in (None, ""):
        return default
    try:
        return type(default)(raw)
    except (TypeError, ValueError):
        log.warning("notify_config.txt: %s=%r is not a number; using %r.", key, raw, default)
        return default


def _audit(channel: str, target: str, ok: bool, attempt: int, detail: str, body: str) -> None:
    """One JSON line per delivery ATTEMPT, so a missing or late alert can be
    checked against when it actually left this machine.

    `ok` means the transport accepted it — for SMS that is Gmail, NOT the
    carrier and definitely not the handset. That distinction is the whole
    reason this file exists, so the field is `accepted`, not `delivered`.
    Never raises: an audit-log problem must not cost us the alert."""
    try:
        AUDIT_LOG.parent.mkdir(parents=True, exist_ok=True)
        now = datetime.now(timezone.utc)
        row = {
            "ts_utc": now.isoformat(timespec="seconds"),
            "ts_local": now.astimezone().isoformat(timespec="seconds"),
            "channel": channel,
            "target": target,
            "accepted": ok,
            "attempt": attempt,
            "detail": detail,
            "body": (body or "")[:400],
        }
        with AUDIT_LOG.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row) + "\n")
    except Exception:
        log.debug("Could not write the notification audit log.", exc_info=True)


@dataclass
class _Job:
    """One outbound message, on one channel, to one target."""
    channel: str                      # "sms" | "push"
    target: str                       # recipient address / ntfy topic (for the audit log)
    body: str                         # human-readable payload, for the audit log
    send: Callable[[], None]          # raises on failure, returns None on success
    attempts: int = 0
    not_before: float = field(default_factory=time.monotonic)


class _Dispatcher(threading.Thread):
    """Serializes and paces every outbound alert on one daemon thread.

    Callers (the scan loop, the maker fill callback) hand a job over and carry
    on immediately, so a 15s inter-message gap or a retry backoff never stalls
    a scan cycle. Jobs carry a `not_before` so a backing-off retry yields to
    other pending alerts instead of blocking the queue head."""

    def __init__(self) -> None:
        super().__init__(name="notify-dispatcher", daemon=True)
        self._q: "queue.Queue[_Job]" = queue.Queue()
        self._pending: list[_Job] = []
        self._last_send: dict[str, float] = {}
        # Outstanding = submitted but not yet finished (delivered or given up).
        # Counting explicitly rather than inferring idleness from "is _pending
        # empty" — a job that has been submitted but not yet drained off the
        # queue is in neither place, and treating that instant as idle would
        # let flush() return before the alert had been attempted at all.
        self._outstanding = 0
        self._count_lock = threading.Lock()
        self._idle = threading.Event()
        self._idle.set()
        self._stopping = threading.Event()

    # -- public ------------------------------------------------------------
    def submit(self, job: _Job) -> None:
        with self._count_lock:
            self._outstanding += 1
            self._idle.clear()
        self._q.put(job)

    def _finished(self) -> None:
        with self._count_lock:
            self._outstanding -= 1
            if self._outstanding <= 0:
                self._outstanding = 0
                self._idle.set()

    def flush(self, timeout: float) -> bool:
        """Block until every queued alert has been attempted (or `timeout`
        elapses). Returns True if the queue drained. Used at process exit so a
        one-shot command (e.g. `settle --notify`) can't die with its text
        still sitting in the queue."""
        return self._idle.wait(timeout)

    def stop(self) -> None:
        self._stopping.set()

    @property
    def outstanding(self) -> int:
        with self._count_lock:
            return self._outstanding

    # -- internals ---------------------------------------------------------
    def _drain_queue(self) -> None:
        while True:
            try:
                self._pending.append(self._q.get_nowait())
            except queue.Empty:
                return

    def _gap_for(self, channel: str) -> float:
        key = "sms_min_gap_seconds" if channel == "sms" else "push_min_gap_seconds"
        return float(_tunable(_cached_config(), key))

    def _next_ready(self, now: float) -> Optional[_Job]:
        """Earliest job whose retry backoff AND channel spacing both allow it."""
        best: Optional[_Job] = None
        for job in self._pending:
            if job.not_before > now:
                continue
            last = self._last_send.get(job.channel)
            if last is not None and now - last < self._gap_for(job.channel):
                continue
            if best is None or job.not_before < best.not_before:
                best = job
        return best

    def run(self) -> None:
        while not self._stopping.is_set():
            self._drain_queue()
            if not self._pending:
                time.sleep(0.05)
                continue
            job = self._next_ready(time.monotonic())
            if job is None:
                time.sleep(0.05)         # everything is waiting on a gap or a backoff
                continue
            self._pending.remove(job)
            self._attempt(job)

    def _attempt(self, job: _Job) -> None:
        cfg = _load_config(DEFAULT_CONFIG)
        max_attempts = int(_tunable(cfg, "max_attempts"))
        backoff = float(_tunable(cfg, "retry_backoff_seconds"))
        job.attempts += 1
        self._last_send[job.channel] = time.monotonic()
        try:
            job.send()
        except Exception as e:
            detail = f"{type(e).__name__}: {e}"
            _audit(job.channel, job.target, False, job.attempts, detail, job.body)
            if job.attempts < max_attempts:
                delay = backoff * (2 ** (job.attempts - 1))
                job.not_before = time.monotonic() + delay
                self._pending.append(job)
                log.warning("%s alert to %s failed (attempt %d/%d: %s); retrying in %gs.",
                            job.channel.upper(), job.target, job.attempts, max_attempts,
                            detail, delay)
            else:
                log.error("%s alert to %s GAVE UP after %d attempts: %s",
                          job.channel.upper(), job.target, job.attempts, detail)
                self._finished()
            return
        self._finished()
        _audit(job.channel, job.target, True, job.attempts, "accepted by transport", job.body)
        if job.channel == "sms":
            # Deliberately not "sent": Gmail accepted it. The carrier gateway is
            # a separate, unacknowledged hop that may still drop or defer it.
            log.info("Text alert handed to SMTP for %s (attempt %d; carrier delivery is "
                     "best-effort and unconfirmed).", job.target, job.attempts)
        else:
            log.info("ntfy push sent to topic %s (attempt %d).", job.target, job.attempts)


_dispatcher: Optional[_Dispatcher] = None
_dispatcher_lock = threading.Lock()


def _get_dispatcher() -> _Dispatcher:
    global _dispatcher
    with _dispatcher_lock:
        if _dispatcher is None:
            _dispatcher = _Dispatcher()
            _dispatcher.start()
            atexit.register(_flush_at_exit)
        return _dispatcher


def _flush_at_exit() -> None:
    d = _dispatcher
    if d is None:
        return
    timeout = float(_tunable(_load_config(DEFAULT_CONFIG), "shutdown_flush_seconds"))
    if not d.flush(timeout):
        # Say so loudly: this is the one path where an alert is genuinely lost
        # by us rather than by the carrier, and the audit log won't show it
        # (nothing was attempted), so the log line is the only record.
        log.error("Exiting with %d alert(s) still queued after %.0fs -- they were NOT "
                  "sent. Raise shutdown_flush_seconds in notify_config.txt if this "
                  "recurs.", d.outstanding, timeout)
    d.stop()


def flush(timeout: Optional[float] = None) -> bool:
    """Wait for queued alerts to be attempted. Safe when nothing is queued."""
    if _dispatcher is None:
        return True
    if timeout is None:
        timeout = float(_tunable(_load_config(DEFAULT_CONFIG), "shutdown_flush_seconds"))
    return _dispatcher.flush(timeout)


class SmsNotifier:
    def __init__(self, config_path: Optional[Path] = None):
        self.path = Path(config_path or DEFAULT_CONFIG)
        self.cfg = _load_config(self.path)

    @staticmethod
    def _load(path: Path) -> dict:
        """Kept for backwards compatibility — PushNotifier and older callers
        reach for this to parse the shared config file."""
        return _load_config(path)

    @property
    def enabled(self) -> bool:
        if not all(self.cfg.get(k) for k in _REQUIRED):
            return False
        # Treat the placeholder password as "not configured yet".
        return "your_" not in self.cfg.get("sender_password", "").lower()

    @property
    def recipients(self) -> list[str]:
        """`recipient` may list several addresses, comma- or semicolon-separated.
        Each gets its own paced message, so adding a second gateway (e.g.
        <number>@vzwpix.com) or a plain inbox is a config-only change."""
        raw = self.cfg.get("recipient", "")
        return [r.strip() for r in raw.replace(";", ",").split(",") if r.strip()]

    def clip(self, message: str) -> str:
        """Carrier gateways truncate long messages silently and mid-word. Cut on
        a word boundary ourselves so the tail is at least legible."""
        limit = int(_tunable(self.cfg, "sms_max_chars"))
        if len(message) <= limit:
            return message
        cut = message[:limit - 1].rstrip()
        space = cut.rfind(" ")
        if space > limit // 2:
            cut = cut[:space]
        return cut.rstrip() + "…"

    def send_now(self, message: str, recipient: Optional[str] = None) -> None:
        """Synchronous single send. Raises on failure — the dispatcher turns
        that into a retry, and `cli.py notify-test` surfaces it to the user."""
        msg = EmailMessage()
        msg["From"] = self.cfg["sender_email"]
        msg["To"] = recipient or self.recipients[0]
        msg["Subject"] = ""                       # SMS gateways prepend subject; keep empty
        msg.set_content(message)
        # Gmail shows App Passwords in groups of four; the real value has no spaces.
        password = self.cfg["sender_password"].replace(" ", "")
        with smtplib.SMTP(self.cfg["smtp_host"], int(self.cfg["smtp_port"]), timeout=30) as s:
            s.starttls(context=ssl.create_default_context())
            s.login(self.cfg["sender_email"], password)
            s.send_message(msg)

    def send(self, message: str) -> bool:
        """Queue the text for paced delivery. The return value says whether it
        was QUEUED, not whether it arrived — nothing on this channel can tell
        us that. Check logs/notifications.jsonl for when each attempt left."""
        if not self.enabled:
            log.info("SMS notifier not configured (%s); skipping alert.", self.path.name)
            return False
        body = self.clip(message)
        for rcpt in self.recipients:
            _get_dispatcher().submit(_Job(
                channel="sms", target=rcpt, body=body,
                send=lambda r=rcpt, b=body: self.send_now(b, r),
            ))
        return True


def format_bet_sms(bet: dict) -> str:
    """Compact, actionable text for a single placed bet: bet, price, wager.
    One bet per SMS (not bundled) -- email-to-SMS carrier gateways (e.g. Verizon's
    vtext.com) silently truncate long messages with no warning and no multipart
    reassembly, so a bundled multi-bet text can get cut off mid-word. Keeping each
    text to one short bet line stays well under any such limit. (Several bets in
    one cycle therefore mean several texts -- which is why the dispatcher above
    paces them apart instead of firing them in the same second.)
    The label already names the game (e.g. "Toronto wins vs Philadelphia" /
    "Cleveland vs Chicago WS Over 5.5"), so no separate matchup prefix is needed."""
    cost_cents = round(bet["entry_price"] * 100)
    wager = bet.get("wager_usd")
    if wager is None:
        wager = round(bet["entry_price"] * (bet.get("contracts") or 0), 2)
    return f"KALSHI: {bet['label']} @ {cost_cents}c, wager ${wager:.2f}"


# --- ntfy push (a richer, phone-first alert with a tap-through order slip) ------
#
# Same recommend-only contract as SMS: this ONLY tells the user what to place.
# It never places, confirms, or submits any order. The tap-through opens Kalshi's
# own market page in the app/browser; the user enters and confirms the order there.

# Kalshi web market URL: /markets/{series}/{slug}/{event}. Verified live (2026-08-31)
# that Kalshi's router resolves on the SERIES + trailing EVENT ticker only and
# ignores the middle slug entirely (a deliberately-wrong slug still loaded the
# correct market), so the slug is cosmetic. The bare `/markets/{event}` form
# (no slug segment) does NOT resolve, so the three-segment form is required.
# Slugs below are just for a human-readable URL; resolution never depends on them.
_KALSHI_SERIES_SLUG = {
    "KXMLBGAME": "professional-baseball-game",
    "KXMLBTOTAL": "professional-baseball-total-runs",
    "KXNFLGAME": "professional-football-game",
    "KXNFLTOTAL": "professional-football-total-points",
    "KXNBAGAME": "pro-basketball-game",
    "KXNBATOTAL": "pro-basketball-total-points",
    "KXNHLGAME": "nhl-game",
    "KXNHLTOTAL": "nhl-goal-total",
}


def kalshi_market_url(ticker: str) -> str:
    """Deep link to a market's Kalshi page (opens the exact game in the app).

    `/markets/{series}/{slug}/{event}` -- resolution is driven by series + the
    trailing event ticker (market ticker minus its -<YESTEAM>/-<K> leaf); the
    slug is decorative. Works uniformly for both sports and both market kinds.
    """
    series = ticker.split("-", 1)[0]
    event = ticker.rsplit("-", 1)[0].lower()   # market ticker minus the leaf segment
    slug = _KALSHI_SERIES_SLUG.get(series, "market")
    return f"https://kalshi.com/markets/{series.lower()}/{slug}/{event}"


def format_bet_push(bet: dict) -> dict:
    """Build a phone-first order slip: what to buy, how many, at what limit, where.

    `bet['side']` is the raw Kalshi contract side (YES/NO) to BUY on `bet['ticker']`
    at `bet['entry_price']` -- the mechanics the user punches into the Kalshi app.
    `bet['label']` is the positively-framed intent (e.g. "Toronto wins vs
    Philadelphia" / "... Over 5.5") used only for the human-readable title.
    Returns {title, body, url, tags, priority} for PushNotifier.send().
    """
    price_c = round(bet["entry_price"] * 100)
    contracts = bet.get("contracts") or 0
    wager = bet.get("wager_usd")
    if wager is None:
        wager = round(bet["entry_price"] * contracts, 2)
    side = str(bet.get("side", "")).upper()
    lines = [
        f"BUY {contracts} x {side} @ {price_c}c limit  (~${wager:.2f})",
        bet.get("ticker", ""),
    ]
    extra = []
    if bet.get("edge_cents") is not None:
        extra.append(f"edge {bet['edge_cents']}c")
    if isinstance(bet.get("confidence"), (int, float)):
        extra.append(f"conf {bet['confidence']:.2f}")
    mb = bet.get("minutes_before")
    if isinstance(mb, (int, float)):
        extra.append(f"{mb:.0f}m to start")
    if extra:
        lines.append(" | ".join(extra))
    return {
        "title": f"KALSHI: {bet.get('label', bet.get('ticker', 'bet'))}",
        "body": "\n".join(l for l in lines if l),
        "url": kalshi_market_url(bet.get("ticker", "")),
        "tags": "moneybag",
        "priority": "high",
    }


def format_order_placed_sms(bet: dict) -> str:
    """Compact receipt for a LIVE order the agent already submitted (contrast
    with format_bet_sms's still-to-place slip)."""
    cost_cents = round(bet["entry_price"] * 100)
    wager = bet.get("wager_usd")
    if wager is None:
        wager = round(bet["entry_price"] * (bet.get("contracts") or 0), 2)
    return (f"KALSHI ORDER PLACED: {bet['label']} @ {cost_cents}c, ${wager:.2f} "
           f"(order {bet.get('order_id') or '?'}, {bet.get('order_status') or 'submitted'})")


def format_order_placed_push(bet: dict) -> dict:
    """Push receipt for a LIVE order the agent already submitted through the
    Kalshi API — past tense, includes order id/status. Contrast with
    format_bet_push, which is a still-to-place slip for the human to act on."""
    price_c = round(bet["entry_price"] * 100)
    contracts = bet.get("contracts") or 0
    wager = bet.get("wager_usd")
    if wager is None:
        wager = round(bet["entry_price"] * contracts, 2)
    side = str(bet.get("side", "")).upper()
    status = bet.get("order_status") or "submitted"
    lines = [
        f"BOUGHT {contracts} x {side} @ {price_c}c  (~${wager:.2f})  status={status}",
        bet.get("ticker", ""),
        f"order {bet.get('order_id') or '?'}",
    ]
    extra = []
    if bet.get("edge_cents") is not None:
        extra.append(f"edge {bet['edge_cents']}c")
    if isinstance(bet.get("confidence"), (int, float)):
        extra.append(f"conf {bet['confidence']:.2f}")
    if extra:
        lines.append(" | ".join(extra))
    return {
        "title": f"KALSHI ORDER PLACED: {bet.get('label', bet.get('ticker', 'bet'))}",
        "body": "\n".join(l for l in lines if l),
        "url": kalshi_market_url(bet.get("ticker", "")),
        "tags": "moneybag,white_check_mark",
        "priority": "high",
    }


class PushNotifier:
    """ntfy push alerts (https://ntfy.sh). Free, no account: the user subscribes
    to a topic in the ntfy phone app and this POSTs to the same topic.

    This is the RELIABLE channel of the two. Unlike the SMS gateway, ntfy
    answers with a real HTTP status code, so a failure here is observable and
    retryable — which is exactly what the dispatcher does with it.

    Config (notify_config.txt, gitignored):
        ntfy_topic  = <a long, private, random topic name>   # required to enable
        ntfy_server = https://ntfy.sh                        # optional; this default

    If ntfy_topic is blank/missing, alerts are silently skipped (like SMS)."""

    def __init__(self, config_path: Optional[Path] = None):
        self.path = Path(config_path or DEFAULT_CONFIG)
        self.cfg = _load_config(self.path)

    @property
    def server(self) -> str:
        return (self.cfg.get("ntfy_server") or "https://ntfy.sh").rstrip("/")

    @property
    def topic(self) -> str:
        return self.cfg.get("ntfy_topic", "").strip()

    @property
    def enabled(self) -> bool:
        return bool(self.topic)

    def send_now(self, slip: dict) -> None:
        """Synchronous single POST. Raises on any non-2xx or network error, so
        the dispatcher can retry it."""
        headers = {
            # HTTP header values must be latin-1 safe; keep title ASCII.
            "Title": slip["title"].encode("ascii", "replace").decode("ascii"),
            "Priority": slip.get("priority", "default"),
            "Tags": slip.get("tags", ""),
        }
        url = slip.get("url")
        if url:
            headers["Click"] = url                       # tapping the notif opens Kalshi
            headers["Actions"] = f"view, Open on Kalshi, {url}, clear=false"
        resp = requests.post(
            f"{self.server}/{self.topic}",
            data=slip["body"].encode("utf-8"),
            headers=headers,
            timeout=20,
        )
        resp.raise_for_status()

    def send(self, slip: dict) -> bool:
        """`slip` is a format_bet_push(...) dict. Queues it; never raises.
        Returns whether it was queued (False only when ntfy isn't configured)."""
        if not self.enabled:
            log.info("ntfy push not configured (ntfy_topic blank); skipping alert.")
            return False
        _get_dispatcher().submit(_Job(
            channel="push", target=self.topic,
            body=f"{slip.get('title', '')} :: {slip.get('body', '')}",
            send=lambda s=slip: self.send_now(s),
        ))
        return True
