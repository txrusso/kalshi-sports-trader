"""Text-message alerts via a carrier email-to-SMS gateway (SMTP).

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
"""
from __future__ import annotations

import logging
import smtplib
import ssl
from email.message import EmailMessage
from pathlib import Path
from typing import Optional

import requests

log = logging.getLogger("engine.notify")

DEFAULT_CONFIG = Path(__file__).resolve().parent.parent / "notify_config.txt"
_REQUIRED = ("smtp_host", "smtp_port", "sender_email", "sender_password", "recipient")


class SmsNotifier:
    def __init__(self, config_path: Optional[Path] = None):
        self.cfg = self._load(Path(config_path or DEFAULT_CONFIG))

    @staticmethod
    def _load(path: Path) -> dict:
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

    @property
    def enabled(self) -> bool:
        if not all(self.cfg.get(k) for k in _REQUIRED):
            return False
        # Treat the placeholder password as "not configured yet".
        return "your_" not in self.cfg.get("sender_password", "").lower()

    def send(self, message: str) -> bool:
        if not self.enabled:
            log.info("SMS notifier not configured (%s); skipping alert.", DEFAULT_CONFIG.name)
            return False
        try:
            msg = EmailMessage()
            msg["From"] = self.cfg["sender_email"]
            msg["To"] = self.cfg["recipient"]
            msg["Subject"] = ""                       # SMS gateways prepend subject; keep empty
            msg.set_content(message)
            # Gmail shows App Passwords in groups of four; the real value has no spaces.
            password = self.cfg["sender_password"].replace(" ", "")
            with smtplib.SMTP(self.cfg["smtp_host"], int(self.cfg["smtp_port"]), timeout=20) as s:
                s.starttls(context=ssl.create_default_context())
                s.login(self.cfg["sender_email"], password)
                s.send_message(msg)
            log.info("Text alert sent to %s", self.cfg["recipient"])
            return True
        except Exception as e:                        # never let a texting failure break the loop
            log.warning("Text alert failed: %s", e)
            return False


def format_bet_sms(bet: dict) -> str:
    """Compact, actionable text for a single placed bet: bet, price, wager.
    One bet per SMS (not bundled) -- email-to-SMS carrier gateways (e.g. Verizon's
    vtext.com) silently truncate long messages with no warning and no multipart
    reassembly, so a bundled multi-bet text can get cut off mid-word. Keeping each
    text to one short bet line stays well under any such limit.
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

    Config (notify_config.txt, gitignored):
        ntfy_topic  = <a long, private, random topic name>   # required to enable
        ntfy_server = https://ntfy.sh                        # optional; this default

    If ntfy_topic is blank/missing, alerts are silently skipped (like SMS)."""

    def __init__(self, config_path: Optional[Path] = None):
        self.cfg = SmsNotifier._load(Path(config_path or DEFAULT_CONFIG))

    @property
    def server(self) -> str:
        return (self.cfg.get("ntfy_server") or "https://ntfy.sh").rstrip("/")

    @property
    def topic(self) -> str:
        return self.cfg.get("ntfy_topic", "").strip()

    @property
    def enabled(self) -> bool:
        return bool(self.topic)

    def send(self, slip: dict) -> bool:
        """`slip` is a format_bet_push(...) dict. Never raises."""
        if not self.enabled:
            log.info("ntfy push not configured (ntfy_topic blank); skipping alert.")
            return False
        try:
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
            log.info("ntfy push sent to topic %s", self.topic)
            return True
        except Exception as e:                               # never let a push failure break the loop
            log.warning("ntfy push failed: %s", e)
            return False
