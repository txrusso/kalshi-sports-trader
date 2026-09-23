"""Offline tests for the alert delivery layer (engine/notify.py).

Nothing here touches SMTP, ntfy, or the network: the transports are stubbed and
the only thing under test is the dispatcher's contract, which is what the
2026-09-22 rewrite actually changed. The bug being guarded against is a silent
one -- an alert that is dropped, or fired in a burst the carrier gateway
rate-limits, looks exactly like a working system from the loop's point of view.

Run:  .venv\\Scripts\\python.exe -m tests.test_notify
"""
from __future__ import annotations

import json
import sys
import tempfile
import threading
import time
from pathlib import Path

from engine import notify


class _Case:
    """Tiny assert harness (this project's tests don't take a pytest dependency)."""

    def __init__(self) -> None:
        self.passed = 0
        self.failed: list[str] = []

    def check(self, name: str, cond: bool, detail: str = "") -> None:
        if cond:
            self.passed += 1
        else:
            self.failed.append(f"{name}: {detail}" if detail else name)

    def report(self) -> int:
        print(f"\n{self.passed} passed, {len(self.failed)} failed")
        for f in self.failed:
            print(f"  FAIL  {f}")
        return 1 if self.failed else 0


_section = [0]


def _fresh_dispatcher(tmp: Path, **tuning) -> None:
    """Point the module at a throwaway config + a FRESH audit log, and reset the
    dispatcher, so each test gets clean pacing state and its own audit rows."""
    _section[0] += 1
    cfg = tmp / f"notify_config_{_section[0]}.txt"
    lines = [
        "smtp_host = smtp.example.com",
        "smtp_port = 587",
        "sender_email = sender@example.com",
        "sender_password = app-password",
        "recipient = 5551234567@vtext.com",
        "ntfy_topic = test-topic",
    ]
    lines += [f"{k} = {v}" for k, v in tuning.items()]
    cfg.write_text("\n".join(lines), encoding="utf-8")
    notify.DEFAULT_CONFIG = cfg
    notify.AUDIT_LOG = tmp / f"notifications_{_section[0]}.jsonl"
    notify._cfg_cache.clear()
    if notify._dispatcher is not None:
        notify._dispatcher.stop()
    notify._dispatcher = None


def main() -> int:
    c = _Case()
    tmpdir = Path(tempfile.mkdtemp(prefix="kalshi-notify-test-"))

    # --- config parsing + tuning knobs ------------------------------------
    _fresh_dispatcher(tmpdir, sms_min_gap_seconds=0.05, push_min_gap_seconds=0.01)
    cfg = notify._load_config(notify.DEFAULT_CONFIG)
    c.check("config parses", cfg.get("smtp_host") == "smtp.example.com", repr(cfg))
    c.check("tuning override read", notify._tunable(cfg, "sms_min_gap_seconds") == 0.05)
    c.check("tuning default when absent",
            notify._tunable({}, "max_attempts") == notify._TUNING_DEFAULTS["max_attempts"])
    c.check("malformed tuning falls back to default, does not raise",
            notify._tunable({"max_attempts": "not-a-number"}, "max_attempts")
            == notify._TUNING_DEFAULTS["max_attempts"])

    # --- multi-recipient support (the config-only second-gateway escape hatch) --
    multi = tmpdir / "multi.txt"
    multi.write_text("smtp_host = h\nsmtp_port = 587\nsender_email = a@b.c\n"
                     "sender_password = pw\n"
                     "recipient = 5551234567@vtext.com, 5551234567@vzwpix.com\n",
                     encoding="utf-8")
    n = notify.SmsNotifier(multi)
    c.check("comma-separated recipients split",
            n.recipients == ["5551234567@vtext.com", "5551234567@vzwpix.com"],
            repr(n.recipients))
    c.check("multi-recipient config is enabled", n.enabled is True)

    # --- disabled config is skipped, not crashed on ------------------------
    blank = tmpdir / "blank.txt"
    blank.write_text("smtp_host = h\n", encoding="utf-8")
    c.check("incomplete config disables SMS", notify.SmsNotifier(blank).enabled is False)
    c.check("placeholder password disables SMS",
            notify.SmsNotifier(_write(tmpdir / "ph.txt",
                                      "smtp_host = h\nsmtp_port = 587\n"
                                      "sender_email = a@b.c\nsender_password = your_app_password\n"
                                      "recipient = x@y.z\n")).enabled is False)
    c.check("no ntfy topic disables push", notify.PushNotifier(blank).enabled is False)
    c.check("disabled send() returns False and does not raise",
            notify.SmsNotifier(blank).send("hi") is False)

    # --- SMS length clipping (gateways truncate mid-word otherwise) --------
    _fresh_dispatcher(tmpdir, sms_max_chars=40)
    s = notify.SmsNotifier()
    short = "KALSHI: Toronto wins"
    c.check("short message untouched", s.clip(short) == short)
    long_msg = "KALSHI: Philadelphia Phillies vs Washington Nationals Under 10.5 @ 57c"
    clipped = s.clip(long_msg)
    c.check("long message clipped to limit", len(clipped) <= 40, f"len={len(clipped)}")
    c.check("clip lands on a word boundary", not clipped[:-1].endswith(" ")
            and " " not in clipped[-2:], repr(clipped))
    c.check("clip marks the truncation", clipped.endswith("\u2026"), repr(clipped))

    # --- the core fix: consecutive sends are PACED, not fired in one second --
    _fresh_dispatcher(tmpdir, sms_min_gap_seconds=0.30, max_attempts=1)
    stamps: list[float] = []
    lock = threading.Lock()

    def _record() -> None:
        with lock:
            stamps.append(time.monotonic())

    for _ in range(4):
        notify._get_dispatcher().submit(notify._Job(
            channel="sms", target="x@y.z", body="b", send=_record))
    drained = notify.flush(timeout=10)
    c.check("paced queue drains", drained and len(stamps) == 4, f"{len(stamps)} sent")
    if len(stamps) == 4:
        gaps = [b - a for a, b in zip(stamps, stamps[1:])]
        c.check("every consecutive SMS is spaced by the configured gap",
                all(g >= 0.28 for g in gaps), f"gaps={[round(g, 3) for g in gaps]}")

    # --- retry with backoff on a transient failure -------------------------
    _fresh_dispatcher(tmpdir, sms_min_gap_seconds=0.01, max_attempts=3,
                      retry_backoff_seconds=0.05)
    tries = {"n": 0}

    def _flaky() -> None:
        tries["n"] += 1
        if tries["n"] < 3:
            raise OSError("temporary gateway failure")

    notify._get_dispatcher().submit(notify._Job(
        channel="sms", target="x@y.z", body="b", send=_flaky))
    notify.flush(timeout=10)
    c.check("a transient failure is retried until it succeeds",
            tries["n"] == 3, f"attempts={tries['n']}")

    # --- a permanently failing alert gives up, and never wedges the queue --
    _fresh_dispatcher(tmpdir, sms_min_gap_seconds=0.01, max_attempts=2,
                      retry_backoff_seconds=0.05)
    dead = {"n": 0}
    survivor = {"sent": False}

    def _always_fail() -> None:
        dead["n"] += 1
        raise OSError("hard failure")

    def _ok() -> None:
        survivor["sent"] = True

    d = notify._get_dispatcher()
    d.submit(notify._Job(channel="sms", target="dead@y.z", body="b", send=_always_fail))
    d.submit(notify._Job(channel="sms", target="ok@y.z", body="b", send=_ok))
    notify.flush(timeout=10)
    c.check("a doomed alert stops at max_attempts", dead["n"] == 2, f"attempts={dead['n']}")
    c.check("a backing-off retry does not block other alerts", survivor["sent"] is True)

    # --- channels are paced independently ----------------------------------
    _fresh_dispatcher(tmpdir, sms_min_gap_seconds=5.0, push_min_gap_seconds=0.01,
                      max_attempts=1)
    got: list[str] = []
    d = notify._get_dispatcher()
    d.submit(notify._Job(channel="sms", target="t", body="b",
                         send=lambda: got.append("sms")))
    for _ in range(3):
        d.submit(notify._Job(channel="push", target="t", body="b",
                             send=lambda: got.append("push")))
    deadline = time.monotonic() + 5
    while got.count("push") < 3 and time.monotonic() < deadline:
        time.sleep(0.02)
    c.check("push alerts are not held behind the SMS gap",
            got.count("push") == 3, f"got={got}")

    # --- the audit log is what makes a late/missing text diagnosable -------
    _fresh_dispatcher(tmpdir, sms_min_gap_seconds=0.01, max_attempts=2,
                      retry_backoff_seconds=0.02)
    d = notify._get_dispatcher()
    d.submit(notify._Job(channel="sms", target="ok@y.z", body="hello", send=lambda: None))
    d.submit(notify._Job(channel="push", target="topic", body="p",
                         send=_raiser(RuntimeError("nope"))))
    notify.flush(timeout=10)
    rows = [json.loads(l) for l in
            notify.AUDIT_LOG.read_text(encoding="utf-8").splitlines() if l.strip()]
    c.check("every attempt is audited", len(rows) == 3, f"{len(rows)} rows")
    ok_rows = [r for r in rows if r["channel"] == "sms"]
    c.check("a success is recorded as accepted",
            len(ok_rows) == 1 and ok_rows[0]["accepted"] is True, repr(ok_rows))
    c.check("the audit row carries a real timestamp and the body",
            ok_rows and ok_rows[0]["ts_utc"] and ok_rows[0]["body"] == "hello",
            repr(ok_rows[:1]))
    bad = [r for r in rows if r["channel"] == "push"]
    c.check("a failure records the error text",
            len(bad) == 2 and all(r["accepted"] is False for r in bad)
            and "nope" in bad[0]["detail"], repr(bad))

    # --- an audit-log problem must never cost us the alert -----------------
    notify.AUDIT_LOG = Path("Z:/definitely/not/a/real/path/notifications.jsonl")
    delivered = {"ok": False}
    notify._get_dispatcher().submit(notify._Job(
        channel="sms", target="x", body="b",
        send=lambda: delivered.__setitem__("ok", True)))
    notify.flush(timeout=10)
    c.check("an unwritable audit log does not block delivery", delivered["ok"] is True)

    # --- formatters still produce what the ledger/push expect --------------
    bet = {"label": "Toronto wins vs Philadelphia", "entry_price": 0.57, "contracts": 2,
           "wager_usd": 1.14, "side": "YES", "ticker": "KXMLBGAME-26SEP16TORPHI-TOR",
           "edge_cents": 8.1, "confidence": 0.52, "minutes_before": 23.0}
    c.check("format_bet_sms is short enough for one segment",
            len(notify.format_bet_sms(bet)) <= 160, notify.format_bet_sms(bet))
    slip = notify.format_bet_push(bet)
    c.check("push slip keeps the tap-through url",
            slip["url"].endswith("kxmlbgame-26sep16torphi"), slip["url"])
    c.check("order receipt is past tense",
            notify.format_order_placed_sms({**bet, "order_id": "abc", "order_status": "resting"})
            .startswith("KALSHI ORDER PLACED:"))

    return c.report()


def _raiser(exc):
    def _f():
        raise exc
    return _f


def _write(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


if __name__ == "__main__":
    sys.exit(main())
