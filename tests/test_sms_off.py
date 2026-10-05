"""Offline tests for the SMS off switch (notify_config.txt::sms_enabled).

Texts were turned off at the user's request on 2026-10-04. The switch lives
on SmsNotifier.enabled, which SmsNotifier.send() already gates on, so it
covers every caller (engine/loop.py's live-order and paper-bet alerts, and
cli.py settle --notify) from one place. ntfy push is a separate channel and
must be unaffected.
"""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from engine.notify import PushNotifier, SmsNotifier   # noqa: E402

CREDS = ("smtp_host = smtp.gmail.com\nsmtp_port = 587\n"
         "sender_email = a@b.com\nsender_password = abcd efgh ijkl mnop\n"
         "recipient = 5551234567@vtext.com\n"
         "ntfy_topic = some-topic\n")


def cfg(extra: str = "") -> Path:
    d = Path(tempfile.mkdtemp())
    p = d / "notify_config.txt"
    p.write_text(CREDS + extra, encoding="utf-8")
    return p


class SmsOffSwitch(unittest.TestCase):

    def test_production_config_has_texts_off(self):
        # The real file, as shipped on this machine.
        real = SmsNotifier()
        self.assertTrue(real.disabled_by_config, "texts should be OFF")
        self.assertFalse(real.enabled)

    def test_off_blocks_send_without_touching_smtp(self):
        n = SmsNotifier(cfg("sms_enabled = false\n"))
        with mock.patch.object(SmsNotifier, "send_now") as sn:
            self.assertFalse(n.send("anything"))
        sn.assert_not_called()

    def test_credentials_are_preserved_so_it_is_reversible(self):
        n = SmsNotifier(cfg("sms_enabled = false\n"))
        self.assertEqual(n.recipients, ["5551234567@vtext.com"])
        self.assertTrue(n.cfg.get("sender_password"))

    def test_true_turns_it_back_on(self):
        self.assertTrue(SmsNotifier(cfg("sms_enabled = true\n")).enabled)

    def test_absent_key_leaves_texts_on(self):
        # A missing switch must never be read as "off" -- the default has to
        # stay on so this can only be disabled deliberately.
        self.assertTrue(SmsNotifier(cfg()).enabled)

    def test_a_typo_does_not_silently_disable(self):
        for val in ("ture", "yes", "1", "on", "enabled", ""):
            with self.subTest(val=val):
                self.assertFalse(SmsNotifier(cfg(f"sms_enabled = {val}\n")).disabled_by_config)

    def test_recognised_off_values(self):
        for val in ("false", "False", "FALSE", " no ", "0", "off", "disabled"):
            with self.subTest(val=val):
                self.assertTrue(SmsNotifier(cfg(f"sms_enabled = {val}\n")).disabled_by_config)

    def test_off_is_distinguishable_from_never_configured(self):
        off = SmsNotifier(cfg("sms_enabled = false\n"))
        self.assertTrue(off.disabled_by_config)
        d = Path(tempfile.mkdtemp()) / "notify_config.txt"
        d.write_text("ntfy_topic = t\n", encoding="utf-8")
        never = SmsNotifier(d)
        self.assertFalse(never.enabled)
        self.assertFalse(never.disabled_by_config)

    def test_push_is_unaffected(self):
        p = PushNotifier(cfg("sms_enabled = false\n"))
        self.assertTrue(p.enabled)
        self.assertEqual(p.topic, "some-topic")


class EveryCallerGoesThroughTheSwitch(unittest.TestCase):
    """send() is the single chokepoint -- if a caller ever reaches send_now()
    directly it would bypass the switch, so pin that none do."""

    def test_no_production_caller_uses_send_now(self):
        root = Path(__file__).resolve().parents[1]
        offenders = []
        for f in [root / "engine" / "loop.py", root / "engine" / "live.py",
                  root / "engine" / "maker.py"]:
            if f.exists() and "send_now(" in f.read_text(encoding="utf-8"):
                offenders.append(f.name)
        self.assertEqual(offenders, [], f"these bypass the SMS switch: {offenders}")

    def test_loop_and_settle_send_paths_are_blocked_when_off(self):
        from engine import notify
        sent = []
        n = SmsNotifier(cfg("sms_enabled = false\n"))
        with mock.patch.object(notify, "_get_dispatcher") as disp:
            for msg in ("KALSHI: order placed", "KALSHI settle: 3-1"):
                self.assertFalse(n.send(msg))
        disp.assert_not_called()
        self.assertEqual(sent, [])


if __name__ == "__main__":
    unittest.main()
