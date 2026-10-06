"""Offline tests for the per-sport live-trading switch
(config/settings.py::live_trade_disabled_sports, enforced in
engine/live.py::run_live_trigger).

NHL was switched off 2026-09-30 after day one went 1-4 on totals, and back ON
2026-10-06 after 33 shadow bets (see config/settings.py for the numbers). The
production default therefore trades every sport TODAY -- but the mechanism has
to keep working, because it is the only brake available if a sport has to come
off real money again in a hurry. These tests exercise the switch with an
EXPLICIT config rather than relying on the default, and separately pin what
the live default actually is, so flipping it can never silently gut the
coverage.
"""
from __future__ import annotations

import sys
import unittest
from dataclasses import replace
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config.settings import DEFAULTS                     # noqa: E402

# The switch under test, applied explicitly. Production's own default is
# asserted separately in LiveTradingDefaults below.
NHL_OFF = replace(DEFAULTS, live_trade_disabled_sports=("nhl",))
from engine import live as live_mod                      # noqa: E402


class _Rec:
    def __init__(self, ticker, side="YES", price=0.50, contracts=2.0):
        self.ticker = ticker
        self.side = side
        self.entry_price = price
        self.suggested_contracts = contracts


class _Ledger:
    def __init__(self):
        self.rows = []

    def record(self, bet):
        self.rows.append(bet)


def _bet(ticker):
    return {"event_key": ticker.rsplit("-", 1)[0], "label": f"bet on {ticker}",
            "minutes_before": 20.0, "first_pitch": "2026-10-01T23:00:00+00:00"}


class LiveSportSwitch(unittest.TestCase):

    def _run(self, tickers, settings):
        recs = [_Rec(t) for t in tickers]
        ledger = _Ledger()
        submitted = []

        def fake_submit(client, req, cfg, client_order_id=None):
            submitted.append(req.ticker)
            return mock.Mock(order_id="oid-" + req.ticker, status="filled",
                             client_order_id=client_order_id)

        with mock.patch.object(live_mod, "iter_trigger_candidates",
                               return_value=[(r, _bet(r.ticker), None) for r in recs]), \
             mock.patch.object(live_mod, "submit_order", side_effect=fake_submit), \
             mock.patch.object(live_mod.bet_inputs, "save"):
            placed = live_mod.run_live_trigger(recs, {}, ledger, mock.Mock(),
                                               settings, 25.0)
        return submitted, placed, ledger

    def test_nhl_order_is_never_submitted(self):
        submitted, placed, ledger = self._run(
            ["KXNHLGAME-26OCT01NYITOR-TOR", "KXNHLTOTAL-26OCT01NYITOR-6"], NHL_OFF)
        self.assertEqual(submitted, [])
        self.assertEqual(placed, [])
        self.assertEqual(ledger.rows, [], "a blocked sport must not write a ledger row")

    def test_nfl_still_trades_live(self):
        submitted, placed, _ = self._run(
            ["KXNFLGAME-26OCT01PITCLE-PIT", "KXNFLTOTAL-26OCT01PITCLE-42",
             "KXNFLSPREAD-26OCT01PITCLE-PIT3"], NHL_OFF)
        self.assertEqual(len(submitted), 3)
        self.assertEqual(len(placed), 3)

    def test_a_mixed_cycle_blocks_only_the_disabled_sport(self):
        submitted, placed, ledger = self._run(
            ["KXNHLGAME-26OCT01NYITOR-TOR", "KXNFLGAME-26OCT01PITCLE-PIT",
             "KXNHLTOTAL-26OCT01NYITOR-6", "KXNFLTOTAL-26OCT01PITCLE-42"], NHL_OFF)
        self.assertEqual(submitted, ["KXNFLGAME-26OCT01PITCLE-PIT",
                                     "KXNFLTOTAL-26OCT01PITCLE-42"])
        self.assertEqual(len(ledger.rows), 2)

    def test_maker_path_is_blocked_too(self):
        # The maker path rests an order on the book instead of crossing; it
        # leaves through the same loop, so the switch must cover it.
        settings = replace(NHL_OFF, maker_mode=True)
        recs = [_Rec("KXNHLTOTAL-26OCT01NYITOR-6")]
        maker = mock.Mock()
        with mock.patch.object(live_mod, "iter_trigger_candidates",
                               return_value=[(recs[0], _bet(recs[0].ticker), None)]), \
             mock.patch.object(live_mod, "submit_order") as sub, \
             mock.patch.object(live_mod.bet_inputs, "save"):
            live_mod.run_live_trigger(recs, {}, _Ledger(), mock.Mock(), settings,
                                      25.0, maker=maker)
        maker.place.assert_not_called()
        sub.assert_not_called()

    def test_clearing_the_switch_re_enables_nhl(self):
        settings = replace(DEFAULTS, live_trade_disabled_sports=())
        submitted, placed, _ = self._run(["KXNHLGAME-26OCT01NYITOR-TOR"], settings)
        self.assertEqual(submitted, ["KXNHLGAME-26OCT01NYITOR-TOR"])

    def test_switch_is_case_insensitive(self):
        settings = replace(DEFAULTS, live_trade_disabled_sports=("NFL",))
        submitted, _, _ = self._run(
            ["KXNFLGAME-26OCT01PITCLE-PIT", "KXNHLGAME-26OCT01NYITOR-TOR"], settings)
        self.assertEqual(submitted, ["KXNHLGAME-26OCT01NYITOR-TOR"])


class LiveTradingDefaults(unittest.TestCase):
    """What production actually does today. Separate from the mechanism tests
    above on purpose: when a sport is switched on or off, exactly this class
    should change and the rest should keep passing untouched."""

    def test_every_sport_trades_live_today(self):
        # NHL came back on 2026-10-06. If this fails, someone barred a sport --
        # check config/settings.py for the evidence they recorded.
        self.assertEqual(DEFAULTS.live_trade_disabled_sports, ())

    def test_the_switch_still_works_even_though_nothing_uses_it(self):
        submitted, _, _ = self._submit_with(NHL_OFF)
        self.assertEqual(submitted, [])
        submitted, _, _ = self._submit_with(DEFAULTS)
        self.assertEqual(submitted, ["KXNHLGAME-26OCT01NYITOR-TOR"])

    def _submit_with(self, settings):
        return LiveSportSwitch._run(self, ["KXNHLGAME-26OCT01NYITOR-TOR"], settings)


class ShadowLedgerSeparation(unittest.TestCase):
    """A barred sport must be SIMULATED, not silently dropped -- and its rows
    must never land in a file that the money reports read."""

    def test_shadow_ledger_is_a_different_file_from_the_paper_ledger(self):
        from engine.paper import PaperLedger, ShadowLedger
        self.assertNotEqual(ShadowLedger().path, PaperLedger().path)
        self.assertEqual(ShadowLedger().path.name, "shadow_ledger.jsonl")

    def test_real_bets_never_reads_the_shadow_ledger(self):
        # output/paper_ledger.jsonl holds REAL hand-placed orders, so
        # load_real_bets counts it as money. The shadow file must not appear
        # anywhere in that module, or simulated bets would inflate P&L.
        import inspect
        from engine import real_bets
        self.assertNotIn("shadow", inspect.getsource(real_bets).lower())

    def test_shadow_and_live_ledgers_dedup_independently(self):
        # A shadow bet must not make the live ledger think that event was bet.
        import tempfile
        from engine.live import LiveLedger
        from engine.paper import ShadowLedger
        with tempfile.TemporaryDirectory() as d:
            shadow = ShadowLedger(Path(d) / "shadow.jsonl")
            live = LiveLedger(Path(d) / "live.jsonl")
            shadow.record({"event_key": "KXNHLGAME-26OCT01NYITOR", "x": 1})
            self.assertTrue(shadow.already_bet("KXNHLGAME-26OCT01NYITOR"))
            self.assertFalse(live.already_bet("KXNHLGAME-26OCT01NYITOR"))


if __name__ == "__main__":
    unittest.main()
