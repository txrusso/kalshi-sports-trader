"""Offline tests for engine/bet_inputs.py: saving each placed bet's model inputs
for the dashboard's Model tab, and deleting them once the bet settles.

The ones that matter most are about the real-order path: the order is placed
and recorded BEFORE anything is saved, and a failing save can never stop or
undo an order.

    .venv\\Scripts\\python.exe -m tests.test_bet_inputs
"""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from engine import bet_inputs                                     # noqa: E402
from engine.execution import OrderResult                          # noqa: E402
from output.model_explainer import explain                        # noqa: E402
from signals.recommendation import Recommendation                 # noqa: E402

NOW = datetime(2026, 9, 24, 3, 0, tzinfo=timezone.utc)


def make_rec(ticker="KXMLBTOTAL-26SEP231840STLPIT-4") -> Recommendation:
    """A real recommendation (STL@PIT Over 3.5 from the 2026-09-23 scan)."""
    return Recommendation(
        ticker=ticker, title="St. Louis vs Pittsburgh Total Runs?", yes_team="Over 3.5",
        side="YES", entry_price=0.87, edge_cents=6.29, confidence=0.4861,
        money_flow_score=0.6237, flow_direction="YES", fair_prob=0.9329,
        fair_source="pregame_totals", suggested_stake_usd=1.41, suggested_contracts=1.62,
        rationale=("Money flow YES (score +0.62, strength 0.56; book +0.69, trades +0.99, "
                   "oi +0.00) | fair P(YES)=93% via pregame_totals"),
        spread_cents=1.0, volume=52000, open_interest=48000,
        headline="St. Louis Cardinals vs Pittsburgh Pirates Over 3.5",
        game_datetime=datetime(2026, 9, 24, 1, 40, tzinfo=timezone.utc),
        debug={"fair_value": {"line": 3.5, "exp_total": 9.52, "starters_known": 2,
                              "park_factor": 0.98, "home_eff_ra9": 4.23, "away_eff_ra9": 5.15},
               "flow_conf": 0.349, "liquidity": 0.917, "calibration_mult": 0.894,
               "money_flow": {"oi": {"available": True, "d_oi": 0.0, "d_price": 0.0}}},
    )


def make_bet(event_key="KXMLBTOTAL-26SEP231840STLPIT", ticker="KXMLBTOTAL-26SEP231840STLPIT-4",
             placed=NOW) -> dict:
    return {"event_key": event_key, "ticker": ticker, "ts": placed.isoformat(),
            "side": "YES", "entry_price": 0.87, "contracts": 1.62, "minutes_before": 20.0,
            "label": "STL vs PIT Over 3.5"}


class Storage(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "bet_inputs.json"

    def tearDown(self):
        self.tmp.cleanup()

    def test_round_trip_explains_the_same_as_the_live_rec(self):
        rec = make_rec()
        bet_inputs.save(make_bet(), rec, self.path)
        entry = bet_inputs.load(self.path)["KXMLBTOTAL-26SEP231840STLPIT"]
        self.assertEqual(entry["ticker"], "KXMLBTOTAL-26SEP231840STLPIT-4")
        live = explain(json.loads(json.dumps(rec.__dict__, default=str)))
        saved = explain(entry["rec"])
        self.assertTrue(saved.repro_ok)
        self.assertEqual((saved.repro_yes, saved.conf_repro, saved.sizing["stake"]),
                         (live.repro_yes, live.conf_repro, live.sizing["stake"]))

    def test_save_never_raises(self):
        # (save() logs a warning for each failure; silence it here)
        warn = mock.patch.object(bet_inputs.log, "warning")
        warn.start()
        self.addCleanup(warn.stop)
        bet_inputs.save(make_bet(), make_rec(), Path(self.tmp.name))     # a directory
        bet_inputs.save({"ticker": "no event key"}, make_rec(), self.path)
        bet_inputs.save(make_bet(), object(), self.path)                 # unserialisable-ish
        self.assertFalse(self.path.exists() and "no event key" in self.path.read_text())

    def test_missing_or_corrupt_file_loads_empty(self):
        self.assertEqual(bet_inputs.load(self.path), {})
        self.path.write_text("{not json", encoding="utf-8")
        self.assertEqual(bet_inputs.load(self.path), {})

    def test_prune_removes_settled_and_stale_only(self):
        bet_inputs.save(make_bet("A", "T-A"), make_rec(), self.path)
        bet_inputs.save(make_bet("B", "T-B"), make_rec(), self.path)
        bet_inputs.save(make_bet("C", "T-C", placed=NOW - timedelta(days=10)), make_rec(), self.path)
        removed = bet_inputs.prune(lambda t: t == "T-A", self.path, now=NOW)
        self.assertEqual(removed, 2)                     # A settled, C too old
        self.assertEqual(set(bet_inputs.load(self.path)), {"B"})

    def test_prune_nothing_to_do_does_not_rewrite(self):
        bet_inputs.save(make_bet("B", "T-B"), make_rec(), self.path)
        before = self.path.stat().st_mtime_ns
        self.assertEqual(bet_inputs.prune(lambda t: False, self.path, now=NOW), 0)
        self.assertEqual(self.path.stat().st_mtime_ns, before)

    def test_default_path_is_resolved_at_call_time(self):
        with mock.patch.object(bet_inputs, "PATH", self.path):
            bet_inputs.save(make_bet(), make_rec())
            self.assertIn("KXMLBTOTAL-26SEP231840STLPIT", bet_inputs.load())


class LiveOrderPath(unittest.TestCase):
    """engine/live.py: record the order first, save inputs second, never block."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "bet_inputs.json"

    def tearDown(self):
        self.tmp.cleanup()

    def _run(self, save_side_effect=None):
        from engine import live
        from config.settings import DEFAULTS
        calls = []
        real_save = bet_inputs.save          # taken BEFORE it is patched below

        class Ledger:
            def record(self, bet):
                calls.append("record")

        def fake_save(bet, rec, path=None):
            calls.append("save")
            if save_side_effect:
                save_side_effect()
            real_save(bet, rec, self.path)

        rec, bet = make_rec(), make_bet()
        with mock.patch.object(live, "iter_trigger_candidates", return_value=[(rec, bet, None)]), \
             mock.patch.object(live, "submit_order",
                               return_value=OrderResult("oid-1", "filled", "coid", {})), \
             mock.patch.object(live.bet_inputs, "save", side_effect=fake_save):
            placed = live.run_live_trigger([rec], {}, Ledger(), client=None,
                                           settings=DEFAULTS, window_minutes=25)
        return placed, calls

    def test_order_recorded_before_inputs_saved(self):
        placed, calls = self._run()
        self.assertEqual(len(placed), 1)
        self.assertEqual(calls, ["record", "save"])
        self.assertIn("KXMLBTOTAL-26SEP231840STLPIT", bet_inputs.load(self.path))

    def test_real_save_failure_cannot_block_the_order(self):
        # Use the REAL save() with a path it cannot write: must not raise.
        from engine import live
        from config.settings import DEFAULTS

        class Ledger:
            recorded = []

            def record(self, bet):
                self.recorded.append(bet)

        rec, bet = make_rec(), make_bet()
        with mock.patch.object(live, "iter_trigger_candidates", return_value=[(rec, bet, None)]), \
             mock.patch.object(live, "submit_order",
                               return_value=OrderResult("oid-2", "filled", "coid", {})), \
             mock.patch.object(bet_inputs, "_write", side_effect=OSError("disk full")), \
             mock.patch.object(bet_inputs.log, "warning"):
            ledger = Ledger()
            placed = live.run_live_trigger([rec], {}, ledger, client=None,
                                           settings=DEFAULTS, window_minutes=25)
        self.assertEqual(len(placed), 1)
        self.assertEqual(len(ledger.recorded), 1)
        self.assertEqual(placed[0]["order_id"], "oid-2")


if __name__ == "__main__":
    unittest.main(verbosity=2)
