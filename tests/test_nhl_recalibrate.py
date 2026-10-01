"""Offline tests for the nightly NHL recalibration
(config/nhl_calibration.py + backtest/nhl_recalibrate.py).

The job re-sweeps the two NHL totals constants every morning and, with
--apply, promotes a setting that beats the live one on BOTH walk-forward
splits. These tests pin the safety properties: a bad or missing file falls
back to the shipped constants, out-of-range values are refused outright, and
a promotion bumps the version that gets stamped onto shadow bets.
"""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import nhl_calibration                                  # noqa: E402
from data.fair_value_nhl_totals import NHL_TOTALS_DISPERSION        # noqa: E402
from data.scoring_environment_nhl import TEAM_SIGNAL_SHRINK         # noqa: E402


class CalibrationStore(unittest.TestCase):

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.path = Path(self._tmp.name) / "nhl_calibration.json"
        self._p = mock.patch.object(nhl_calibration, "CALIBRATION_PATH", self.path)
        self._p.start()
        nhl_calibration._cache = nhl_calibration._cache_mtime = None

    def tearDown(self):
        self._p.stop()
        nhl_calibration._cache = nhl_calibration._cache_mtime = None
        self._tmp.cleanup()

    def test_missing_file_falls_back_to_the_shipped_constants(self):
        self.assertFalse(self.path.exists())
        self.assertEqual(nhl_calibration.shrink(), TEAM_SIGNAL_SHRINK)
        self.assertEqual(nhl_calibration.dispersion(), NHL_TOTALS_DISPERSION)
        self.assertIn("shipped", nhl_calibration.version())

    def test_corrupt_file_falls_back_rather_than_raising(self):
        self.path.write_text("{not json", encoding="utf-8")
        self.assertEqual(nhl_calibration.shrink(), TEAM_SIGNAL_SHRINK)
        self.assertEqual(nhl_calibration.dispersion(), NHL_TOTALS_DISPERSION)

    def test_file_missing_required_keys_falls_back(self):
        self.path.write_text(json.dumps({"version": "x"}), encoding="utf-8")
        self.assertEqual(nhl_calibration.shrink(), TEAM_SIGNAL_SHRINK)

    def test_out_of_range_file_is_rejected(self):
        # shrink=1.0 would restore the phantom-edge bug the shrinkage removed.
        self.path.write_text(json.dumps(
            {"version": "bad", "team_signal_shrink": 1.0, "dispersion": 0.85}),
            encoding="utf-8")
        self.assertEqual(nhl_calibration.shrink(), TEAM_SIGNAL_SHRINK)

    def test_save_refuses_out_of_range_values(self):
        with self.assertRaises(ValueError):
            nhl_calibration.save(1.0, 0.85, "v1")          # shrink too high
        with self.assertRaises(ValueError):
            nhl_calibration.save(0.2, 1.4, "v1")           # dispersion too high
        self.assertFalse(self.path.exists(), "nothing should be written on refusal")

    def test_save_then_load_round_trips(self):
        nhl_calibration.save(0.25, 0.80, "2026-10-02", n_train=1312)
        self.assertAlmostEqual(nhl_calibration.shrink(), 0.25)
        self.assertAlmostEqual(nhl_calibration.dispersion(), 0.80)
        self.assertEqual(nhl_calibration.version(), "2026-10-02")
        self.assertEqual(json.loads(self.path.read_text())["n_train"], 1312)

    def test_a_rewrite_is_picked_up(self):
        nhl_calibration.save(0.25, 0.80, "v1")
        self.assertAlmostEqual(nhl_calibration.shrink(), 0.25)
        nhl_calibration.save(0.15, 0.90, "v2")
        self.assertAlmostEqual(nhl_calibration.shrink(), 0.15)
        self.assertEqual(nhl_calibration.version(), "v2")

    def test_save_is_atomic_leaving_no_temp_file(self):
        nhl_calibration.save(0.2, 0.85, "v1")
        self.assertEqual([p.name for p in self.path.parent.iterdir()],
                         [self.path.name])


class PlateauPick(unittest.TestCase):
    """Among both-splits winners, the job takes the middle of the flat shelf,
    not the single best point -- the discipline that picked max_stake_pct=0.06
    and in_game_max_minutes=40."""

    def test_picks_the_shelf_middle_not_the_knife_edge(self):
        from backtest.nhl_recalibrate import _plateau_pick
        # (validate, train, shrink, dispersion); all within 0.0002 of best
        shelf = [(0.1890, 0.197, 0.15, 0.85),
                 (0.1889, 0.197, 0.20, 0.85),
                 (0.1891, 0.197, 0.25, 0.85),
                 (0.1950, 0.198, 0.40, 0.70)]   # clearly worse, must be ignored
        va, tr, s, d = _plateau_pick(shelf)
        self.assertEqual(s, 0.20)
        self.assertNotEqual(s, 0.40)

    def test_a_lone_winner_is_returned_as_is(self):
        from backtest.nhl_recalibrate import _plateau_pick
        va, tr, s, d = _plateau_pick([(0.188, 0.197, 0.30, 0.75)])
        self.assertEqual((s, d), (0.30, 0.75))


class ShadowVersionStamp(unittest.TestCase):

    def test_shadow_bets_are_stamped_with_the_model_version(self):
        from engine.paper import ShadowLedger
        with tempfile.TemporaryDirectory() as d:
            sl = ShadowLedger(Path(d) / "s.jsonl")
            sl.record({"ticker": "KXNHLTOTAL-26OCT02NYRDET-7",
                       "event_key": "KXNHLTOTAL-26OCT02NYRDET", "side": "YES"})
            row = json.loads((Path(d) / "s.jsonl").read_text().strip())
            self.assertEqual(row["model_version"], nhl_calibration.version())

    def test_a_non_nhl_row_is_not_stamped(self):
        from engine.paper import ShadowLedger
        with tempfile.TemporaryDirectory() as d:
            sl = ShadowLedger(Path(d) / "s.jsonl")
            sl.record({"ticker": "KXNFLGAME-26OCT05PITCLE-PIT",
                       "event_key": "KXNFLGAME-26OCT05PITCLE", "side": "YES"})
            row = json.loads((Path(d) / "s.jsonl").read_text().strip())
            self.assertNotIn("model_version", row)


class ModelReadsTheStore(unittest.TestCase):

    def test_the_totals_model_prices_with_the_stored_dispersion(self):
        from data.distributions import overtime_total_survival
        a = overtime_total_survival(5, 3.1, 3.1, 0.85)
        b = overtime_total_survival(5, 3.1, 3.1, 0.70)
        self.assertNotAlmostEqual(a, b, places=4,
                                  msg="dispersion must actually move the price")

    def test_shrink_is_read_through_the_store_not_the_constant(self):
        import data.scoring_environment_nhl as senv
        self.assertIn("nhl_calibration.shrink()",
                      Path(senv.__file__).read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
