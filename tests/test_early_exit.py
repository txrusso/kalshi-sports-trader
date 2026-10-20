"""A bet cashed out before settlement (the user selling it back by hand).

The agent never sells to close, so every report assumed every fill on the
account opened a position held to settlement. The first manual cash-out
(2026-10-18, HOU@JAC Under 42.5: bought 2.8 NO @ 0.52, sold back @ 0.05
mid-game) broke that: the exit fill was summed in as a second opening fill,
making the bet 5.6 contracts @ ~0.73 graded on the final score. These are the
real fills Kalshi recorded for it.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from engine import real_bets                                     # noqa: E402
from engine.account_summary import _grade                        # noqa: E402
from engine.real_bets import apply_real_fill, grade              # noqa: E402

TK = "KXNFLTOTAL-26OCT18HOUJAC-43"
REAL_FILLS = [   # newest first, as the API returns them
    {"action": "sell", "book_side": "bid", "count_fp": "1.80", "created_time": "2026-10-18T15:45:37.473432Z",
     "fee_cost": "0.006000", "no_price_dollars": "0.0500", "outcome_side": "yes", "side": "yes",
     "ticker": TK, "yes_price_dollars": "0.9500"},
    {"action": "sell", "book_side": "bid", "count_fp": "1.00", "created_time": "2026-10-18T15:45:37.473432Z",
     "fee_cost": "0.003400", "no_price_dollars": "0.0500", "outcome_side": "yes", "side": "yes",
     "ticker": TK, "yes_price_dollars": "0.9500"},
    {"action": "sell", "book_side": "ask", "count_fp": "2.80", "created_time": "2026-10-18T13:08:41.107481Z",
     "fee_cost": "0.049000", "no_price_dollars": "0.5200", "outcome_side": "no", "side": "no",
     "ticker": TK, "yes_price_dollars": "0.4800"},
]
BET = {"ticker": TK, "side": "NO", "entry_price": 0.52, "contracts": 2.8,
       "first_pitch": "2026-10-18T13:30:00+00:00", "edge_cents": 9.67}


def _client(fills):
    return type("C", (), {"get_all_fills": lambda self_: fills})()


class EarlyExit(unittest.TestCase):
    def setUp(self):
        real_bets.reset_fill_cache()

    def tearDown(self):
        real_bets.reset_fill_cache()

    def _graded_bet(self):
        got = real_bets._ticker_fills(_client(REAL_FILLS))[TK]
        return apply_real_fill(BET, got)

    def test_exit_fill_is_not_counted_as_an_opening(self):
        got = real_bets._ticker_fills(_client(REAL_FILLS))[TK]
        self.assertAlmostEqual(got["contracts"], 2.8)        # not 5.6
        self.assertAlmostEqual(got["price"], 0.52)           # not ~0.735
        self.assertAlmostEqual(got["closed"], 2.8)
        self.assertAlmostEqual(got["exit_price"], 0.05)      # the NO leg, not 0.95
        self.assertAlmostEqual(got["fees"], 0.0584, places=4)  # open + close fees

    def test_fully_cashed_out_bet_is_final_before_the_game_is(self):
        won, pnl = grade(self._graded_bet(), {})             # no outcome yet
        self.assertFalse(won)
        self.assertAlmostEqual(pnl, 2.8 * (0.05 - 0.52), places=4)   # -1.316

    def test_the_final_score_does_not_change_a_cashed_out_bet(self):
        b = self._graded_bet()
        self.assertEqual(grade(b, {TK: True}), grade(b, {TK: False}))

    def test_profitable_cash_out_is_a_win(self):
        b = {**BET, "closed_contracts": 2.8, "exit_price": 0.80}
        won, pnl = grade(b, {})
        self.assertTrue(won)
        self.assertAlmostEqual(pnl, 2.8 * 0.28, places=4)

    def test_partial_cash_out_waits_for_settlement_on_the_rest(self):
        b = {**BET, "closed_contracts": 1.0, "exit_price": 0.30}
        self.assertIsNone(grade(b, {}))
        won, pnl = grade(b, {TK: False})                     # NO wins
        self.assertAlmostEqual(pnl, 1.0 * (0.30 - 0.52) + 1.8 * 0.48, places=4)
        self.assertTrue(won)

    def test_held_bets_grade_exactly_as_before(self):
        won, pnl = grade(BET, {TK: False})
        self.assertTrue(won)
        self.assertAlmostEqual(pnl, 0.48 * 2.8, places=4)
        self.assertIsNone(grade(BET, {}))

    def test_every_report_agrees(self):
        from output.dashboard_stats import graded_rows
        from output.live_dashboard import pending_bets, profit_curves, settled_bets
        b = self._graded_bet()
        w, l, net = _grade([b], {})
        self.assertEqual((w, l), (0, 1))
        self.assertAlmostEqual(net, -1.316, places=3)
        self.assertAlmostEqual(graded_rows([b], {})[0]["profit"], -1.316, places=3)
        self.assertEqual(pending_bets([b], {}), [])
        rows = settled_bets([b], {}, days=36500)
        self.assertAlmostEqual(rows[0]["_net"], -1.316 - 0.0584, places=3)
        self.assertAlmostEqual(profit_curves([b], {})["Overall"][-1][1], -1.316, places=3)


if __name__ == "__main__":
    unittest.main()
