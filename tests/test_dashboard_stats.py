"""Offline tests for output/dashboard_stats.py -- the numbers behind the
dashboard's DRAWDOWN / TODAY / OVERALL / LOOP HEALTH cards and the Rec Table
and Cal Table tabs.

    .venv\\Scripts\\python.exe -m tests.test_dashboard_stats
"""
from __future__ import annotations

import json
import sys
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from engine.account_summary import _grade                        # noqa: E402
from output.dashboard_stats import (untracked_pnl,                              # noqa: E402
    alert_status, calibration_table, drawdown, graded_rows, log_health,
    parse_calibration_log, parse_tasks, rec_table, resting_orders, sim_table, tally, today_risk,
)

UTC = timezone.utc
T0 = datetime(2026, 9, 1, 23, 0, tzinfo=UTC)


def bet(ticker, side="YES", price=0.40, contracts=2.0, edge=6.0, fair=None, hours=0, **kw):
    if fair is None:                       # consistent with the edge by default
        p_side = price + edge / 100.0
        fair = p_side if side == "YES" else 1.0 - p_side
    return {"ticker": ticker, "side": side, "entry_price": price, "contracts": contracts,
            "wager_usd": round(price * contracts, 2), "edge_cents": edge, "fair_prob": fair,
            "first_pitch": (T0 + timedelta(hours=hours)).isoformat(),
            "ts": (T0 + timedelta(hours=hours) - timedelta(minutes=20)).isoformat(), **kw}


BETS = [bet("KXMLBGAME-A-X", hours=0), bet("KXMLBTOTAL-B-8", "NO", 0.6, 1.0, hours=1),
        bet("KXNFLSPREAD-C-X3", hours=2), bet("KXNFLTOTAL-D-40", hours=3),
        bet("KXMLBGAME-E-X", hours=4)]
OUTCOMES = {"KXMLBGAME-A-X": True, "KXMLBTOTAL-B-8": True, "KXNFLSPREAD-C-X3": False,
            "KXNFLTOTAL-D-40": True, "KXMLBGAME-E-X": False}


class Graded(unittest.TestCase):
    def test_profit_matches_shared_grading_and_nofill_is_skipped(self):
        rows = graded_rows(BETS + [bet("KXMLBGAME-Z-X", _nofill=True)],
                           {**OUTCOMES, "KXMLBGAME-Z-X": True})
        self.assertEqual(len(rows), 5)
        self.assertAlmostEqual(sum(r["profit"] for r in rows), _grade(BETS, OUTCOMES)[2])

    def test_expected_is_edge_times_contracts(self):
        r = graded_rows([bet("KXMLBGAME-A-X", edge=8.0, contracts=2.5)],
                        {"KXMLBGAME-A-X": True})[0]
        self.assertAlmostEqual(r["expected"], 0.08 * 2.5)

    def test_p_side_flips_for_no(self):
        r = graded_rows([bet("KXMLBTOTAL-B-8", "NO", fair=0.3)], {"KXMLBTOTAL-B-8": False})[0]
        self.assertAlmostEqual(r["p_side"], 0.7)
        self.assertTrue(r["won"])


class RecTable(unittest.TestCase):
    def test_structure_and_sums(self):
        rows = rec_table(graded_rows(BETS, OUTCOMES))
        labels = [(lvl, lab) for lvl, lab, _ in rows]
        self.assertEqual(labels, [("all", "ALL SPORTS"),
                                  ("sport", "MLB"), ("kind", "winner"), ("kind", "total"),
                                  ("sport", "NFL"), ("kind", "total"), ("kind", "spread")])
        by = {(lvl, lab): t for lvl, lab, t in rows}
        total = by[("all", "ALL SPORTS")]
        self.assertEqual((total.wins, total.losses), (2, 3))   # B is a NO bet on a YES result
        self.assertAlmostEqual(total.profit, by[("sport", "MLB")].profit + by[("sport", "NFL")].profit)
        self.assertAlmostEqual(total.net, total.profit - total.fees)
        self.assertAlmostEqual(total.vs_expected, total.profit - total.expected)
        self.assertAlmostEqual(total.roi, total.net / total.staked)

    def test_sim_table_relabels_total_and_keeps_breakdown(self):
        g = graded_rows(BETS, OUTCOMES)
        real, sim = rec_table(g), sim_table(g)
        self.assertEqual(sim[0][:2], ("sim", "SIMULATED"))
        self.assertNotIn("all", [lvl for lvl, _, _ in sim])
        self.assertEqual([r[:2] for r in sim[1:]], [r[:2] for r in real[1:]])
        self.assertEqual(sim[0][2].profit, real[0][2].profit)

    def test_sim_table_empty(self):
        self.assertEqual(sim_table([]), [])

    def test_empty(self):
        rows = rec_table([])
        self.assertEqual(len(rows), 1)
        self.assertIsNone(rows[0][2].roi)
        self.assertIsNone(rows[0][2].vs_expected)


class DrawdownCard(unittest.TestCase):
    def _curve(self, values):
        return [(T0 + timedelta(hours=i), v) for i, v in enumerate(values)]

    def test_peak_now_down(self):
        dd = drawdown(self._curve([1.0, 5.0, 16.0, 9.0, 12.0, 6.75]))
        self.assertEqual((dd.peak, dd.now), (16.0, 6.75))
        self.assertAlmostEqual(dd.down, -9.25)
        self.assertEqual(dd.trough, 6.75)
        self.assertEqual(dd.peak_at, T0 + timedelta(hours=2))
        self.assertFalse(dd.at_peak)

    def test_at_peak(self):
        self.assertTrue(drawdown(self._curve([1.0, 2.0, 3.0])).at_peak)

    def test_never_positive_measures_from_zero(self):
        dd = drawdown(self._curve([-1.0, -3.0, -2.0]))
        self.assertEqual(dd.peak, 0.0)
        self.assertIsNone(dd.peak_at)
        self.assertAlmostEqual(dd.down, -2.0)

    def test_empty(self):
        self.assertIsNone(drawdown([]))


class TodayRiskCard(unittest.TestCase):
    def test_staked_and_open(self):
        bets = [bet("A-1"), bet("A-2", price=0.5, contracts=3.0), bet("B-1"),
                bet("A-3", _nofill=True)]
        outcomes = {"A-1": True}                       # A-2 still open
        game_date = lambda t: "2026-09-23" if t.startswith("A") else "2026-09-22"
        r = today_risk(bets, outcomes, "2026-09-23", game_date)
        self.assertEqual((r.n_bets, r.n_open), (2, 1))
        self.assertAlmostEqual(r.staked, 0.80 + 1.50)
        self.assertAlmostEqual(r.open_risk, 1.50)


class Calibration(unittest.TestCase):
    def test_buckets(self):
        rows = graded_rows([bet("KXMLBGAME-A-X", fair=0.55), bet("KXMLBGAME-B-X", fair=0.58),
                            bet("KXMLBGAME-C-X", fair=0.72)],
                           {"KXMLBGAME-A-X": True, "KXMLBGAME-B-X": False, "KXMLBGAME-C-X": True})
        table = {b["bucket"]: b for b in calibration_table(rows)}
        self.assertEqual(set(table), {"50%-60%", "70%-80%"})
        self.assertEqual(table["50%-60%"]["n"], 2)
        self.assertAlmostEqual(table["50%-60%"]["predicted"], 0.565)
        self.assertAlmostEqual(table["50%-60%"]["actual"], 0.5)
        self.assertAlmostEqual(table["50%-60%"]["diff"], -0.065)

    def test_parse_log_takes_the_last_line(self):
        text = ("INFO signals.calibration: Calibration built: {'mlb:total': {'n': 1, 'multiplier': 1.0}}\n"
                "noise\n"
                "INFO signals.calibration: Calibration built: {'mlb:total': {'n': 376, "
                "'win_rate': 0.718, 'expected_wr': 0.808, 'multiplier': 0.894}}\n")
        cal = parse_calibration_log(text)
        self.assertEqual(cal["mlb:total"]["n"], 376)
        self.assertEqual(parse_calibration_log("nothing here"), {})
        self.assertEqual(parse_calibration_log("Calibration built: {not python"), {})


class LoopHealth(unittest.TestCase):
    LOG = "\n".join([
        "===== supervisor 2026-09-22 10:00:01 start; stop at 2026-09-22 23:00 =====",
        "ERROR engine.loop: yesterday's error must NOT count",
        "INFO engine.loop: Cycle 90 done in 250.0s; next in 350s",
        "===== supervisor 2026-09-23 11:29:07 start; stop at 2026-09-23 23:00 =====",
        "INFO engine.loop: Cycle 1 done in 280.1s; next in 320s",
        "WARNING engine.real_bets: Fill lookup failed",
        "===== supervisor 2026-09-23 11:39:10 loop exited (code 1) BEFORE the daily stop -- restart 1/50 in 30s =====",
        "ERROR kalshi.client: HTTP 429 Too Many Requests",
        "INFO engine.loop: Cycle 2 done in 291.7s; next in 309s",
    ])

    def test_counts_only_the_current_session(self):
        h = log_health(self.LOG)
        self.assertEqual(h.session_start.strftime("%Y-%m-%d %H:%M"), "2026-09-23 11:29")
        self.assertEqual((h.restarts, h.errors, h.warnings, h.rate_limits), (1, 1, 1, 1))
        self.assertEqual((h.cycles, h.last_cycle_secs), (2, 291.7))
        self.assertIn("429", h.last_error)

    def test_empty_log(self):
        h = log_health("")
        self.assertIsNone(h.session_start)
        self.assertEqual((h.restarts, h.errors), (0, 0))


class Alerts(unittest.TestCase):
    def test_last_per_channel_and_today_counts(self):
        rows = [
            {"ts_utc": "2026-09-22T20:00:00+00:00", "ts_local": "2026-09-22T16:00:00-04:00",
             "channel": "sms", "accepted": False},
            {"ts_utc": "2026-09-23T16:45:12+00:00", "ts_local": "2026-09-23T12:45:12-04:00",
             "channel": "sms", "accepted": True},
            {"ts_utc": "2026-09-23T17:00:00+00:00", "ts_local": "2026-09-23T13:00:00-04:00",
             "channel": "push", "accepted": False},
        ]
        text = "\n".join(json.dumps(r) for r in rows) + "\nnot json\n"
        st = alert_status(text, "2026-09-23")
        self.assertTrue(st["sms"].last_ok)
        self.assertEqual((st["sms"].sent_today, st["sms"].failures_today), (1, 0))
        self.assertFalse(st["push"].last_ok)
        self.assertEqual(st["push"].failures_today, 1)


class Tasks(unittest.TestCase):
    def test_parse_real_probe_output(self):
        lines = [
            "KalshiPaperLoop|2026-09-23T11:29:04.0000000-04:00|267009|2026-09-24T10:00:00.0000000-04:00",
            "KalshiLoopStop|1999-11-30T00:00:00.0000000-05:00|267011|2026-09-23T23:00:00.0000000-04:00",
            "KalshiSettle|2026-09-22T23:00:01.0000000-04:00|1|2026-09-23T23:00:00.0000000-04:00",
            "KalshiSnapshotCompress|||",
            "garbage",
        ]
        tasks = {t.name: t for t in parse_tasks(lines)}
        self.assertEqual(len(tasks), 4)
        self.assertTrue(tasks["KalshiPaperLoop"].ok)           # running
        self.assertTrue(tasks["KalshiLoopStop"].ok)            # never ran yet
        self.assertIsNone(tasks["KalshiLoopStop"].last_run)    # 1999 sentinel
        self.assertFalse(tasks["KalshiSettle"].ok)             # exit code 1
        self.assertIsNone(tasks["KalshiSnapshotCompress"].result)


class Resting(unittest.TestCase):
    def test_count_and_oldest(self):
        now = datetime(2026, 9, 23, 18, 0, tzinfo=UTC)
        orders = [{"created_time": "2026-09-23T17:50:00Z"}, {"created_time": "2026-09-23T17:15:00Z"}]
        r = resting_orders(orders, now)
        self.assertEqual(r.count, 2)
        self.assertAlmostEqual(r.oldest_age_min, 45.0)
        self.assertEqual(resting_orders([], now).count, 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)


class UntrackedReconciliation(unittest.TestCase):
    """Realized money on the account that no ledger row covers, so the OVERALL
    card can be checked straight against the balance.

    Computed from Kalshi's settlements every refresh rather than being a
    constant: a hardcoded offset would be silently wrong the first time
    another untracked trade or scalar refund landed.
    """

    @staticmethod
    def s(ticker, revenue_cents, yes_cost=0.0, no_cost=0.0, fee=0.0):
        return {"ticker": ticker, "revenue": revenue_cents,
                "yes_total_cost_dollars": f"{yes_cost:.6f}",
                "no_total_cost_dollars": f"{no_cost:.6f}",
                "fee_cost": f"{fee:.6f}"}

    def test_known_tickers_are_excluded(self):
        u = untracked_pnl([self.s("A", 100, yes_cost=0.40)], {"A"})
        self.assertEqual((u.n, u.net), (0, 0.0))

    def test_an_unknown_winner_adds_its_realized_net(self):
        # paid $0.40, market paid $1.00, 2c fee
        u = untracked_pnl([self.s("B", 100, yes_cost=0.40, fee=0.02)], set())
        self.assertEqual(u.n, 1)
        self.assertAlmostEqual(u.net, 0.58, places=4)

    def test_an_unknown_loser_subtracts(self):
        u = untracked_pnl([self.s("C", 0, yes_cost=0.75, fee=0.01)], set())
        self.assertAlmostEqual(u.net, -0.76, places=4)

    def test_revenue_is_cents_and_costs_are_dollars(self):
        # The factor-of-100 trap: revenue 254 is $2.54, not $254.
        u = untracked_pnl([self.s("D", 254, yes_cost=0.762)], set())
        self.assertAlmostEqual(u.net, 2.54 - 0.762, places=4)

    def test_no_side_costs_are_counted(self):
        u = untracked_pnl([self.s("E", 153, no_cost=0.7038, fee=0.0267)], set())
        self.assertAlmostEqual(u.net, 1.53 - 0.7038 - 0.0267, places=4)

    def test_malformed_rows_are_skipped_not_fatal(self):
        rows = [self.s("F", 100, yes_cost=0.40),
                {"ticker": "G", "revenue": "banana"},
                {"revenue": 100},                      # no ticker
                None if False else {"ticker": "H", "revenue": 100,
                                    "yes_total_cost_dollars": None,
                                    "no_total_cost_dollars": None,
                                    "fee_cost": None}]
        u = untracked_pnl(rows, set())
        self.assertEqual(u.n, 2)                       # F and H; G and the no-ticker row skipped
        self.assertAlmostEqual(u.net, 0.60 + 1.00, places=4)

    def test_empty_or_none_is_zero(self):
        self.assertEqual(untracked_pnl([], set()).net, 0.0)
        self.assertEqual(untracked_pnl(None, set()).n, 0)

    def test_card_net_is_tracked_plus_untracked(self):
        from output.dashboard_stats import Tally, Untracked
        from output.live_dashboard import totals_lines
        t = Tally(); t.n = 3; t.profit = 12.85; t.fees = 6.19
        line = totals_lines(t, Untracked(n=10, net=-0.63)).plain
        self.assertIn("net +6.03", line)               # 12.85 - 6.19 - 0.63
        self.assertIn("untracked -0.63", line)
        self.assertIn("(10)", line)

    def test_no_untracked_money_leaves_the_line_alone(self):
        from output.dashboard_stats import Tally
        from output.live_dashboard import totals_lines
        t = Tally(); t.n = 3; t.profit = 12.85; t.fees = 6.19
        line = totals_lines(t, None).plain
        self.assertIn("net +6.66", line)
        self.assertNotIn("untracked", line)
