"""Offline tests for the live TUI dashboard's data layer (run_dashboard.py).

Everything here is pure-function: no network, no Kalshi client, no terminal.
The rendering layer is exercised separately by Textual's own headless driver;
what matters for correctness is that the numbers the dashboard SHOWS are the
numbers the loop WROTE, so that is what these cover.

    .venv\\Scripts\\python.exe -m tests.test_dashboard
"""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from run_dashboard import (  # noqa: E402
    LATEST, calibration_mult, flow_components, fmt_countdown, fmt_duration,
    fmt_start_offset, load_scan, pending_bets, record_text, _mode_from_args,
    signed, truncate,
)

# A real row out of output/recommendations_latest.json, trimmed to the fields
# the dashboard reads. The rationale string is verbatim from the loop.
REAL_REC = {
    "ticker": "KXMLBGAME-26SEP221940CWSKC-CWS",
    "headline": "Kansas City Royals wins vs Chicago White Sox",
    "side": "NO",
    "entry_price": 0.01,
    "edge_cents": 51.2,
    "confidence": 0.5908,
    "money_flow_score": 0.05,
    "flow_direction": "FLAT",
    "fair_prob": 0.478,
    "fair_source": "live",
    "suggested_stake_usd": 1.19,
    "suggested_contracts": 118.62,
    "rationale": ("Money flow FLAT (score +0.05, strength 0.27; book +1.00, "
                  "trades -1.00, oi +0.00) | fair P(YES)=48% via live | edge "
                  "+51.2c on Kansas City Royals wins vs Chicago White Sox @ "
                  "0.01 | calibration 0.95x from past winner bets"),
    "debug": {"calibration_mult": 0.954},
}


class ScanParsing(unittest.TestCase):
    def _write(self, payload) -> Path:
        tmp = Path(tempfile.mkdtemp()) / "recommendations_latest.json"
        tmp.write_text(json.dumps(payload, default=str), encoding="utf-8")
        return tmp

    def test_reads_a_real_payload(self):
        path = self._write({
            "generated_at": "2026-09-22T22:22:46.240867-04:00",
            "scanned": 1155, "deep_scanned": 1152, "recommendations": [REAL_REC],
        })
        scan = load_scan(path)
        self.assertIsNotNone(scan)
        self.assertEqual(scan.scanned, 1155)
        self.assertEqual(scan.deep_scanned, 1152)
        self.assertEqual(len(scan.recs), 1)
        self.assertEqual(scan.generated_at.year, 2026)
        self.assertIsNotNone(scan.generated_at.tzinfo)

    def test_partial_write_is_not_an_error(self):
        """The loop rewrites this file while we may be reading it. A truncated
        read must mean 'try again next poll', never a crash or a blank screen."""
        tmp = Path(tempfile.mkdtemp()) / "half.json"
        tmp.write_text('{"scanned": 10, "recomm', encoding="utf-8")
        self.assertIsNone(load_scan(tmp))

    def test_missing_file_is_not_an_error(self):
        self.assertIsNone(load_scan(Path(tempfile.mkdtemp()) / "nope.json"))

    def test_empty_cycle_parses(self):
        path = self._write({"generated_at": None, "scanned": 0,
                            "deep_scanned": 0, "recommendations": []})
        scan = load_scan(path)
        self.assertEqual(scan.recs, [])
        self.assertIsNone(scan.generated_at)

    def test_production_file_parses_if_present(self):
        """Guard against the loop's own payload shape drifting away from us."""
        if not LATEST.exists():
            self.skipTest("no scan file yet")
        scan = load_scan(LATEST)
        self.assertIsNotNone(scan, "the live recommendations file no longer parses")


class FlowBreakdown(unittest.TestCase):
    def test_book_trades_oi_come_out_of_the_rationale(self):
        self.assertEqual(flow_components(REAL_REC), (1.0, -1.0, 0.0))

    def test_negative_and_fractional_values(self):
        rec = {"rationale": "Money flow NO (score -0.45, strength 0.6; "
                            "book -0.26, trades -0.78, oi -0.30) | rest"}
        self.assertEqual(flow_components(rec), (-0.26, -0.78, -0.30))

    def test_missing_rationale_degrades_to_none(self):
        self.assertEqual(flow_components({}), (None, None, None))
        self.assertEqual(flow_components({"rationale": "flow-only"}),
                         (None, None, None))

    def test_calibration_prefers_debug_precision(self):
        # debug has 0.954; the rationale string only has the rounded 0.95
        self.assertAlmostEqual(calibration_mult(REAL_REC), 0.954)

    def test_calibration_falls_back_to_the_rationale(self):
        rec = {k: v for k, v in REAL_REC.items() if k != "debug"}
        self.assertAlmostEqual(calibration_mult(rec), 0.95)

    def test_calibration_absent(self):
        self.assertIsNone(calibration_mult({"rationale": "no calibration here"}))


class ModeDetection(unittest.TestCase):
    """The mode card must never under-state what the loop is doing -- showing
    PAPER while real orders are being submitted would be the worst possible
    error in this panel."""

    def test_live_in_game(self):
        self.assertEqual(
            _mode_from_args("python -u cli.py loop --interval 600 --live --in-game"),
            "LIVE + IN-GAME")

    def test_live_in_game_maker(self):
        self.assertEqual(_mode_from_args("cli.py loop --live --in-game --maker"),
                         "LIVE + IN-GAME + MAKER")

    def test_plain_live(self):
        self.assertEqual(_mode_from_args("cli.py loop --live"), "LIVE")

    def test_paper(self):
        self.assertEqual(_mode_from_args("cli.py loop --paper"), "PAPER")

    def test_bare_loop_is_recommend_only(self):
        self.assertEqual(_mode_from_args("cli.py loop --interval 600"),
                         "RECOMMEND-ONLY")


class PendingBets(unittest.TestCase):
    def _bet(self, ticker, hours_ago, **extra):
        start = datetime.now(timezone.utc) - timedelta(hours=hours_ago)
        return {"ticker": ticker, "first_pitch": start.isoformat(),
                "side": "YES", "entry_price": 0.5, "contracts": 2.0, **extra}

    def test_settled_bets_are_excluded(self):
        bets = [self._bet("A", 1), self._bet("B", 1)]
        out = pending_bets(bets, {"A": True})
        self.assertEqual([b["ticker"] for b in out], ["B"])

    def test_unsettleable_old_rows_are_cut_off(self):
        """NFL preseason rows can never settle (nflverse excludes preseason), so
        without the 36h cutoff they would pile up in this panel forever."""
        bets = [self._bet("OLD", 100), self._bet("NEW", 2)]
        out = pending_bets(bets, {})
        self.assertEqual([b["ticker"] for b in out], ["NEW"])

    def test_future_games_are_included(self):
        bets = [self._bet("SOON", -3)]        # starts in 3 hours
        self.assertEqual(len(pending_bets(bets, {})), 1)

    def test_sorted_most_recent_start_first(self):
        bets = [self._bet("OLDER", 5), self._bet("NEWER", 1)]
        self.assertEqual([b["ticker"] for b in pending_bets(bets, {})],
                         ["NEWER", "OLDER"])

    def test_unparseable_start_is_dropped_not_crashed(self):
        self.assertEqual(pending_bets([{"ticker": "X", "first_pitch": "garbage"}], {}), [])
        self.assertEqual(pending_bets([{"ticker": "X"}], {}), [])


class Formatting(unittest.TestCase):
    def test_sign_colors(self):
        self.assertEqual(signed(1.0).style, "bold green")
        self.assertEqual(signed(-1.0).style, "bold red")
        self.assertEqual(signed(None).plain, "--")

    def test_edge_format_matches_the_console(self):
        self.assertEqual(signed(51.2, "{:+.1f}c").plain, "+51.2c")
        self.assertEqual(signed(-8.0, "{:+.1f}c").plain, "-8.0c")

    def test_duration(self):
        self.assertEqual(fmt_duration(0), "0:00")
        self.assertEqual(fmt_duration(65), "1:05")
        self.assertEqual(fmt_duration(599), "9:59")
        self.assertEqual(fmt_duration(3700), "1h01m")
        self.assertEqual(fmt_duration(-5), "0:00")

    def test_countdown_past_due_reads_as_scanning(self):
        past = datetime.now(timezone.utc) - timedelta(seconds=5)
        self.assertEqual(fmt_countdown(past)[0], "scanning...")
        self.assertEqual(fmt_countdown(None)[0], "--")

    def test_countdown_future(self):
        future = datetime.now(timezone.utc) + timedelta(seconds=300)
        text, _ = fmt_countdown(future)
        self.assertTrue(text.startswith("4:5") or text.startswith("5:00"), text)

    def test_started_games_read_as_live(self):
        started = datetime.now(timezone.utc) - timedelta(minutes=12)
        self.assertTrue(fmt_start_offset(started).plain.startswith("live"))
        upcoming = datetime.now(timezone.utc) + timedelta(minutes=12)
        self.assertTrue(fmt_start_offset(upcoming).plain.startswith("in"))

    def test_record(self):
        self.assertEqual(record_text((0, 0, 0.0)).plain, "no settled bets")
        self.assertIn("95-70", record_text((95, 70, 3.2)).plain)
        self.assertIn("+3.20", record_text((95, 70, 3.2)).plain)

    def test_truncate_keeps_short_strings_intact(self):
        self.assertEqual(truncate("short", 10), "short")
        self.assertEqual(len(truncate("x" * 50, 10)), 10)
        self.assertEqual(truncate(None, 5), "")


if __name__ == "__main__":
    unittest.main(verbosity=2)
