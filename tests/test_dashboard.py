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
    GREEN, RED, AccountData, LATEST, calibration_mult, flow_components, fmt_countdown,
    fmt_duration, fmt_start_offset, load_scan, mark_pending, name_cell,
    name_column_width, pending_bets, record_card, record_text, short_source,
    sport_records, _mode_from_args, signed, truncate,
)
from engine.account_summary import _grade          # noqa: E402
from kalshi.normalize import position_size         # noqa: E402

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
        self.assertEqual(signed(1.0).style, GREEN)
        self.assertEqual(signed(-1.0).style, RED)
        self.assertNotEqual(GREEN, RED)
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

class SportSplit(unittest.TestCase):
    """MLB/NFL/NBA/NHL share one account and one ledger but are independently
    validated models at very different stages, so the blended record hides
    which one is actually working."""

    def _bet(self, ticker, side="YES", price=0.5, contracts=2.0):
        return {"ticker": ticker, "side": side, "entry_price": price,
                "contracts": contracts}

    def test_splits_by_ticker_prefix(self):
        bets = [self._bet("KXMLBGAME-26SEP221940MIACHC-MIA"),
                self._bet("KXMLBTOTAL-26SEP222145MINSF-4"),
                self._bet("KXNFLSPREAD-26SEP27NYJDET-DET8")]
        outcomes = {b["ticker"]: True for b in bets}
        got = dict(sport_records(bets, outcomes, _grade))
        self.assertEqual(got["MLB"][:2], (2, 0))
        self.assertEqual(got["NFL"][:2], (1, 0))

    def test_per_sport_sums_to_the_blended_total(self):
        bets = [self._bet("KXMLBGAME-A"), self._bet("KXNFLGAME-B"),
                self._bet("KXNHLGAME-C")]
        outcomes = {"KXMLBGAME-A": True, "KXNFLGAME-B": False, "KXNHLGAME-C": True}
        total = _grade(bets, outcomes)
        parts = sport_records(bets, outcomes, _grade)
        self.assertEqual(sum(r[0] for _, r in parts), total[0])
        self.assertEqual(sum(r[1] for _, r in parts), total[1])
        self.assertAlmostEqual(sum(r[2] for _, r in parts), total[2])

    def test_sports_with_nothing_settled_are_omitted(self):
        bets = [self._bet("KXMLBGAME-A"), self._bet("KXNBAGAME-B")]
        parts = sport_records(bets, {"KXMLBGAME-A": True}, _grade)
        self.assertEqual([name for name, _ in parts], ["MLB"])

    def test_card_shows_total_then_each_sport(self):
        card = record_card((5, 3, 1.25), [("MLB", (4, 2, 1.0)), ("NFL", (1, 1, 0.25))])
        lines = card.plain.splitlines()
        self.assertIn("5-3", lines[0])
        self.assertIn("MLB 4-2", lines[1])
        self.assertIn("NFL 1-1", lines[2])


class UnrealizedPnl(unittest.TestCase):
    """Marked off the POSITION, not the ledger row: the position is what the
    exchange actually holds, so a partial fill is reflected honestly."""

    class FakeClient:
        def __init__(self, markets):
            self.markets = markets
            self.asked = []

        def get_market(self, ticker):
            self.asked.append(ticker)
            return self.markets[ticker]

    @staticmethod
    def _market(bid, ask):
        return {"ticker": "T", "yes_bid_dollars": str(bid), "yes_ask_dollars": str(ask)}

    def _run(self, pending, positions, markets):
        data = AccountData(pending=pending)
        client = self.FakeClient(markets)
        mark_pending(data, client, positions, position_size)
        return data, client

    def test_long_yes_marks_at_the_yes_bid(self):
        pending = [{"ticker": "T", "side": "YES"}]
        positions = [{"ticker": "T", "position_fp": "2.00",
                      "market_exposure_dollars": "1.000000"}]
        data, _ = self._run(pending, positions, {"T": self._market(0.80, 0.85)})
        self.assertAlmostEqual(pending[0]["_mark"], 0.80)
        self.assertAlmostEqual(pending[0]["_unrealized"], 2 * 0.80 - 1.0)
        self.assertAlmostEqual(data.unrealized, 0.6)

    def test_long_no_marks_at_one_minus_the_yes_ask(self):
        """position_fp is negative for a NO position, and a NO contract is only
        worth (1 - yes_ask) -- marking it on the yes side would invert the P&L."""
        pending = [{"ticker": "T", "side": "NO"}]
        positions = [{"ticker": "T", "position_fp": "-1.37",
                      "market_exposure_dollars": "0.616500"}]
        data, _ = self._run(pending, positions, {"T": self._market(0.10, 0.13)})
        self.assertAlmostEqual(pending[0]["_mark"], 0.87)
        self.assertAlmostEqual(pending[0]["_unrealized"], 1.37 * 0.87 - 0.6165, places=6)

    def test_mark_is_never_negative_on_a_wide_or_stale_ask(self):
        pending = [{"ticker": "T", "side": "NO"}]
        positions = [{"ticker": "T", "position_fp": "-1.00",
                      "market_exposure_dollars": "0.500000"}]
        self._run(pending, positions, {"T": self._market(0.99, 1.05)})
        self.assertGreaterEqual(pending[0]["_mark"], 0.0)

    def test_unfilled_order_is_left_unmarked_not_shown_as_a_loss(self):
        """A live row is written at SUBMIT time, so it may still be resting."""
        pending = [{"ticker": "RESTING", "side": "YES"}]
        data, client = self._run(pending, [], {})
        self.assertNotIn("_unrealized", pending[0])
        self.assertIsNone(data.unrealized)
        self.assertEqual(client.asked, [], "no quote should be fetched without a position")

    def test_a_failed_quote_skips_that_row_only(self):
        outer = self

        class Flaky(self.FakeClient):
            def get_market(self, ticker):
                if ticker == "BAD":
                    raise RuntimeError("boom")
                return super().get_market(ticker)

        pending = [{"ticker": "BAD"}, {"ticker": "GOOD"}]
        positions = [{"ticker": "BAD", "position_fp": "1.00",
                      "market_exposure_dollars": "0.500000"},
                     {"ticker": "GOOD", "position_fp": "1.00",
                      "market_exposure_dollars": "0.500000"}]
        data = AccountData(pending=pending)
        mark_pending(data, Flaky({"GOOD": outer._market(0.70, 0.72)}), positions,
                     position_size)
        self.assertNotIn("_unrealized", pending[0])
        self.assertAlmostEqual(pending[1]["_unrealized"], 0.20)
        self.assertAlmostEqual(data.unrealized, 0.20)

    def test_total_sums_only_marked_rows(self):
        pending = [{"ticker": "A"}, {"ticker": "B"}, {"ticker": "RESTING"}]
        positions = [{"ticker": "A", "position_fp": "1.00",
                      "market_exposure_dollars": "0.400000"},
                     {"ticker": "B", "position_fp": "1.00",
                      "market_exposure_dollars": "0.900000"}]
        markets = {"A": self._market(0.60, 0.62), "B": self._market(0.50, 0.55)}
        data, _ = self._run(pending, positions, markets)
        self.assertAlmostEqual(data.unrealized, (0.60 - 0.40) + (0.50 - 0.90))


class ColumnSizing(unittest.TestCase):
    """The bet must never be cut off -- that is the whole point of measuring."""

    def test_widens_to_the_longest_name(self):
        longest = "St. Louis Cardinals vs San Francisco Giants Under 9.5"
        self.assertEqual(name_column_width([longest, "short"], ["KXMLBTOTAL-X"]),
                         len(longest))

    def test_accounts_for_the_ticker_on_the_second_line(self):
        """The ticker sits under the name, so it drives the width too."""
        ticker = "KXNFLSPREAD-26SEP27NYJDET-DET8"          # 30 chars
        self.assertEqual(name_column_width(["tiny"], [ticker]), 34,
                         "below the floor, the floor should win")
        longer = ticker + "-EXTRALONGSUFFIX"
        self.assertEqual(name_column_width(["tiny"], [longer]), len(longer))

    def test_floor_keeps_the_header_readable_when_empty(self):
        self.assertEqual(name_column_width([], []), 34)

    def test_cap_stops_one_freak_row_eating_the_screen(self):
        self.assertEqual(name_column_width(["x" * 500], []), 64)

    def test_cell_stacks_name_over_ticker(self):
        cell = name_cell("Miami Marlins wins vs Chicago Cubs", "KXMLBGAME-X")
        self.assertEqual(cell.plain.splitlines(),
                         ["Miami Marlins wins vs Chicago Cubs", "KXMLBGAME-X"])


class SourceLabel(unittest.TestCase):
    def test_pregame_prefix_is_dropped(self):
        self.assertEqual(short_source("pregame_nfl_totals"), "nfl_totals")
        self.assertEqual(short_source("pregame_elo"), "elo")

    def test_live_is_preserved_because_it_is_the_meaningful_case(self):
        self.assertEqual(short_source("live"), "live")

    def test_missing_source(self):
        self.assertEqual(short_source(""), "flow-only")
        self.assertEqual(short_source(None), "flow-only")

if __name__ == "__main__":
    unittest.main(verbosity=2)
