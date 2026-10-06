"""Offline tests for engine/real_bets.py -- the ledger view every results report
grades: both ledgers, live orders corrected to what actually filled.

    .venv\\Scripts\\python.exe -m tests.test_real_bets
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from engine import real_bets                                   # noqa: E402
from engine.account_summary import _grade                      # noqa: E402
from engine.real_bets import _FILL_CACHE, apply_fill, load_real_bets   # noqa: E402


class FakeClient:
    """order_id -> (status, filled). Counts calls so caching can be checked."""

    def __init__(self, orders):
        self.orders, self.calls = orders, 0

    def get_order(self, oid):
        self.calls += 1
        return {"status": self.orders[oid][0]}

    def get_fills(self, limit=100, order_id=None):
        filled = self.orders[order_id][1]
        return [{"count_fp": str(filled)}] if filled else []


class GoneClient(FakeClient):
    """get_order 404s (order aged out); get_fills still answers."""

    def __init__(self, fills):
        super().__init__({k: ("?", v) for k, v in fills.items()})

    def get_order(self, oid):
        raise RuntimeError("HTTP 404 not_found")


class DeadClient(GoneClient):
    """Neither endpoint answers -- nothing can be concluded."""

    def __init__(self):
        super().__init__({})

    def get_fills(self, limit=100, order_id=None):
        raise RuntimeError("HTTP 500")


def _live(oid, status="resting", contracts=1.71, price=0.43, ticker=None):
    return {"ticker": ticker or f"KXMLBGAME-{oid}-X", "side": "YES", "entry_price": price,
            "contracts": contracts, "wager_usd": round(price * contracts, 2),
            "order_id": oid, "order_status": status, "event_key": f"KXMLBGAME-{oid}"}


def _paper(ticker="KXMLBTOTAL-P-8"):
    return {"ticker": ticker, "side": "NO", "entry_price": 0.6, "contracts": 2.0,
            "wager_usd": 1.2, "event_key": ticker.rsplit("-", 1)[0]}


class _Ledgers:
    """Patch both ledger classes to return fixed rows (no files touched)."""

    def __init__(self, paper, live):
        self.paper, self.live = paper, live

    def __enter__(self):
        p = mock.patch("engine.paper.PaperLedger.load", return_value=list(self.paper))
        l = mock.patch("engine.live.LiveLedger.load", return_value=list(self.live))
        self._ps = [p, l]
        for x in self._ps:
            x.start()
        return self

    def __exit__(self, *exc):
        for x in self._ps:
            x.stop()


class ApplyFill(unittest.TestCase):
    def test_never_filled_is_marked(self):
        out = apply_fill(_live("a"), "canceled", 0.0)
        self.assertTrue(out["_nofill"])

    def test_full_fill_is_unchanged(self):
        b = _live("b")
        out = apply_fill(b, "executed", 1.71)
        self.assertNotIn("_nofill", out)
        self.assertEqual(out["contracts"], b["contracts"])

    def test_partial_fill_is_cut_to_what_filled(self):
        out = apply_fill(_live("c", contracts=2.0, price=0.5), "canceled", 0.8)
        self.assertEqual(out["contracts"], 0.8)
        self.assertEqual(out["requested_contracts"], 2.0)
        self.assertEqual(out["wager_usd"], 0.4)

    def test_still_resting_is_left_as_recorded(self):
        out = apply_fill(_live("d"), "resting", 0.0)
        self.assertNotIn("_nofill", out)
        self.assertEqual(out["contracts"], 1.71)


class LoadRealBets(unittest.TestCase):
    def setUp(self):
        _FILL_CACHE.clear()

    def test_both_ledgers_are_included_and_tagged(self):
        with _Ledgers([_paper()], [_live("e", status="filled")]):
            bets = load_real_bets(client=FakeClient({}))
        self.assertEqual(sorted(b["_src"] for b in bets), ["live", "paper"])

    def test_never_filled_is_dropped_unless_asked_for(self):
        client = FakeClient({"f": ("canceled", 0)})
        with _Ledgers([], [_live("f"), _live("g", status="filled")]):
            self.assertEqual([b["order_id"] for b in load_real_bets(client)], ["g"])
            kept = load_real_bets(client, include_unfilled=True)
        self.assertEqual(len(kept), 2)
        self.assertTrue(next(b for b in kept if b["order_id"] == "f")["_nofill"])

    def test_the_phantom_loss_no_longer_counts(self):
        # A YES bet that lost -- but its order was canceled unfilled.
        rows = [_live("h"), _live("i", status="filled", ticker="KXMLBGAME-i-X")]
        outcomes = {"KXMLBGAME-h-X": False, "KXMLBGAME-i-X": True}
        with _Ledgers([], rows):
            before = _grade(rows, outcomes)
            after = _grade(load_real_bets(FakeClient({"h": ("canceled", 0)})), outcomes)
        self.assertEqual(before[:2], (1, 1))
        self.assertEqual(after[:2], (1, 0))
        self.assertAlmostEqual(after[2] - before[2], 0.43 * 1.71)

    def test_filled_at_submit_needs_no_lookup(self):
        client = FakeClient({})
        with _Ledgers([], [_live("j", status="filled")]):
            load_real_bets(client)
        self.assertEqual(client.calls, 0)

    def test_terminal_orders_are_looked_up_once(self):
        client = FakeClient({"k": ("executed", 1.71)})
        with _Ledgers([], [_live("k")]):
            load_real_bets(client)
            load_real_bets(client)
            load_real_bets(None)             # cache applies with no client at all
        self.assertEqual(client.calls, 1)

    def test_lookup_failure_grades_as_recorded(self):
        class Broken:
            def get_order(self, oid):
                raise RuntimeError("429")
        with _Ledgers([], [_live("m")]), mock.patch.object(real_bets.log, "warning"):
            bets = load_real_bets(Broken())
        self.assertEqual(len(bets), 1)
        self.assertNotIn("_nofill", bets[0])


class AgedOutOrder(unittest.TestCase):
    """Kalshi 404s /portfolio/orders/{id} for orders only days old. That used to
    abort the lookup, and "can't read" was graded as "assume it filled" -- which
    silently resurrected never-filled bets into the record days after the fact.
    Fills outlive the order record, so they decide it."""

    def setUp(self):
        _FILL_CACHE.clear()

    def test_gone_order_with_no_fills_is_a_no_fill(self):
        with _Ledgers([], [_live("gone")]):
            bets = load_real_bets(client=GoneClient({"gone": 0.0}))
        self.assertEqual(bets, [])

    def test_gone_order_is_kept_and_marked_when_asked(self):
        with _Ledgers([], [_live("gone")]):
            bets = load_real_bets(client=GoneClient({"gone": 0.0}),
                                  include_unfilled=True)
        self.assertTrue(bets[0]["_nofill"])

    def test_gone_order_that_did_fill_is_graded_on_its_fills(self):
        with _Ledgers([], [_live("gone", contracts=2.0, price=0.5)]):
            bets = load_real_bets(client=GoneClient({"gone": 2.0}))
        self.assertEqual(bets[0]["contracts"], 2.0)
        self.assertNotIn("_nofill", bets[0])

    def test_gone_order_partial_fill_is_cut_to_what_filled(self):
        with _Ledgers([], [_live("gone", contracts=2.0, price=0.5)]):
            bets = load_real_bets(client=GoneClient({"gone": 0.8}))
        self.assertEqual(bets[0]["contracts"], 0.8)
        self.assertEqual(bets[0]["requested_contracts"], 2.0)

    def test_unreadable_fills_still_grade_as_recorded(self):
        """Only when the FILLS call fails too is the bet genuinely unverifiable;
        it must stay in the report rather than vanish."""
        with _Ledgers([], [_live("gone")]):
            bets = load_real_bets(client=DeadClient())
        self.assertEqual(len(bets), 1)
        self.assertNotIn("_nofill", bets[0])


if __name__ == "__main__":
    unittest.main(verbosity=2)


class RepriceFromRealFills(unittest.TestCase):
    """Grade at the price that actually traded, not the limit we submitted.

    Found 2026-10-06 reconciling the dashboard against the account balance:
    the card read +$1.85 where the account was really +$6.04. A live ledger
    row is written at SUBMIT time with the LIMIT price, and Kalshi routinely
    fills better (one row recorded 0.31 and filled at 0.13), so grading off
    the limit overstated every loss and understated every win.
    """

    def setUp(self):
        real_bets.reset_fill_cache()

    def tearDown(self):
        real_bets.reset_fill_cache()

    def _client(self, fills):
        return type("C", (), {"get_all_fills": lambda self_: fills})()

    def test_yes_fill_reprices_to_the_executed_price(self):
        idx = real_bets._ticker_fills(self._client([
            {"ticker": "T1", "action": "buy", "outcome_side": "yes",
             "count_fp": "4.56", "yes_price_dollars": "0.1300",
             "no_price_dollars": "0.8700", "fee_cost": "0.0362"},
        ]))
        self.assertAlmostEqual(idx["T1"]["price"], 0.13, places=4)
        self.assertAlmostEqual(idx["T1"]["contracts"], 4.56, places=2)

    def test_no_side_fill_is_counted_and_priced_from_the_no_leg(self):
        # THE BUG THIS CLASS EXISTS FOR. Buying NO is submitted as a YES ASK,
        # so Kalshi records it action="sell", outcome_side="no". A filter of
        # action == "buy" drops every NO bet and leaves it at its limit.
        idx = real_bets._ticker_fills(self._client([
            {"ticker": "T2", "action": "sell", "outcome_side": "no",
             "count_fp": "1.53", "yes_price_dollars": "0.5400",
             "no_price_dollars": "0.4600", "fee_cost": "0.0267"},
        ]))
        self.assertIn("T2", idx, "a NO-side fill must not be skipped")
        self.assertAlmostEqual(idx["T2"]["price"], 0.46, places=4)

    def test_several_fills_average_by_size(self):
        idx = real_bets._ticker_fills(self._client([
            {"ticker": "T3", "action": "buy", "outcome_side": "yes", "count_fp": "1.00",
             "yes_price_dollars": "0.2000", "fee_cost": "0.01"},
            {"ticker": "T3", "action": "buy", "outcome_side": "yes", "count_fp": "3.00",
             "yes_price_dollars": "0.4000", "fee_cost": "0.02"},
        ]))
        self.assertAlmostEqual(idx["T3"]["contracts"], 4.0, places=4)
        self.assertAlmostEqual(idx["T3"]["price"], 0.35, places=4)   # size-weighted
        self.assertAlmostEqual(idx["T3"]["fees"], 0.03, places=4)

    def test_apply_real_fill_keeps_what_was_recorded(self):
        row = real_bets.apply_real_fill(
            {"ticker": "T1", "entry_price": 0.31, "contracts": 4.56, "wager_usd": 1.41},
            {"price": 0.13, "contracts": 4.56, "fees": 0.0362})
        self.assertAlmostEqual(row["entry_price"], 0.13)
        self.assertAlmostEqual(row["recorded_entry_price"], 0.31)
        self.assertAlmostEqual(row["wager_usd"], 0.5928, places=4)
        self.assertAlmostEqual(row["fees_usd"], 0.0362, places=6)

    def test_a_better_fill_turns_a_smaller_loss(self):
        from output.dashboard_stats import graded_rows
        bet = {"ticker": "KXMLBGAME-26SEP232040AZCOL-COL", "side": "YES",
               "entry_price": 0.31, "contracts": 4.56, "edge_cents": 5.0}
        at_limit = graded_rows([bet], {bet["ticker"]: False})[0]
        filled = real_bets.apply_real_fill(bet, {"price": 0.13, "contracts": 4.56,
                                                 "fees": 0.0362})
        at_fill = graded_rows([filled], {bet["ticker"]: False})[0]
        self.assertAlmostEqual(at_limit["profit"], -1.4136, places=3)
        self.assertAlmostEqual(at_fill["profit"], -0.5928, places=3)
        self.assertGreater(at_fill["profit"], at_limit["profit"])

    def test_no_client_or_old_client_degrades_to_recorded_prices(self):
        self.assertEqual(real_bets._ticker_fills(None), {})
        real_bets.reset_fill_cache()
        self.assertEqual(real_bets._ticker_fills(object()), {})   # no get_all_fills

    def test_a_failing_fetch_is_not_fatal(self):
        class Boom:
            def get_all_fills(self_):
                raise RuntimeError("HTTP 500")
        self.assertEqual(real_bets._ticker_fills(Boom()), {})

    def test_cache_expires_so_a_long_run_sees_new_fills(self):
        calls = []

        class C:
            def get_all_fills(self_):
                calls.append(1)
                return [{"ticker": "T", "outcome_side": "yes", "count_fp": "1",
                         "yes_price_dollars": "0.50", "fee_cost": "0"}]
        real_bets._ticker_fills(C())
        real_bets._ticker_fills(C())
        self.assertEqual(len(calls), 1, "second call within the TTL should be cached")
        real_bets._TICKER_FILLS_AT -= real_bets._TICKER_FILLS_TTL + 1
        real_bets._ticker_fills(C())
        self.assertEqual(len(calls), 2, "a stale cache must refetch")
