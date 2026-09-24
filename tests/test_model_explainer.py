"""Offline tests for output/model_explainer.py -- the dashboard's Model tab.

The explainer re-derives each recommendation's fair value, confidence, gates
and Kelly stake from the inputs the loop saved. The tests that matter most
check those rebuilds against the loop's own recorded numbers -- on fixtures
taken from real recommendations, and on the live scan file when present.

    .venv\\Scripts\\python.exe -m tests.test_model_explainer
"""
from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config.settings import DEFAULTS                             # noqa: E402
from output.dashboard_stats import tally                          # noqa: E402
from output.model_explainer import explain, model_name, parse_flow, segment_rows  # noqa: E402

LATEST = Path(__file__).resolve().parents[1] / "output" / "recommendations_latest.json"


def rec(ticker, source, fv, fair, side="YES", price=0.45, edge=None, conf=0.5,
        flow_conf=0.3, liq=0.8, cal=1.0, conflict=False, stake=1.0, contracts=2.0,
        spread=1.0, rationale=None):
    if fair is None:
        edge_val = edge
    else:
        p_side = fair if side == "YES" else 1 - fair
        edge_val = round((p_side - price) * 100, 2) if edge is None else edge
    return {
        "ticker": ticker, "side": side, "entry_price": price, "fair_prob": fair,
        "fair_source": source, "confidence": conf, "conflict": conflict,
        "edge_cents": edge_val,
        "suggested_stake_usd": stake, "suggested_contracts": contracts,
        "spread_cents": spread, "flow_direction": side,
        "rationale": rationale or ("Money flow YES (score +0.40, strength 0.60; book +0.50, "
                                   "trades +0.40, oi +0.00) | fair ..."),
        "debug": {"fair_value": fv, "flow_conf": flow_conf, "liquidity": liq,
                  "calibration_mult": cal, "money_flow": {"oi": {"available": False}}},
    }


# A real MLB winner fair-value detail (LAA @ SEA, 2026-09-23 scan).
LOG5 = {"yes_win_pct": 0.382, "opp_win_pct": 0.465, "neutral": 0.4156, "log5_prob": 0.3773,
        "elo_prob": 0.4072, "elo_yes_rating": 1441.0, "elo_opp_rating": 1478.4,
        "yes_is_home": False, "yes_name": "Los Angeles Angels", "opp_name": "Seattle Mariners"}


class FairValueRebuild(unittest.TestCase):
    def test_mlb_log5_elo_blend(self):
        ex = explain(rec("KXMLBGAME-26SEP232140LAASEA-LAA", "pregame_log5", LOG5, 0.3833))
        self.assertAlmostEqual(ex.repro_yes, 0.8 * 0.3773 + 0.2 * 0.4072, places=6)
        self.assertTrue(ex.repro_ok)
        self.assertAlmostEqual(ex.fv_conf, round(0.35 + min(abs(0.382 - 0.465) * 1.5, 0.25), 3))
        labels = {d.label for d in ex.drivers}
        self.assertEqual(labels, {"Elo blend", "Team records (log5)", "Elo rating gap", "Home field"})

    def test_home_field_helps_the_home_side(self):
        # YES team is AWAY here, so removing home field must RAISE its probability:
        # home field's contribution to a YES bet is negative, to a NO bet positive.
        yes = explain(rec("KXMLBGAME-X-LAA", "pregame_log5", LOG5, 0.3833, side="YES", price=0.35))
        no = explain(rec("KXMLBGAME-X-LAA", "pregame_log5", LOG5, 0.3833, side="NO", price=0.55))
        hf_yes = next(d.cents for d in yes.drivers if d.label == "Home field")
        hf_no = next(d.cents for d in no.drivers if d.label == "Home field")
        self.assertLess(hf_yes, 0)
        self.assertAlmostEqual(hf_yes, -hf_no, places=6)

    def test_live_feed(self):
        fv = {"home_win_prob": 0.522, "yes_is_home": False, "state": "In Progress"}
        ex = explain(rec("KXMLBGAME-X-KC", "live", fv, 0.478))
        self.assertAlmostEqual(ex.repro_yes, 0.478)
        self.assertEqual(ex.fv_conf, 0.80)
        self.assertTrue(ex.in_game)
        self.assertEqual(ex.drivers, [])

    def test_mlb_totals_and_park_driver(self):
        # Real STL@PIT Over 3.5 detail; the loop recorded P(over) = 0.9329
        fv = {"line": 3.5, "exp_total": 9.52, "starters_known": 2, "park_factor": 0.98,
              "home_eff_ra9": 4.23, "away_eff_ra9": 5.15}
        ex = explain(rec("KXMLBTOTAL-26SEP231840STLPIT-4", "pregame_totals", fv, 0.9329, price=0.87))
        self.assertTrue(ex.repro_ok, ex.repro_yes)
        self.assertEqual(ex.fv_conf, 0.50)
        self.assertIn("Park factor", {d.label for d in ex.drivers})
        self.assertEqual(len(ex.sensitivities), 2)

    def test_nfl_spread_with_key_number(self):
        from data.distributions import normal_sf
        from data.fair_value_nfl_spread import KEY_NUMBER_BUMP, MARGIN_SIGMA
        fv = {"line": 2.5, "exp_margin": 1.0, "yes_is_home": True}
        fair = min(1.0, normal_sf(2.5, 1.0, MARGIN_SIGMA) + KEY_NUMBER_BUMP[2.5])
        ex = explain(rec("KXNFLSPREAD-X-KC3", "pregame_nfl_spread", fv, round(fair, 4)))
        self.assertTrue(ex.repro_ok)
        self.assertIn("Key-number bump", {d.label for d in ex.drivers})

    def test_nfl_elo_winner(self):
        from signals.elo_nfl import win_prob
        fv = {"yes_elo": 1532.8, "opp_elo": 1379.6, "yes_is_home": True}
        ex = explain(rec("KXNFLGAME-X-CHI", "pregame_elo", fv, round(win_prob(1532.8, 1379.6, True), 4)))
        self.assertTrue(ex.repro_ok)
        # 0.35 + 153.2/800 = 0.5415, which rounds either way depending on float representation
        self.assertAlmostEqual(ex.fv_conf, 0.35 + min(153.2 / 800, 0.25), delta=0.0006)

    def test_every_live_recommendation_rebuilds(self):
        if not LATEST.exists():
            self.skipTest("no scan file")
        recs = json.loads(LATEST.read_text(encoding="utf-8")).get("recommendations") or []
        for r in recs:
            ex = explain(r)
            if ex.fair_yes is not None and ex.repro_yes is not None:
                self.assertTrue(ex.repro_ok, f"{r['ticker']}: {ex.fair_yes} vs {ex.repro_yes}")
            if ex.conf_ok is not None:
                self.assertTrue(ex.conf_ok, f"{r['ticker']}: conf {ex.conf_recorded} vs {ex.conf_repro}")


class ConfidenceRebuild(unittest.TestCase):
    def test_parts_and_calibration(self):
        r = rec("KXMLBGAME-X-LAA", "pregame_log5", LOG5, 0.3833, price=0.30, flow_conf=0.4,
                liq=0.9, cal=0.9)
        ex = explain(r)
        edge_conf = min(r["edge_cents"] / 12.0, 1.0) * ex.fv_conf
        raw = 0.45 * 0.4 + 0.35 * edge_conf + 0.20 * 0.9 + 0.15
        self.assertAlmostEqual(ex.conf_raw, round(raw, 4), places=4)
        self.assertAlmostEqual(ex.conf_repro, round(raw * 0.9, 4), places=4)

    def test_conflict_halves(self):
        r = rec("KXMLBGAME-X-LAA", "pregame_log5", LOG5, 0.3833, price=0.30, conflict=True)
        ex = explain(r)
        self.assertTrue(any("conflict" in label for label, _ in ex.conf_parts))
        no_bonus = [v for label, v in ex.conf_parts if "agreement" in label][0]
        self.assertEqual(no_bonus, 0.0)


class Gates(unittest.TestCase):
    def _gate(self, ex, name):
        return next(g for g in ex.gates if g.name == name)

    def test_edge_pass_and_margin(self):
        ex = explain(rec("KXMLBGAME-X-LAA", "pregame_log5", LOG5, 0.3833, price=0.30))
        g = self._gate(ex, "Minimum edge")
        self.assertEqual(g.status, "PASS")
        self.assertIn("margin", g.note)

    def test_strong_flow_bypasses_the_edge_gate(self):
        ex = explain(rec("KXMLBGAME-X-LAA", "pregame_log5", LOG5, 0.3833, price=0.35, flow_conf=0.6))
        self.assertEqual(self._gate(ex, "Minimum edge").status, "BYPASS")

    def test_small_edge_weak_flow_fails(self):
        ex = explain(rec("KXMLBGAME-X-LAA", "pregame_log5", LOG5, 0.3833, price=0.35, flow_conf=0.2))
        self.assertEqual(self._gate(ex, "Minimum edge").status, "FAIL")

    def test_in_game_uses_the_stiffer_bar(self):
        fv = {"home_win_prob": 0.40, "yes_is_home": True}
        ex = explain(rec("KXMLBGAME-X-KC", "live", fv, 0.40, price=0.34))     # +6c: ok pregame, not in-game
        g = self._gate(ex, "Minimum edge")
        self.assertIn(f"{DEFAULTS.in_game_min_edge_cents:g}c", g.rule)
        self.assertEqual(g.status, "FAIL")

    def test_calibration_is_never_a_gate(self):
        ex = explain(rec("KXMLBGAME-X-LAA", "pregame_log5", LOG5, 0.3833, cal=0.7))
        self.assertEqual(self._gate(ex, "Calibration").status, "INFO")


class Sizing(unittest.TestCase):
    def test_kelly_chain_and_implied_bankroll(self):
        # p=0.6233 (NO side of a 37.7% YES), c=0.52 -> the real PIT +3.5 case
        fv = {"line": 3.5, "exp_margin": -0.75, "yes_is_home": False}
        ex = explain(rec("KXNFLSPREAD-26SEP27CINPIT-CIN4", "pregame_nfl_spread", fv, 0.3767,
                         side="NO", price=0.52, stake=1.01, contracts=1.95))
        z = ex.sizing
        self.assertAlmostEqual(z["f_star"], (0.6233 - 0.52) / 0.48, places=4)
        self.assertAlmostEqual(z["f_frac"], z["f_star"] * DEFAULTS.kelly_fraction)
        self.assertFalse(z["capped"])
        self.assertAlmostEqual(z["bankroll"], 1.01 / z["f_used"])
        self.assertAlmostEqual(z["ev_net_ct"], z["ev_ct"] - z["fee_ct"])
        self.assertAlmostEqual(z["breakeven"], 0.52 + z["fee_ct"])

    def test_cap_binds_on_a_huge_edge(self):
        fv = {"home_win_prob": 0.9, "yes_is_home": True}
        ex = explain(rec("KXMLBGAME-X-KC", "live", fv, 0.9, price=0.30))
        self.assertTrue(ex.sizing["capped"])
        self.assertEqual(ex.sizing["f_used"], DEFAULTS.max_stake_pct)


class FlowOnlyAndParsing(unittest.TestCase):
    def test_no_fair_value(self):
        r = rec("KXNFLGAME-X-NYG", "none", {}, None)
        r["fair_prob"], r["edge_cents"] = None, None
        ex = explain(r)
        self.assertIsNone(ex.p_side)
        self.assertEqual(ex.sizing, {})
        self.assertTrue(any(g.name == "Flow-only strength" for g in ex.gates))
        self.assertTrue(any("money-flow-only" in n for n in ex.notes))

    def test_parse_flow_from_a_real_rationale(self):
        f = parse_flow("Money flow NO (score -0.54, strength 0.48; book -0.58, trades -0.87, "
                       "oi +0.00) | fair P(YES)=38% via pregame_nfl_spread")
        self.assertEqual((f["score"], f["strength"], f["book"], f["trades"], f["oi"]),
                         (-0.54, 0.48, -0.58, -0.87, 0.0))

    def test_model_names(self):
        self.assertIn("log5", model_name("MLB", "winner", False))
        self.assertIn("live", model_name("MLB", "winner", True))
        self.assertIn("negative binomial", model_name("NFL", "total", False))


class Segments(unittest.TestCase):
    def test_grouping(self):
        rows = [{"sport": "MLB", "kind": "total", "in_game": False, "won": True, "wager": 1.0,
                 "profit": 0.5, "fee": 0.01, "expected": 0.1},
                {"sport": "MLB", "kind": "total", "in_game": True, "won": False, "wager": 1.0,
                 "profit": -1.0, "fee": 0.01, "expected": 0.1},
                {"sport": "MLB", "kind": "total", "in_game": False, "won": False, "wager": 1.0,
                 "profit": -1.0, "fee": 0.01, "expected": 0.1}]
        segs = dict(segment_rows(rows))
        self.assertEqual(segs[("MLB", "total", False)].n, 2)
        self.assertEqual(segs[("MLB", "total", True)].n, 1)


class Renderers(unittest.TestCase):
    """Every model path renders without raising (the UI must never crash)."""

    def test_all_sources_render(self):
        from rich.console import Console
        from output.live_dashboard import model_left, model_right
        cases = [rec("KXMLBGAME-X-LAA", "pregame_log5", LOG5, 0.3833),
                 rec("KXMLBGAME-X-KC", "live", {"home_win_prob": 0.5, "yes_is_home": True}, 0.5),
                 rec("KXNFLGAME-X-CHI", "pregame_elo", {"yes_elo": 1500, "opp_elo": 1450,
                                                         "yes_is_home": False}, 0.53)]
        none_rec = rec("KXNFLGAME-X-NYG", "none", {}, None)
        none_rec["fair_prob"], none_rec["edge_cents"] = None, None
        cases.append(none_rec)
        segs = [(("MLB", "winner", False), tally([]))]
        console = Console(width=100, file=open("NUL" if sys.platform == "win32" else "/dev/null", "w"))
        for r in cases:
            ex = explain(r)
            console.print(model_left(ex))
            console.print(model_right(ex, segs))


if __name__ == "__main__":
    unittest.main(verbosity=2)
