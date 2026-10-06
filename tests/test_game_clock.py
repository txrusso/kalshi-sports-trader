"""Offline tests for data/fair_value.py::has_started -- which clock decides
whether a game is under way.

Found 2026-10-06. Kalshi's `occurrence_datetime` runs ~3h LATE on NFL and
NHL (a long-documented skew). engine/paper.py's trigger used the real
kickoff/puck-drop from each sport's own schedule, so it correctly saw games
already in progress and offered them as in-game bets -- while every pregame
fair-value model judged "started" against the SKEWED field, believed the
game had not begun, and returned a FULL-GAME estimate for it.

Two clocks, hours apart, and the safety net keyed off the wrong one:
signals/recommendation.py refuses a started game whose `fair_source` starts
with "pregame", but it reads `game_state`, which the model had just set to
"Preview" from the same skewed field. 17 real bets went on in-progress
games that way -- 10 NFL, 7 NHL, up to 39 minutes in.
"""
from __future__ import annotations

import pathlib
import sys
import unittest
from datetime import datetime, timezone

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from data.fair_value import has_started                     # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]

# The real NSH@TOR game this was found on.
REAL = datetime(2026, 10, 6, 23, 0, tzinfo=timezone.utc)     # 19:00 ET puck drop
SKEWED = datetime(2026, 10, 7, 2, 0, tzinfo=timezone.utc)    # Kalshi said 22:00 ET
MID = datetime(2026, 10, 6, 23, 14, tzinfo=timezone.utc)     # 14 min in: the bet fired here


class WhichClockDecides(unittest.TestCase):

    def test_the_skewed_field_alone_gets_it_wrong(self):
        self.assertFalse(has_started(SKEWED, None, MID))

    def test_the_schedule_start_gets_it_right(self):
        self.assertTrue(has_started(SKEWED, REAL, MID))

    def test_pregame_is_still_pregame(self):
        before = datetime(2026, 10, 6, 22, 40, tzinfo=timezone.utc)   # 20 min early
        self.assertFalse(has_started(SKEWED, REAL, before))

    def test_exactly_at_start_counts_as_started(self):
        self.assertTrue(has_started(SKEWED, REAL, REAL))

    def test_one_second_before_start_does_not(self):
        from datetime import timedelta
        self.assertFalse(has_started(SKEWED, REAL, REAL - timedelta(seconds=1)))

    def test_no_schedule_time_falls_back_to_occurrence(self):
        # NBA's case: no independent tip-off source exists, so occurrence is
        # all there is. Must behave, not crash or say "never started".
        self.assertFalse(has_started(SKEWED, None, MID))
        self.assertTrue(has_started(REAL, None, MID))

    def test_no_times_at_all_is_not_started(self):
        self.assertFalse(has_started(None, None, MID))

    def test_schedule_time_wins_even_when_it_is_LATER(self):
        # The skew could in principle go the other way; the schedule is the
        # authority either way, not whichever is earlier.
        early_occurrence = datetime(2026, 10, 6, 20, 0, tzinfo=timezone.utc)
        self.assertFalse(has_started(early_occurrence, REAL,
                                     datetime(2026, 10, 6, 22, 0, tzinfo=timezone.utc)))


class EveryModelIsWiredToTheScheduleClock(unittest.TestCase):
    """If a model goes back to comparing against occurrence alone, that is
    this bug returning -- it cost 17 real bets and would do so silently."""

    def test_nfl_and_nhl_models_use_the_schedule_start(self):
        for name, field in [("fair_value_nhl", "puck_drop_utc"),
                            ("fair_value_nhl_totals", "puck_drop_utc"),
                            ("fair_value_nfl", "kickoff_utc"),
                            ("fair_value_nfl_totals", "kickoff_utc"),
                            ("fair_value_nfl_spread", "kickoff_utc")]:
            src = (ROOT / "data" / f"{name}.py").read_text(encoding="utf-8")
            with self.subTest(model=name):
                self.assertIn(f"has_started(occurrence, g.{field})", src)

    def test_no_model_compares_against_occurrence_directly_any_more(self):
        for f in sorted((ROOT / "data").glob("fair_value*.py")):
            with self.subTest(model=f.name):
                self.assertNotIn("datetime.now(timezone.utc) >= occurrence",
                                 f.read_text(encoding="utf-8"))

    def test_nba_is_the_documented_exception_and_says_so(self):
        # NBA genuinely has no independent clock; it must still go through
        # has_started (so the fallback is explicit) and explain why.
        for name in ("fair_value_nba", "fair_value_nba_totals"):
            src = (ROOT / "data" / f"{name}.py").read_text(encoding="utf-8")
            with self.subTest(model=name):
                self.assertIn("has_started(occurrence)", src)
                self.assertIn("no independent tip-off source", src.lower()
                              .replace("no independent tip-off source",
                                       "no independent tip-off source"))


if __name__ == "__main__":
    unittest.main()
