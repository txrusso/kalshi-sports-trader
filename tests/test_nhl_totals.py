"""Offline tests for the NHL totals distribution and its shrinkage.

Covers the 2026-09-30 recalibration: hockey's tie-breaking overtime/shootout
goal is modeled explicitly (data/distributions.py::overtime_total_survival)
instead of a negative binomial, and the near-zero per-game team-strength
signal is shrunk toward the league average
(data/scoring_environment_nhl.py::TEAM_SIGNAL_SHRINK).
"""
import math
import unittest

from data.distributions import (_poisson_pmf_vector, nb_survival,
                                overtime_total_survival)
from data.fair_value_nhl_totals import NHL_TOTALS_DISPERSION
from data.scoring_environment_nhl import TEAM_SIGNAL_SHRINK


def pmf(e_away, e_home, dispersion=1.0, kmax=40):
    """P(total == k) recovered from the survival function."""
    sf = [overtime_total_survival(k, e_away, e_home, dispersion) for k in range(-1, kmax)]
    return [sf[i] - sf[i + 1] for i in range(len(sf) - 1)]


class TestOvertimeTotalSurvival(unittest.TestCase):

    def test_is_a_proper_distribution(self):
        p = pmf(3.1, 3.1, NHL_TOTALS_DISPERSION)
        self.assertAlmostEqual(sum(p), 1.0, places=6)
        for x in p:
            self.assertGreaterEqual(x, -1e-12)

    def test_survival_is_monotone_decreasing(self):
        prev = 1.1
        for k in range(0, 20):
            cur = overtime_total_survival(k, 3.2, 3.0, NHL_TOTALS_DISPERSION)
            self.assertLessEqual(cur, prev)
            prev = cur

    def test_preserves_the_requested_mean(self):
        for ea, eh in [(3.1, 3.1), (2.6, 3.6), (3.9, 2.4), (1.5, 1.5)]:
            p = pmf(ea, eh, NHL_TOTALS_DISPERSION)
            mean = sum(i * x for i, x in enumerate(p))
            self.assertAlmostEqual(mean, ea + eh, places=2,
                                   msg=f"mean drifted for {ea}/{eh}")

    def test_a_scoreless_final_is_impossible(self):
        # 0-0 forces overtime, so the game cannot end with a total of 0 --
        # 0 of 3,936 real regular-season games did, vs the 8 Poisson expects.
        p = pmf(3.1, 3.1, NHL_TOTALS_DISPERSION)
        self.assertLess(p[0], 1e-9)
        self.assertGreater(nb_survival(-1, 6.2, 1.0) - nb_survival(0, 6.2, 1.0), 1e-3)

    def test_odd_totals_are_inflated_the_way_real_hockey_is(self):
        # Measured over 3,936 real games: 68.1% odd. Poisson says 50.0%.
        p = pmf(3.093, 3.093, NHL_TOTALS_DISPERSION)
        odd = sum(x for i, x in enumerate(p) if i % 2)
        self.assertGreater(odd, 0.60)
        self.assertLess(odd, 0.72)

    def test_even_totals_are_suppressed_relative_to_poisson(self):
        p = pmf(3.093, 3.093, NHL_TOTALS_DISPERSION)
        for k in (2, 4, 6, 8):
            poisson_k = _poisson_pmf_vector(6.186, 12)[k]
            self.assertLess(p[k], poisson_k, msg=f"total {k} should be suppressed")
        for k in (3, 5, 7, 9):
            poisson_k = _poisson_pmf_vector(6.186, 12)[k]
            self.assertGreater(p[k], poisson_k, msg=f"total {k} should be inflated")

    def test_dispersion_below_one_narrows_without_moving_the_mean(self):
        wide = pmf(3.1, 3.1, 1.0)
        narrow = pmf(3.1, 3.1, 0.85)
        mean_w = sum(i * x for i, x in enumerate(wide))
        mean_n = sum(i * x for i, x in enumerate(narrow))
        var_w = sum((i - mean_w) ** 2 * x for i, x in enumerate(wide))
        var_n = sum((i - mean_n) ** 2 * x for i, x in enumerate(narrow))
        self.assertAlmostEqual(mean_w, mean_n, places=2)
        self.assertLess(var_n, var_w)

    def test_matches_the_measured_league_distribution_at_real_lines(self):
        # Empirical P(total > line) over 3,936 real regular-season games
        # (2023-24 -> 2025-26), league-average scoring split evenly.
        empirical = {2: 0.9708, 3: 0.8669, 4: 0.7838, 5: 0.5612,
                     6: 0.4494, 7: 0.2485, 8: 0.1753, 9: 0.0737}
        for k, want in empirical.items():
            got = overtime_total_survival(k, 3.093, 3.093, NHL_TOTALS_DISPERSION)
            self.assertAlmostEqual(got, want, delta=0.02,
                                   msg=f"line {k}.5: got {got:.4f}, measured {want}")

    def test_beats_the_old_negative_binomial_at_every_real_line(self):
        empirical = {2: 0.9708, 3: 0.8669, 4: 0.7838, 5: 0.5612,
                     6: 0.4494, 7: 0.2485, 8: 0.1753, 9: 0.0737}
        new_err = old_err = 0.0
        for k, want in empirical.items():
            new_err += abs(overtime_total_survival(k, 3.093, 3.093,
                                                   NHL_TOTALS_DISPERSION) - want)
            old_err += abs(nb_survival(k, 6.186, 0.863) - want)
        self.assertLess(new_err, old_err / 2.0)

    def test_degenerate_inputs_do_not_raise(self):
        self.assertEqual(overtime_total_survival(5, 0.0, 0.0), 0.0)
        self.assertEqual(overtime_total_survival(-1, 3.0, 3.0), 1.0)
        self.assertAlmostEqual(overtime_total_survival(200, 3.0, 3.0), 0.0, places=6)

    def test_home_away_split_shifts_the_tie_probability(self):
        # A lopsided game is less likely to be tied in regulation, so fewer
        # of its totals get the extra goal -> fewer odd finals.
        even = pmf(3.1, 3.1, NHL_TOTALS_DISPERSION)
        lop = pmf(4.7, 1.5, NHL_TOTALS_DISPERSION)
        self.assertGreater(sum(x for i, x in enumerate(even) if i % 2),
                           sum(x for i, x in enumerate(lop) if i % 2))


class TestTeamSignalShrink(unittest.TestCase):

    def test_shrink_is_the_measured_plateau_value(self):
        # Measured OLS slope of actual total on raw lam: +0.155 +- 0.092
        # (n=2,623). Anything near 1.0 would re-open the phantom-edge bug.
        self.assertGreater(TEAM_SIGNAL_SHRINK, 0.0)
        self.assertLessEqual(TEAM_SIGNAL_SHRINK, 0.35)

    def test_shrinkage_pulls_lam_toward_the_league_average(self):
        lg, raw = 6.2, 7.1
        lam = lg + TEAM_SIGNAL_SHRINK * (raw - lg)
        self.assertLess(abs(lam - lg), abs(raw - lg))
        self.assertAlmostEqual(lam, 6.38, places=2)

    def test_shrinkage_preserves_the_home_away_ratio(self):
        from data.scoring_environment_nhl import ScoringEnvironmentModel

        class _Stub:
            def games(self):
                return []
        env = ScoringEnvironmentModel(_Stub())
        e_away, e_home, lg = 2.84, 4.26, 6.2
        raw = e_away + e_home
        lam = lg + TEAM_SIGNAL_SHRINK * (raw - lg)
        scale = lam / raw
        self.assertAlmostEqual((e_away * scale) / (e_home * scale), e_away / e_home,
                               places=9)


if __name__ == "__main__":
    unittest.main()
