"""Shared probability-distribution helpers for totals (over/under) models.

Extracted from data/fair_value_totals.py so data/fair_value_nfl_totals.py can
reuse the exact same math instead of copy-pasting it -- both MLB run totals and
NFL point totals are modeled as negative-binomial (overdispersed vs Poisson;
each sport tunes its own phi empirically).
"""
from __future__ import annotations

import math


def nb_survival(k: int, mean: float, phi: float) -> float:
    """P(N > k) for a negative-binomial with given mean and variance = phi*mean.

    phi <= 1.0 degenerates to the Poisson survival function (phi=1 is the
    Poisson variance=mean case).
    """
    if mean <= 0:
        return 0.0
    if phi <= 1.0:
        cdf = 0.0
        term = math.exp(-mean)
        for i in range(0, k + 1):
            if i > 0:
                term *= mean / i
            cdf += term
        return max(0.0, 1.0 - cdf)
    r = mean / (phi - 1.0)          # size parameter
    p = r / (r + mean)              # success prob
    # CDF(k) = sum_{i=0}^{k} pmf(i);  pmf via lgamma for stability
    log_p = math.log(p)
    log_1mp = math.log(1 - p)
    cdf = 0.0
    for i in range(0, k + 1):
        log_pmf = (math.lgamma(i + r) - math.lgamma(r) - math.lgamma(i + 1)
                   + r * log_p + i * log_1mp)
        cdf += math.exp(log_pmf)
    return max(0.0, min(1.0, 1.0 - cdf))

# NOTE (2026-09-01): a heavier-right-tail "blowout" mixture on the survival function was
# built and tested here to fix the totals model's low-P(over)/high-line miscalibration
# (season_backtest 0-10% bucket predicts ~5.6% over, hits ~36%). A MEAN-PRESERVING mixture
# (fatten the right tail, thin the base, keep E[N]) did improve held-out calibration cleanly
# (validate Brier 0.1955->0.1942, high-half guard also better) -- but it LOST money and was
# REVERTED. A targeted deep-tail ROI test (real pregame prices on all baseline-P(over)<0.25
# lines) showed ROI got WORSE on both splits (validate -0.012 baseline -> -0.062 with the
# fix): the fix mechanically shifts bets under->over as intended, but the MARKET already
# prices those high-scoring games correctly, so moving the model toward reality just moves it
# toward the market's prices -- removing disagreement without creating profitable edges (the
# same "calibration != profit" lesson as the reverted Pythagorean blend). The high-line
# miscalibration is real but not exploitable. See backtest/baseline_20260831.txt.
