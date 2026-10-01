"""Shared probability-distribution helpers for totals (over/under) models.

Extracted from data/fair_value_totals.py so data/fair_value_nfl_totals.py can
reuse the exact same math instead of copy-pasting it -- both MLB run totals and
NFL point totals are modeled as negative-binomial (overdispersed vs Poisson;
each sport tunes its own phi empirically).
"""
from __future__ import annotations

import math
from functools import lru_cache


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



def _poisson_pmf_vector(mean: float, kmax: int) -> list[float]:
    out = [0.0] * (kmax + 1)
    term = math.exp(-mean)
    out[0] = term
    for i in range(1, kmax + 1):
        term *= mean / i
        out[i] = term
    return out


@lru_cache(maxsize=8192)
def _overtime_total_pmf(e_away: float, e_home: float,
                        dispersion: float, kmax: int) -> tuple[float, ...]:
    """P(final total == i), i in 0..kmax+1, for a sport where a REGULATION
    TIE is broken by exactly one more scoring event -- i.e. NHL, where
    overtime ends on the first goal and a shootout win is recorded as a
    single goal too (see data/nhl_data.py's module docstring for the
    empirical check).

    Cached, because one game's whole ladder of Kalshi lines shares a single
    build.

    Why this exists instead of nb_survival(): that rule makes the final total
    ODD far more often than any Poisson/negative-binomial can express.
    Measured over 3,936 real regular-season games (2023-24 -> 2025-26),
    68.1% of NHL games end on an odd total, against the 50.0% a Poisson at
    the same mean implies, and a 0-0 final is impossible. The old
    nb_survival() model was therefore mis-stating P(over) by up to 4.5
    points at real Kalshi lines -- and with alternating SIGN by line, which
    is exactly the shape that manufactures phantom edges.

    Construction: regulation goals are independent Poisson per team; a
    regulation tie (which is what makes the regulation total even) adds
    exactly +1. Team means are solved so that E[final total] equals the
    caller's e_away + e_home, since the extra goal is already inside the
    scoring rates those were measured from.

    `dispersion` < 1 contracts the distribution around its mean (preserving
    the mean) to match hockey's measured UNDER-dispersion -- real team
    scores are mildly negatively correlated within a game (score effects,
    empty nets), which independent Poissons cannot capture.
    """
    target = e_away + e_home
    if target <= 0:
        return (1.0,) + (0.0,) * (kmax + 1)
    share_away = e_away / target
    a, h = e_away, e_home
    for _ in range(8):                      # converges in ~3; cheap and fixed-cost
        pa, ph = _poisson_pmf_vector(a, kmax), _poisson_pmf_vector(h, kmax)
        p_tie = sum(pa[s] * ph[s] for s in range(kmax + 1))
        reg = max(1e-9, target - p_tie)
        a, h = reg * share_away, reg * (1.0 - share_away)

    pa, ph = _poisson_pmf_vector(a, kmax), _poisson_pmf_vector(h, kmax)
    p_reg = _poisson_pmf_vector(a + h, kmax)
    pmf = [0.0] * (kmax + 2)
    for r in range(kmax + 1):
        tie = pa[r // 2] * ph[r // 2] if r % 2 == 0 else 0.0
        pmf[r] += p_reg[r] - tie            # not tied in regulation -> total stands
        pmf[r + 1] += tie                   # tied -> exactly one more goal

    if dispersion != 1.0:
        mu = sum(i * p for i, p in enumerate(pmf))
        if mu > 0:
            w = [p * math.exp(-(1.0 - dispersion) * (i - mu) ** 2 / (2.0 * mu))
                 for i, p in enumerate(pmf)]
            tot = sum(w)
            pmf = [x / tot for x in w]
            # Restore the mean exactly with an exponential tilt exp(theta*i).
            # The tilted mean is monotone increasing in theta, so bisect --
            # a single first-order correction leaves a ~0.01-goal drift,
            # which is enough to move P(over) at a near-the-money line.
            def _tilted_mean(theta: float) -> float:
                w2 = [p * math.exp(theta * i) for i, p in enumerate(pmf)]
                t2 = sum(w2)
                return sum(i * x for i, x in enumerate(w2)) / t2 if t2 > 0 else mu

            lo, hi = -1.0, 1.0
            for _ in range(80):
                mid = (lo + hi) / 2.0
                if _tilted_mean(mid) < mu:
                    lo = mid
                else:
                    hi = mid
            theta = (lo + hi) / 2.0
            w = [p * math.exp(theta * i) for i, p in enumerate(pmf)]
            tot = sum(w)
            pmf = [x / tot for x in w]

    return tuple(pmf)


def overtime_total_survival(k: int, e_away: float, e_home: float,
                            dispersion: float = 1.0, kmax: int = 32) -> float:
    """P(final total > k) -- see _overtime_total_pmf for the construction."""
    if k < 0:
        return 1.0 if e_away + e_home > 0 else 0.0
    if e_away + e_home <= 0:
        return 0.0
    pmf = _overtime_total_pmf(e_away, e_home, dispersion, kmax)
    return max(0.0, min(1.0, sum(pmf[k + 1:])))


def normal_sf(x: float, mu: float, sigma: float) -> float:
    """P(X > x) for X ~ Normal(mu, sigma). Used by the NFL spread model
    (data/fair_value_nfl_spread.py) to convert an Elo-implied expected point
    margin into P(team covers a given line)."""
    if sigma <= 0:
        return 1.0 if mu > x else 0.0
    z = (x - mu) / (sigma * math.sqrt(2))
    return 0.5 * math.erfc(z)

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
