"""Adaptive confidence calibration from historical bet outcomes.

Pools settled bets from two sources -- the backtest snapshot history and the
paper-trade ledger -- and buckets them by market type (winner vs. totals).
For each bucket, compares the realized win rate to what the model itself
expected (the mean of its own pre-bet win probabilities) and derives a small
multiplier applied to `Recommendation.confidence`.

This intentionally does NOT touch money-flow or fair-value math -- it only
scales how much weight a recommendation's confidence carries, using the same
signals the agent already computes plus a memory of how similar past bets
turned out. Two guardrails keep tiny samples from steering it:

  1. MIN_BUCKET_N -- a bucket stays at a neutral 1.0x multiplier until it has
     accumulated at least this many settled bets.
  2. Bayesian shrinkage -- above that floor, the realized win rate is pulled
     toward the model's own expectation by PRIOR_STRENGTH pseudo-observations,
     so the multiplier only fully trusts the data once evidence is ample.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

log = logging.getLogger("signals.calibration")

MIN_BUCKET_N = 15          # below this many settled bets, multiplier stays neutral (1.0)
PRIOR_STRENGTH = 20        # pseudo-count anchoring the multiplier toward 1.0 until n is large
MULTIPLIER_CLAMP = (0.7, 1.3)


@dataclass
class Calibration:
    multipliers: dict[str, float] = field(default_factory=dict)
    bucket_stats: dict[str, dict] = field(default_factory=dict)   # for rationale/debug

    def multiplier_for(self, market_type: str) -> float:
        return self.multipliers.get(market_type, 1.0)


def _normalize_backtest_bets(rows: list[dict], outcomes: dict[str, bool]) -> list[dict]:
    from backtest.evaluate import graded_bets
    from config.sports import calibration_bucket
    out = []
    for b in graded_bets(rows, outcomes):
        out.append({"market": calibration_bucket(b["ticker"]),
                    "won": b["won"], "win_prob": b["win_prob"]})
    return out


def _normalize_ledger_bets(ledger_bets: list[dict], outcomes: dict[str, bool]) -> list[dict]:
    from config.sports import calibration_bucket
    out = []
    for b in ledger_bets:
        if b["ticker"] not in outcomes:
            continue                                     # not yet final
        yes_won = outcomes[b["ticker"]]
        won = yes_won if b["side"] == "YES" else (not yes_won)
        fp = b.get("fair_prob")
        win_prob = (fp if b["side"] == "YES" else (1 - fp)) if fp is not None else None
        # Derived from the ticker itself, not the ledger's legacy "market" field
        # (bare "winner"/"total", no sport dimension) -- keeps old ledger rows
        # (which lack a "market" field entirely on the earliest lines) working too.
        out.append({"market": calibration_bucket(b["ticker"]), "won": won, "win_prob": win_prob})
    return out


def load_settled_bets(clients: Optional[dict] = None) -> list[dict]:
    """Settled bets from both the backtest snapshot history and the paper ledger,
    normalized to {market: 'mlb:winner'|'nfl:total'|..., won: bool, win_prob: float|None}."""
    from backtest.evaluate import load_rows
    from data.games import build_clients, resolve_outcomes
    from engine.paper import PaperLedger

    clients = clients or build_clients()
    out: list[dict] = []

    try:
        rows = load_rows(None)
        if rows:
            outcomes = resolve_outcomes({r["ticker"] for r in rows}, clients)
            out.extend(_normalize_backtest_bets(rows, outcomes))
    except Exception as e:
        log.warning("Calibration: could not load backtest history (%s)", e)

    try:
        ledger_bets = PaperLedger().load()
        if ledger_bets:
            outcomes = resolve_outcomes({b["ticker"] for b in ledger_bets}, clients)
            out.extend(_normalize_ledger_bets(ledger_bets, outcomes))
    except Exception as e:
        log.warning("Calibration: could not load paper ledger (%s)", e)

    return out


def build_calibration(bets: Optional[list[dict]] = None) -> Calibration:
    if bets is None:
        bets = load_settled_bets()

    cal = Calibration()
    buckets: dict[str, list[dict]] = {}
    for b in bets:
        buckets.setdefault(b["market"], []).append(b)

    for key, bucket_bets in buckets.items():
        n = len(bucket_bets)
        wins = sum(1 for b in bucket_bets if b["won"])
        win_rate = wins / n
        expected_wr = sum(b["win_prob"] if b["win_prob"] is not None else 0.5
                          for b in bucket_bets) / n

        if n < MIN_BUCKET_N:
            cal.multipliers[key] = 1.0
            cal.bucket_stats[key] = {"n": n, "win_rate": round(win_rate, 3),
                                     "expected_wr": round(expected_wr, 3), "multiplier": 1.0,
                                     "note": f"below min sample ({MIN_BUCKET_N})"}
            continue

        posterior_wr = (wins + PRIOR_STRENGTH * expected_wr) / (n + PRIOR_STRENGTH)
        raw_mult = (posterior_wr / expected_wr) if expected_wr > 0 else 1.0
        mult = round(max(MULTIPLIER_CLAMP[0], min(MULTIPLIER_CLAMP[1], raw_mult)), 3)
        cal.multipliers[key] = mult
        cal.bucket_stats[key] = {"n": n, "win_rate": round(win_rate, 3),
                                 "expected_wr": round(expected_wr, 3),
                                 "posterior_wr": round(posterior_wr, 3), "multiplier": mult}

    log.info("Calibration built: %s", cal.bucket_stats)
    return cal
