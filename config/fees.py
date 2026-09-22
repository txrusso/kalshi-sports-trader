"""Kalshi trading fees — the first fee model in this project.

Until 2026-09-22 nothing here modeled fees at all: the edge calc, Kelly sizing,
the ledger and every backtest were gross of fees. That's fine for ranking bets
(fees barely change which side you pick) but it makes any taker-vs-maker
comparison meaningless, since the whole point of resting an order is the fee.

FORMULA — verified, not quoted from docs
----------------------------------------
    fee = ceil_4dp(fee_multiplier * RATE * C * P * (1 - P))

where C = contracts, P = the price in dollars of the side actually traded, and
RATE is 0.07 for a taker fill. Checked against all 144 real fills on this
account (GET /portfolio/fills, which reports `fee_cost` and `is_taker` per
fill): every NFL fill matched `0.07 * C * P * (1-P)` to the cent, and every MLB
fill came back at exactly HALF that -- which is `/series/KXMLBGAME` reporting
`fee_multiplier: 0.5`. So the multiplier scales the taker fee too, not just the
maker fee. Rounding is UP to $0.0001 (not to the cent); that also reproduced
exactly on all 144 fills.

Note P is the traded side's own price: a NO fill at 66c is P=0.66, not 0.34.
P*(1-P) is symmetric so the fee is the same either way, but keep it explicit.

MAKER FEES — two regimes, one verified and one not
--------------------------------------------------
Each series reports `fee_type` and `fee_multiplier` (GET /series/{ticker}):

  * `quadratic`                  -> maker fills are FREE. VERIFIED: the two
    real maker fills on this account (both KXMLBTOTAL, 2026-09-18) each
    reported `fee_cost: 0.000000` where the taker fee would have been ~1.9-2.7c.
    Series seen in this regime: KXMLBTOTAL, KXNHLTOTAL.

  * `quadratic_with_maker_fees`  -> maker fills are charged at a reduced rate.
    NOT VERIFIED: no maker fill exists on any such series on this account, so
    MAKER_RATE below is taken from Kalshi's published maker formula (0.0175,
    i.e. one quarter of the taker rate) rather than measured. Treat it as an
    upper bound on the saving until a real maker fill lands on one of these
    series. `audit_fills()` re-checks the model against reality on every run and
    will flag it the moment one does.

Series fee params are fetched live and cached (they change -- Kalshi has moved
sports fee multipliers before); _FALLBACK_SERIES holds the values measured
2026-09-22 for when the fetch fails, so a network blip degrades to a stale
number rather than a crash.
"""
from __future__ import annotations

import logging
import math
from typing import Optional

log = logging.getLogger("config.fees")

TAKER_RATE = 0.07
# One quarter of the taker rate, per Kalshi's published maker formula.
# UNVERIFIED on this account -- see the module docstring.
MAKER_RATE = 0.0175

# fee_type values that mean "resting orders are not charged at all".
_FREE_MAKER_TYPES = {"quadratic"}

# Measured live 2026-09-22 via GET /series/{ticker}. Used only if the live
# fetch fails. (series -> (fee_type, fee_multiplier))
_FALLBACK_SERIES: dict[str, tuple[str, float]] = {
    "KXMLBGAME":   ("quadratic_with_maker_fees", 0.5),
    "KXMLBTOTAL":  ("quadratic", 0.5),
    "KXNFLGAME":   ("quadratic_with_maker_fees", 1.0),
    "KXNFLTOTAL":  ("quadratic_with_maker_fees", 1.0),
    "KXNFLSPREAD": ("quadratic_with_maker_fees", 1.0),
    "KXNBAGAME":   ("quadratic_with_maker_fees", 1.0),
    "KXNBATOTAL":  ("quadratic_with_maker_fees", 1.0),
    "KXNHLGAME":   ("quadratic_with_maker_fees", 1.0),
    "KXNHLTOTAL":  ("quadratic", 1.0),
}

_cache: dict[str, tuple[str, float]] = {}


def series_of(ticker: str) -> str:
    """'KXMLBTOTAL-26SEP...-8' -> 'KXMLBTOTAL'."""
    return (ticker or "").split("-", 1)[0].upper()


def series_fee_params(ticker: str, client=None) -> tuple[str, float]:
    """(fee_type, fee_multiplier) for a ticker's series, cached per process.

    Falls back to the values measured 2026-09-22 when no client is given or the
    fetch fails -- never raises, since this sits in the sizing path."""
    srs = series_of(ticker)
    if srs in _cache:
        return _cache[srs]
    params: Optional[tuple[str, float]] = None
    if client is not None:
        try:
            d = client._get(f"/series/{srs}").get("series", {}) or {}
            ft = d.get("fee_type")
            fm = d.get("fee_multiplier")
            if ft:
                params = (str(ft), float(fm if fm is not None else 1.0))
        except Exception:
            log.debug("Fee params fetch failed for %s; using fallback.", srs, exc_info=True)
    if params is None:
        params = _FALLBACK_SERIES.get(srs, ("quadratic_with_maker_fees", 1.0))
    _cache[srs] = params
    return params


def _ceil_4dp(x: float) -> float:
    """Kalshi rounds fees UP to $0.0001 (verified on 144 real fills)."""
    return math.ceil(x * 10000 - 1e-9) / 10000.0


def fee_for(count: float, price: float, ticker: str, is_taker: bool = True,
            client=None) -> float:
    """Fee in dollars for filling `count` contracts at `price` on `ticker`.

    `price` is the traded side's own price in dollars (a NO fill at 66c is 0.66).
    Returns 0.0 for a maker fill on a `quadratic` (no-maker-fee) series."""
    if count <= 0 or not (0.0 < price < 1.0):
        return 0.0
    fee_type, mult = series_fee_params(ticker, client)
    if is_taker:
        rate = TAKER_RATE
    elif fee_type in _FREE_MAKER_TYPES:
        return 0.0
    else:
        rate = MAKER_RATE
    return _ceil_4dp(mult * rate * count * price * (1.0 - price))


def fee_cents_per_contract(price: float, ticker: str, is_taker: bool = True,
                           client=None) -> float:
    """Per-contract fee in CENTS -- the unit the edge/threshold math uses.

    Not just fee_for(1,...)*100: the 4-decimal round-up is applied to the whole
    order, so on a 1-contract order it would overstate the per-contract cost.
    Computed unrounded here so it composes correctly with edge_cents."""
    if not (0.0 < price < 1.0):
        return 0.0
    fee_type, mult = series_fee_params(ticker, client)
    if is_taker:
        rate = TAKER_RATE
    elif fee_type in _FREE_MAKER_TYPES:
        return 0.0
    else:
        rate = MAKER_RATE
    return mult * rate * price * (1.0 - price) * 100.0


def maker_saving_cents(price: float, ticker: str, client=None) -> float:
    """Cents per contract saved by resting as a maker instead of taking."""
    return (fee_cents_per_contract(price, ticker, True, client)
            - fee_cents_per_contract(price, ticker, False, client))


def audit_fills(fills: list[dict], client=None) -> dict:
    """Re-check this model against real fills. Returns
    {n, n_taker, n_maker, mismatches:[...], total_fee, total_notional}.

    The maker rate is unverified (no maker fill exists on a
    `quadratic_with_maker_fees` series on this account), so this is how we'll
    learn the real number the first time maker mode fills one."""
    out: dict = {"n": 0, "n_taker": 0, "n_maker": 0, "mismatches": [],
                 "total_fee": 0.0, "total_notional": 0.0}
    for f in fills:
        try:
            count = float(f["count_fp"])
            side = (f.get("outcome_side") or "yes").lower()
            price = float(f["yes_price_dollars"] if side == "yes" else f["no_price_dollars"])
            actual = float(f["fee_cost"])
            taker = bool(f.get("is_taker"))
        except (KeyError, TypeError, ValueError):
            continue
        pred = fee_for(count, price, f.get("ticker", ""), taker, client)
        out["n"] += 1
        out["n_taker" if taker else "n_maker"] += 1
        out["total_fee"] += actual
        out["total_notional"] += count * price
        if abs(pred - actual) > 1e-9:
            out["mismatches"].append({
                "ticker": f.get("ticker"), "is_taker": taker, "count": count,
                "price": price, "predicted": pred, "actual": actual,
            })
    return out
