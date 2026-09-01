"""Leak-safe backtest of the NFL Elo/scoring-rate model against real historical
game outcomes -- the NFL analog of backtest/season_backtest.py, adapted for the
NFL model's inputs (signals/elo_nfl.py, data/scoring_environment_nfl.py).

Unlike MLB's season_backtest.py, there's no separate point-in-time ledger to
build here: EloRatings.rating_before() and data/nfl_data.py's
team_games_before() are ALREADY leak-safe by construction -- every query
filters to games strictly before the target date, from a static, already-final
historical CSV. MLB needed its own point-in-time ledger because the MLB Stats
API's "as of date" fields turned out to leak that date's own result into the
input (a real bug found and fixed 2026-08-13, see season_backtest.py's
docstring). nflverse's games.csv has no such "as of" field to accidentally
trust: we always filter by date ourselves, so validating against ANY past
season needs no separate data truncation step.

Two different validations against nflverse's historical seasons, because the
ground truth differs:
  - WINNER: real game outcomes (who won) are ground truth on their own --
    Brier/log-loss/accuracy against Elo's win probability needs no market data.
  - TOTALS: nflverse doesn't carry a betting line, so instead of faking one this
    validates the thing that actually matters before ever trading it: is the
    expected-points mean (`lam`) itself unbiased and better than a naive
    baseline, and what does the *actual* residual variance/mean ratio in the
    validation season say NFL_TOTALS_PHI should be (reported as "implied_phi",
    to compare against the in-sample value the constant was originally seeded
    from).

A THIRD validation, `--kalshi-settled`, grades against real settled Kalshi NFL
markets directly (confirmed live 2026-08-29: 48 settled KXNFLGAME games + 920
settled KXNFLTOTAL lines already exist) -- mirrors season_backtest.py's
methodology (real settled results, real pregame trade prices, a real ROI pass
at the production min-edge gate) rather than this file's other two modes' use
of nflverse's historical seasons. The important caveat: every settled NFL
market so far is from PRESEASON (Aug 6-29, 2026) -- nflverse's games.csv
doesn't carry preseason games at all, so the model's win-prob/scoring-rate
inputs for these games are each team's regular-season rating/rate (2025's,
since no 2026 regular-season game has been played yet), not anything
preseason-specific. Preseason rosters lean heavily on backups, especially in
later weeks, so predictive accuracy here is a real but weaker signal than
regular-season games will be -- reported honestly either way, not smoothed
over. A real regular-season ROI read has to wait for Week 1 (kicks off
2026-09-04) to accumulate settled markets.

Usage:
    py -3 -m backtest.nfl_model_backtest
    py -3 -m backtest.nfl_model_backtest --validate-season 2024
    py -3 -m backtest.nfl_model_backtest --sweep k_factor 10 15 20 25 30
    py -3 -m backtest.nfl_model_backtest --sweep season_blend_games 3 6 9 12
    py -3 -m backtest.nfl_model_backtest --kalshi-settled
    py -3 -m backtest.nfl_model_backtest --kalshi-settled --no-roi
"""
from __future__ import annotations

import argparse
import math
import sys
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import data.fair_value_nfl_totals as fv_nfl_totals_mod
import data.scoring_environment_nfl as scoring_env_mod
import signals.elo_nfl as elo_mod
from config.settings import DEFAULTS
from data.distributions import nb_survival
from data.fair_value_nfl import parse_ticker as parse_nfl_ticker
from data.fair_value_nfl_totals import parse_total_ticker as parse_nfl_total_ticker
from data.nfl_data import NflDataClient, NflGame
from data.scoring_environment_nfl import ScoringEnvironmentModel
from kalshi.client import KalshiClient
from signals.elo_nfl import EloRatings, win_prob

SWEEPABLE = {
    "k_factor": elo_mod, "home_field_elo": elo_mod, "season_regression": elo_mod,
    "season_blend_games": scoring_env_mod,
}
_ATTR_NAME = {
    "k_factor": "K_FACTOR", "home_field_elo": "HOME_FIELD_ELO",
    "season_regression": "SEASON_REGRESSION", "season_blend_games": "SEASON_BLEND_GAMES",
}


def grade_winner(elo: EloRatings, games: list[NflGame]) -> list[dict]:
    out = []
    for g in games:
        if g.state != "Final" or g.home_score == g.away_score:
            continue    # ties excluded (~0.1% of games) -- not a binary outcome
        r_home = elo.rating_before(g.home_abbr, g.date_str)
        r_away = elo.rating_before(g.away_abbr, g.date_str)
        pred = win_prob(r_home, r_away, a_is_home=True)
        actual = 1 if g.home_score > g.away_score else 0
        out.append({"date": g.date_str, "pred": pred, "actual": actual, "game_id": g.game_id})
    return out


def grade_totals(env: ScoringEnvironmentModel, games: list[NflGame]) -> list[dict]:
    out = []
    for g in games:
        if g.state != "Final":
            continue
        et = env.expected_points(g.away_abbr, g.home_abbr, g.date_str, roof=g.roof)
        if et is None:
            continue
        out.append({"date": g.date_str, "lam": et["lam"], "actual": g.total_points,
                    "game_id": g.game_id})
    return out


def _brier(rows: list[dict]) -> float | None:
    if not rows:
        return None
    return sum((r["pred"] - r["actual"]) ** 2 for r in rows) / len(rows)


def _log_loss(rows: list[dict]) -> float | None:
    if not rows:
        return None
    eps = 1e-6
    total = 0.0
    for r in rows:
        p = min(max(r["pred"], eps), 1 - eps)
        total += -(r["actual"] * math.log(p) + (1 - r["actual"]) * math.log(1 - p))
    return total / len(rows)


def _accuracy(rows: list[dict]) -> float | None:
    if not rows:
        return None
    return sum(1 for r in rows if (r["pred"] > 0.5) == (r["actual"] == 1)) / len(rows)


def _calibration_table(rows: list[dict], n_buckets: int = 10) -> list[dict]:
    buckets: list[list[dict]] = [[] for _ in range(n_buckets)]
    for r in rows:
        buckets[min(int(r["pred"] * n_buckets), n_buckets - 1)].append(r)
    table = []
    for i, b in enumerate(buckets):
        if not b:
            continue
        table.append({"range": f"{i*100//n_buckets}-{(i+1)*100//n_buckets}%", "n": len(b),
                      "mean_pred": round(sum(r["pred"] for r in b) / len(b), 3),
                      "actual_rate": round(sum(r["actual"] for r in b) / len(b), 3)})
    return table


def totals_report(rows: list[dict]) -> dict:
    if not rows:
        return {}
    n = len(rows)
    errs = [r["actual"] - r["lam"] for r in rows]
    bias = sum(errs) / n
    rmse = math.sqrt(sum(e * e for e in errs) / n)
    mean_lam = sum(r["lam"] for r in rows) / n
    mean_actual = sum(r["actual"] for r in rows) / n
    var_resid = sum((e - bias) ** 2 for e in errs) / n
    naive_rmse = math.sqrt(sum((r["actual"] - mean_actual) ** 2 for r in rows) / n)
    return {
        "n": n, "mean_lam": round(mean_lam, 2), "mean_actual": round(mean_actual, 2),
        "bias (actual - lam)": round(bias, 3), "rmse": round(rmse, 3),
        "naive_rmse (always predict sample mean)": round(naive_rmse, 3),
        "implied_phi (var(resid)/mean_lam)": round(var_resid / mean_lam, 3) if mean_lam else None,
    }


def _games_for(nfl_data: NflDataClient, season: int) -> list[NflGame]:
    return [g for g in nfl_data.games() if g.season == season and g.state == "Final"]


def run_validation(nfl_data: NflDataClient, season: int) -> None:
    games = _games_for(nfl_data, season)
    print(f"Validating season {season}: {len(games)} completed games.\n")

    elo = EloRatings(nfl_data)
    winner_rows = grade_winner(elo, games)
    print("=== WINNER (Elo win probability) ===")
    print(f"  n                : {len(winner_rows)}")
    print(f"  brier            : {round(_brier(winner_rows), 4) if winner_rows else None}")
    print(f"  log_loss         : {round(_log_loss(winner_rows), 4) if winner_rows else None}")
    print(f"  accuracy         : {round(_accuracy(winner_rows), 3) if winner_rows else None}")
    coinflip = [{"pred": 0.5, "actual": r["actual"]} for r in winner_rows]
    home_baseline = [{"pred": 0.58, "actual": r["actual"]} for r in winner_rows]  # ~league home-win rate
    print(f"  baseline: always-0.5 brier      : {round(_brier(coinflip), 4) if coinflip else None}")
    print(f"  baseline: always-home-58% brier : {round(_brier(home_baseline), 4) if home_baseline else None}")
    print("\n  calibration:")
    for row in _calibration_table(winner_rows):
        print(f"    {row['range']:>8}  n={row['n']:<4} mean_pred={row['mean_pred']:<6} "
              f"actual_rate={row['actual_rate']}")

    env = ScoringEnvironmentModel(nfl_data)
    totals_rows = grade_totals(env, games)
    print(f"\n=== TOTALS (expected points, no real Kalshi line to test P(over) against yet) ===")
    for k, v in totals_report(totals_rows).items():
        print(f"  {k:42}: {v}")
    print("\n  Guide: rmse < naive_rmse means the model beats guessing the league average;\n"
          "  compare 'implied_phi' above to NFL_TOTALS_PHI in data/fair_value_nfl_totals.py\n"
          "  (currently seeded from a raw 2015-2025 in-sample measurement, not yet\n"
          "  validated out-of-sample the way this number is).")


def run_sweep(nfl_data: NflDataClient, season: int, param: str, values: list[str]) -> None:
    module = SWEEPABLE[param]
    attr = _ATTR_NAME[param]
    original = getattr(module, attr)
    games = _games_for(nfl_data, season)
    is_elo_param = module is elo_mod
    print(f"Sweeping {param} ({attr} on {module.__name__}) against season {season} "
          f"({len(games)} games); original value = {original}\n")
    try:
        for raw in values:
            val = float(raw)
            setattr(module, attr, val)
            if is_elo_param:
                elo = EloRatings(nfl_data)     # fresh instance -- constants read at _build() time
                rows = grade_winner(elo, games)
                print(f"  {attr}={val:<8} n={len(rows):<4} brier={round(_brier(rows), 4)} "
                      f"log_loss={round(_log_loss(rows), 4)} accuracy={round(_accuracy(rows), 3)}")
            else:
                env = ScoringEnvironmentModel(nfl_data)
                rows = grade_totals(env, games)
                r = totals_report(rows)
                print(f"  {attr}={val:<8} n={r.get('n')} bias={r.get('bias (actual - lam)')} "
                      f"rmse={r.get('rmse')} implied_phi={r.get('implied_phi (var(resid)/mean_lam)')}")
    finally:
        setattr(module, attr, original)    # never leave the module mutated after the script exits


# --------------------------------------------------------------------------- #
# --kalshi-settled: grade against real settled Kalshi NFL markets (preseason
# only so far -- see module docstring).
# --------------------------------------------------------------------------- #
def fetch_settled_markets(client: KalshiClient, series: str) -> list[dict]:
    """Full, uncapped pagination -- mirrors season_backtest.py's identical helper
    (KalshiClient.get_markets() caps at 40 pages / 8,000 markets by default)."""
    params = {"series_ticker": series, "status": "settled", "limit": 200}
    markets: list[dict] = []
    while True:
        data = client._get("/markets", params)
        markets.extend(data.get("markets", []) or [])
        cursor = data.get("cursor")
        if not cursor:
            break
        params["cursor"] = cursor
    return markets


def fetch_entry_price(client: KalshiClient, ticker: str, cutoff_ts: int) -> float | None:
    """Last traded YES price at/before cutoff_ts, our pregame-entry proxy
    (identical to season_backtest.py's helper)."""
    data = client._get("/markets/trades", {"ticker": ticker, "limit": 3, "max_ts": cutoff_ts})
    trades = data.get("trades") or []
    if not trades:
        return None
    try:
        return float(trades[0]["yes_price_dollars"])
    except (KeyError, ValueError, TypeError):
        return None


def grade_winner_kalshi(markets: list[dict], elo: EloRatings) -> list[dict]:
    out, seen_events = [], set()
    for m in markets:
        pt = parse_nfl_ticker(m["ticker"])
        if not pt or pt.event_ticker in seen_events:
            continue
        seen_events.add(pt.event_ticker)
        r_yes = elo.rating_before(pt.yes_team, pt.date_str)
        r_opp = elo.rating_before(pt.opponent, pt.date_str)
        pred = win_prob(r_yes, r_opp, a_is_home=pt.yes_is_home)
        actual = 1 if m.get("result") == "yes" else 0
        out.append({"pred": pred, "actual": actual, "ticker": m["ticker"],
                    "event": pt.event_ticker, "occurrence_datetime": m.get("occurrence_datetime")})
    return out


def grade_totals_kalshi(markets: list[dict], env: ScoringEnvironmentModel) -> list[dict]:
    out = []
    for m in markets:
        pt = parse_nfl_total_ticker(m["ticker"])
        if not pt:
            continue
        et = env.expected_points(pt.away_abbr, pt.home_abbr, pt.date_str, roof="")
        if et is None:
            continue
        prob_over = nb_survival(int(pt.line), et["lam"], fv_nfl_totals_mod.NFL_TOTALS_PHI)
        actual = 1 if m.get("result") == "yes" else 0
        out.append({"pred": prob_over, "actual": actual, "ticker": m["ticker"],
                    "event": pt.event_ticker, "occurrence_datetime": m.get("occurrence_datetime"),
                    "lam": et["lam"], "line": pt.line})
    return out


def _select_representative_totals(rows: list[dict]) -> list[dict]:
    """One line per game for the ROI pass -- the line closest to the model's own
    predicted total, the one actually near-the-money/tradeable in practice
    (mirrors season_backtest.py's identical helper)."""
    best: dict[str, dict] = {}
    for r in rows:
        k = r["event"]
        if k not in best or abs(r["line"] - r["lam"]) < abs(best[k]["line"] - best[k]["lam"]):
            best[k] = r
    return list(best.values())


PREGAME_BUFFER_MIN = 15.0   # cutoff before the market's own occurrence_datetime
MIN_EDGE = DEFAULTS.min_edge_cents / 100   # tracks live production, not a fixed snapshot of it


def compute_roi_kalshi(rows: list[dict], client: KalshiClient) -> dict:
    """Would betting the model's edge, at the real pregame trade price, at the
    production min-edge gate, have made money? Flat $1-notional stake per
    contract (season_backtest.py's convention). Unlike MLB's ticker-derived
    cutoff (a fixed UTC offset assumption), NFL's cutoff comes straight from
    the market's own occurrence_datetime -- more precise, and it's the only
    option since NFL tickers don't encode a kickoff time at all."""
    graded, no_time, no_price, no_edge = [], 0, 0, 0
    for r in rows:
        occ = r.get("occurrence_datetime")
        if not occ:
            no_time += 1
            continue
        cutoff = int((datetime.fromisoformat(occ.replace("Z", "+00:00"))
                      - timedelta(minutes=PREGAME_BUFFER_MIN)).timestamp())
        yes_price = fetch_entry_price(client, r["ticker"], cutoff)
        if yes_price is None:
            no_price += 1
            continue
        side = "YES" if r["pred"] > 0.5 else "NO"
        entry = yes_price if side == "YES" else round(1 - yes_price, 4)
        model_p = r["pred"] if side == "YES" else 1 - r["pred"]
        edge = model_p - entry
        if edge < MIN_EDGE:
            no_edge += 1
            continue
        yes_won = r["actual"] == 1
        won = yes_won if side == "YES" else not yes_won
        pnl = (1 - entry) if won else -entry
        graded.append({**r, "side": side, "entry": entry, "edge": edge, "won": won, "pnl": pnl})

    n = len(graded)
    wins = sum(1 for g in graded if g["won"])
    staked = sum(g["entry"] for g in graded)
    pnl_total = sum(g["pnl"] for g in graded)
    return {
        "n_bets": n, "no_kickoff_time": no_time, "no_price_data": no_price, "no_edge": no_edge,
        "win_rate": round(wins / n, 3) if n else None,
        "pnl_per_contract_total": round(pnl_total, 3),
        "roi": round(pnl_total / staked, 4) if staked else None, "rows": graded,
    }


def run_kalshi_settled_validation(no_roi: bool = False) -> None:
    client = KalshiClient(DEFAULTS)
    nfl_data = NflDataClient()
    elo = EloRatings(nfl_data)
    env = ScoringEnvironmentModel(nfl_data)

    print("Fetching settled Kalshi NFL markets (all preseason so far -- see module "
          "docstring)...")
    winner_markets = fetch_settled_markets(client, "KXNFLGAME")
    total_markets = fetch_settled_markets(client, "KXNFLTOTAL")
    print(f"  {len(winner_markets)} winner market-sides, {len(total_markets)} total-line markets.\n")

    winner_rows = grade_winner_kalshi(winner_markets, elo)
    print(f"=== WINNER ({len(winner_rows)} games, real Kalshi settlement) ===")
    print(f"  brier    : {round(_brier(winner_rows), 4) if winner_rows else None}")
    print(f"  log_loss : {round(_log_loss(winner_rows), 4) if winner_rows else None}")
    print(f"  accuracy : {round(_accuracy(winner_rows), 3) if winner_rows else None}")
    print("  calibration:")
    for row in _calibration_table(winner_rows):
        print(f"    {row['range']:>8}  n={row['n']:<4} mean_pred={row['mean_pred']:<6} "
              f"actual_rate={row['actual_rate']}")

    totals_rows = grade_totals_kalshi(total_markets, env)
    print(f"\n=== TOTALS ({len(totals_rows)} lines, real Kalshi settlement) ===")
    print(f"  brier    : {round(_brier(totals_rows), 4) if totals_rows else None}")
    print(f"  log_loss : {round(_log_loss(totals_rows), 4) if totals_rows else None}")
    print(f"  accuracy : {round(_accuracy(totals_rows), 3) if totals_rows else None}")

    if no_roi:
        return

    print(f"\nFetching real pregame trade prices for the ROI pass "
          f"({PREGAME_BUFFER_MIN:.0f}min-before-kickoff cutoff, "
          f"{MIN_EDGE*100:.1f}c min-edge gate, matches live production)...")
    winner_roi = compute_roi_kalshi(winner_rows, client)
    totals_repr = _select_representative_totals(totals_rows)
    print(f"  totals: {len(totals_rows)} lines -> {len(totals_repr)} representative (1/game)")
    totals_roi = compute_roi_kalshi(totals_repr, client)
    combined_rows = winner_roi["rows"] + totals_roi["rows"]
    combined_pnl = sum(g["pnl"] for g in combined_rows)
    combined_staked = sum(g["entry"] for g in combined_rows)

    print(f"\n=== ROI (real pregame trade prices, {MIN_EDGE*100:.1f}c edge gate, flat $1/contract) ===")
    for label, roi in (("WINNER", winner_roi), ("TOTALS", totals_roi)):
        print(f"--- {label} ---")
        print(f"  candidates            : {roi['n_bets'] + roi['no_price_data'] + roi['no_edge'] + roi['no_kickoff_time']}")
        print(f"  no kickoff time       : {roi['no_kickoff_time']}")
        print(f"  no pregame trade data : {roi['no_price_data']}")
        print(f"  below {MIN_EDGE*100:.1f}c edge (no bet): {roi['no_edge']}")
        print(f"  bets placed           : {roi['n_bets']}")
        print(f"  win rate              : {roi['win_rate']}")
        print(f"  pnl (per $1 contract) : {roi['pnl_per_contract_total']}")
        print(f"  roi                   : {roi['roi']}")
        print()
    print("--- COMBINED ---")
    print(f"  bets placed           : {len(combined_rows)}")
    print(f"  pnl (per $1 contract) : {round(combined_pnl, 3)}")
    print(f"  roi                   : {round(combined_pnl / combined_staked, 4) if combined_staked else None}")
    print("\nCaveat: preseason only (see module docstring) -- treat this as a plumbing/\n"
          "calibration check, not a verdict on the model's regular-season edge.")


def main() -> None:
    ap = argparse.ArgumentParser(description="Leak-safe backtest of the NFL Elo/scoring-rate model")
    ap.add_argument("--validate-season", type=int, default=2025,
                    help="NFL season to grade predictions against (default: last full season, 2025)")
    ap.add_argument("--sweep", nargs="+", metavar=("PARAM", "VALUES"),
                    help=f"sweep one parameter: --sweep <param> <v1> <v2> ... "
                         f"(param in {sorted(SWEEPABLE)})")
    ap.add_argument("--kalshi-settled", action="store_true",
                    help="grade against real settled Kalshi NFL markets instead of nflverse "
                         "historical seasons (preseason-only data so far, see module docstring)")
    ap.add_argument("--no-roi", action="store_true",
                    help="with --kalshi-settled, skip the real-money ROI pass (fetches a "
                         "pregame trade price per game)")
    args = ap.parse_args()

    if args.kalshi_settled:
        run_kalshi_settled_validation(no_roi=args.no_roi)
        return

    nfl_data = NflDataClient()

    if args.sweep:
        param, *values = args.sweep
        if param not in SWEEPABLE:
            print(f"Unknown sweep param {param!r}. Choose from: {sorted(SWEEPABLE)}")
            return
        if not values:
            print("Provide at least one value to sweep, e.g. --sweep k_factor 10 15 20 25 30")
            return
        run_sweep(nfl_data, args.validate_season, param, values)
        return

    run_validation(nfl_data, args.validate_season)


if __name__ == "__main__":
    main()
