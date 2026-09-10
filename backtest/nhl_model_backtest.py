"""Leak-safe backtest of the NHL Elo/scoring-rate model against real historical
game outcomes -- the NHL analog of backtest/nba_model_backtest.py, adapted for
the NHL model's inputs (signals/elo_nhl.py, data/scoring_environment_nhl.py).

EloRatings.rating_before() and data/nhl_data.py's team_games_before() are
leak-safe by construction: every query filters to games strictly before the
target date, from already-final historical results.

No --kalshi-settled mode yet: no KXNHLGAME/KXNHLTOTAL market has been open to
check, let alone settled, as of 2026-09-10 (this far from the season). Add
that mode once real settled NHL markets exist, mirroring
nfl_model_backtest.py's/nba_model_backtest.py's structure.

Usage:
    py -3 -m backtest.nhl_model_backtest
    py -3 -m backtest.nhl_model_backtest --validate-season 2024
    py -3 -m backtest.nhl_model_backtest --sweep k_factor 10 15 20 25 30
    py -3 -m backtest.nhl_model_backtest --sweep home_ice_elo 0 15 30 45 60
    py -3 -m backtest.nhl_model_backtest --sweep season_blend_games 4 8 12 16
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import data.scoring_environment_nhl as scoring_env_mod
import signals.elo_nhl as elo_mod
from data.nhl_data import NhlDataClient, NhlGame, GAME_TYPE_REGULAR
from data.scoring_environment_nhl import ScoringEnvironmentModel
from signals.elo_nhl import EloRatings, win_prob

SWEEPABLE = {
    "k_factor": elo_mod, "home_ice_elo": elo_mod, "season_regression": elo_mod,
    "season_blend_games": scoring_env_mod,
}
_ATTR_NAME = {
    "k_factor": "K_FACTOR", "home_ice_elo": "HOME_ICE_ELO",
    "season_regression": "SEASON_REGRESSION", "season_blend_games": "SEASON_BLEND_GAMES",
}


def grade_winner(elo: EloRatings, games: list[NhlGame]) -> list[dict]:
    out = []
    for g in games:
        if g.state != "Final" or g.home_score == g.away_score:
            continue    # a tie is impossible in the NHL's real rules (OT/SO), so this is just defensive
        r_home = elo.rating_before(g.home_abbr, g.date_str)
        r_away = elo.rating_before(g.away_abbr, g.date_str)
        pred = win_prob(r_home, r_away, a_is_home=True)
        actual = 1 if g.home_score > g.away_score else 0
        out.append({"date": g.date_str, "pred": pred, "actual": actual, "game_id": g.game_id})
    return out


def grade_totals(env: ScoringEnvironmentModel, games: list[NhlGame]) -> list[dict]:
    out = []
    for g in games:
        if g.state != "Final":
            continue
        et = env.expected_goals(g.away_abbr, g.home_abbr, g.date_str)
        if et is None:
            continue
        out.append({"date": g.date_str, "lam": et["lam"], "actual": g.total_goals,
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


def _games_for(nhl_data: NhlDataClient, season: int) -> list[NhlGame]:
    return [g for g in nhl_data.games()
            if g.season == season and g.state == "Final" and g.game_type == GAME_TYPE_REGULAR]


def run_validation(nhl_data: NhlDataClient, season: int) -> None:
    games = _games_for(nhl_data, season)
    print(f"Validating season {season}-{str(season+1)[2:]}: {len(games)} completed regular-season games.\n")

    elo = EloRatings(nhl_data)
    winner_rows = grade_winner(elo, games)
    print("=== WINNER (Elo win probability) ===")
    print(f"  n                : {len(winner_rows)}")
    print(f"  brier            : {round(_brier(winner_rows), 4) if winner_rows else None}")
    print(f"  log_loss         : {round(_log_loss(winner_rows), 4) if winner_rows else None}")
    print(f"  accuracy         : {round(_accuracy(winner_rows), 3) if winner_rows else None}")
    coinflip = [{"pred": 0.5, "actual": r["actual"]} for r in winner_rows]
    home_baseline = [{"pred": 0.54, "actual": r["actual"]} for r in winner_rows]  # ~league home-win rate
    print(f"  baseline: always-0.5 brier      : {round(_brier(coinflip), 4) if coinflip else None}")
    print(f"  baseline: always-home-54% brier : {round(_brier(home_baseline), 4) if home_baseline else None}")
    print("\n  calibration:")
    for row in _calibration_table(winner_rows):
        print(f"    {row['range']:>8}  n={row['n']:<4} mean_pred={row['mean_pred']:<6} "
              f"actual_rate={row['actual_rate']}")

    env = ScoringEnvironmentModel(nhl_data)
    totals_rows = grade_totals(env, games)
    print(f"\n=== TOTALS (expected goals, no real Kalshi line to test P(over) against yet) ===")
    for k, v in totals_report(totals_rows).items():
        print(f"  {k:42}: {v}")
    print("\n  Guide: rmse < naive_rmse means the model beats guessing the league average;\n"
          "  compare 'implied_phi' above to NHL_TOTALS_PHI in data/fair_value_nhl_totals.py\n"
          "  (currently seeded from a raw 4-season in-sample measurement, not yet\n"
          "  validated out-of-sample the way this number is).")


def run_sweep(nhl_data: NhlDataClient, season: int, param: str, values: list[str]) -> None:
    module = SWEEPABLE[param]
    attr = _ATTR_NAME[param]
    original = getattr(module, attr)
    games = _games_for(nhl_data, season)
    is_elo_param = module is elo_mod
    print(f"Sweeping {param} ({attr} on {module.__name__}) against season {season} "
          f"({len(games)} games); original value = {original}\n")
    try:
        for raw in values:
            val = float(raw)
            setattr(module, attr, val)
            if is_elo_param:
                elo = EloRatings(nhl_data)     # fresh instance -- constants read at _build() time
                rows = grade_winner(elo, games)
                print(f"  {attr}={val:<8} n={len(rows):<4} brier={round(_brier(rows), 4)} "
                      f"log_loss={round(_log_loss(rows), 4)} accuracy={round(_accuracy(rows), 3)}")
            else:
                env = ScoringEnvironmentModel(nhl_data)
                rows = grade_totals(env, games)
                r = totals_report(rows)
                print(f"  {attr}={val:<8} n={r.get('n')} bias={r.get('bias (actual - lam)')} "
                      f"rmse={r.get('rmse')} implied_phi={r.get('implied_phi (var(resid)/mean_lam)')}")
    finally:
        setattr(module, attr, original)    # never leave the module mutated after the script exits


def main() -> None:
    ap = argparse.ArgumentParser(description="Leak-safe backtest of the NHL Elo/scoring-rate model")
    ap.add_argument("--validate-season", type=int, default=2024,
                    help="NHL season (starting year) to grade predictions against "
                         "(default: last full season, 2024 = 2024-25)")
    ap.add_argument("--sweep", nargs="+", metavar=("PARAM", "VALUES"),
                    help=f"sweep one parameter: --sweep <param> <v1> <v2> ... "
                         f"(param in {sorted(SWEEPABLE)})")
    ap.add_argument("--history-seasons", type=int, default=4,
                    help="how many season files to load for Elo ratings depth (default 4)")
    args = ap.parse_args()

    nhl_data = NhlDataClient(history_seasons=args.history_seasons)

    if args.sweep:
        param, *values = args.sweep
        if param not in SWEEPABLE:
            print(f"Unknown sweep param {param!r}. Choose from: {sorted(SWEEPABLE)}")
            return
        if not values:
            print("Provide at least one value to sweep, e.g. --sweep k_factor 10 15 20 25 30")
            return
        run_sweep(nhl_data, args.validate_season, param, values)
        return

    run_validation(nhl_data, args.validate_season)


if __name__ == "__main__":
    main()
