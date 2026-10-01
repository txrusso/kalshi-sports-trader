"""Nightly NHL totals recalibration.

Re-runs the walk-forward sweep that shipped on 2026-09-30 (see CLAUDE.md,
"NHL TOTALS RECALIBRATED") against every completed regular-season game
available, including the current season as it accumulates, and -- with
--apply -- promotes a setting that beats the live one on BOTH splits.

    .venv\\Scripts\\python.exe -m backtest.nhl_recalibrate --apply

WHAT IT SWEEPS
  data/scoring_environment_nhl.py::TEAM_SIGNAL_SHRINK  (how much of the
      per-game team-strength signal to keep; measured at ~0.155 +- 0.092)
  data/fair_value_nhl_totals.py::NHL_TOTALS_DISPERSION (how far to contract
      the tie-break-aware goal distribution around its mean)

THE BAR IS DELIBERATELY STRICT, and it is the project's own: a candidate
must beat the LIVE calibration on the training split AND the held-out split.
That is what every production constant here moved on -- never a single split,
never a full-sample result (the reverted winner_regress, Pythagorean, weather
and QB-injury experiments all died on exactly this bar). Among candidates that
clear it, this picks the PLATEAU MIDDLE rather than the single best point:
the shipped max_stake_pct, in_game_max_minutes and NBA HOME_FIELD_ELO were all
chosen that way, because a knife-edge peak on a few thousand games is usually
noise.

EXPECT IT TO CHANGE NOTHING FOR WEEKS. The sweep runs over ~2,600 games and a
night of NHL adds roughly 5-12, so the optimum will not move quickly. A job
that reports "no change" most nights is the job working, not failing.

WHAT IT DOES NOT DO: touch NFL, or re-enable NHL for real money. NHL is
shadow-traded (config/settings.py::live_trade_disabled_sports); this only
sharpens the model whose simulated record will eventually answer whether that
should change. Nothing here can place an order.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import statistics as st
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import nhl_calibration                                  # noqa: E402
from config.settings import OUTPUT_DIR, PROJECT_ROOT, EASTERN       # noqa: E402
from data.distributions import overtime_total_survival              # noqa: E402
import data.scoring_environment_nhl as senv                         # noqa: E402
from data.nhl_data import NhlDataClient                             # noqa: E402
from data.scoring_environment_nhl import ScoringEnvironmentModel    # noqa: E402

REPORT_DIR = PROJECT_ROOT / "backtest"
LINES = [3.5, 4.5, 5.5, 6.5, 7.5]
SHRINK_GRID = [0.0, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40]
DISPERSION_GRID = [0.95, 0.90, 0.85, 0.80, 0.75, 0.70]
MIN_VALIDATE_GAMES = 200      # below this a "held-out split" is not one


def _rows_for(shrink: float, games) -> dict:
    """Per-game (e_away, e_home, actual) at a given shrink, by season.
    Rebuilt per shrink because it rescales the per-side expectations."""
    # Passed explicitly, NOT by setting senv.TEAM_SIGNAL_SHRINK: production
    # reads the live calibration store, so mutating the module constant would
    # silently sweep nothing and every value would score identically.
    env = ScoringEnvironmentModel(NhlDataClient())
    out: dict[int, list] = {}
    for g in games:
        et = env.expected_goals(g.away_abbr, g.home_abbr, g.date_str, shrink=shrink)
        if et:
            out.setdefault(g.season, []).append(
                (et["e_away"], et["e_home"], g.total_goals))
    return out


def _brier(rows, disp: float) -> float:
    total, n = 0.0, 0
    for ea, eh, actual in rows:
        for line in LINES:
            p = overtime_total_survival(int(line), ea, eh, disp)
            total += (p - (1.0 if actual > line else 0.0)) ** 2
            n += 1
    return total / n if n else float("nan")


def _calibration_table(rows, disp: float) -> list[tuple]:
    out = []
    for line in LINES:
        preds = [overtime_total_survival(int(line), ea, eh, disp) for ea, eh, _ in rows]
        act = [1.0 if a > line else 0.0 for _, _, a in rows]
        p, a = st.mean(preds), st.mean(act)
        se = 2 * (a * (1 - a) / len(act)) ** 0.5 if act else 0.0
        out.append((line, len(act), p, a, p - a, se))
    return out


def _signal_strength(rows) -> tuple:
    """OLS slope of actual total on the model's own expectation -- the number
    that showed the per-game signal was ~zero in the first place."""
    lams = [ea + eh for ea, eh, _ in rows]
    acts = [a for _, _, a in rows]
    if len(rows) < 30 or st.pstdev(lams) == 0:
        return (float("nan"),) * 4
    mx, my = st.mean(lams), st.mean(acts)
    cov = sum((l - mx) * (a - my) for l, a in zip(lams, acts)) / len(rows)
    sdl, sda = st.pstdev(lams), st.pstdev(acts)
    slope = cov / st.pvariance(lams)
    se = sda / sdl / len(rows) ** 0.5
    rmse = (sum((l - a) ** 2 for l, a in zip(lams, acts)) / len(rows)) ** 0.5
    base = (sum((my - a) ** 2 for a in acts) / len(acts)) ** 0.5
    return slope, se, cov / (sdl * sda), rmse - base


def _plateau_pick(winners: list[tuple]) -> tuple:
    """Middle of the widest flat shelf, not the single best point. `winners`
    is [(validate, train, shrink, dispersion)], already both-splits wins."""
    best_v = min(w[0] for w in winners)
    shelf = [w for w in winners if w[0] <= best_v + 0.0002]     # within noise
    shrinks = sorted({w[2] for w in shelf})
    disps = sorted({w[3] for w in shelf})
    want_s, want_d = shrinks[len(shrinks) // 2], disps[len(disps) // 2]
    # the shelf member closest to that (shrink, dispersion) centre
    return min(shelf, key=lambda w: (abs(w[2] - want_s), abs(w[3] - want_d)))


def grade_shadow_ledger() -> dict:
    """Simulated NHL record. Never touches real P&L -- see
    engine/paper.py::ShadowLedger for why this is a separate file."""
    from data.games import build_clients, resolve_outcomes
    from engine.paper import ShadowLedger
    bets = ShadowLedger().load()
    if not bets:
        return {"n": 0, "settled": 0}
    try:
        outcomes = resolve_outcomes({b["ticker"] for b in bets}, build_clients())
    except Exception:
        return {"n": len(bets), "settled": 0, "error": "could not resolve outcomes"}
    w = l = 0
    pnl = wagered = 0.0
    by_ver: Counter = Counter()
    for b in bets:
        o = outcomes.get(b["ticker"])
        if o is None:
            continue
        won = (o == (b["side"] == "YES"))
        wager = float(b.get("wager_usd") or 0.0)
        ct = float(b.get("contracts") or 0.0)
        pnl += (ct - wager) if won else -wager
        wagered += wager
        w, l = (w + 1, l) if won else (w, l + 1)
        by_ver[b.get("model_version", "?")] += 1
    return {"n": len(bets), "settled": w + l, "w": w, "l": l, "pnl": pnl,
            "wagered": wagered, "roi": (pnl / wagered) if wagered else 0.0,
            "by_version": dict(by_ver)}


def run(apply: bool = False, quiet: bool = False) -> dict:
    client = NhlDataClient()
    client.clear_cache()                       # pick up last night's finals
    games = [g for g in client.games() if g.game_type == 2 and g.state == "Final"]
    by_season = Counter(g.season for g in games)
    seasons = sorted(by_season)
    out = {"ran_at": dt.datetime.now(EASTERN).isoformat(timespec="seconds"),
           "games": len(games), "by_season": dict(by_season)}

    if len(seasons) < 2:
        out["status"] = "not enough seasons to split"
        return out

    validate_season = seasons[-1]
    if by_season[validate_season] < MIN_VALIDATE_GAMES:
        # The current season is too young to hold out against; fall back to
        # the last two COMPLETED seasons so the job still reports a real
        # measurement instead of a meaningless one.
        validate_season = seasons[-2] if by_season[seasons[-2]] >= MIN_VALIDATE_GAMES else None
    if validate_season is None:
        out["status"] = "no season large enough to validate on"
        return out
    train_season = max(s for s in seasons if s < validate_season) \
        if any(s < validate_season for s in seasons) else None
    if train_season is None:
        out["status"] = "no season before the validate split"
        return out
    out.update(train_season=train_season, validate_season=validate_season,
               n_train=by_season[train_season], n_validate=by_season[validate_season])

    live = nhl_calibration.load()
    live_s, live_d = float(live["team_signal_shrink"]), float(live["dispersion"])

    cache: dict[float, dict] = {}

    def rows(shrink, season):
        if shrink not in cache:
            cache[shrink] = _rows_for(shrink, games)
        return cache[shrink].get(season, [])

    base_tr = _brier(rows(live_s, train_season), live_d)
    base_va = _brier(rows(live_s, validate_season), live_d)
    out.update(live_version=live["version"], live_shrink=live_s, live_dispersion=live_d,
               live_train_brier=base_tr, live_validate_brier=base_va)

    winners, grid = [], []
    for s in SHRINK_GRID:
        for d in DISPERSION_GRID:
            tr = _brier(rows(s, train_season), d)
            va = _brier(rows(s, validate_season), d)
            grid.append((s, d, tr, va))
            if tr < base_tr and va < base_va:
                winners.append((va, tr, s, d))
    out["grid"] = grid
    out["n_both_splits_winners"] = len(winners)

    slope, se, corr, rmse_gap = _signal_strength(rows(1.0, validate_season))
    out["signal"] = {"slope": slope, "se": se, "corr": corr,
                     "rmse_minus_baseline": rmse_gap}

    if not winners:
        out["status"] = "no change -- live calibration still best on both splits"
        out["chosen"] = {"team_signal_shrink": live_s, "dispersion": live_d}
    else:
        va, tr, s, d = _plateau_pick(winners)
        out["status"] = "candidate beats live on both splits"
        out["chosen"] = {"team_signal_shrink": s, "dispersion": d,
                         "train_brier": tr, "validate_brier": va,
                         "train_gain": base_tr - tr, "validate_gain": base_va - va}
        if apply:
            ver = dt.datetime.now(EASTERN).strftime("%Y-%m-%d")
            if ver == live["version"]:
                ver += "b"
            saved = nhl_calibration.save(
                s, d, ver, source="backtest/nhl_recalibrate.py",
                train_season=train_season, validate_season=validate_season,
                n_train=by_season[train_season], n_validate=by_season[validate_season],
                train_brier=round(tr, 6), validate_brier=round(va, 6),
                previous={"version": live["version"], "team_signal_shrink": live_s,
                          "dispersion": live_d})
            out["applied"] = saved
            out["status"] = f"APPLIED -> version {ver}"

    eff = out["chosen"]
    out["calibration_table"] = _calibration_table(
        rows(eff["team_signal_shrink"], validate_season), eff["dispersion"])
    out["shadow"] = grade_shadow_ledger()
    if not quiet:
        print(render(out))
    return out


def render(r: dict) -> str:
    L = ["=" * 78, "NHL TOTALS NIGHTLY RECALIBRATION", "=" * 78,
         f"ran            : {r.get('ran_at')}",
         f"games          : {r.get('games')}  {r.get('by_season')}"]
    if "train_season" in r:
        L += [f"train split    : {r['train_season']}  (n={r['n_train']})",
              f"validate split : {r['validate_season']}  (n={r['n_validate']})", "",
              f"live           : v{r['live_version']}  shrink={r['live_shrink']} "
              f"dispersion={r['live_dispersion']}",
              f"                 train Brier {r['live_train_brier']:.5f}  "
              f"validate {r['live_validate_brier']:.5f}"]
    s = r.get("signal") or {}
    if s and s.get("slope") == s.get("slope"):
        L += ["", "per-game team-strength signal (held-out, unshrunk):",
              f"  OLS slope {s['slope']:+.3f} +- {s['se']:.3f}   corr {s['corr']:+.4f}"
              f"   RMSE vs league-avg baseline {s['rmse_minus_baseline']:+.3f}"
              f"  ({'beats' if s['rmse_minus_baseline'] < 0 else 'LOSES TO'} baseline)"]
    L += ["", f"both-splits winners in grid: {r.get('n_both_splits_winners')}",
          f"STATUS: {r.get('status')}"]
    c = r.get("chosen") or {}
    if "train_gain" in c:
        L += [f"  chosen  shrink={c['team_signal_shrink']} dispersion={c['dispersion']}"
              f"   train {c['train_gain']:+.5f}  validate {c['validate_gain']:+.5f}"]
    if r.get("applied"):
        L.append(f"  WROTE {nhl_calibration.CALIBRATION_PATH}")
    ct = r.get("calibration_table") or []
    if ct:
        L += ["", "held-out calibration at the effective setting:",
              f"  {'line':>5} {'n':>5} {'pred':>8} {'actual':>8} {'err':>8} {'2se':>7}"]
        for line, n, p, a, diff, se in ct:
            flag = "" if abs(diff) <= se else "   <-- outside 2se"
            L.append(f"  {line:>5} {n:>5} {p:>8.4f} {a:>8.4f} {diff:>+8.4f} {se:>7.4f}{flag}")
    sh = r.get("shadow") or {}
    L += ["", f"shadow ledger (SIMULATED, no money): {sh.get('n', 0)} bets, "
              f"{sh.get('settled', 0)} settled"]
    if sh.get("settled"):
        L.append(f"  {sh['w']}-{sh['l']}   net ${sh['pnl']:+.2f} on ${sh['wagered']:.2f}"
                 f"   ROI {sh['roi']:+.1%}   by model version: {sh.get('by_version')}")
    L.append("=" * 78)
    return "\n".join(L)


def shadow_summary_line() -> str:
    """One-line NHL shadow record for the nightly settle text. Always marked
    SIM so it can never be read as real money."""
    try:
        sh = grade_shadow_ledger()
    except Exception:
        return ""
    if not sh.get("settled"):
        return f"[NHL SIM {sh.get('n', 0)} pending]" if sh.get("n") else ""
    return f"[NHL SIM {sh['w']}-{sh['l']} ${sh['pnl']:+.2f} ROI {sh['roi']:+.0%} (no money)]"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--apply", action="store_true",
                    help="promote a candidate that beats the live calibration on BOTH splits")
    ap.add_argument("--json", action="store_true", help="print the raw result as JSON")
    ap.add_argument("--no-report", action="store_true", help="don't write the dated report file")
    a = ap.parse_args()
    r = run(apply=a.apply, quiet=a.json)
    if a.json:
        print(json.dumps(r, indent=2, default=str))
    if not a.no_report:
        p = REPORT_DIR / f"nhl_recalibrate_{dt.datetime.now(EASTERN):%Y%m%d}.txt"
        p.write_text(render(r), encoding="utf-8")
        print(f"\nreport -> {p}")


if __name__ == "__main__":
    main()
