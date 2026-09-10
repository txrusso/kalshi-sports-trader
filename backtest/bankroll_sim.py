"""Walk-forward, sizing-aware backtest: a real $-denominated Kelly bankroll
simulation, not just calibration (that's backtest/season_backtest.py).

Built 2026-08-13 to answer "maximize ROI" properly, after backtest/season_backtest.py
showed calibration/flat-$1 ROI but nothing about compounding or risk. Two things this
adds that season_backtest.py doesn't:

1. A TRAIN/VALIDATE split (train = 2026-06-07 to TRAIN_END, validate = the rest) so
   parameter changes are judged on held-out data, not the same sample they were picked
   from. This caught a real overfit: a Pythagorean pitching-matchup blend on the winner
   model looked like a genuine improvement on a single full-sample test (Brier 0.2500 ->
   0.2475) but validation performance got monotonically WORSE as its weight increased --
   best at weight 0, i.e. no blend at all. See the sweep this module ran on 2026-08-13
   in the session history; the blend was reverted from data/fair_value.py as a result.

2. A day-batched Kelly bankroll simulation (starting at $20, matching the live account)
   using REAL pregame trade prices, not a flat $1/contract convention -- this is what
   "would this have grown the account" actually means, and it's what surfaced that
   kelly_fraction=0.35 carried a 50-70%+ max drawdown with no better held-out ROI than
   a lower fraction (Kelly overbetting given real estimation error in the edge).

Caching: the expensive part (schedule ledger, pitcher gameLogs, and ~1,700 real pregame
trade-price lookups) is fetched once and cached to bankroll_sim_cache.json next to this
file. Re-run with --refresh to rebuild it (e.g. once enough new settled markets exist to
be worth including). Every parameter sweep after that is pure in-memory computation --
seconds, not the ~10 minute network-bound cost of building the cache.

Usage:
    py -3 -m backtest.bankroll_sim                      # report current production params
    py -3 -m backtest.bankroll_sim --refresh             # rebuild the cache first
    py -3 -m backtest.bankroll_sim --sweep kelly_fraction 0.15 0.20 0.25 0.30 0.35 0.45
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backtest.season_backtest import (
    SeasonLedger, PitcherLogs, fetch_settled_markets, fetch_entry_price, KALSHI_DATA_START,
    GAME_UTC_OFFSET_HOURS, PREGAME_BUFFER_MIN, _shift,
)
from data.fair_value import parse_ticker, log5, _apply_home_field, _regress_win_pct, WINNER_REGRESS, ELO_WEIGHT
from data.fair_value_totals import parse_total_ticker, park_factor, SP_WEIGHT, PITCHER_REG_IP, LEAGUE_SHRINK, TOTALS_PHI
from data.mlb_stats import RECENT_WINDOW_GAMES, fetch_historical_games
from kalshi.client import KalshiClient
from config.settings import DEFAULTS
from signals.elo_mlb import EloRatings, win_prob as elo_win_prob

CACHE_PATH = Path(__file__).resolve().parent / "bankroll_sim_cache.json"
ELO_CACHE_PATH = Path(__file__).resolve().parent / "elo_mlb_cache.json"
TRAIN_END = "2026-07-15"   # inclusive. train = KALSHI_DATA_START..TRAIN_END, validate = after
HOME_FIELD_ODDS_MULT = 0.54 / 0.46
# ~3.5 seasons before the earliest backtest date (2026-06-07) so team Elo ratings have
# converged well before the walk-forward window opens, not still sitting near 1500.
ELO_HISTORY_START = "2023-01-01"

DEFAULT_PARAMS = {
    "sp_weight": SP_WEIGHT, "pitcher_reg_ip": PITCHER_REG_IP, "league_shrink": LEAGUE_SHRINK,
    "totals_phi": TOTALS_PHI, "min_edge": DEFAULTS.min_edge_cents / 100,
    "kelly_fraction": DEFAULTS.kelly_fraction, "max_stake_pct": DEFAULTS.max_stake_pct,
    "winner_regress": WINNER_REGRESS,
    "sp_winner_beta": 0.0,   # starting-pitcher matchup weight on the WINNER side (log-odds
                              # per run-per-9 of starter advantage vs each team's own
                              # baseline). 0.0 = production: the winner model uses no
                              # pitcher information at all, unlike the totals model.
    "elo_weight": ELO_WEIGHT,   # pulled from data/fair_value.py so this stays in sync with
                                # production (was hardcoded 0.0 / stale "current production"
                                # comment until 2026-09-02, even after ELO_WEIGHT shipped at
                                # 0.2 on 2026-09-01 -- the baseline report wasn't actually
                                # testing production. 0 = pure log5; 1 = pure Elo; blended between.
}


# --- cache build (network-bound; run once, reused by every sweep) ----------------

def build_cache(path: Path = CACHE_PATH) -> dict:
    session = requests.Session()
    session.headers.update({"User-Agent": "kalshi-agent research"})
    season_end = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    print("Building ledger...")
    ledger = SeasonLedger(session, "2026-03-01", season_end, recency_window=9999)
    plogs = PitcherLogs(session)
    client = KalshiClient(DEFAULTS)

    print("Fetching settled markets...")
    winner_markets = [m for m in fetch_settled_markets(client, "KXMLBGAME")
                       if (m.get("close_time") or "") >= KALSHI_DATA_START]
    total_markets = [m for m in fetch_settled_markets(client, "KXMLBTOTAL")
                      if (m.get("close_time") or "") >= KALSHI_DATA_START]
    print(f"  {len(winner_markets)} winner sides, {len(total_markets)} total lines")

    def game_ingredients(game, date_str):
        away_rate = ledger.entering_run_rate(game["away_abbr"], date_str)
        home_rate = ledger.entering_run_rate(game["home_abbr"], date_str)
        away_wp = ledger.entering_win_pct(game["away_abbr"], date_str)
        home_wp = ledger.entering_win_pct(game["home_abbr"], date_str)
        if not away_rate or not home_rate or away_wp is None or home_wp is None:
            return None
        lg_total = ledger.league_avg_total(date_str)
        home_sp = plogs.ra9_entering(game["home_pitcher_id"], date_str) if game["home_pitcher_id"] else None
        away_sp = plogs.ra9_entering(game["away_pitcher_id"], date_str) if game["away_pitcher_id"] else None
        return {
            "away_abbr": game["away_abbr"], "home_abbr": game["home_abbr"],
            "away_rs_pg": away_rate[0], "away_ra_pg": away_rate[1],
            "home_rs_pg": home_rate[0], "home_ra_pg": home_rate[1],
            "away_win_pct": away_wp, "home_win_pct": home_wp,
            "lg_team": lg_total / 2.0, "park_factor": park_factor(game["home_abbr"]),
            "home_sp_ra9": home_sp[0] if home_sp else None, "home_sp_ip": home_sp[1] if home_sp else None,
            "away_sp_ra9": away_sp[0] if away_sp else None, "away_sp_ip": away_sp[1] if away_sp else None,
        }

    print("Building winner rows...")
    games_cache, winner_rows, seen = {}, [], set()
    for m in winner_markets:
        pt = parse_ticker(m["ticker"])
        if not pt or pt.event_ticker in seen or not pt.opponent:
            continue
        seen.add(pt.event_ticker)
        date_str = pt.date.strftime("%Y-%m-%d")
        game = ledger.find_game(date_str, pt.yes_team, pt.opponent)
        if not game:
            continue
        ing = game_ingredients(game, date_str)
        if ing is None:
            continue
        games_cache[pt.event_ticker] = ing
        winner_rows.append({"event": pt.event_ticker, "ticker": m["ticker"], "date": date_str,
                             "game_dt": pt.date.isoformat(), "yes_is_home": game["home_abbr"] == pt.yes_team,
                             "actual": 1 if m.get("result") == "yes" else 0})
    print(f"  {len(winner_rows)} winner rows with ingredients")

    print("Building totals rows (representative line per game)...")
    totals_by_event = {}

    def lam_only(ing):
        lg_team = ing["lg_team"]

        def eff(team_ra_pg, sp_ra9, sp_ip):
            if sp_ra9 is None:
                return team_ra_pg
            reliab = sp_ip / (sp_ip + PITCHER_REG_IP)
            sp_reg = reliab * sp_ra9 + (1 - reliab) * lg_team
            return SP_WEIGHT * sp_reg + (1 - SP_WEIGHT) * team_ra_pg

        home_eff = eff(ing["home_ra_pg"], ing["home_sp_ra9"], ing["home_sp_ip"])
        away_eff = eff(ing["away_ra_pg"], ing["away_sp_ra9"], ing["away_sp_ip"])
        e_away = ing["away_rs_pg"] * (home_eff / lg_team)
        e_home = ing["home_rs_pg"] * (away_eff / lg_team)
        base = (1 - LEAGUE_SHRINK) * (e_away + e_home) + LEAGUE_SHRINK * (2 * lg_team)
        return base * ing["park_factor"]

    for m in total_markets:
        pt = parse_total_ticker(m["ticker"])
        if not pt:
            continue
        date_str, game = pt.date.strftime("%Y-%m-%d"), None
        for d in (pt.date.strftime("%Y-%m-%d"), _shift(pt.date.strftime("%Y-%m-%d"), -1),
                  _shift(pt.date.strftime("%Y-%m-%d"), 1)):
            for g in ledger.games_by_date.get(d, []):
                if pt.teams in (g["away_abbr"] + g["home_abbr"], g["home_abbr"] + g["away_abbr"]):
                    game = g
                    break
            if game:
                date_str = d
                break
        if not game:
            continue
        ing = games_cache.get(pt.event_ticker) or game_ingredients(game, date_str)
        if ing is None:
            continue
        games_cache[pt.event_ticker] = ing
        lam = lam_only(ing)
        row = {"event": pt.event_ticker, "ticker": m["ticker"], "date": date_str,
               "game_dt": pt.date.isoformat(), "line": pt.line,
               "actual": 1 if m.get("result") == "yes" else 0, "lam": lam}
        cur = totals_by_event.get(pt.event_ticker)
        if cur is None or abs(row["line"] - lam) < abs(cur["line"] - cur["lam"]):
            totals_by_event[pt.event_ticker] = row
    totals_rows = list(totals_by_event.values())
    print(f"  {len(totals_rows)} totals rows (1/game)")

    print(f"Fetching entry prices for {len(winner_rows)} winner + {len(totals_rows)} totals tickers...")

    def entry_price_for(ticker, game_dt_iso):
        game_dt = datetime.fromisoformat(game_dt_iso)
        cutoff = int((game_dt.replace(tzinfo=timezone.utc) + timedelta(hours=GAME_UTC_OFFSET_HOURS)
                      - timedelta(minutes=PREGAME_BUFFER_MIN)).timestamp())
        return fetch_entry_price(client, ticker, cutoff)

    for i, r in enumerate(winner_rows):
        r["entry_price"] = entry_price_for(r["ticker"], r["game_dt"])
        if (i + 1) % 200 == 0:
            print(f"  winner entry prices {i + 1}/{len(winner_rows)}")
    for i, r in enumerate(totals_rows):
        r["entry_price"] = entry_price_for(r["ticker"], r["game_dt"])
        if (i + 1) % 200 == 0:
            print(f"  totals entry prices {i + 1}/{len(totals_rows)}")

    winner_rows = [r for r in winner_rows if r["entry_price"] is not None]
    totals_rows = [r for r in totals_rows if r["entry_price"] is not None]
    print(f"After entry-price filter: {len(winner_rows)} winner, {len(totals_rows)} totals")

    out = {"games": games_cache, "winner_rows": winner_rows, "totals_rows": totals_rows,
           "built_at": datetime.now(timezone.utc).isoformat()}
    out = _merge_with_existing(out, path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f)
    print("saved", path)
    return out


def _merge_with_existing(fresh: dict, path: Path) -> dict:
    """Union the freshly-fetched rows with whatever the cache already held.

    Kalshi's /markets?status=settled endpoint serves only a ROLLING ~68-day window of
    settled markets, so a plain overwrite silently AMPUTATES the oldest games on every
    refresh. The 2026-09-06 rebuild lost Jun 22-30 outright (111 winner + 109 totals
    events) and cut the train split from n=175 to n=128 bets. Left alone this erodes the
    train window (KALSHI_DATA_START..TRAIN_END) from the back until it is empty and the
    project's both-splits discipline breaks WITHOUT ever raising an error.

    Merging is safe because a settled row is immutable: across the 743 winner / 741
    totals events present in both the 2026-08-29 and 2026-09-06 builds, every
    entry_price and every outcome matched exactly (0 differences). Rows are keyed by
    event -- the same dedupe key build_cache uses upstream -- so a re-fetched event
    replaces its own older copy instead of duplicating it.
    """
    if not path.exists():
        return fresh
    try:
        with open(path, encoding="utf-8") as f:
            prior = json.load(f)
    except (OSError, ValueError) as e:
        print(f"  (could not read prior cache to merge: {e}) -- writing fresh rows only")
        return fresh
    merged = {"games": {**prior.get("games", {}), **fresh["games"]},
              "built_at": fresh["built_at"]}
    for kind in ("winner_rows", "totals_rows"):
        by_event = {r["event"]: r for r in prior.get(kind, [])}
        by_event.update({r["event"]: r for r in fresh[kind]})
        merged[kind] = sorted(by_event.values(), key=lambda r: (r["date"], r["ticker"]))
        retained = len(merged[kind]) - len(fresh[kind])
        print(f"  {kind}: {len(fresh[kind])} fetched + {retained} retained from prior "
              f"cache = {len(merged[kind])}")
    return merged


def _warn_if_train_thin(cache: dict) -> None:
    """The train split only means anything while the cache still reaches back to
    KALSHI_DATA_START; see _merge_with_existing() for why it can silently shrink."""
    dates = [r["date"] for r in cache["winner_rows"] + cache["totals_rows"]]
    if not dates:
        return
    earliest, n_train = min(dates), sum(1 for d in dates if d <= TRAIN_END)
    if earliest > KALSHI_DATA_START or n_train < 100:
        print(f"WARNING: cache reaches back only to {earliest} (wanted {KALSHI_DATA_START}); "
              f"{n_train} rows fall in the train window (<= {TRAIN_END}). Train-split "
              f"numbers are weakened or meaningless -- see _merge_with_existing().")


def load_cache(path: Path = CACHE_PATH) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# --- Elo training-history cache (separate from the above: raw game scores across
# multiple seasons, not tied to Kalshi settlement or trade prices, so it doesn't need
# refreshing on the same cadence) ------------------------------------------------

def build_elo_cache(path: Path = ELO_CACHE_PATH) -> dict:
    end = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    print(f"Fetching MLB history {ELO_HISTORY_START}..{end} for Elo training...")
    games = fetch_historical_games(ELO_HISTORY_START, end)
    print(f"  {len(games)} games")
    out = {"games": games, "built_at": datetime.now(timezone.utc).isoformat()}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f)
    print("saved", path)
    return out


def load_elo_cache(path: Path = ELO_CACHE_PATH) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# --- pure recompute + Kelly bankroll sim (no network; fast) -----------------------

def _log5(pa, pb):
    denom = pa + pb - 2 * pa * pb
    return 0.5 if denom <= 0 else (pa - pa * pb) / denom


def _apply_hf(p, yes_is_home):
    odds = p / (1 - p) if p < 1 else 999.0
    odds = odds * HOME_FIELD_ODDS_MULT if yes_is_home else odds / HOME_FIELD_ODDS_MULT
    return odds / (1 + odds)


def _nb_sf(k, mean, phi):
    if mean <= 0:
        return 0.0
    if phi <= 1.0:
        cdf, term = 0.0, math.exp(-mean)
        for i in range(k + 1):
            if i > 0:
                term *= mean / i
            cdf += term
        return max(0.0, 1.0 - cdf)
    r = mean / (phi - 1.0)
    p_ = r / (r + mean)
    log_p, log_1mp = math.log(p_), math.log(1 - p_)
    cdf = 0.0
    for i in range(k + 1):
        log_pmf = math.lgamma(i + r) - math.lgamma(r) - math.lgamma(i + 1) + r * log_p + i * log_1mp
        cdf += math.exp(log_pmf)
    return max(0.0, min(1.0, 1.0 - cdf))


def _lam(ing, p):
    lg_team = ing["lg_team"]

    def eff(team_ra_pg, sp_ra9, sp_ip):
        if sp_ra9 is None:
            return team_ra_pg
        reliab = sp_ip / (sp_ip + p["pitcher_reg_ip"])
        sp_reg = reliab * sp_ra9 + (1 - reliab) * lg_team
        return p["sp_weight"] * sp_reg + (1 - p["sp_weight"]) * team_ra_pg

    home_eff = eff(ing["home_ra_pg"], ing["home_sp_ra9"], ing["home_sp_ip"])
    away_eff = eff(ing["away_ra_pg"], ing["away_sp_ra9"], ing["away_sp_ip"])
    e_away = ing["away_rs_pg"] * (home_eff / lg_team)
    e_home = ing["home_rs_pg"] * (away_eff / lg_team)
    base = (1 - p["league_shrink"]) * (e_away + e_home) + p["league_shrink"] * (2 * lg_team)
    return base * ing["park_factor"]


def elo_prob(row, ratings: EloRatings) -> float:
    """P(YES team wins) per Elo rating entering this game -- ticker re-parsed for the
    team abbrevs rather than stored in bankroll_sim_cache.json, so this needs no cache
    rebuild (no re-fetch of the ~1,700 real trade prices) to add."""
    pt = parse_ticker(row["ticker"])
    r_yes = ratings.rating_before(pt.yes_team, row["date"])
    r_opp = ratings.rating_before(pt.opponent, row["date"])
    return elo_win_prob(r_yes, r_opp, a_is_home=row["yes_is_home"],
                        home_field_elo=ratings.home_field_elo)


def starter_edge_runs(row, ing, p) -> float:
    """Runs-per-9 advantage the YES team gets from TODAY'S starting-pitcher matchup,
    measured RELATIVE TO each team's own typical run prevention (so it adds pitcher
    information without re-litigating team strength, which log5/Elo already cover).

    Each starter's RA9 is regressed toward league average by innings pitched, exactly
    as data/fair_value_totals.py does (`pitcher_reg_ip`), then compared to that team's
    season run-prevention rate. A starter better than his team's baseline is a negative
    adjustment (fewer runs allowed). Positive return = today's matchup favors YES.
    Returns 0.0 when either starter is unknown (same graceful-degradation path the
    totals model takes)."""
    lg_team = ing["lg_team"]

    def sp_vs_baseline(sp_ra9, sp_ip, team_ra_pg):
        if sp_ra9 is None or sp_ip is None:
            return None
        reliab = sp_ip / (sp_ip + p["pitcher_reg_ip"])
        sp_reg = reliab * sp_ra9 + (1 - reliab) * lg_team
        return sp_reg - team_ra_pg

    if row["yes_is_home"]:
        yes_adj = sp_vs_baseline(ing["home_sp_ra9"], ing["home_sp_ip"], ing["home_ra_pg"])
        opp_adj = sp_vs_baseline(ing["away_sp_ra9"], ing["away_sp_ip"], ing["away_ra_pg"])
    else:
        yes_adj = sp_vs_baseline(ing["away_sp_ra9"], ing["away_sp_ip"], ing["away_ra_pg"])
        opp_adj = sp_vs_baseline(ing["home_sp_ra9"], ing["home_sp_ip"], ing["home_ra_pg"])
    if yes_adj is None or opp_adj is None:
        return 0.0
    return opp_adj - yes_adj


def winner_prob(row, ing, p, ratings: EloRatings = None):
    if row["yes_is_home"]:
        pa, pb = ing["home_win_pct"], ing["away_win_pct"]
    else:
        pa, pb = ing["away_win_pct"], ing["home_win_pct"]
    r = p.get("winner_regress", 0.0)
    log5_p = _apply_hf(_log5(_regress_win_pct(pa, r), _regress_win_pct(pb, r)), row["yes_is_home"])
    w = p.get("elo_weight", 0.0)
    prob = log5_p if (w <= 0.0 or ratings is None) else \
        (1 - w) * log5_p + w * elo_prob(row, ratings)

    # Starting-pitcher matchup as a log-odds nudge on top of the team-strength estimate.
    # 0.0 = production (no pitcher input on the winner side at all).
    beta = p.get("sp_winner_beta", 0.0)
    if beta > 0.0:
        edge_runs = starter_edge_runs(row, ing, p)
        if edge_runs:
            odds = prob / (1 - prob) if prob < 1 else 999.0
            odds *= math.exp(beta * edge_runs)
            prob = odds / (1 + odds)
    return prob


def totals_prob(row, ing, p):
    return _nb_sf(int(row["line"]), _lam(ing, p), p["totals_phi"])


def kelly_contracts(win_prob, cost, bankroll, kelly_fraction, max_stake_pct):
    if cost <= 0 or cost >= 1:
        return 0.0, 0
    f = max(0.0, (win_prob - cost) / (1 - cost)) * kelly_fraction
    f = min(f, max_stake_pct)
    stake = bankroll * f
    return round(stake, 2), int(stake / cost) if cost > 0 else 0


def simulate(cache: dict, params: dict, split: str = "all", start_bankroll: float = 20.0,
             ratings: EloRatings = None) -> dict:
    """split: 'train' (KALSHI_DATA_START..TRAIN_END), 'validate' (after), or 'all'."""
    p = {**DEFAULT_PARAMS, **params}
    games = cache["games"]
    rows = []
    for r in cache["winner_rows"]:
        if split == "train" and r["date"] > TRAIN_END:
            continue
        if split == "validate" and r["date"] <= TRAIN_END:
            continue
        ing = games.get(r["event"])
        if ing:
            rows.append({"date": r["date"], "kind": "winner", "prob": winner_prob(r, ing, p, ratings),
                         "actual": r["actual"], "entry_price_yes": r["entry_price"]})
    for r in cache["totals_rows"]:
        if split == "train" and r["date"] > TRAIN_END:
            continue
        if split == "validate" and r["date"] <= TRAIN_END:
            continue
        ing = games.get(r["event"])
        if ing:
            rows.append({"date": r["date"], "kind": "total", "prob": totals_prob(r, ing, p),
                         "actual": r["actual"], "entry_price_yes": r["entry_price"]})

    by_date = defaultdict(list)
    for r in rows:
        by_date[r["date"]].append(r)

    bankroll, n_bets, wins = start_bankroll, 0, 0
    total_staked = total_pnl = brier_sum = 0.0
    brier_n, peak, max_dd = 0, start_bankroll, 0.0
    by_kind = {"winner": {"n": 0, "wins": 0, "pnl": 0.0, "staked": 0.0},
               "total": {"n": 0, "wins": 0, "pnl": 0.0, "staked": 0.0}}

    for date in sorted(by_date):
        day_pnl = 0.0
        for r in by_date[date]:
            prob, actual = r["prob"], r["actual"]
            brier_sum += (prob - actual) ** 2
            brier_n += 1
            side = "YES" if prob > 0.5 else "NO"
            entry = r["entry_price_yes"] if side == "YES" else round(1 - r["entry_price_yes"], 4)
            model_p = prob if side == "YES" else 1 - prob
            if model_p - entry < p["min_edge"]:
                continue
            _, contracts = kelly_contracts(model_p, entry, bankroll, p["kelly_fraction"], p["max_stake_pct"])
            if contracts <= 0:
                continue
            wager = contracts * entry
            won = (actual == 1) if side == "YES" else (actual == 0)
            pnl = contracts * (1 - entry) if won else -wager
            day_pnl += pnl
            total_staked += wager
            total_pnl += pnl
            n_bets += 1
            wins += int(won)
            bk = by_kind[r["kind"]]
            bk["n"] += 1
            bk["wins"] += int(won)
            bk["pnl"] += pnl
            bk["staked"] += wager
        bankroll += day_pnl
        peak = max(peak, bankroll)
        if peak > 0:
            max_dd = max(max_dd, (peak - bankroll) / peak)

    return {
        "n_bets": n_bets, "win_rate": round(wins / n_bets, 3) if n_bets else None,
        "final_bankroll": round(bankroll, 2),
        "total_return_pct": round((bankroll / start_bankroll - 1) * 100, 2),
        "roi_on_staked": round(total_pnl / total_staked, 4) if total_staked else None,
        "brier": round(brier_sum / brier_n, 4) if brier_n else None,
        "max_drawdown_pct": round(max_dd * 100, 2),
        "by_kind": {k: {"n": v["n"], "win_rate": round(v["wins"] / v["n"], 3) if v["n"] else None,
                        "roi": round(v["pnl"] / v["staked"], 4) if v["staked"] else None}
                    for k, v in by_kind.items()},
    }


def report(cache: dict, params: dict, label: str = "", ratings: EloRatings = None) -> None:
    tr = simulate(cache, params, "train", ratings=ratings)
    va = simulate(cache, params, "validate", ratings=ratings)
    overrides = {k: v for k, v in params.items() if DEFAULT_PARAMS.get(k) != v}
    print(f"\n=== {label or 'params'} ===  overrides: {overrides or '(none -- current production)'}")
    for name, r in (("TRAIN", tr), ("VALID", va)):
        print(f"  {name}: n={r['n_bets']:<4} win_rate={r['win_rate']} roi_on_staked={r['roi_on_staked']} "
              f"bankroll=${r['final_bankroll']} ({r['total_return_pct']:+.1f}%) "
              f"brier={r['brier']} maxDD={r['max_drawdown_pct']}%  by_kind={r['by_kind']}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Sizing-aware walk-forward bankroll backtest")
    ap.add_argument("--refresh", action="store_true", help="rebuild the data cache (network-bound, ~10 min)")
    ap.add_argument("--refresh-elo", action="store_true",
                     help="rebuild the Elo training-history cache (network-bound, ~1 min)")
    ap.add_argument("--sweep", nargs="+", metavar=("PARAM", "VALUES"),
                     help="e.g. --sweep kelly_fraction 0.2 0.25 0.3 0.35")
    args = ap.parse_args()

    cache = build_cache() if (args.refresh or not CACHE_PATH.exists()) else load_cache()
    _warn_if_train_thin(cache)
    elo_cache = build_elo_cache() if (args.refresh_elo or not ELO_CACHE_PATH.exists()) else load_elo_cache()
    ratings = EloRatings(elo_cache["games"])

    if args.sweep:
        param, values = args.sweep[0], args.sweep[1:]
        for v in values:
            try:
                v_typed = float(v)
            except ValueError:
                v_typed = v
            # elo_k_factor / elo_home_field rebuild ratings per value (both live inside
            # EloRatings' chronological build, not simulate()'s per-row params) rather than
            # being ordinary DEFAULT_PARAMS entries -- see EloRatings.__init__'s overrides.
            rebuild_params = {"elo_k_factor": "k_factor", "elo_home_field": "home_field_elo"}
            if param in rebuild_params:
                sweep_ratings = EloRatings(elo_cache["games"], **{rebuild_params[param]: v_typed})
                report(cache, {}, f"{param}={v_typed}", sweep_ratings)
            else:
                report(cache, {param: v_typed}, f"{param}={v_typed}", ratings)
        return

    report(cache, {}, "current production parameters", ratings)


if __name__ == "__main__":
    main()
