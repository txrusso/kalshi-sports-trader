"""Season-long, leak-safe backtest of the fair-value model against real Kalshi settlement.

Money-flow can't be backtested this way -- it needs point-in-time order-book/trade
history that Kalshi doesn't retroactively expose, and Kalshi's MLB game/total markets
only go back to 2026-06-07 anyway (confirmed by probing /markets with max_close_ts;
KXMLBGAME/KXMLBTOTAL simply didn't exist before then -- there's no earlier series).
This backtests the OTHER half: is the pregame fair-value model (log5 win-prob /
negative-binomial totals) itself accurate, graded against real settled outcomes.

Leakage traps found and avoided (both are real, both would silently inflate results):

1. MLB Stats API's `leagueRecord` on a historical `/schedule?date=` query reflects the
   record AFTER that date's games finish, not before -- confirmed empirically (NYY
   41-26 on 2026-06-10 after a win that day, not 40-26 entering it). Fine for the LIVE
   scanner (it always queries pregame, before today's games are final), but querying it
   for a past date -- which a backtest must do -- leaks that day's own result into the
   input. Same risk suspected for `/standings?date=` team run rates.
   Fix: never trust an "as of date" API field. Build a chronological per-team game
   ledger from the full-season schedule ourselves, and only aggregate games with
   date_str STRICTLY LESS THAN the target game's date (whole-day granularity, so
   doubleheaders can't leak into each other either).

2. `pitcher_ra9()` / `team_run_rates()` (data/mlb_stats.py, used live) call the season
   /stats and /standings endpoints with no date scope at all -- always "as of right
   now". Fix: pull each starter's full-season gameLog (per-appearance, individually
   dated -- unambiguous) and sum only appearances before the target date.

Everything else reuses the exact production math (log5, home-field odds adjustment,
the negative-binomial survival function, park factors, starter/team RA9 blend) via
direct imports from data/fair_value.py and data/fair_value_totals.py, so this measures
the real model, not a reimplementation of it. Team stats default to full-season-to-date;
a last-30-games recency window is available via --recency-window for experiments but is
NOT the default -- an ablation here on 2026-08-13 showed it made the winner model's
calibration worse (Brier 0.2500 -> 0.2608: a 30-game win/loss sample is too noisy for
log5), which is why production (data/fair_value.py) doesn't use it either.

A Pythagorean pitching-matchup blend on the winner model was also tried and reverted
the same day -- see data/fair_value.py's comment for why (it failed a walk-forward
train/validate test after looking like a win on a single full-sample test). A separate,
more careful walk-forward parameter search (train Jun 7-Jul 15, validate Jul 16-Aug 13,
including a real Kelly bankroll simulation, not just calibration) lives in
backtest/bankroll_sim.py -- that's what settled the final production constants
(TOTALS_PHI, LEAGUE_SHRINK, min_edge_cents, kelly_fraction, max_stake_pct).

Usage:
    py -3 -m backtest.season_backtest
    py -3 -m backtest.season_backtest --since 2026-06-07 --season-start 2026-03-01
"""
from __future__ import annotations

import argparse
import math
import sys
import time
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.fair_value import log5, _apply_home_field, parse_ticker, _regress_win_pct
from data.fair_value_totals import (
    parse_total_ticker, park_factor, SP_WEIGHT, PITCHER_REG_IP, LEAGUE_SHRINK,
    TOTALS_PHI, _nb_sf,
)
from data.mlb_stats import RECENT_WINDOW_GAMES, RECENT_MIN_GAMES
from kalshi.client import KalshiClient
from config.settings import DEFAULTS

MLB_BASE = "https://statsapi.mlb.com/api/v1"
KALSHI_DATA_START = "2026-06-07"   # earliest settled KXMLBGAME/KXMLBTOTAL market found


def _ip_to_float(ip_str) -> float:
    try:
        whole, _, frac = str(ip_str).partition(".")
        return int(whole) + (int(frac) / 3.0 if frac else 0.0)
    except (ValueError, TypeError):
        return 0.0


class SeasonLedger:
    """Point-in-time-safe team results, built once from the full-season schedule."""

    def __init__(self, session: requests.Session, season_start: str, season_end: str,
                 recency_window: int = RECENT_WINDOW_GAMES):
        self.session = session
        self.team_games: dict[str, list[tuple]] = defaultdict(list)   # (date, is_win, rf, ra)
        self.games_by_date: dict[str, list[dict]] = defaultdict(list)
        self._league_avg_cache: dict[str, float] = {}
        self.recency_window = recency_window   # ablation knob; 9999 == effectively full-season
        self._load(season_start, season_end)

    def _load(self, season_start: str, season_end: str) -> None:
        r = self.session.get(f"{MLB_BASE}/schedule", params={
            "sportId": 1, "startDate": season_start, "endDate": season_end,
            "hydrate": "team,linescore,probablePitcher",
        }, timeout=60)
        r.raise_for_status()
        data = r.json()
        for de in data.get("dates", []):
            for g in de.get("games", []):
                if g.get("gameType") != "R":
                    continue
                away, home = g["teams"]["away"], g["teams"]["home"]
                away_abbr = (away["team"].get("abbreviation") or "").upper()
                home_abbr = (home["team"].get("abbreviation") or "").upper()
                date_str = g.get("officialDate")
                state = g["status"].get("abstractGameState")
                away_score, home_score = away.get("score"), home.get("score")
                if state == "Final" and away_score is not None and home_score is not None and date_str:
                    self.team_games[away_abbr].append((date_str, away_score > home_score, away_score, home_score))
                    self.team_games[home_abbr].append((date_str, home_score > away_score, home_score, away_score))
                if date_str:
                    self.games_by_date[date_str].append({
                        "away_abbr": away_abbr, "home_abbr": home_abbr,
                        "away_pitcher_id": (away.get("probablePitcher") or {}).get("id"),
                        "home_pitcher_id": (home.get("probablePitcher") or {}).get("id"),
                    })
        for v in self.team_games.values():
            v.sort(key=lambda x: x[0])

    def _entering(self, team: str, date_str: str) -> list[tuple]:
        return [g for g in self.team_games.get(team, []) if g[0] < date_str]

    def _entering_recent(self, team: str, date_str: str) -> list[tuple]:
        """Last RECENT_WINDOW_GAMES entering this date; falls back to the full
        entering set when there's less than RECENT_MIN_GAMES of history (mirrors
        MlbStatsClient.team_recent_stats()'s season-to-date fallback -- in this
        backtest both sources come from the same ledger, so "fallback" just means
        "use everything available" rather than truncating to the last 30)."""
        games = self._entering(team, date_str)
        if len(games) < RECENT_MIN_GAMES:
            return games
        return games[-self.recency_window:]

    def entering_win_pct(self, team: str, date_str: str) -> float | None:
        games = self._entering_recent(team, date_str)
        if not games:
            return None
        return sum(1 for g in games if g[1]) / len(games)

    def entering_run_rate(self, team: str, date_str: str) -> tuple[float, float] | None:
        games = self._entering_recent(team, date_str)
        if not games:
            return None
        return (sum(g[2] for g in games) / len(games), sum(g[3] for g in games) / len(games))

    def league_avg_total(self, date_str: str) -> float:
        if date_str not in self._league_avg_cache:
            rates = [self.entering_run_rate(t, date_str) for t in self.team_games]
            rates = [r for r in rates if r]
            self._league_avg_cache[date_str] = (
                (sum(rf for rf, _ in rates) + sum(ra for _, ra in rates)) / len(rates)) if rates else 9.0
        return self._league_avg_cache[date_str]

    def find_game(self, date_str: str, team_a: str, team_b: str) -> dict | None:
        """Matches production's team-set logic; tries the ticker date then +/-1 day
        as a small buffer against rare officialDate/ticker-date mismatches."""
        for d in (date_str, _shift(date_str, -1), _shift(date_str, 1)):
            for g in self.games_by_date.get(d, []):
                if {g["away_abbr"], g["home_abbr"]} == {team_a, team_b}:
                    return g
        return None


def _shift(date_str: str, days: int) -> str:
    return (datetime.strptime(date_str, "%Y-%m-%d") + timedelta(days=days)).strftime("%Y-%m-%d")


class PitcherLogs:
    """Per-appearance (date, IP, runs) history, fetched once per pitcher and reused
    across every backtest date -- summing only appearances before the target date
    keeps this leak-safe without needing a call per (pitcher, date) pair."""

    def __init__(self, session: requests.Session):
        self.session = session
        self._cache: dict[int, list[tuple]] = {}

    def _fetch(self, pitcher_id: int) -> list[tuple]:
        if pitcher_id in self._cache:
            return self._cache[pitcher_id]
        entries: list[tuple] = []
        try:
            r = self.session.get(f"{MLB_BASE}/people/{pitcher_id}/stats",
                                  params={"stats": "gameLog", "group": "pitching", "season": 2026},
                                  timeout=20)
            if r.status_code == 200:
                for sp in r.json().get("stats", [{}])[0].get("splits", []):
                    date_str = sp.get("date")
                    st = sp.get("stat", {})
                    if date_str:
                        entries.append((date_str, _ip_to_float(st.get("inningsPitched", "0")),
                                        float(st.get("runs") or 0)))
        except requests.RequestException:
            pass
        entries.sort(key=lambda x: x[0])
        self._cache[pitcher_id] = entries
        time.sleep(0.08)
        return entries

    def ra9_entering(self, pitcher_id: int | None, date_str: str) -> tuple[float, float] | None:
        if not pitcher_id:
            return None
        past = [e for e in self._fetch(pitcher_id) if e[0] < date_str]
        ip = sum(e[1] for e in past)
        if ip <= 0:
            return None
        return (sum(e[2] for e in past) * 9.0 / ip, ip)


def _effective_ra9(team_ra_pg: float, pitcher_id, date_str: str, lg_team: float,
                    plogs: PitcherLogs) -> float:
    stat = plogs.ra9_entering(pitcher_id, date_str)
    if not stat:
        return team_ra_pg
    sp_ra9, ip = stat
    reliab = ip / (ip + PITCHER_REG_IP)
    sp_reg = reliab * sp_ra9 + (1 - reliab) * lg_team
    return SP_WEIGHT * sp_reg + (1 - SP_WEIGHT) * team_ra_pg


def expected_runs_backtest(game: dict, date_str: str, ledger: SeasonLedger,
                            plogs: PitcherLogs) -> dict | None:
    """Mirrors data/run_environment.py's RunEnvironmentModel.expected_runs() --
    shared by grade_winner (pitching-matchup nudge) and grade_totals (the line
    itself) so both use identical, leak-safe, entering-game-only run math."""
    away_rate = ledger.entering_run_rate(game["away_abbr"], date_str)
    home_rate = ledger.entering_run_rate(game["home_abbr"], date_str)
    if not away_rate or not home_rate:
        return None
    lg_total = ledger.league_avg_total(date_str)
    lg_team = lg_total / 2.0
    home_eff = _effective_ra9(home_rate[1], game["home_pitcher_id"], date_str, lg_team, plogs)
    away_eff = _effective_ra9(away_rate[1], game["away_pitcher_id"], date_str, lg_team, plogs)
    e_away = away_rate[0] * (home_eff / lg_team)
    e_home = home_rate[0] * (away_eff / lg_team)
    base = (1 - LEAGUE_SHRINK) * (e_away + e_home) + LEAGUE_SHRINK * lg_total
    lam = base * park_factor(game["home_abbr"])
    return {"lam": lam, "e_away": e_away, "e_home": e_home}


def fetch_settled_markets(client: KalshiClient, series: str) -> list[dict]:
    """Full, uncapped pagination -- KalshiClient.get_markets() caps at 40 pages
    (8,000 markets) via its default _paginate(max_pages=40), which silently
    truncates KXMLBTOTAL (10,500+ settled markets, 53 pages)."""
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


def grade_winner(markets: list[dict], ledger: SeasonLedger) -> list[dict]:
    out, seen_events = [], set()
    for m in markets:
        pt = parse_ticker(m["ticker"])
        if not pt or pt.event_ticker in seen_events or not pt.opponent:
            continue
        seen_events.add(pt.event_ticker)
        date_str = pt.date.strftime("%Y-%m-%d")
        pa = ledger.entering_win_pct(pt.yes_team, date_str)
        pb = ledger.entering_win_pct(pt.opponent, date_str)
        if pa is None or pb is None:
            continue
        game = ledger.find_game(date_str, pt.yes_team, pt.opponent)
        yes_is_home = bool(game and game["home_abbr"] == pt.yes_team)
        prob = _apply_home_field(log5(_regress_win_pct(pa), _regress_win_pct(pb)), yes_is_home)
        actual = 1 if m.get("result") == "yes" else 0
        out.append({"kind": "winner", "date": date_str, "pred": prob, "actual": actual,
                     "ticker": m["ticker"], "game_dt": pt.date, "event": pt.event_ticker})
    return out


def grade_totals(markets: list[dict], ledger: SeasonLedger, plogs: PitcherLogs) -> list[dict]:
    out = []
    for m in markets:
        pt = parse_total_ticker(m["ticker"])
        if not pt:
            continue
        date_str = pt.date.strftime("%Y-%m-%d")
        game = None
        for d in (date_str, _shift(date_str, -1), _shift(date_str, 1)):
            for g in ledger.games_by_date.get(d, []):
                if pt.teams in (g["away_abbr"] + g["home_abbr"], g["home_abbr"] + g["away_abbr"]):
                    game = g
                    break
            if game:
                date_str = d
                break
        if not game:
            continue
        env = expected_runs_backtest(game, date_str, ledger, plogs)
        if env is None:
            continue
        lam = env["lam"]
        prob_over = _nb_sf(int(pt.line), lam, TOTALS_PHI)
        actual = 1 if m.get("result") == "yes" else 0
        out.append({"kind": "total", "date": date_str, "pred": prob_over, "actual": actual,
                     "ticker": m["ticker"], "game_dt": pt.date, "event": pt.event_ticker,
                     "line": pt.line, "lam": lam})
    return out


def select_representative_totals(rows: list[dict]) -> list[dict]:
    """One line per game for the ROI test -- the line closest to the model's own
    predicted total (~pick'em), the one that's actually liquid/tradeable in
    practice. Grading all ~12 lines/game would count many trivially-obvious
    extreme strikes as independent "bets", which no one would actually place."""
    best: dict[str, dict] = {}
    for r in rows:
        k = r["event"]
        if k not in best or abs(r["line"] - r["lam"]) < abs(best[k]["line"] - best[k]["lam"]):
            best[k] = r
    return list(best.values())


GAME_UTC_OFFSET_HOURS = 4   # ET->UTC during EDT (Jun-Aug, no DST edge in this window)
PREGAME_BUFFER_MIN = 33     # matches the live loop's paper-bet trigger window
MIN_EDGE = DEFAULTS.min_edge_cents / 100   # tracks live production, not a fixed snapshot of it


def fetch_entry_price(client: KalshiClient, ticker: str, cutoff_ts: int) -> float | None:
    """Last traded YES price at/before cutoff_ts, our pregame-entry proxy."""
    data = client._get("/markets/trades", {"ticker": ticker, "limit": 3, "max_ts": cutoff_ts})
    trades = data.get("trades") or []
    if not trades:
        return None
    try:
        return float(trades[0]["yes_price_dollars"])
    except (KeyError, ValueError, TypeError):
        return None


def compute_roi(rows: list[dict], client: KalshiClient) -> dict:
    """Would betting the model's edge, at the real pregame trade price, at the
    production min-edge gate (MIN_EDGE, tracks settings.min_edge_cents), have
    made money? Flat $1-notional stake per contract, matching
    backtest/evaluate.py's existing PnL convention."""
    graded, no_price, no_edge = [], 0, 0
    for r in rows:
        cutoff = int((r["game_dt"].replace(tzinfo=timezone.utc)
                      + timedelta(hours=GAME_UTC_OFFSET_HOURS)
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
        "n_bets": n, "no_price_data": no_price, "no_edge": no_edge,
        "win_rate": round(wins / n, 3) if n else None,
        "pnl_per_contract_total": round(pnl_total, 3),
        "avg_pnl_per_bet": round(pnl_total / n, 4) if n else None,
        "roi": round(pnl_total / staked, 4) if staked else None,
        "rows": graded,
    }


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
    correct = sum(1 for r in rows if (r["pred"] > 0.5) == (r["actual"] == 1))
    return correct / len(rows)


def _calibration_table(rows: list[dict], n_buckets: int = 10) -> list[dict]:
    buckets = [[] for _ in range(n_buckets)]
    for r in rows:
        idx = min(int(r["pred"] * n_buckets), n_buckets - 1)
        buckets[idx].append(r)
    table = []
    for i, b in enumerate(buckets):
        if not b:
            continue
        table.append({
            "range": f"{i*100//n_buckets}-{(i+1)*100//n_buckets}%",
            "n": len(b),
            "mean_pred": round(sum(r["pred"] for r in b) / len(b), 3),
            "actual_rate": round(sum(r["actual"] for r in b) / len(b), 3),
        })
    return table


def _monthly(rows: list[dict]) -> dict:
    by_month = defaultdict(list)
    for r in rows:
        by_month[r["date"][:7]].append(r)
    return {m: {"n": len(rs), "brier": round(_brier(rs), 4), "accuracy": round(_accuracy(rs), 3)}
            for m, rs in sorted(by_month.items())}


def main() -> None:
    ap = argparse.ArgumentParser(description="Leak-safe season backtest of the fair-value model")
    ap.add_argument("--since", default=KALSHI_DATA_START, help="grade Kalshi markets closing on/after this date")
    ap.add_argument("--season-start", default="2026-03-01", help="schedule fetch start (for building the ledger)")
    ap.add_argument("--season-end", default=None, help="schedule fetch end (default: today)")
    ap.add_argument("--no-roi", action="store_true",
                     help="skip the real-money ROI pass (fetches a pregame trade price per game)")
    ap.add_argument("--recency-window", type=int, default=9999,
                     help="team-stats lookback in games; default is full-season "
                          "(a 30-game window was tried and reverted -- ablation showed it "
                          "hurt the winner model's calibration; kept as a knob for experiments)")
    args = ap.parse_args()
    season_end = args.season_end or datetime.now(timezone.utc).strftime("%Y-%m-%d")

    session = requests.Session()
    session.headers.update({"User-Agent": "kalshi-agent research"})

    print(f"Building point-in-time team ledger from {args.season_start} to {season_end} "
          f"(recency_window={args.recency_window})...")
    ledger = SeasonLedger(session, args.season_start, season_end, recency_window=args.recency_window)
    print(f"  {sum(len(v) for v in ledger.team_games.values())} completed team-games across "
          f"{len(ledger.team_games)} teams.")

    plogs = PitcherLogs(session)
    client = KalshiClient(DEFAULTS)

    print("Fetching settled Kalshi markets...")
    winner_markets = fetch_settled_markets(client, "KXMLBGAME")
    total_markets = fetch_settled_markets(client, "KXMLBTOTAL")
    winner_markets = [m for m in winner_markets if (m.get("close_time") or "") >= args.since]
    total_markets = [m for m in total_markets if (m.get("close_time") or "") >= args.since]
    print(f"  {len(winner_markets)} winner market-sides, {len(total_markets)} total-line markets"
          f" since {args.since}.")

    print("Grading winner markets (log5 + home field, entering-game records only)...")
    winner_rows = grade_winner(winner_markets, ledger)
    print(f"  graded {len(winner_rows)} games ({len(winner_markets) - len(winner_rows)} skipped:"
          f" no prior record or unparseable).")

    print("Grading totals markets (negative-binomial, entering-game rates + starter RA9)...")
    total_rows = grade_totals(total_markets, ledger, plogs)
    print(f"  graded {len(total_rows)} lines ({len(total_markets) - len(total_rows)} skipped:"
          f" no matching game or missing rates).")

    all_rows = winner_rows + total_rows

    print("\n=== SEASON BACKTEST REPORT (leak-safe, entering-game data only) ===")
    print(f"Window: {args.since} to {season_end}\n")
    for label, rows in (("WINNER", winner_rows), ("TOTALS", total_rows), ("COMBINED", all_rows)):
        print(f"--- {label} ---")
        print(f"  n                : {len(rows)}")
        print(f"  brier            : {_brier(rows)}")
        print(f"  log_loss         : {round(_log_loss(rows), 4) if rows else None}")
        print(f"  accuracy         : {round(_accuracy(rows), 3) if rows else None}")
        print()

    print("--- Calibration (winner) ---")
    for row in _calibration_table(winner_rows):
        print(f"  {row['range']:>8}  n={row['n']:<5} mean_pred={row['mean_pred']:<6} actual_rate={row['actual_rate']}")
    print("\n--- Calibration (totals) ---")
    for row in _calibration_table(total_rows):
        print(f"  {row['range']:>8}  n={row['n']:<5} mean_pred={row['mean_pred']:<6} actual_rate={row['actual_rate']}")

    print("\n--- By month (winner) ---")
    for m, s in _monthly(winner_rows).items():
        print(f"  {m}: n={s['n']:<5} brier={s['brier']:<7} accuracy={s['accuracy']}")
    print("\n--- By month (totals) ---")
    for m, s in _monthly(total_rows).items():
        print(f"  {m}: n={s['n']:<5} brier={s['brier']:<7} accuracy={s['accuracy']}")

    # Naive baselines for context.
    home_favorite_rows = [{"pred": 0.54, "actual": r["actual"]} for r in winner_rows]
    coinflip_rows = [{"pred": 0.5, "actual": r["actual"]} for r in winner_rows]
    print("\n--- Baselines (winner only) ---")
    print(f"  always-0.5 brier      : {round(_brier(coinflip_rows), 4) if coinflip_rows else None}")
    print(f"  always-home-0.54 brier: {round(_brier(home_favorite_rows), 4) if home_favorite_rows else None}")
    print(f"  model brier            : {round(_brier(winner_rows), 4) if winner_rows else None}")

    if args.no_roi:
        return

    print("\nFetching real pregame trade prices for the ROI pass "
          f"(1 line/game for totals, {PREGAME_BUFFER_MIN}min-before-first-pitch cutoff, "
          f"{MIN_EDGE*100:.0f}c min-edge gate)...")
    winner_roi = compute_roi(winner_rows, client)
    totals_repr = select_representative_totals(total_rows)
    print(f"  totals: {len(total_rows)} lines -> {len(totals_repr)} representative (1/game)")
    totals_roi = compute_roi(totals_repr, client)
    combined_roi_rows = winner_roi["rows"] + totals_roi["rows"]
    combined_n = len(combined_roi_rows)
    combined_pnl = sum(g["pnl"] for g in combined_roi_rows)
    combined_staked = sum(g["entry"] for g in combined_roi_rows)

    print(f"\n=== ROI (real pregame trade prices, production {MIN_EDGE*100:.1f}c edge gate, "
          f"flat $1/contract) ===")
    for label, roi in (("WINNER", winner_roi), ("TOTALS", totals_roi)):
        print(f"--- {label} ---")
        print(f"  candidates          : {roi['n_bets'] + roi['no_price_data'] + roi['no_edge']}")
        print(f"  no pregame trade data: {roi['no_price_data']}")
        print(f"  below {MIN_EDGE*100:.1f}c edge (no bet): {roi['no_edge']}")
        print(f"  bets placed         : {roi['n_bets']}")
        print(f"  win rate            : {roi['win_rate']}")
        print(f"  pnl (per $1 contract): {roi['pnl_per_contract_total']}")
        print(f"  avg pnl / bet       : {roi['avg_pnl_per_bet']}")
        print(f"  roi                 : {roi['roi']}")
        print()
    print("--- COMBINED ---")
    print(f"  bets placed         : {combined_n}")
    print(f"  pnl (per $1 contract): {round(combined_pnl, 3)}")
    print(f"  roi                 : {round(combined_pnl / combined_staked, 4) if combined_staked else None}")


if __name__ == "__main__":
    main()
