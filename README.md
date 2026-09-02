# Kalshi MLB Money-Flow Agent

A **recommend-only** trading agent for Kalshi MLB game markets. It scans markets
on a schedule, follows where the money is going (order-flow + book imbalance +
open-interest momentum), cross-checks against a fair-value model built from the
MLB Stats API, and emits **ranked trade recommendations**.

> **It never places, cancels, or modifies orders.** Every "stake"/"contracts"
> number is advisory. You decide what to actually trade.

---

## What it does each cycle (~every 20 min)

1. Pulls open MLB markets from Kalshi: game winners (`KXMLBGAME*`) and total-runs
   over/under ladders (`KXMLBTOTAL*`, filtered to near-the-money lines).
2. Picks the most liquid **games** and deep-scans them (order book + recent trades).
   Winner markets pair the two teams for cross-market flow; totals use within-market
   flow (over-buyers vs under-buyers).
3. Computes a **money-flow score** per market from three signals:
   - **Book imbalance** — near-touch, distance-decayed resting capital, compared
     *across the two sides of the game* (de-biases the structural NO-wall skew).
   - **Trade flow** — recency/blocks-weighted aggressive (taker) dollars, compared
     *across the two sides of the game* (de-biases the structural "retail buys YES" skew).
   - **OI momentum** — change in open interest in the direction of price (new
     conviction money vs. churn); needs a prior snapshot, so it warms up over cycles.
4. Estimates a **fair YES probability**:
   - **Winner, live** → MLB Stats API live win probability.
   - **Winner, pre-game** → log5 on season records + home-field adjustment.
   - **Totals (O/U)** → expected total runs from a multiplicative runs model:
     team offense × opponent run-prevention, where run-prevention folds in the
     **probable starting pitcher's RA9** (regressed toward league by innings, ~62%
     starter / 38% bullpen weight) plus team rates. Modeled as a negative-binomial
     (overdispersed vs Poisson) → P(total > line). Confidence rises when both
     starters are announced. Falls back to team rates when a starter isn't posted yet.
5. Fuses flow (primary) + fair-value edge (filter) + liquidity into a ranked list
   with an edge (in cents), a composite confidence, and fractional-Kelly sizing.
6. Writes recommendations + records a snapshot of every market for later backtesting.

## Why this design (vs. a naive "follow the money")

The raw Kalshi data is **structurally skewed**: nearly every market shows heavy
YES-buying and heavy NO-side resting depth, regardless of which team is favored.
A naive per-market "net flow" is therefore almost always positive and tells you
nothing. This agent compares the **two sides of the same game head-to-head**, so
the signal measures *which team the money is actually piling into*.

---

## Setup

Already done once (Python 3.12 venv + deps). To recreate:

```bash
py -3 -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements.txt
```

Credentials are read from `C:\Users\txrus\Portfolio\Kal_API.txt` (API key id + RSA
private key). Override with the `KALSHI_KEY_FILE` env var. **Never commit this file.**

## Commands (terminal CLI)

The agent is command-driven via `cli.py`. All commands accept `--allow-live`
(include in-progress games; default is pre-game only), `--demo`, and `--bankroll`.

```bash
.venv/Scripts/python.exe cli.py search               # scan markets, show money-flow reads
.venv/Scripts/python.exe cli.py rank                 # ranked, actionable recommendations
.venv/Scripts/python.exe cli.py inspect <TICKER>     # deep dive: quote, book, flow, fair value
.venv/Scripts/python.exe cli.py status               # account + exchange + positions/orders
.venv/Scripts/python.exe cli.py positions            # open positions
.venv/Scripts/python.exe cli.py orders               # resting orders
.venv/Scripts/python.exe cli.py results [YYYY-MM-DD] # MLB outcomes for a date
.venv/Scripts/python.exe cli.py outcome [YYYY-MM-DD] # how the agent's bets did on ONE day
.venv/Scripts/python.exe cli.py backtest [--since D] # validate the signal (cumulative)
.venv/Scripts/python.exe cli.py loop [--interval S] [--paper]   # continuous loop
.venv/Scripts/python.exe cli.py settle [YYYY-MM-DD] # grade the paper-trade ledger
.venv/Scripts/python.exe cli.py place <TICKER> --side yes --count 10 --price 0.57
```

### Paper-trading (forward test)

`loop --paper` runs the scan on a schedule and **paper-bets each game on the last
cycle before its first pitch** — the moment pitchers are set, money has flowed, and
OI-momentum is populated. One bet per game, recorded to `output/paper_ledger.jsonl`.
After games finish, `settle` grades it (win rate, ROI, net P&L, winner vs O/U).
No real orders are ever placed. Typical use: start it in the morning and let it run
through the day's games.

```bash
.venv/Scripts/python.exe cli.py loop --interval 1800 --paper   # 30-min cadence, paper on
.venv/Scripts/python.exe cli.py settle                          # grade after games finish
```

`place` is **guarded**: it previews the order and runs every safety check
(size/cost caps, price bounds, market status, funds), but **does not submit**.
Live submission is intentionally disabled in this build — the agent is
recommend-only. To actually trade, place the previewed order yourself in Kalshi,
or wire up your own execution module (see `engine/execution.py`).

Health check (auth + sample data): `.venv/Scripts/python.exe -m tests.smoke_auth`

## Commands (Claude Code slash commands)

Open a Claude Code session in this folder and use:

| Command | Does |
|---|---|
| `/searchtrades` | scan markets, show where the money is flowing |
| `/rank` | ranked, actionable recommendations with rationale |
| `/inspect <TICKER>` | deep dive on one market |
| `/status` | account + exchange summary |
| `/positions` | open positions |
| `/results [date]` | MLB game outcomes |
| `/backtest [--since D]` | validate the signal (honestly) |
| `/placetrade <TICKER> <yes\|no> <count> <price>` | **preview only** — never auto-submits |

`/placetrade` deliberately runs the dry-run preview and hands you the exact
`--confirm` command to run yourself; it never places a live order.

## Validate the edge (backtest)

After the agent has recorded snapshots and games have settled:
```bash
.venv/Scripts/python.exe -m backtest.evaluate
.venv/Scripts/python.exe -m backtest.evaluate --since 2026-08-05
```
Reports **rec win-rate & ROI**, **signal IC** (does money-flow predict YES wins?),
and **fair-value calibration** (Brier). This is how you confirm the thesis has edge
before ever risking real money.

## Outputs

- `output/recommendations_latest.json` — latest cycle (full detail).
- `output/recommendations_history.jsonl` — every cycle, appended.
- `snapshots/YYYY-MM-DD.jsonl` — per-market snapshots (the backtest dataset).
- `snapshots/_last_state.json` — prior mid/OI per market (feeds OI momentum).
- `logs/agent.log` — run log.

## Tuning

All thresholds live in `config/settings.py`:
signal weights, `min_edge_cents`, `min_confidence`, `max_spread_cents`,
`scan_interval_seconds`, `bankroll_usd`, `kelly_fraction`, `max_deep_markets`.

## Layout

```
config/      credentials loader + settings
kalshi/      signed API client, request signing, payload normalization
data/        MLB Stats + nflverse clients, fair-value models (winner + totals, per sport)
signals/     money-flow signal, Elo ratings, recommendation fusion, calibration
engine/      scan cycle, loop, snapshot store, paper ledger, notifications
output/      console + JSON reporting
backtest/    outcome resolution + signal evaluation
tests/       auth smoke test + data-shape explorers
```

## Roadmap / next steps

- Let the agent run through several game days, then backtest to measure real IC/ROI.
- Tune signal weights from backtest results.
- Generalize beyond MLB (NBA/NFL/NHL) once the MLB signal is validated.
- Optional: add a paper-trading ledger that "fills" recs at quoted prices and tracks P&L live.
```
