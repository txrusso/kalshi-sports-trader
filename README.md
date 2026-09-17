# Kalshi Sports Trader

A trading agent for Kalshi's sports markets. It watches where money is moving in the order
book, checks that against its own fair-value model of the game, and when the two agree by
enough it sizes a trade.

By default (`loop`) it's still recommend-only: it works out the size, texts me the order
slip, and I tap buy in the Kalshi app myself. `loop --live` (added 2026-09-17, at my
explicit request) switches that same trigger to actually submit the order through Kalshi's
API — no confirmation step, straight to the live account, gated only by the per-order
caps in `config/settings.py` (`max_order_contracts`, `max_order_cost_usd`). I text/push
myself a receipt after each order lands, not a slip to act on. See `engine/live.py` and
`engine/execution.py` for exactly what is and isn't guarded on that path — it's a real
reversal of the original design boundary below, made deliberately and not lightly, given
how thin the edge here is.

Covers MLB and NFL right now. Both run through the same pipeline in the same scan cycle;
only the fair-value half is sport-specific.

![The pipeline, end to end](docs/images/pipeline.png)

## The signal

Two independent things have to line up before a market is worth a bet.

**Money flow** is the primary signal and it's the reason the project exists. Every 30
minutes it reads the order book, recent trades, and open interest for the most liquid
games, and scores which side the money is actually leaning. Three components, weighted
40/35/25: near-touch book imbalance, recency-weighted aggressive (taker) dollars, and
change in open interest in the direction of price.

The tricky part is that raw Kalshi flow is structurally biased — retail buys YES, market
makers rest NO walls — so reading one market in isolation mostly tells you about the venue,
not the game. Winner markets get de-biased by comparing the *two sides of the same game*
against each other. Totals compare over-buyers against under-buyers within the line.

There's also a large-print detector that up-weights trades big relative to a market's own
median trade size. Kalshi trades are anonymous, so you can't follow specific accounts the
way you can on-chain; inferring a whale from print size is the closest available proxy.
`cli.py whales` shows that tape.

**Fair value** is the filter. It never picks the side — it decides whether the price is
wrong enough to bother.

| | Winner | Totals (over/under) |
|---|---|---|
| **MLB** | log5 on season records + home field, blended 80/20 with an Elo rating built from every game back to 2023 | Expected runs from team offense × opponent run prevention, where prevention folds in the starter's RA9 regressed by innings, plus a park factor. Negative binomial → P(over) |
| **NFL** | Elo only. 17 games a season is far too few for a stable in-season win%, so ratings carry over from the prior season with a third regressed to the mean | Same shape as MLB, but blended with last season's full rate weighted by games played so far |

A recommendation needs the flow and the edge to agree, the edge to clear 5.5¢, the spread
to be tradeable, and the confidence to clear its floor. Size is quarter-ish Kelly against
the live account balance, floored to whole contracts.

## Does it work

Cautiously yes, on a sample that is still too small to be sure.

![Cumulative profit across the MLB season](docs/images/pnl-curve.png)

The backtest is walk-forward against real settled Kalshi markets at real pregame trade
prices. Constants were tuned on games through July 15 and validated on everything after,
which is the only half that means anything: **+12.2% ROI on 308 held-out bets**. Over the
whole 505-bet window it's +10.4% on a 51.9% win rate, with a 32.6% peak-to-trough drawdown.

The forward test is the paper ledger — bets the agent recommended near first pitch that I
then actually placed. As of September 3, that's **39-20 (66%), +12% ROI, +$9.07** on a $50
account. Totals are 22-8, winners 17-12. NFL has no settled record yet; the regular season
starts September 4.

Two honest caveats. One partial season is one partial season — 500 simulated bets with a
per-bet standard deviation near 0.9 is encouraging, not conclusive. And the P&L chart was
generated at `kelly_fraction` 0.25; I've since cut it to 0.20 to pull drawdown down, which
lowers the compounded dollar figure while leaving ROI-on-staked about where it was.

## Things I tried that didn't work

This is most of the actual work, so it's worth writing down. Every change has to beat the
current model on *both* the training split and the held-out split before it ships. That bar
has killed almost everything I've built:

- **A pitching matchup blend on the winner side.** Looked like a clean improvement on a
  single full-sample test. Validation performance got monotonically worse as I raised its
  weight, best at zero.
- **Starting pitcher on the winner model.** The largest obviously-missing input, and it
  made the probabilities measurably *more accurate* while making them less profitable. The
  starter is the single most publicized input to a baseball line, so the market has it
  fully priced — adding it just moved my model toward the market's price and dissolved the
  disagreements that were making money.
- **Weather on totals.** Built the whole wind-and-temperature hydrate. Moved the number
  about 10%, nowhere near enough to close the gap I was chasing, and hurt held-out results.
- **A heavier right tail on the totals distribution.** This one genuinely improved
  calibration and still lost money, because the market prices high-scoring games correctly.
  The miscalibration is real and not exploitable.
- **Resting limit orders instead of paying the ask.** Fills happen precisely when the price
  falls to you, and by then the last-cycle taker has usually captured more of that decline
  than the limit would have.
- **Filtering winner bets for bigger edges.** Backwards: the model's biggest disagreements
  with the market are disproportionately its own errors.

The pattern that keeps showing up is that better Brier scores and better ROI are not the
same objective, and when they conflict the market is usually right.

## Running it

```
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
```

Credentials go in a `Kal_API.txt` one directory above the repo — outside the working tree
so it can't be committed — or anywhere you point `KALSHI_KEY_FILE`:

```
APIkeyid = <your key id>
Privatekey = -----BEGIN RSA PRIVATE KEY-----
<base64>
-----END RSA PRIVATE KEY-----
```

Then:

```
.venv\Scripts\python cli.py status          # account, exchange, positions
.venv\Scripts\python cli.py rank            # today's ranked recommendations
.venv\Scripts\python cli.py inspect TICKER  # quote, book, flow, fair value for one market
.venv\Scripts\python cli.py loop --paper    # scan, alert, log to the paper ledger (no real orders)
.venv\Scripts\python cli.py loop --live     # scan and SUBMIT REAL ORDERS automatically (real money)
.venv\Scripts\python cli.py settle          # grade the ledger against outcomes
```

Alerts are optional. Drop SMTP credentials and/or an [ntfy](https://ntfy.sh) topic in
`notify_config.txt` (gitignored) and you get a text and a phone push per triggered bet,
with a tap-through to the market. Leave it out and the loop runs fine and silent.

The backtests take longer and are worth it:

```
python -m backtest.bankroll_sim      # walk-forward, Kelly-sized, real prices
python -m backtest.season_backtest   # leak-safe calibration against settled markets
python -m backtest.ledger_stats      # slices of the real ledger
python -m backtest.nfl_model_backtest
```

`bankroll_sim --sweep <param> <values...>` re-runs a single parameter against the cached
data without hitting the network. That's how every constant in `config/settings.py` got
where it is, and most of those comments explain which sweep put it there.

## Layout

```
config/     settings, credential loading, sport dispatch by ticker prefix
kalshi/     signed API client (RSA-PSS) and response normalization
data/       per-sport game data and fair-value models
signals/    money flow, Elo, calibration, recommendation assembly
engine/     scanner, scan loop, snapshots, paper ledger, notifications
backtest/   the validation suite
output/     console reporting and the PNG dashboard
```

Kalshi's API uses a fixed-point schema (`*_dollars` as strings, `*_fp` for sizes) that is
easy to get subtly wrong — everything goes through `kalshi/normalize.py` for that reason.

## Where this is going

**NBA and NHL are next.** Both fit the existing shape: money flow is entirely Kalshi-native
and needed no changes at all when NFL was added, so the work is a data source and two
fair-value models per sport. NBA should be the easier of the two — 82 games gives a stable
in-season signal, possessions and pace make totals fairly tractable, and rest/back-to-backs
are a real and well-documented effect the market doesn't always price. NHL is harder;
scoring is low enough that game outcomes are noisy, and I expect goalie starts to matter to
totals roughly the way starting pitchers do in baseball. Given what the pitcher experiment
found, that's an argument for goalies on totals and against them on winners.

Other things on the list:

- Money flow picks the side for every bet, and the two backtests disagree about how well it
  does that on winner markets specifically. That disagreement is the most interesting
  unresolved thing in the project and where I'd dig next.
- Open-interest momentum carries 25% of the flow weight and scores an information
  coefficient close to noise. Dropping it didn't improve ROI either, which is its own
  puzzle.
- The per-print size data needed to properly validate the large-print detector only started
  being recorded recently. It needs a few more weeks before it can be tuned like everything
  else.
- Bullpen quality is proxied by team rate, and NFL has no quarterback or injury signal at
  all.

## Disclaimer

This is a personal project and it's been profitable over a sample small enough that it
could still be luck. Nothing here is financial advice. Run it recommend-only (`loop`,
`loop --paper`) and you're the one placing every order. Run it with `loop --live` and it
places real orders itself, automatically, with real money, subject only to the per-order
caps in `config/settings.py` — you are still the one who owns the outcome and the risk,
you've just moved the trigger finger from yourself to the code.
