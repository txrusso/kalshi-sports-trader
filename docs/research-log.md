# Research log

Working notes on why the model is shaped the way it is. Most of these entries are things
that didn't work — that's the useful part, and it's why several source files point here
instead of carrying a three-paragraph comment.

The rule everything is held to: a change ships only if it beats the current model on
**both** the training split (settled markets through July 15) and the held-out validation
split (everything after). A win on one split and a loss on the other is an overfit, not an
improvement, and I've reverted a lot of ideas on exactly that evidence.

---

## Position sizing

`kelly_fraction` started at 0.35 with `max_stake_pct` 0.15. The walk-forward Kelly bankroll
simulation in `backtest/bankroll_sim.py` showed 0.35 carrying 50–70% peak-to-trough
drawdowns on real settled prices with no better held-out return than a smaller fraction —
textbook Kelly overbetting, which happens because full-Kelly math assumes your edge estimate
has no error and mine obviously does.

Dropped to 0.30, then refined to 0.25 the same day on a finer sweep. 0.25 beat 0.30 on both
halves of the dataset independently, higher ROI and lower drawdown, so it wasn't a
train-flattering pick.

Cut again to 0.20 on 2026-09-03, this time deliberately trading growth for calm. Drawdown
falls monotonically with the fraction on both splits; 0.20 took validation drawdown from
23.9% to 19.5% and training from 32.6% to 25.0%, with ROI-on-staked flat to slightly better
and the bet count barely moving. The cost is compounding, not edge — terminal bankroll on
the validation split drops from $97 to $70, because ROI-on-staked is size-invariant and
growth is not.

**Don't go below about 0.15.** With a small bankroll and ~$0.50 contracts, the
whole-contract floor starts silently eating bets: at 0.10 only 96 of 308 validation bets get
taken at all, because Kelly wants less than one contract and it rounds to zero. Below that
point the rounding is the sizer, not the model.

`min_edge_cents` went 4.0 → 3.0 → 5.5. The last move was the surprising one: after the cache
picked up about three more weeks of settled markets, 5–6.5¢ beat 3¢ on both splits at once,
with drawdown falling on both too. 3¢ was leaving money on the table by taking marginal
bets. Landed at 5.5 from the middle of the supported band rather than the single best grid
point.

**`max_stake_pct` was doing nothing at all.** A full grid over every sweepable parameter on
2026-09-07 turned up exactly one that wasn't near its optimum, and the interesting part is
*why*: at 0.25 the cap was **inert**. Values of 0.12, 0.16, 0.20 and 0.25 produce
byte-identical results, because fractional Kelly never asks for more than about 11% of
bankroll — the cap simply never binds, so the number was decorative. It had been raised to
0.25 by request on 2026-08-31 and never validated, and the sweep says the level was not so
much wrong as unused.

Values that *do* bind form a plateau at 0.045–0.06 that beats production on both splits, and
it replicated independently on two separate cache builds. At 0.06 training ROI goes +0.0018
→ +0.0278, validation +0.0507 → +0.0518, validation drawdown 40.7% → 39.8%, bet count
unchanged at 345. Shipped 0.06 rather than the nominally better 0.045 (validation +0.0583)
because 0.045's immediate neighbours 0.04 and 0.05 *both* fail the both-splits bar — a knife
edge — while 0.055/0.06/0.065 sit on a flat shelf (training +0.0287/+0.0278/+0.0288). 0.06 is
also the least aggressive point in the band, so it stays non-binding on bet count as the
bankroll grows.

Read this one honestly: 0.06 **ties** on validation (+0.001) and wins clearly on training.
The argument for it is that it replaces a provably dead value with a live one, not that it's
a big edge gain.

The same grid re-confirmed the rest. `sp_weight` 0.62, `pitcher_reg_ip` 40, `totals_phi` 2.2,
`min_edge` 5.5¢ and `elo_home_field` 27.85 are all clean validation peaks. `winner_regress`
and `sp_winner_beta` stay at 0 — both still improve validation and fail training, the exact
shape that got them reverted before. `league_shrink` 0.05 cleared the bar on one cache build
and lost on the other, so it's noise. `elo_weight`/`K_FACTOR` stay at 0.2/3.5 despite
0.1/4.0 edging them on one build; the neighbouring cells disagree between builds, which is
the same knife edge noted on 2026-09-02.

The money-flow weights, `min_confidence` and `max_spread_cents` still can't be tuned at all.
Replaying snapshots through the real recommendation logic at the current 5.5¢ gate yields
n=15 training / 73 validation bets, and nothing wins on both splits. `max_spread_cents` is
also close to inert — 2¢ through 6¢ barely changes the bet set.

---

## The winner model

MLB winner probability is near a coin flip by nature — Brier hovers around 0.247 no matter
what I do to it. That isn't a fixable overconfidence bug, it's what predicting baseball
games from team records looks like. Checked directly: across all settled winner markets the
favorite gap is only about 2.7 points (58.0% predicted vs 55.3% actual). Mildly
overconfident, not badly.

**Elo (shipped).** Fair value is now `0.8 × log5 + 0.2 × Elo`, with the Elo ratings built
from a chronological walk over every game back to 2023. The motivation was smoothing
early-season noise, since 162 games already give log5 a strong signal and there wasn't much
room for Elo to add accuracy. And it didn't add accuracy — Brier barely moved across the
entire weight sweep. What it changed was *which* bets clear the edge gate and how they get
sized, and that lifted held-out ROI from 9.8% to 11.0% while cutting drawdown on both
splits. Weight past about 0.5 overfits in the familiar shape (training keeps improving,
validation degrades), which is why production sits at a moderate 0.2.

**Elo K factor.** Seeded at 6.0 by analogy to the NFL model's K=20 scaled down for a longer
season, and never actually swept until later. Sweeping it found a flat plateau at 3.2–3.8
beating 6.0 on both splits and every metric at once — held-out winner ROI 0.0995 → 0.127,
training drawdown 42.2% → 32.6%. Shipped at 3.5, the middle of the plateau rather than the
single-point peak. Joint `elo_weight` × `K_FACTOR` grid afterwards confirmed the pair is a
local optimum; notably the weight dimension is knife-edge (0.18 and 0.22 both underperform
0.2 at every K tested), so re-tuning the weight alone at a different K could land somewhere
misleadingly bad.

**Home field in Elo** was originally derived from log5's fixed 0.54/0.46 assumption purely so
an early Elo-vs-log5 comparison wasn't confounded by two independently fitted numbers. Once
Elo was carrying real weight there was no reason it had to match, so I swept it 0–70. The
derived value (27.85) was already the best point tested, and several others were much worse.
Accidentally well-tuned.

### Reverted

- **Recency window.** Last-30-games win% instead of season-to-date. Too noisy at n=30 for a
  binary sample.
- **Pythagorean pitching blend.** Full-sample Brier improved 0.2500 → 0.2475, which looked
  real. Validation got monotonically worse as the weight rose; best at weight zero.
- **Regressing win% toward .500.** Motivated by a 21-bet diagnostic that looked like severe
  favorite overconfidence, which the full sample didn't support. Trivial Brier gain, worse
  accuracy, splits disagreed. Kept as an inert tunable — a mild regression did cut validation
  drawdown 32% → 23%, which is the one result worth revisiting with more data.
- **A bigger edge threshold for winner bets only.** The hypothesis was that noisier winner
  probabilities need a bigger cushion. The result was the opposite and dramatic: filtering
  for bigger modeled edges made winner ROI far worse, win rate falling from 0.47 to 0.20–0.30
  as the bonus rose. The model's biggest disagreements with the market are mostly its own
  errors.
- **Blending winner fair value toward the market price.** Helped training a lot, hurt
  validation by about the same amount. Same shape as every other overfit here.
- **Starting pitcher on the winner side.** The most interesting failure in the project. The
  winner model uses no pitcher information at all, while the totals model has regressed
  starter RA9 baked in, so this was the largest obviously-unexploited input. Implemented
  properly — each starter's RA9 regressed toward league average by innings, compared against
  that team's own run-prevention rate so it adds "who's on the mound today" without
  re-litigating team strength — and swept 0.0–0.4. Every non-zero value was worse on both
  splits, monotonically. Held-out winner ROI fell 0.127 → 0.062 → 0.009 → −0.031 as beta went
  0 → 0.05 → 0.1 → 0.3.

  And **Brier improved while ROI degraded**. The pitcher information makes the probabilities
  genuinely more accurate and simultaneously less profitable, because the starter is the
  single most publicized input to a baseball line. The market has it fully priced; adding it
  moves the model toward the market's price and dissolves exactly the disagreements that were
  making money. The winner model should stay pitcher-free.

---

## The totals model

This is where the profitable edge actually lives. `LEAGUE_SHRINK` went to 0.0 (was 0.15,
removed on held-out evidence) and `TOTALS_PHI` sits at 2.2.

### The low-bucket miscalibration, and why it's a dead end

There's a real and large calibration failure at the extreme low-predicted-probability end.
In the 0–10% bucket the model says about 5.6% and reality is about 36% across 630 markets;
the 10–20% bucket is off too. Everything from 30% up is well calibrated.

I chased this twice.

**Weather.** A game-time wind and temperature multiplier on lambda, fully plumbed through
the backtest cache. A bounded wind factor only shifts lambda 9–12%, nowhere near enough to
close a 5.6% → 36% gap, and it made validation Brier, ROI and drawdown all worse. Reverted.
Details in `backtest/baseline_20260831.txt`.

**Tail shape.** The hypothesis that the negative binomial's right tail is too thin at high
lines. A mean-preserving heavier-right-tail mixture *did* improve held-out calibration
cleanly — validation Brier 0.1955 → 0.1942 without breaking the buckets that were already
fine. Then a targeted ROI test on real pregame prices across all 2,586 low-probability lines
showed it loses money, worse on both splits.

So the miscalibration is real but not exploitable. The market prices high-scoring games
correctly, and moving the model toward reality just moves it toward the market's prices.
Calibration up, profit down — the same lesson as the pitcher experiment. **Leave the totals
tail alone.** The edge is in the near-money lines where the model is already well calibrated
and occasionally disagrees with the market correctly.

### Known gaps

No weather signal. Bullpen quality is proxied by the team rate. Both are real, both have
resisted a profitable fix so far.

---

## Money flow

The 40/35/25 weights were hand-set and went a long time without being scored. Two tools now
exist to check them.

`backtest/evaluate.py` reports each component's information coefficient separately. First
read over 2,085 graded markets: book 0.545, trades 0.286, OI momentum 0.090. OI momentum
looks like near-noise and is inert on about half of markets, which makes its 25% weight look
like dead weight.

`backtest/replay_moneyflow.py` replays stored snapshots through the *real* recommendation
builder — exact side, gate and sizing logic — and grades realized P&L, so the strategy being
measured is the strategy being traded. It's the only backtest that exercises the money-flow
layer and confidence gate at all; the others bypass them entirely. Two results that the IC
alone got exactly wrong:

1. Dropping OI momentum, the *lowest*-IC component, does not improve ROI.
2. Book-only, the *highest*-IC component, is the worst ROI of any weighting tested.

Book imbalance at the latest snapshot largely re-reads the market price, so its IC is
inflated by favorite bias rather than reflecting real edge. The weights stay as they are.

The same harness surfaced something more interesting: money-flow **winner** bets lose across
every weighting tested, while totals carry all the positive ROI. But the fair-value backtest
now shows winner ROI roughly matching totals on the held-out split. The two backtests
disagree, and the difference between them is money flow's side selection on winner markets.
That's the most promising unexplored thread in the project.

Real-ledger slices point the same way: ROI *falls* as money-flow strength rises (+38% flat,
+21% mild, +4% strong). Small samples, but it rhymes.

### Large prints

Kalshi is a centralized, CFTC-regulated exchange and its trades are anonymous — no account
id, no per-trader P&L, no leaderboard. Following specific whales the way you can on-chain is
simply not possible. The buildable proxy is to infer a whale from a print that's large
relative to a market's *own* flow: size at least 4× the median trade (or an absolute floor,
whichever is larger), or Kalshi's formal block flag.

This is deliberately modest and **not walk-forward validated**, because it can't be — the
snapshot history only ever stored the derived flow score, never the raw per-print sizes.
Per-snapshot large-print stats are now being recorded so it can be tuned properly once a few
weeks accumulate. Setting the weight to 1.0 disables it.

Live sanity check: the two sides of a game confirm each other, e.g. whales +1.00 on one
team's ticker and −0.86 on the opposing ticker for the same game.

---

## Execution

Does resting a limit order beat paying the ask? `backtest/limit_entry.py` reconstructs each
bet's pregame ask time series from snapshots, simulates a limit some cents inside the ask at
two placement times, and grades taker vs limit-skip vs limit-fallback against real outcomes.

Limits modestly **hurt**. Taker ROI 0.088 against 0.039–0.069 for limit-fallback across every
offset and both placements, roughly flat on winners and worse on totals. The reason is
visible in the data: the recorded saving is negative. A limit fills precisely because the
price fell to it, and the last-cycle taker captures that same decline and usually more. On
this model's bets the entry price already drifts favorably into first pitch, so there's no
cheaper price left for a limit to catch. No adverse-selection signal either way.

Taking the ask on the last pregame cycle is already close to optimal.

---

## NFL

Money flow needed zero changes — it's Kalshi-native and sport-agnostic. Only fair value is
sport-specific.

Data comes from nflverse's `games.csv`, which covers in one free file what MLB needs three
separate API endpoints for: 1999 to present, every score, starting quarterbacks, and roof,
surface, temperature and wind. Two team codes differ from Kalshi's and are mapped
explicitly (`JAX`/`LA` vs `JAC`/`LAR`) — the same class of bug as the Athletics' `OAK` → `ATH`
abbreviation change between 2023 and 2026, which would otherwise have reset that franchise's
Elo rating at the rename.

Winner is Elo rather than log5-on-record because a 17-game season is too noisy for a stable
in-season win percentage. Ratings regress a third of the way to the mean at each season
boundary, and `rating_before(team, date)` only ever reads games strictly before the query
date, so leak-safety is structural rather than something the backtest has to be careful
about. K=20 was swept 10–30 against real 2025 outcomes and left at the 538-standard value;
differences were small and only tested on one season.

### Contract-shape gotchas, both found live

1. NFL tickers have no time-of-day segment where MLB's do. Kickoff comes from the market's
   own `occurrence_datetime` field instead, which turns out to be present on MLB markets too
   and is now what the paper trigger uses for both sports.
2. NFL's market `title` is just `"<Team> wins"` with no opponent, unlike MLB's shared
   `"A vs B Winner?"`. Matchup names come from `rules_primary` instead, which is present on
   every market in the bulk list response at no extra API cost. The competition-type phrase
   between the team names varies ("Pro Football" vs "professional football"), and an
   open-ended word-class match in that regex silently truncated "LA Rams" to "LA" until it
   was anchored on both forms explicitly.

### Preseason is a real blind spot, and it's handled

Settled Kalshi NFL markets existed well before the regular season, so I graded against them.
The model applied each team's *regular-season* strength to preseason games and did badly:
winner Brier 0.284, which is worse than a coin flip, 52% accuracy, and at the production edge
gate only 4 of 25 real-money-priced bets won. Totals lost 38.7% ROI.

The cause is obvious in hindsight — preseason rosters lean on backups, so regular-season team
strength doesn't transfer, and the market prices that while the model didn't. Both NFL
fair-value models now refuse to estimate whenever the game can't be found in `games.csv`,
which excludes preseason entirely and so doubles as a reliable preseason detector. Same
"unverifiable means unsafe" treatment as an unmatched MLB game.

The same gap means NFL preseason bets can never settle — they resolve through `games.csv`
too, so such rows sit pending forever. That's the data gap, not a settlement bug.

NFL therefore has **no real regular-season track record yet**. MLB and NFL share one account
and one ledger, so `settle` breaks results out by sport to keep one model's record from
hiding inside the other's. Confidence calibration is likewise bucketed by sport.

---

## NBA, and the case for/against other sports

Scoped 2026-09-10 by auditing live Kalshi markets and candidate free data sources before
writing any model code, the same way NFL's feasibility was checked first. The audit covered
NBA, NHL, and briefly why soccer and college sports don't fit the current architecture.

**NBA** checked out cleanly. `KXNBAGAME`/`KXNBATOTAL` are real, live, and ticker-shaped
exactly like NFL's (no time-of-day segment, `rules_primary` carries the matchup since the
market `title` doesn't). Kalshi's 30 team codes matched a candidate data source
(`sportsdataverse-data`'s NBA game logs) exactly, with no NFL-style remap needed. So it was
built the same day — see the "NBA model" section of CLAUDE.md for the full build, the
already-corrected `HOME_FIELD_ELO` finding, and the one real open gap (the 2026-27 season's
game-log file doesn't exist upstream yet, five weeks before Kalshi's own listed openers;
should resolve on its own before the season starts trading, but worth a live recheck).

**NHL** built the same session, right after NBA. `KXNHLGAME`/`KXNHLTOTAL` are real
(confirmed by search, not yet live-open this far from October), and hockey always has a
winner — no draw problem the way soccer would have. The data source is the best of the
three sports added this way: `api-web.nhle.com` is a live official NHL API, confirmed
reachable, the NHL analog of MLB Stats API rather than a third-party mirror — and unlike
the NBA mirror, it already had the full 2026-27 season schedule published even though the
season hadn't started. The shootout-goal question (does Kalshi's total-goals settlement
count a shootout-winning goal?) got answered empirically before writing the totals model:
pulled a real shootout game's play-by-play and confirmed the official final score already
bakes in the deciding goal, the same convention every broadcast uses, so no special-casing
was needed. The Elo model had no published methodology to start from the way NFL's and
NBA's did (538 never shipped a public NHL model), so it shipped as a standard Elo, swept
against two real seasons before shipping — K_FACTOR corrected from a borrowed NBA seed of
20 down to 8, landing on real Brier ~0.238, a modest but genuine signal, appropriately
weaker than NBA's given hockey's well-known single-game variance. See CLAUDE.md's "NHL
model" section for the full build and the honest list of what's still unconfirmed against
a live market (ticker format and matchup-text phrasing are carried over from the other
three sports' convention, not yet independently verified — no NHL market has been open to
check against, and last season's settled markets have already rolled off Kalshi's
~68-day settled-market window).

**Why not soccer**: draws break the binary win/lose assumption baked into the whole
recommendation pipeline (`headline_for`, side selection, `resolve_outcomes`) — Kalshi's
soccer markets are 3-way (win/draw/lose), and that's a real restructuring, not a new data
source plugged into the existing shape.

**Why not college football/basketball**: real Kalshi volume, but a noisier team-strength
signal (roster churn every year, weaker historical baselines than the pros) for a worse
return on the same build effort as NBA/NHL.

---

## Operational notes

The loop needs the machine awake and online, which is more fragile than it sounds. Two
failure modes cost real trigger windows before they were diagnosed:

- **Modern Standby throttling.** This hardware only supports S0 low-power idle, which throttles
  background processes about ten minutes after the system goes idle even with a keep-awake
  call held. Fixed at the OS level with `powercfg` monitor and standby timeouts, and in code
  by what `_keep_awake()` asserts.
- **Lid close**, which is a completely separate Windows setting (`SUB_BUTTONS` / `LIDACTION`)
  from the idle timeouts, was hidden from the default `powercfg` listing and set to sleep.

Both fixes are AC-only by design. `_keep_awake()` currently requests only
`ES_SYSTEM_REQUIRED`, which blocks sleep but lets the display idle off — that reintroduces
some throttling risk, and if delayed scans come back the fix is to add `ES_DISPLAY_REQUIRED`
back to the flags.

Notifications are best-effort by construction: the announce-and-notify block is wrapped in
its own try/except so a formatting or send failure can't kill a scan cycle, and bets reach
the ledger whether or not the message goes out. Worth knowing that the carrier SMS gateway
has silently dropped or delayed messages more than once even when the SMTP transaction
completed normally, which is why there's a push channel as well.

### Scan coverage is a silent limit

The scanner lists every market cheaply, then deep-scans only the richest events by volume
and open interest, capped by `max_deep_markets`. A market that isn't deep-scanned can never
produce a recommendation no matter how large its edge, so that cap is a coverage limit
rather than a performance knob.

It was costing real bets. With MLB and NFL both live, a cap of 50 was exhausted by about
rank 15 of the liquidity-ranked event list, and a totals line sitting at rank 19 with a
+9.9¢ edge and agreeing money flow was never evaluated at all. The squeeze is structural: an
MLB totals event costs 8–10 slots because it has one market per over/under rung, while an
NFL game event costs 2, so NFL volume crowds out exactly the expensive MLB totals events
that carry most of the realized profit.

Raised to 100. Deep-scan coverage went from 50 markets across 15 games to 100 across 26,
recommendations from 4 to 8–9, and cycle time from ~12s to ~30s, which is nothing against a
1800-second interval. If both sports ever run heavy slates concurrently again, check this
first — the symptom is a market with a live edge missing from `rank` while `inspect` on its
ticker shows the edge is real.

### `--refresh` used to eat the training split

Found on 2026-09-07 while rebuilding the cache before a parameter grid. Kalshi's
`/markets?status=settled` endpoint serves only a **rolling window of roughly 68 days** of
settled markets. Local pagination is uncapped, so this isn't a truncation bug in
`fetch_settled_markets` — the data is simply gone from the API. That means a plain overwrite
on `--refresh` amputates the oldest games every time: the 2026-09-06 rebuild silently lost
Jun 22–30 entirely, 111 winner and 109 totals events, dropping the training split from n=175
to n=128 bets.

The failure is quiet and it compounds. `TRAIN_END` is 2026-07-15, so a couple more refreshes
would have left the training split empty, and every "beats production on both splits" check
in this file would have degraded to a validation-only test **without raising an error**. The
whole discipline the project runs on would have quietly stopped working.

`build_cache` now merges instead of overwriting. That's safe because a settled row is
immutable — across the 743 winner and 741 totals events present in both the 2026-08-29 and
2026-09-06 builds, every entry price and every outcome matched exactly, zero differences.
Rows key by event, the same dedupe key the builder already uses upstream. `_warn_if_train_thin()`
prints a warning if the cache ever stops reaching back to `KALSHI_DATA_START` anyway.

Merging the two surviving builds recovered a cache spanning Jun 22 – Sep 5, 961 winner and
957 totals rows, about 13% more data than either build alone. The Jun 7–21 games are gone for
good, so the warning fires today and correctly keeps firing.

One thing the fuller data changed: validation totals ROI collapsed from +0.127 to roughly
zero once Aug 29 – Sep 5 was included, and validation max drawdown rose to 40.7%. Winner ROI
held at +0.119. Worth watching rather than acting on — it's one bad week, not a signal.
