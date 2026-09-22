"""Central configuration: endpoints, loop cadence, and risk / signal thresholds.

By default this agent is recommend-only: it scans markets, scores them, and
writes ranked recommendations, and all 'sizing' numbers are advisory only.
Setting `live_trade=True` (via `loop --live`) switches the T-minus-start
trigger from paper-recording to actually submitting orders through the Kalshi
API — see engine/live.py for what that does and does not guard against.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from zoneinfo import ZoneInfo

# --- Display timezone --------------------------------------------------------
# All user-facing timestamps (console, texts, logs, date defaults) use this.
# Kalshi tickers, MLB game times, and "today's slate" are all naturally ET.
EASTERN = ZoneInfo("America/New_York")

# --- Kalshi endpoints -------------------------------------------------------
# Production trade API. Signing paths must include the full '/trade-api/v2' prefix.
KALSHI_API_BASE = "https://api.elections.kalshi.com/trade-api/v2"
KALSHI_DEMO_BASE = "https://demo-api.kalshi.co/trade-api/v2"

# --- Project paths ----------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOGS_DIR = PROJECT_ROOT / "logs"
SNAPSHOTS_DIR = PROJECT_ROOT / "snapshots"
OUTPUT_DIR = PROJECT_ROOT / "output"


@dataclass
class Settings:
    # Environment
    use_demo: bool = False

    # Loop cadence (seconds). ~20 min = balanced money-flow tracking.
    scan_interval_seconds: int = 20 * 60

    # --- Market universe (MLB + NFL + NBA + NHL) ---
    # Kalshi series tickers for game markets. Discovered/verified at runtime; these
    # are the prefixes we filter events by.
    # KX**GAME = per-game winner markets; KX**TOTAL = total-runs/points/goals
    # (over/under) ladders. (KXMLB*/KXNFL*/KXNBA*/KXNHL* futures are excluded — the
    # fair-value model is game-based.) All sports scan in the same cycle;
    # config/sports.py dispatches each ticker to its sport's fair-value model and
    # calibration bucket.
    # NBA added 2026-09-10; KXNBAGAME confirmed live (real Oct 20 2026 openers
    # already listed), KXNBATOTAL confirmed real but not yet listed this far out
    # (same as MLB/NFL totals) -- see data/nba_data.py for the model build notes.
    # NHL added 2026-09-10; neither series is listed yet this far from the season
    # (confirmed live) -- ticker shape confirmed via web search against real
    # historical markets only, see data/nhl_data.py/data/fair_value_nhl.py.
    # NFL spread added 2026-09-13 (KXNFLSPREAD, confirmed live) -- one ladder of
    # "team wins by over K.5" rungs PER TEAM per game (unlike totals' one shared
    # ladder), so an event costs up to ~24 markets, worse than MLB totals' 8-10 --
    # see spread_min_mid/max_mid below, and the deep-scan-budget note above.
    sport_series_prefixes: tuple[str, ...] = (
        "KXMLBGAME", "KXMLBTOTAL", "KXNFLGAME", "KXNFLTOTAL", "KXNFLSPREAD",
        "KXNBAGAME", "KXNBATOTAL", "KXNHLGAME", "KXNHLTOTAL",
    )
    include_totals: bool = True           # scan over/under (total runs) markets
    totals_min_mid: float = 0.12          # only consider near-the-money O/U lines
    totals_max_mid: float = 0.88          #   (skip deep ITM/OTM ladder rungs)
    include_spreads: bool = True          # scan NFL point-spread markets
    spread_min_mid: float = 0.12          # only consider near-the-money spread rungs
    spread_max_mid: float = 0.88          #   (same rationale as totals_min_mid/max_mid)
    market_status: str = "open"          # only scan open markets
    min_market_volume: int = 50          # ignore illiquid markets (contracts traded)
    min_open_interest: int = 50
    max_deep_markets: int = 1500         # cap markets deep-scanned/cycle (bounds API load).
                                         # Raised 50 -> 100 on 2026-09-02: with MLB and NFL
                                         # both live, the 50 cap was being exhausted by rank
                                         # ~15 of the liquidity-ranked event list, and real
                                         # qualifying bets were never being evaluated (found
                                         # live: TOR@CLE Under 8.5 sat at rank 19 with a
                                         # +9.9c edge and agreeing money flow, unscanned).
                                         # The squeeze is structural, not incidental: an MLB
                                         # totals event costs 8-10 slots (one per O/U rung)
                                         # while an NFL game event costs 2, so cheap NFL
                                         # events crowd out exactly the expensive MLB totals
                                         # events that carry most of the realized profit.
                                         # Deep scan is 2 API calls/market (book + trades),
                                         # so this doubles per-cycle calls (~100 -> ~200),
                                         # still far inside a 30-min cycle.
                                         #
                                         # Raised 100 -> 500 on 2026-09-18, at the user's
                                         # request, after NFL spread markets (added
                                         # 2026-09-13, up to ~24 rungs/event) made the squeeze
                                         # far worse: a live scan the same day found 83 of 85
                                         # deep-scanned markets were NFL and only 2 were MLB,
                                         # despite 15 real MLB games on the slate that night --
                                         # NFL spread ladders alone were exhausting the 100 cap
                                         # before the liquidity ranking ever reached most MLB
                                         # events. Measured against that live candidate list
                                         # (1,053 tradeable markets/166 events) before shipping:
                                         # budget 100 -> 1 MLB event covered / 16 NFL; budget
                                         # 500 -> 14 MLB / 37 NFL. MLB games are genuinely
                                         # lower-liquidity than an NFL Sunday slate right now, so
                                         # they'll never fully catch up in a liquidity-ranked
                                         # scan, but 500 gets meaningfully more MLB coverage
                                         # without a real cycle-time cost (~5x the ~100-market
                                         # cycle time, still on the order of minutes against the
                                         # 1800s loop interval). Re-check this balance if a
                                         # sixth market kind or another high-rung-count sport
                                         # ships and squeezes things further.
                                         #
                                         # Raised 500 -> 1500 same day (2026-09-18), at the
                                         # user's request, after confirming there were only
                                         # ~1,055 total tradeable markets live at the time --
                                         # so 900-1000 already gave FULL coverage of both
                                         # sports (60/60 MLB events, 63/63 NFL events measured
                                         # live), and 1500 changes nothing about today's actual
                                         # cycle behavior (still capped by the real candidate
                                         # count, not this ceiling). It's pure headroom: buys
                                         # room for the candidate count to grow (more sports
                                         # live, more markets opening) before this cap starts
                                         # binding again, at negligible cost since deep-scan
                                         # cost scales with the ACTUAL number of candidate
                                         # markets, not the ceiling itself (measured cycle time
                                         # at full coverage of the live candidate list was
                                         # ~288s, ~16% of the 1800s loop interval).

    # --- Money-flow signal weights (sum need not be 1; normalized internally) ---
    w_book_imbalance: float = 0.40       # dollar-weighted resting depth skew
    w_trade_flow: float = 0.35           # recent aggressor-side pressure
    w_oi_momentum: float = 0.25          # open-interest change w/ price direction

    trade_flow_lookback: int = 100       # recent trades to analyze per market
    book_depth_levels: int = 10          # order book levels to weight

    # --- Large-print ("whale") detection, folded into the trade-flow component ---
    # Kalshi trades are anonymous (no account id, no leaderboard -- confirmed 2026-09-01),
    # so "follow the whales" can only be approximated: flag prints that are large relative
    # to a market's OWN flow and up-weight them in the aggressor-dollar signal, since a big
    # conviction print is likelier to be informed/sized money than a retail click. This
    # replaces the old hard-coded 2.0x weight that fired ONLY on Kalshi's rare formal
    # block-trade flag; now any print >= large_print_mult x the market's median trade size
    # (or >= the absolute floor, whichever is larger) also qualifies, plus formal blocks.
    # NOTE ON VALIDATION: unlike every shipped signal constant, this one is NOT yet
    # walk-forward validated -- it CAN'T be on existing history, because snapshots only
    # stored the derived mf_trades, never the raw per-print sizes the detector needs (the
    # same "money flow needs point-in-time trades Kalshi doesn't expose historically" limit
    # backtest/replay_moneyflow.py documents). So the weight is deliberately MODEST (2.5,
    # barely above the old 2.0 block weight) and the scanner now records lp_* per snapshot
    # (engine/scanner.py) to accumulate the raw large-print data needed to tune/validate
    # this the same both-splits way as kelly_fraction/min_edge_cents once a few weeks
    # exist. `cli.py whales` shows the live detector output. Set large_print_weight=1.0 to
    # disable the up-weight entirely (reverts trade flow to size-agnostic).
    large_print_mult: float = 4.0          # size >= this x median trade size => a whale print
    large_print_min_contracts: float = 25.0  # absolute floor so tiny-median markets don't over-flag
    large_print_weight: float = 2.5        # weight multiplier applied to whale prints in flow

    # --- Edge / recommendation thresholds ---
    # Minimum modeled edge (fair prob - market prob, in cents) to surface a rec.
    # Was 4.0, then 3.0 (2026-08-13). Raised to 5.5 on 2026-08-29 after a fresh
    # walk-forward sweep (train Jun7-Jul15, validate Jul16+, cache rebuilt with
    # ~3 more weeks of settled markets than the prior sweep) showed 3c was actually
    # leaving money on the table: 5-6.5c beat 3c on BOTH splits at once (not just
    # validate) -- e.g. at 5.5c, train ROI improved -15.6%->-10.7% and validate
    # +221.8%->+262.7%, with max drawdown falling on both splits too (34.0%->25.8%
    # validate, 39.8%->30.1% train). Chosen from the middle of the well-supported
    # 5-6.5c band rather than the single best point (5.5c wasn't literally the top
    # of the grid) to avoid picking a sample-specific peak.
    min_edge_cents: float = 5.5
    min_confidence: float = 0.35         # 0..1 composite confidence floor
    max_spread_cents: float = 6.0        # skip markets wider than this (untradeable)
    # Raised 15 -> 50 on 2026-09-18, at the user's request, after the max_deep_markets
    # 500->1500 bump (full scan coverage, same day) took the number of distinct
    # qualifying games from ~8-9 (this cap's era) to 36 measured live -- 15 was quietly
    # cutting off real, already-deduped (one bet/game) recommendations. This slice is
    # pure display/trigger-eligibility, not an API-cost knob like max_deep_markets, so
    # there's no scan-time cost to raising it; sized with headroom above today's 36 the
    # same way max_deep_markets was sized with headroom above its measured need.
    # NOTE: unlike max_deep_markets, this DOES interact with live risk -- the loop has no
    # daily loss cap or max-concurrent-positions limit (a deliberate omission, see
    # CLAUDE.md's "Live order execution"), so more simultaneously-eligible recs means more
    # games CAN trigger a real order in the same cycle if their trigger windows overlap.
    # Each order is still independently capped by max_stake_pct/max_order_cost_usd and
    # submit_order() refuses if live balance can't cover it, so this raises how many
    # DISTINCT games can bet, not the size of any one bet.
    top_n_recommendations: int = 50

    # --- Advisory position sizing (NOT executed) ---
    dynamic_bankroll: bool = True        # pull live account balance as the bankroll each run
    bankroll_usd: float = 50.0           # fallback stake base if the balance fetch fails
                                         # (= total deposited capital: $20 + $30 added 2026-09-01)
    # kelly_fraction was 0.35, max_stake_pct was 0.15 (raised together 2026-08-08 --
    # see docs/research-log.md). Both lowered 2026-08-13 after backtest/bankroll_sim.py's
    # walk-forward sweep: a Kelly-bankroll simulation on real settled markets/prices
    # showed max drawdown climbing steeply above ~0.25-0.30 fraction (73% max
    # drawdown at 0.5) while dollar-weighted ROI on held-out data did not keep pace --
    # classic Kelly overbetting given our edge estimates carry real estimation error,
    # not the zero error full-Kelly math assumes. kelly_fraction refined 0.30 -> 0.25
    # same day after a finer sweep: 0.25 beat 0.30 on BOTH halves of the dataset
    # independently (so it isn't sensitive to which half is "train"), with both higher
    # ROI and lower drawdown -- not just a train-set-flattering pick.
    # 0.25 -> 0.20 on 2026-09-03, explicitly to cut drawdown. Sweep 0.05-0.35 showed
    # max drawdown falling monotonically with the fraction on BOTH splits; 0.20 cut it
    # 23.9% -> 19.5% (validate) and 32.6% -> 25.0% (train) while ROI-on-staked held flat
    # to slightly better on both (validate 0.122 -> 0.133, train -0.006 -> +0.002) and
    # bet count barely moved (validate 308 -> 304, train 197 -> 175). The cost is
    # compounding, not edge: validate terminal bankroll $97 -> $70, since ROI-on-staked
    # is size-invariant while growth is not -- a deliberate variance-for-growth trade at
    # the user's request, not a free win. Do NOT go below ~0.15: with a ~$58 bankroll and
    # ~$0.50 contracts the integer-contract floor starts dropping real bets (at 0.10 only
    # 96/308 validate and 39/197 train bets get taken at all -- Kelly wants <1 contract
    # and it rounds to zero), so below that the rounding, not Kelly, becomes the sizer.
    kelly_fraction: float = 0.20         # fractional Kelly for suggested size
    # max_stake_pct history: 0.15 (orig) -> 0.08 (2026-08-13) -> 0.05 (2026-08-29),
    # each step down after backtest/bankroll_sim.py walk-forward sweeps showed lower
    # values cut max drawdown on held-out data without giving up ROI. Raised to 0.25 on
    # 2026-08-31 per explicit user request, unvalidated; cut back to 0.06 on 2026-09-07
    # after a full-grid sweep found 0.25 was not merely suboptimal but INERT -- 0.12,
    # 0.16, 0.20 and 0.25 all produce byte-identical results, because fractional Kelly
    # never asks for more than ~11% of bankroll, so the cap never binds and the number
    # was doing nothing. Values that DO bind form a plateau at 0.045-0.06 beating
    # production on both splits, replicated independently on two cache builds: at 0.06,
    # train ROI +0.0018 -> +0.0278 and validate +0.0507 -> +0.0518, with validate max
    # drawdown 40.7% -> 39.8% and the bet count unchanged (345). Chose 0.06 over the
    # nominally better 0.045 (validate +0.0583) because 0.045's immediate neighbours
    # 0.04 and 0.05 BOTH fail the both-splits bar -- it's a knife edge -- whereas
    # 0.055/0.06/0.065 sit on a flat shelf (train +0.0287/+0.0278/+0.0288). 0.06 is also
    # the least aggressive point in the band, so it stays non-binding on bet count as
    # the bankroll grows. Honest read: 0.06 TIES on validate (+0.001) and clearly wins on
    # train -- the argument is picking a live value over a provably dead one, not a
    # tuned value over a good one.
    #
    # Raised 0.06 -> 0.10 on 2026-09-18, at the user's request, NOT re-validated by a
    # fresh bankroll_sim sweep. The walk-forward grid above found 0.12-0.25 historically
    # INERT (fractional Kelly rarely asked for >~11% of bankroll in that sample), which
    # would put 0.10 still inside the "should rarely bind" range on average -- but that
    # finding predates NFL spread markets (2026-09-13) and today's live `rank` run showed
    # several real recs already hitting the old 0.06 cap on unusually large edges (e.g.
    # Atlanta +1.5 at +21.3c), so the cap is demonstrably binding more often now than the
    # historical sample suggests. Treat 0.10 the same as the 2026-08-31 unvalidated 0.25
    # bump was treated: live at the user's explicit request, due for a proper
    # `bankroll_sim --sweep max_stake_pct` re-check (ideally after re-running with
    # fractional contract sizing, shipped 2026-09-17 -- see CLAUDE.md's "Live order
    # execution" section -- since that sweep hasn't been redone under it either) rather
    # than assumed safe.
    #
    # Reverted 0.10 -> 0.06 same day (2026-09-18), at the user's request, immediately
    # after max_deep_markets (500->1500, full scan coverage) and top_n_recommendations
    # (15->50) were both raised. Those two changes took the number of simultaneously-
    # eligible recommendations from ~8-9 to 36+ measured live -- and since the loop has
    # no daily loss cap or max-concurrent-positions limit (a deliberate omission, see
    # CLAUDE.md's "Live order execution"), a higher per-bet cap combined with many more
    # bets eligible to trigger in the same cycle meant materially more potential
    # simultaneous exposure than existed when 0.10 was set a few hours earlier under the
    # old ~8-9-rec regime. Back to the walk-forward-validated 0.06 rather than the
    # unvalidated 0.10, restoring the original per-bet safety margin under the new,
    # much wider scan/rec footprint.
    max_stake_pct: float = 0.06          # cap suggested stake at 6% of bankroll

    # --- Trade policy ---
    pregame_only: bool = True             # only recommend games in "Preview" (no live/final)

    # --- In-game trading (added 2026-09-22) --------------------------------
    # `pregame_only` and `in_game_trade` are deliberately SEPARATE switches:
    #   pregame_only=False  -> fair value is estimated for in-progress games, so
    #                          they appear in `rank`/`search` (what --allow-live
    #                          has always done). Display/eligibility only.
    #   in_game_trade=True  -> the T-minus trigger additionally accepts games
    #                          that have ALREADY STARTED, so they can fire a real
    #                          order. This is the one that risks money.
    # Both are needed to actually trade a live game; `loop --in-game` sets both.
    #
    # MLB-ONLY IN PRACTICE. Only MLB has a live model (home_win_probability from
    # the MLB Stats API, fair_source="live", confidence 0.80). NFL/NBA/NHL all
    # explicitly refuse once a game starts ("no live NFL model yet"), returning
    # prob=None -- and signals/recommendation.py now refuses to bet a started
    # game with no fair value rather than falling through to a money-flow-only
    # bet on a game whose score the model cannot see.
    #
    # UNVALIDATED. There is no backtest for this and effectively no history to
    # build one from: across 252,687 recent snapshot rows exactly 18 carry
    # fair_source="live", because pregame_only=True has suppressed them all
    # along. CLAUDE.md asserts "live bets were the junk" but nothing in
    # docs/research-log.md backs that up, so treat in-game as untested in BOTH
    # directions rather than either proven bad or safe. Default off.
    in_game_trade: bool = False           # `loop --live --in-game` turns this on
    # 90 -> 40 on 2026-09-22, the same day, after backtest/in_game_backtest.py made
    # this measurable for the first time. The sweep found a CLIFF, not a gradient:
    # held-out VALIDATE ROI (net of fees) holds +0.23..+0.34 for every window <= 50
    # min and collapses to ~+0.09 at 60, 75 and 90. Production's 90 sat well past
    # that cliff, capturing the worst bets available.
    #   window  25     30     35     40     50  |  60     75     90
    #   TRAIN  .400   .357   .232   .246   .246 | .374   .351   .281
    #   VALID  .307   .335   .238   .268   .251 | .094   .095   .077
    # Chose 40 as the upper-middle of the flat 25-50 shelf rather than 30's
    # single-point VALIDATE peak -- the same "prefer a shelf to a knife edge, and
    # prefer the point with more data" discipline that picked max_stake_pct=0.06
    # over 0.045. 40 carries n=22/28 vs 30's n=17/17.
    # HONEST CAVEAT: against production's 90, 40 wins VALIDATE decisively
    # (+0.268 vs +0.077) but is 3.5pts WORSE on TRAIN (+0.246 vs +0.281), so it
    # does not pass a strict both-splits bar. TRAIN is non-monotone across the
    # shelf (.400/.357/.232/.246/.246) at n=15-22, i.e. noise; VALIDATE is clean
    # and monotone. 25 and 30 DO pass strictly if you want the conservative pick.
    in_game_max_minutes: float = 40.0     # stop betting this long after first pitch
    # Seeded at 8.0 by judgment, then swept (at max_minutes=45) and CONFIRMED:
    #   edge    5.0    6.5    8.0    10.0   12.0
    #   TRAIN  .114   .212   .246   .418   .091
    #   VALID  .114   .166   .231   .185   .127
    # 8.0 is the VALIDATE peak and sits mid-shelf (6.5/8/10 all +0.17..+0.23),
    # so it's kept. Below 6.5 the bar lets in too much noise; above 10 the sample
    # collapses and both splits fall.
    in_game_min_edge_cents: float = 8.0
    paper_trade: bool = False             # (loop) paper-bet each game near its first pitch
    live_trade: bool = False              # (loop --live) submit REAL orders near first pitch --
                                          # see engine/live.py. No confirmation step; gated only
                                          # by max_order_contracts/max_order_cost_usd below.
    paper_trigger_buffer_min: float = 15.0 # extra minutes on the trigger window (vs interval)
    # Changed 2026-08-30 (was 3.0): with a 30-min scan interval, a 3-min buffer meant
    # a 33-min trigger window -- worst case (cycle lands just past the window) the next
    # cycle is 30 min later, leaving only ~3 min before first pitch. Confirmed in the
    # ledger (several bets fired with 2.9-4.5 min notice) -- not enough time to actually
    # place the bet on the Kalshi app before the game goes live. 15-min buffer -> 45-min
    # window -> worst case ~15 min notice, best case up to 45 min.

    # --- Order safety caps (used by the guarded `place` preview) ---
    max_order_contracts: int = 4000        # hard cap on contracts per order
    max_order_cost_usd: float = 1000.0     # hard cap on $ risked per order

    # --- Maker mode (added 2026-09-22) -------------------------------------
    # Rest orders on the book as a MAKER (post-only, priced at best_bid+1c)
    # instead of crossing the spread and paying the taker fee, with a re-peg /
    # timeout / taker-fallback lifecycle managed by engine/maker.py.
    #
    # DEFAULT OFF, and that default is evidence-based, not caution: the fee
    # model added the same day (config/fees.py) let backtest/limit_entry.py be
    # re-run with maker fees properly credited on every filled limit, over 289
    # settled bets. Result -- taker ROI +0.0419 vs the best limit policy's
    # -0.0114. Crediting maker fees narrowed taker's lead by only 0.46 ROI
    # points (5.79 -> 5.33), because the fee saving (~0.6-1.2c/contract) is an
    # order of magnitude smaller than what resting costs you: `save_c = -11.8c`,
    # i.e. a limit anchored one cycle early fills ~12c WORSE than just waiting
    # for the last-cycle ask. Entry prices drift down hard into game time and
    # the last-minute taker already captures that drift. Notably this is NOT
    # adverse selection (filled win% 0.625 vs missed 0.622 -- no selection
    # effect at all); it is purely the cost of committing to a price early.
    #
    # What that backtest does NOT settle, and why this ships anyway: its
    # `_simulate_fill()` anchors the limit ONCE and never moves it. The
    # lifecycle below re-pegs every `maker_poll_seconds` as the book moves, so
    # it FOLLOWS the falling ask instead of locking in a stale price, and would
    # forfeit far less of that 11.8c. Snapshots are 20-30 min apart, so a 30s
    # re-peg cadence is not simulable on this data at all -- it can only be
    # measured live. Enable narrowly (MLB totals first: `fee_type: "quadratic"`
    # means maker fills there are provably FREE, verified against two real
    # maker fills on this account) and compare the live ledger against the
    # paper one before trusting it broadly.
    maker_mode: bool = False               # `loop --live --maker` turns this on
    maker_poll_seconds: float = 30.0       # re-peg/fill-check cadence
    maker_improve_cents: float = 1.0       # rest this far above the best bid
    maker_taker_fallback_min: float = 5.0  # cross the spread at T-minus this many min
    maker_max_repegs: int = 40             # churn guard: stop re-pricing after this many
    maker_post_only: bool = True           # let the exchange reject would-be taker fills
    maker_require_post_only: bool = False  # True = refuse to rest if post_only is rejected

    def base_url(self) -> str:
        return KALSHI_DEMO_BASE if self.use_demo else KALSHI_API_BASE


DEFAULTS = Settings()
