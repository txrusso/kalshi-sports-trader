"""Central configuration: endpoints, loop cadence, and risk / signal thresholds.

Recommend-only mode: this agent NEVER places orders. It scans markets, scores
them, and writes ranked recommendations. All 'sizing' numbers are advisory only.
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

    # --- Market universe (MLB + NFL + NBA) ---
    # Kalshi series tickers for game markets. Discovered/verified at runtime; these
    # are the prefixes we filter events by.
    # KX**GAME = per-game winner markets; KX**TOTAL = total-runs/points (over/under)
    # ladders. (KXMLB*/KXNFL*/KXNBA* futures are excluded — the fair-value model is
    # game-based.) All sports scan in the same cycle; config/sports.py dispatches
    # each ticker to its sport's fair-value model and calibration bucket.
    # NBA added 2026-09-10; KXNBAGAME confirmed live (real Oct 20 2026 openers
    # already listed), KXNBATOTAL confirmed real but not yet listed this far out
    # (same as MLB/NFL totals) -- see data/nba_data.py for the model build notes.
    sport_series_prefixes: tuple[str, ...] = (
        "KXMLBGAME", "KXMLBTOTAL", "KXNFLGAME", "KXNFLTOTAL", "KXNBAGAME", "KXNBATOTAL",
    )
    include_totals: bool = True           # scan over/under (total runs) markets
    totals_min_mid: float = 0.12          # only consider near-the-money O/U lines
    totals_max_mid: float = 0.88          #   (skip deep ITM/OTM ladder rungs)
    market_status: str = "open"          # only scan open markets
    min_market_volume: int = 50          # ignore illiquid markets (contracts traded)
    min_open_interest: int = 50
    max_deep_markets: int = 100          # cap markets deep-scanned/cycle (bounds API load).
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
    top_n_recommendations: int = 15

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
    max_stake_pct: float = 0.06          # cap suggested stake at 6% of bankroll

    # --- Trade policy ---
    pregame_only: bool = True             # only recommend games in "Preview" (no live/final)
    paper_trade: bool = False             # (loop) paper-bet each game near its first pitch
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

    def base_url(self) -> str:
        return KALSHI_DEMO_BASE if self.use_demo else KALSHI_API_BASE


DEFAULTS = Settings()
