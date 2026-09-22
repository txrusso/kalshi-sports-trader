"""Fuse money-flow + fair-value + liquidity into a ranked trade recommendation.

Design priority (per the user's mandate): FOLLOW THE MONEY. Money flow picks the
side; fair value is the edge filter / confirmation; liquidity gates tradeability.
This module produces recommendations only — nothing is ever executed.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from config.settings import Settings, DEFAULTS
from config.sports import calibration_bucket, is_total, market_kind
from data.fair_value import FairValue
from kalshi.normalize import MarketQuote
from signals.calibration import Calibration
from signals.money_flow import MoneyFlow


@dataclass
class Recommendation:
    ticker: str
    title: str
    yes_team: str
    side: str                    # "YES" or "NO" -- the actual Kalshi contract side (used for
                                  # sizing/pricing/order placement; NEVER change this for display)
    entry_price: float           # cost to enter one contract (dollars)
    edge_cents: Optional[float]  # modeled edge vs fair value (None if no model)
    confidence: float            # 0..1
    money_flow_score: float
    flow_direction: str
    fair_prob: Optional[float]
    fair_source: str
    suggested_stake_usd: float
    suggested_contracts: float
    rationale: str
    spread_cents: float
    volume: float
    open_interest: float
    headline: str = ""           # display-only, always positively framed (see headline_for())
    conflict: bool = False
    game_datetime: Optional[datetime] = None   # from the market's own occurrence_datetime
                                                # (kalshi/normalize.py) -- authoritative kickoff/
                                                # first-pitch time for both sports; lets
                                                # engine/paper.py's trigger work without a
                                                # sport-specific schedule lookup
    debug: dict = field(default_factory=dict)

    # ranking key: prioritize confidence, then |edge|, then flow strength
    def rank_key(self) -> tuple:
        return (round(self.confidence, 4), abs(self.edge_cents or 0.0), abs(self.money_flow_score))


def _liquidity_score(q: MarketQuote, settings: Settings) -> float:
    # Tighter spread + more volume/OI => more tradeable => higher score (0..1).
    spread_term = max(0.0, 1.0 - q.spread_cents / max(settings.max_spread_cents, 1.0))
    vol_term = min(1.0, q.volume / 2000.0)
    oi_term = min(1.0, q.open_interest / 2000.0)
    return max(0.0, min(1.0, 0.5 * spread_term + 0.25 * vol_term + 0.25 * oi_term))


def _kelly_contracts(win_prob: float, cost: float, bankroll: float,
                     settings: Settings) -> tuple[float, float]:
    if cost <= 0 or cost >= 1:
        return 0.0, 0.0
    f_star = (win_prob - cost) / (1 - cost)          # full-Kelly fraction for a $1-payout contract
    f = max(0.0, f_star) * settings.kelly_fraction
    f = min(f, settings.max_stake_pct)
    stake = bankroll * f
    # Kalshi supports fractional contract counts (0.01 granularity), so size to
    # the full Kelly-suggested stake instead of flooring to a whole contract.
    contracts = round(stake / cost, 2) if cost > 0 else 0.0
    return round(stake, 2), contracts


def _teams_from_title(title: str, suffix: str) -> Optional[tuple[str, str]]:
    """('Team A', 'Team B') from a Kalshi market title like 'A vs B <suffix>'."""
    t = (title or "").strip()
    if t.endswith(suffix):
        t = t[: -len(suffix)]
    if " vs " not in t:
        return None
    a, b = t.split(" vs ", 1)
    a, b = a.strip(), b.strip()
    return (a, b) if a and b else None


def headline_for(ticker: str, side: str, yes_team: str, title: str,
                  away_name: str = "", home_name: str = "",
                  yes_name: str = "", opp_name: str = "",
                  line: Optional[float] = None) -> str:
    """Display-only phrasing. Rules (never change the underlying side/price for this):
      - Winner markets: "<winning team> wins vs <losing team>" -- always the team we
        think WINS, never "<team> loses", regardless of which team's own contract the
        actual side sits on.
      - Totals, betting the over: "<game> Over <line>".
      - Totals, betting the under: "<game> Under <line>" -- no YES/NO word anywhere.
        There's no separate "Under" contract on Kalshi (one contract per line,
        YES=over/NO=under), so calling an under pick "Over" would tell the user to
        buy the opposite of what's recommended.
      - Spread markets: each KXNFLSPREAD ticker is one team's own "wins by over
        L" ladder rung (YES=that team covers). Betting YES: "<team> covers -<L>".
        Betting NO (that team does NOT cover) is displayed from the OTHER team's
        implied side instead of a negative "fails to cover" framing -- the same
        "never say a team loses" spirit as winner markets -- so it reads
        "<opponent> covers +<L>", i.e. standard "+3.5 underdog" spread notation.

    away_name/home_name (totals) and yes_name/opp_name (winner, spread) come from
    whichever sport's fair-value model produced `fv` (data/fair_value.py /
    data/fair_value_totals.py for MLB via the matched MLB game; data/
    fair_value_nfl.py / data/fair_value_nfl_totals.py / data/fair_value_nfl_spread.py
    for NFL via its rules_primary matchup text) -- the real source of team names
    either way. Kalshi's market `title` is per-contract text like "Over 5.5 runs
    scored" / "Colorado wins" (MLB) or "Full Game: over 58.5 points scored?" /
    "New York G wins" (NFL), NOT a game-level "A vs B ..." string, so
    _teams_from_title below almost never matches; it's kept only as a
    last-resort fallback for old replayed rows that predate these fields (see
    backtest/evaluate.py::_label_for).
    """
    if market_kind(ticker) == "spread":
        if yes_name and opp_name and line is not None:
            line_str = f"{line:g}"
            if side == "YES":
                return f"{yes_name} covers -{line_str}"
            return f"{opp_name} covers +{line_str}"
        team = yes_team if side == "YES" else "the opponent"  # nothing parsed; safe fallback
        return f"{team} spread bet"

    if is_total(ticker):
        y = (yes_team or "").lower()
        line = y.replace("over", "").replace("runs scored", "").replace(
            "points scored", "").strip()
        if away_name and home_name:
            matchup = f"{away_name} vs {home_name}"
        else:
            teams = _teams_from_title(title, " Total Runs?")
            matchup = " vs ".join(teams) if teams else ""
        direction = "Over" if side == "YES" else "Under"
        return f"{matchup} {direction} {line}".strip()

    if yes_name and opp_name:
        winner, loser = (yes_name, opp_name) if side == "YES" else (opp_name, yes_name)
        return f"{winner} wins vs {loser}"

    teams = _teams_from_title(title, " Winner?")
    if teams and yes_team in teams:
        a, b = teams
        other = b if yes_team == a else a
        winner, loser = (yes_team, other) if side == "YES" else (other, yes_team)
        return f"{winner} wins vs {loser}"
    team = yes_team if side == "YES" else "the opponent"  # nothing parsed; safe fallback
    return f"{team} wins"


def build_recommendation(q: MarketQuote, mf: MoneyFlow, fv: FairValue,
                         settings: Settings = DEFAULTS,
                         calibration: Optional[Calibration] = None) -> Optional[Recommendation]:
    # --- gate: pre-game-only policy ---
    # fair_value.py/fair_value_totals.py already decide this correctly (they trust
    # the game's actual scheduled first pitch, not just MLB's status string, since
    # that string can flip early -- see their "effective_pregame" comments). Trust
    # their verdict via fv.source instead of re-checking fv.game_state here: a
    # source of "skip_non_pregame" means the game has genuinely started, and a
    # game_state of None means the market couldn't even be matched (unverifiable,
    # so treated as unsafe). Re-checking game_state directly here would silently
    # re-introduce the exact bug that fix was for.
    # Unverifiable is ALWAYS unsafe, independent of policy. game_state=None means
    # the market couldn't be matched to a real game at all (NFL preseason, an
    # unmatched ticker), so there is nothing to price against. This check used to
    # sit behind `pregame_only`, which meant --allow-live silently ALSO permitted
    # money-flow-only bets on unmatched markets -- harmless while nothing traded
    # in-game, a real money risk once in_game_trade exists. Split out 2026-09-22.
    if fv.game_state is None:
        return None
    if settings.pregame_only and fv.source == "skip_non_pregame":
        return None

    # A game that has already started and has NO fair value would be a bet placed
    # on money flow alone, on a game whose score the model cannot see. NFL/NBA/NHL
    # all return prob=None once a game starts ("no live <sport> model yet"), and a
    # finished game falls through the same way -- refuse all of them.
    started = fv.game_state != "Preview"
    if started and fv.prob is None:
        return None
    # A PREGAME-sourced estimate on a game that has already started is stale by
    # construction -- it was computed from season rates/ratings and knows nothing
    # about the score. Only a genuinely live source (MLB's in-game win
    # probability, fair_source="live") may price a started game. This is the
    # general form of the MLB-totals bug found 2026-09-22: that model returned a
    # full-game expected_runs() for an in-progress game, which `prob is None`
    # above could not catch because the number was real, just wrong.
    if started and (fv.source or "").startswith("pregame"):
        return None
    # A settled game has no uncertainty left to price. MLB never returns a prob
    # for a Final game today (it falls through to prob=None above), so this is
    # defence in depth rather than a live path -- but betting a decided outcome
    # is the single worst failure this gate could allow.
    if fv.game_state == "Final":
        return None

    # --- gate: need a tradeable two-sided market ---
    if not q.has_two_sided_quote or q.spread_cents > settings.max_spread_cents:
        return None

    # --- side selection: follow the money ---
    side = mf.direction
    value_side = None
    if fv.prob is not None:
        # Which side does fair value favor, and by how much at the ask?
        edge_yes = (fv.prob - q.yes_ask) * 100
        edge_no = ((1 - fv.prob) - q.no_ask) * 100
        value_side = "YES" if edge_yes >= edge_no else "NO"

    if side == "FLAT":
        # No money-flow lean; only proceed if fair value shows a real edge.
        if value_side is None:
            return None
        side = value_side

    entry_price = q.yes_ask if side == "YES" else q.no_ask
    if entry_price <= 0 or entry_price >= 1:
        return None

    # --- edge vs fair value ---
    edge_cents: Optional[float] = None
    win_prob: Optional[float] = None
    conflict = False
    if fv.prob is not None:
        win_prob = fv.prob if side == "YES" else 1 - fv.prob
        edge_cents = round(win_prob * 100 - entry_price * 100, 2)
        conflict = (value_side is not None and value_side != side)

    # --- confidence composite ---
    flow_conf = mf.strength * abs(mf.score)                     # 0..1
    liq = _liquidity_score(q, settings)
    if edge_cents is not None:
        edge_conf = max(0.0, min(1.0, edge_cents / 12.0)) * fv.confidence
        agree_bonus = 0.0 if conflict else 0.15
        confidence = 0.45 * flow_conf + 0.35 * edge_conf + 0.20 * liq + agree_bonus
        if conflict:
            confidence *= 0.5                                    # money flow fights the model
    else:
        # Flow-only: demand stronger flow, no model edge to lean on.
        confidence = 0.60 * flow_conf + 0.40 * liq
    confidence = round(max(0.0, min(1.0, confidence)), 4)

    # --- adaptive calibration: scale confidence by how similar past bets did ---
    cal_mult = 1.0
    if calibration is not None:
        cal_mult = calibration.multiplier_for(calibration_bucket(q.ticker))
        if cal_mult != 1.0:
            confidence = round(max(0.0, min(1.0, confidence * cal_mult)), 4)

    # --- filters ---
    # In-game bets clear a stiffer edge bar than pregame ones: prices move fast
    # mid-game, our book read is already stale, and a live win-prob swings far
    # more per minute than a pregame estimate does.
    min_edge = settings.in_game_min_edge_cents if started else settings.min_edge_cents
    if edge_cents is not None and not conflict and edge_cents < min_edge \
            and flow_conf < 0.5:
        return None
    if edge_cents is None and flow_conf < 0.45:
        return None
    if confidence < settings.min_confidence:
        return None

    # --- advisory sizing ---
    stake, contracts = (0.0, 0)
    if win_prob is not None:
        stake, contracts = _kelly_contracts(win_prob, entry_price, settings.bankroll_usd, settings)

    headline = headline_for(
        q.ticker, side, q.yes_sub_title, q.title,
        away_name=fv.detail.get("away_name", ""), home_name=fv.detail.get("home_name", ""),
        yes_name=fv.detail.get("yes_name", ""), opp_name=fv.detail.get("opp_name", ""),
        line=fv.detail.get("line"))
    rationale = _rationale(headline, mf, fv, edge_cents, conflict, q, cal_mult, entry_price)

    return Recommendation(
        ticker=q.ticker, title=q.title, yes_team=q.yes_sub_title, side=side,
        entry_price=round(entry_price, 4), edge_cents=edge_cents, confidence=confidence,
        money_flow_score=mf.score, flow_direction=mf.direction,
        fair_prob=fv.prob, fair_source=fv.source,
        suggested_stake_usd=stake, suggested_contracts=contracts,
        rationale=rationale, spread_cents=q.spread_cents, volume=q.volume,
        open_interest=q.open_interest, headline=headline, conflict=conflict,
        game_datetime=q.occurrence_datetime,
        debug={"money_flow": mf.components, "fair_value": fv.detail,
               "flow_conf": round(flow_conf, 3), "liquidity": round(liq, 3),
               "calibration_mult": cal_mult},
    )


def _rationale(headline, mf, fv, edge_cents, conflict, q, cal_mult: float, entry_price: float) -> str:
    bits = [f"Money flow {mf.direction} (score {mf.score:+.2f}, strength {mf.strength:.2f}; "
            f"book {mf.book_imbalance:+.2f}, trades {mf.trade_flow:+.2f}, oi {mf.oi_momentum:+.2f})"]
    if fv.prob is not None:
        bits.append(f"fair P(YES)={fv.prob:.0%} via {fv.source}")
        if edge_cents is not None:
            bits.append(f"edge {edge_cents:+.1f}c on {headline} @ {entry_price:.2f}")
    else:
        bits.append("no external model (flow-only)")
    if conflict:
        bits.append("⚠ flow conflicts with fair value")
    if cal_mult != 1.0:
        bits.append(f"calibration {cal_mult:.2f}x from past {market_kind(q.ticker)} bets")
    return " | ".join(bits)
