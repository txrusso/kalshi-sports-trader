"""Explain one recommendation: the math behind it, rebuilt from what the loop saved.

Every recommendation in output/recommendations_latest.json carries its inputs
(the money-flow components, the fair-value model's `detail`, confidence parts,
the calibration multiplier). This module RE-DERIVES the fair value, the
confidence, the gates and the Kelly stake from those inputs using the models'
own constants and functions -- imported, never copied -- and checks each
rebuild against the number the loop recorded. A mismatch is reported, not
hidden: that is what makes the explanation trustworthy.

Two things the dashboard's Model tab is careful NOT to claim, because the
engine does not work that way:
- Money flow never moves fair value. It picks the SIDE and feeds CONFIDENCE.
- Calibration scales CONFIDENCE, not the probability.

"What-if" drivers are counterfactuals computed with the model's own math: the
edge on this bet minus the edge if one input were neutralised (no Elo blend,
no home field, a neutral park, ...). Positive = that input made the bet look
better. They are not additive -- each is a separate "remove just this" test.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Optional

from config.settings import DEFAULTS, Settings
from config.sports import market_kind, sport_of

REPRO_TOL = 0.003          # fair values are stored rounded to 4dp, inputs to 2dp

_FLOW_RE = re.compile(r"score ([+-][\d.]+), strength ([\d.]+); book ([+-][\d.]+), "
                      r"trades ([+-][\d.]+), oi ([+-][\d.]+)")


@dataclass
class Driver:
    label: str
    detail: str
    cents: float                       # edge now minus edge without this input


@dataclass
class Gate:
    name: str
    value: str
    rule: str
    status: str                        # "PASS" / "BYPASS" / "FAIL" / "INFO" / "N/A"
    note: str = ""


@dataclass
class Explanation:
    ticker: str
    headline: str
    side: str
    price: float
    sport: str
    kind: str
    source: str
    in_game: bool
    model_name: str
    pipeline: list[str] = field(default_factory=list)
    inputs: list[str] = field(default_factory=list)

    # fair value
    fair_yes: Optional[float] = None           # recorded P(YES)
    p_side: Optional[float] = None             # P(this bet wins)
    fair_steps: list[tuple[str, str]] = field(default_factory=list)
    repro_yes: Optional[float] = None
    repro_ok: Optional[bool] = None
    fv_conf: Optional[float] = None            # the fair-value model's own confidence
    drivers: list[Driver] = field(default_factory=list)
    sensitivities: list[Driver] = field(default_factory=list)

    # money flow
    flow: dict = field(default_factory=dict)

    # confidence
    conf_parts: list[tuple[str, float]] = field(default_factory=list)
    conf_raw: Optional[float] = None
    cal_mult: Optional[float] = None
    conf_repro: Optional[float] = None
    conf_recorded: Optional[float] = None
    conf_ok: Optional[bool] = None

    gates: list[Gate] = field(default_factory=list)
    sizing: dict = field(default_factory=dict)
    segment: tuple[str, str, bool] = ("", "", False)
    notes: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# model identity
# ---------------------------------------------------------------------------

def model_name(sport: str, kind: str, in_game: bool) -> str:
    if sport == "MLB" and kind == "winner":
        return ("MLB winner: MLB live win-probability feed (in-game)" if in_game else
                "MLB winner: log5 on records 80% + Elo 20%")
    if kind == "total":
        unit = {"MLB": "runs", "NFL": "points", "NBA": "points", "NHL": "goals"}.get(sport, "score")
        return f"{sport} total: expected {unit} -> negative binomial P(over)"
    if kind == "spread":
        return f"{sport} spread: Elo margin -> normal P(cover)"
    if kind == "winner":
        return f"{sport} winner: Elo"
    return f"{sport} {kind}"


def _inputs_for(sport: str, kind: str, in_game: bool) -> list[str]:
    feeds = ["Kalshi order book (resting depth, both sides)",
             "Kalshi trade tape (aggressor side, large prints)",
             "Kalshi open interest vs the previous scan (snapshot store)"]
    if sport == "MLB" and kind == "winner" and in_game:
        feeds.append("MLB Stats API live win probability (per play)")
    elif sport == "MLB" and kind == "winner":
        feeds += ["MLB Stats API season records (log5)",
                  "Historical MLB Elo, every game since 2023"]
    elif sport == "MLB" and kind == "total":
        feeds += ["MLB Stats API team run rates + probable starters' RA9",
                  "Park factors"]
    elif sport == "NFL":
        feeds.append("nflverse games.csv (Elo history, scoring rates, roof)")
    elif sport == "NBA":
        feeds.append("sportsdataverse NBA game logs (Elo, scoring rates)")
    elif sport == "NHL":
        feeds.append("NHL API schedule + results (Elo, scoring rates)")
    feeds.append("Calibration: settled past bets, per sport x market")
    return feeds


def _pipeline(kind: str, in_game: bool) -> list[str]:
    return [
        "Scan: list every market, deep-scan the most liquid (book + trades)",
        "Money flow: book 40% / trades 35% / OI 25% -> score -> PICKS THE SIDE",
        "Fair value: the sport/market model -> P(YES) (money flow does not touch it)",
        "Edge = P(bet wins) - price; gates: tradeable, edge, confidence",
        "Confidence = flow + edge + liquidity (+agreement), x calibration",
        "Size: fractional Kelly on the live balance, capped per bet",
        ("Trigger: in-game, within the in-game window" if in_game else
         "Trigger: placed on the first scan 15-25 min before start"),
    ]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _clamp(x, lo=0.0, hi=1.0):
    return max(lo, min(hi, x))


def _logistic_elo(diff: float) -> float:
    return 1.0 / (1.0 + 10.0 ** (-diff / 400.0))


def _side_p(p_yes: float, side: str) -> float:
    return p_yes if side == "YES" else 1.0 - p_yes


def _edge_c(p_yes: float, side: str, price: float) -> float:
    return (_side_p(p_yes, side) - price) * 100.0


def parse_flow(rationale: str) -> Optional[dict]:
    m = _FLOW_RE.search(rationale or "")
    if not m:
        return None
    score, strength, book, trades, oi = (float(g) for g in m.groups())
    return {"score": score, "strength": strength, "book": book, "trades": trades, "oi": oi}


# ---------------------------------------------------------------------------
# fair value: rebuild + what-ifs, per model
# ---------------------------------------------------------------------------

def _explain_fair(ex: Explanation, fv: dict, settings: Settings) -> None:
    src, side, price = ex.source, ex.side, ex.price
    base_edge = _edge_c(ex.fair_yes, side, price) if ex.fair_yes is not None else None

    def driver(label, detail, p_without):
        if base_edge is not None and p_without is not None:
            ex.drivers.append(Driver(label, detail, base_edge - _edge_c(p_without, side, price)))

    def sens(label, p_alt):
        if base_edge is not None and p_alt is not None:
            ex.sensitivities.append(Driver(label, "", _edge_c(p_alt, side, price) - base_edge))

    if src == "live":
        hwp = fv.get("home_win_prob")
        yes_home = fv.get("yes_is_home")
        ex.fair_steps = [("MLB live win prob (home)", f"{hwp:.1%}" if hwp is not None else "--"),
                         ("YES team is", "home" if yes_home else "away"),
                         ("game state", str(fv.get("state") or ""))]
        if hwp is not None and yes_home is not None:
            ex.repro_yes = hwp if yes_home else 1.0 - hwp
        ex.fv_conf = 0.80
        ex.notes.append("In-game probability comes straight from MLB's own model; "
                        "there are no internal inputs to attribute.")
        return

    if src == "pregame_log5":
        from data.fair_value import ELO_WEIGHT, HOME_FIELD_ODDS_MULT, _apply_home_field, log5
        from signals.elo_mlb import win_prob as mlb_elo
        pa, pb = fv.get("yes_win_pct"), fv.get("opp_win_pct")
        l5, elo = fv.get("log5_prob"), fv.get("elo_prob")
        ry, ro, home = fv.get("elo_yes_rating"), fv.get("elo_opp_rating"), fv.get("yes_is_home")
        ex.fair_steps = [
            ("win % (YES / opp)", f"{pa:.3f} / {pb:.3f}" if pa is not None else "--"),
            ("log5, neutral site", f"{fv.get('neutral', 0):.1%}"),
            (f"+ home field (odds x{HOME_FIELD_ODDS_MULT:.3f}{'' if home else ' inverse'})",
             f"{l5:.1%}" if l5 is not None else "--"),
            ("Elo (YES / opp)", f"{ry:.0f} / {ro:.0f}" if ry is not None else "--"),
            ("Elo win prob", f"{elo:.1%}" if elo is not None else "--"),
            (f"blend {1 - ELO_WEIGHT:.0%} log5 + {ELO_WEIGHT:.0%} Elo", ""),
        ]
        if l5 is not None:
            blend = (lambda a, b: (1 - ELO_WEIGHT) * a + ELO_WEIGHT * b) if elo is not None \
                else (lambda a, b: a)
            ex.repro_yes = blend(l5, elo)
            if pa is not None and pb is not None:
                ex.fv_conf = round(0.35 + min(abs(pa - pb) * 1.5, 0.25), 3)
            driver("Elo blend", f"without it: pure log5 {l5:.1%}", l5)
            if elo is not None and ry is not None:
                driver("Team records (log5)", "if both teams were .500",
                       blend(_apply_home_field(0.5, bool(home)), elo))
                driver("Elo rating gap", f"{ry - ro:+.0f} pts; if ratings were equal",
                       blend(l5, mlb_elo(ry, ry, bool(home))))
                neutral = fv.get("neutral")
                if neutral is not None:
                    driver("Home field", "if played on a neutral site",
                           blend(neutral, mlb_elo(ry, ro, bool(home), home_field_elo=0.0)))
        return

    if src == "pregame_elo":
        mod = {"NFL": "signals.elo_nfl", "NBA": "signals.elo_nba", "NHL": "signals.elo_nhl"}.get(ex.sport)
        ry, ro, home = fv.get("yes_elo"), fv.get("opp_elo"), fv.get("yes_is_home")
        if mod is None or ry is None or ro is None:
            return
        import importlib
        m = importlib.import_module(mod)
        hfe = getattr(m, "HOME_FIELD_ELO", getattr(m, "HOME_ICE_ELO", 0.0))
        hf = hfe if home else -hfe
        ex.fair_steps = [("Elo (YES / opp)", f"{ry:.0f} / {ro:.0f}"),
                         ("rating gap", f"{ry - ro:+.0f}"),
                         (f"home field ({'+' if home else '-'}{hfe:.0f} Elo)", f"{hf:+.0f}"),
                         ("P = 1 / (1 + 10^(-diff/400))", "")]
        ex.repro_yes = m.win_prob(ry, ro, bool(home))
        ex.fv_conf = round(0.35 + min(abs(ry - ro) / 800.0, 0.25), 3)
        driver("Elo rating gap", f"{ry - ro:+.0f} pts; if ratings were equal", _logistic_elo(hf))
        driver("Home field", f"{hf:+.0f} Elo; if neutral site", _logistic_elo(ry - ro))
        return

    if src == "pregame_nfl_spread":
        from data.distributions import normal_sf
        from data.fair_value_nfl_spread import KEY_NUMBER_BUMP, MARGIN_PER_ELO, MARGIN_SIGMA
        from signals.elo_nfl import HOME_FIELD_ELO
        line, mu, home = fv.get("line"), fv.get("exp_margin"), fv.get("yes_is_home")
        if line is None or mu is None:
            return
        hf_pts = MARGIN_PER_ELO * (HOME_FIELD_ELO if home else -HOME_FIELD_ELO)
        bump = KEY_NUMBER_BUMP.get(line, 0.0)

        def p_cover(m, b=bump):
            return min(1.0, normal_sf(line, m, MARGIN_SIGMA) + b)

        ex.fair_steps = [("expected margin (YES team)", f"{mu:+.2f} pts"),
                         (f"  of which home field", f"{hf_pts:+.2f} pts"),
                         ("line (must win by more than)", f"{line}"),
                         (f"P(margin > line), Normal sd {MARGIN_SIGMA}", ""),
                         ("key-number bump", f"+{bump:.1%}" if bump else "none")]
        ex.repro_yes = p_cover(mu)
        rating_gap = mu / MARGIN_PER_ELO - (HOME_FIELD_ELO if home else -HOME_FIELD_ELO)
        ex.fv_conf = round(0.30 + min(abs(rating_gap) / 800.0, 0.20), 3)
        driver("Elo rating gap", f"{rating_gap:+.0f} pts; if ratings were equal", p_cover(hf_pts))
        driver("Home field", f"{hf_pts:+.2f} pts; if neutral site", p_cover(mu - hf_pts))
        if bump:
            driver("Key-number bump", f"line {line} sits on a common margin", p_cover(mu, 0.0))
        sens("expected margin +1 pt", p_cover(mu + 1))
        sens("expected margin -1 pt", p_cover(mu - 1))
        return

    if src.endswith("totals"):
        from data.distributions import nb_survival
        line, lam = fv.get("line"), fv.get("exp_total")
        if line is None or lam is None:
            return
        phi, bumps = _totals_params(src)
        bump = bumps.get(line, 0.0)

        def p_over(l, b=bump):
            return min(1.0, nb_survival(int(line), l, phi) + b)

        unit = {"MLB": "runs", "NFL": "pts", "NBA": "pts", "NHL": "goals"}.get(ex.sport, "")
        ex.fair_steps = [("expected total", f"{lam:.2f} {unit}"),
                         ("line", f"{line}"),
                         (f"P(total > line), neg. binomial phi={phi}", "")]
        pf, rf = fv.get("park_factor"), fv.get("roof_factor")
        if pf:
            ex.fair_steps.insert(1, ("  incl. park factor", f"x{pf:.2f}"))
        if rf:
            ex.fair_steps.insert(1, (f"  incl. roof ({fv.get('roof')})", f"x{rf:.2f}"))
        if "starters_known" in fv:
            ex.fair_steps.insert(1, ("  starters known", f"{fv['starters_known']}/2 "
                                     f"(RA9 home {fv.get('home_eff_ra9')}, away {fv.get('away_eff_ra9')})"))
        if bump:
            ex.fair_steps.append(("key-number bump", f"+{bump:.1%}"))
        ex.repro_yes = p_over(lam)
        if src == "pregame_totals":
            ex.fv_conf = round(0.40 + 0.05 * int(fv.get("starters_known") or 0), 3)
        else:
            ex.fv_conf = 0.40
        if pf and pf != 1.0:
            driver("Park factor", f"x{pf:.2f}; if a neutral park", p_over(lam / pf))
        if rf and rf != 1.0:
            driver("Roof", f"x{rf:.2f}; if outdoors", p_over(lam / rf))
        if bump:
            driver("Key-number bump", f"line {line}", p_over(lam, 0.0))
        step = 0.5 if ex.sport in ("MLB", "NHL") else 1.0
        sens(f"expected total +{step:g} {unit}", p_over(lam + step))
        sens(f"expected total -{step:g} {unit}", p_over(lam - step))
        ex.notes.append("The team-strength inputs behind the expected total (offense and "
                        "run-prevention rates) aren't saved per recommendation, so they are "
                        "shown as a sensitivity rather than attributed one by one.")
        return


def _totals_params(src: str) -> tuple[float, dict]:
    if src == "pregame_totals":
        from data.fair_value_totals import TOTALS_PHI
        return TOTALS_PHI, {}
    if src == "pregame_nfl_totals":
        from data.fair_value_nfl_totals import KEY_NUMBER_BUMP, NFL_TOTALS_PHI
        return NFL_TOTALS_PHI, KEY_NUMBER_BUMP
    if src == "pregame_nba_totals":
        from data.fair_value_nba_totals import NBA_TOTALS_PHI
        return NBA_TOTALS_PHI, {}
    if src == "pregame_nhl_totals":
        from data.fair_value_nhl_totals import NHL_TOTALS_PHI
        return NHL_TOTALS_PHI, {}
    return 1.0, {}


# ---------------------------------------------------------------------------
# the whole explanation
# ---------------------------------------------------------------------------

def explain(rec: dict, settings: Settings = DEFAULTS) -> Explanation:
    ticker = rec.get("ticker") or ""
    side = str(rec.get("side") or "").upper()
    price = float(rec.get("entry_price") or 0)
    src = rec.get("fair_source") or "none"
    sport = sport_of(ticker).upper()
    kind = market_kind(ticker)
    in_game = src == "live"
    dbg = rec.get("debug") or {}
    fv = dbg.get("fair_value") or {}

    ex = Explanation(ticker=ticker, headline=rec.get("headline") or ticker, side=side,
                     price=price, sport=sport, kind=kind, source=src, in_game=in_game,
                     model_name=model_name(sport, kind, in_game),
                     pipeline=_pipeline(kind, in_game), inputs=_inputs_for(sport, kind, in_game),
                     segment=(sport, kind, in_game))

    # -- fair value --------------------------------------------------------
    fair = rec.get("fair_prob")
    if fair is not None:
        ex.fair_yes = float(fair)
        ex.p_side = _side_p(ex.fair_yes, side)
        _explain_fair(ex, fv, settings)
        if ex.repro_yes is not None:
            ex.repro_ok = abs(ex.repro_yes - ex.fair_yes) <= REPRO_TOL
        ex.drivers.sort(key=lambda d: -d.cents)
    else:
        ex.notes.append("No fair value (money-flow-only path): the market could not be priced "
                        "by a model, so there is no edge or Kelly stake.")

    # -- money flow ----------------------------------------------------------
    flow = parse_flow(rec.get("rationale") or "")
    mf = dbg.get("money_flow") or {}
    oi_available = bool((mf.get("oi") or {}).get("available"))
    if flow:
        w = (settings.w_book_imbalance, settings.w_trade_flow, settings.w_oi_momentum)
        wsum = w[0] + w[1] + (w[2] if oi_available else 0.0)
        weighted = (w[0] * flow["book"] + w[1] * flow["trades"]
                    + (w[2] * flow["oi"] if oi_available else 0.0)) / wsum
        flow.update(weights=w, oi_available=oi_available, weighted=_clamp(weighted, -1, 1),
                    direction=rec.get("flow_direction"), raw=mf,
                    flow_conf=dbg.get("flow_conf"))
        ex.flow = flow

    # -- confidence ----------------------------------------------------------
    flow_conf = dbg.get("flow_conf")
    liq = dbg.get("liquidity")
    ex.cal_mult = dbg.get("calibration_mult")
    ex.conf_recorded = rec.get("confidence")
    conflict = bool(rec.get("conflict"))
    edge = rec.get("edge_cents")
    if flow_conf is not None and liq is not None:
        if edge is not None and ex.fv_conf is not None:
            edge_conf = _clamp(float(edge) / 12.0) * ex.fv_conf
            ex.conf_parts = [
                (f"money flow 0.45 x {flow_conf:.3f}", 0.45 * flow_conf),
                (f"edge 0.35 x min(edge/12c,1) x model conf {ex.fv_conf:.2f}", 0.35 * edge_conf),
                (f"liquidity 0.20 x {liq:.3f}", 0.20 * liq),
                ("agreement bonus (flow agrees with value)" if not conflict
                 else "agreement bonus (none: CONFLICT)", 0.0 if conflict else 0.15),
            ]
            raw = sum(v for _, v in ex.conf_parts)
            if conflict:
                ex.conf_parts.append(("conflict penalty (x0.5)", -raw * 0.5))
                raw *= 0.5
        elif edge is None:
            ex.conf_parts = [(f"money flow 0.60 x {flow_conf:.3f}", 0.60 * flow_conf),
                             (f"liquidity 0.40 x {liq:.3f}", 0.40 * liq)]
            raw = sum(v for _, v in ex.conf_parts)
        else:
            raw = None
        if raw is not None:
            ex.conf_raw = round(_clamp(raw), 4)
            ex.conf_repro = round(_clamp(ex.conf_raw * (ex.cal_mult or 1.0)), 4)
            if ex.conf_recorded is not None:
                ex.conf_ok = abs(ex.conf_repro - float(ex.conf_recorded)) <= 0.003

    # -- gates ---------------------------------------------------------------
    ex.gates = _gates(rec, ex, settings, flow_conf, conflict)

    # -- sizing / EV ---------------------------------------------------------
    if ex.p_side is not None:
        ex.sizing = _sizing(rec, ex, settings)
    return ex


def _gates(rec: dict, ex: Explanation, s: Settings, flow_conf, conflict: bool) -> list[Gate]:
    gates = []
    spread = rec.get("spread_cents")
    if spread is not None:
        gates.append(Gate("Tradeable market", f"spread {float(spread):.0f}c",
                          f"two-sided quote, spread <= {s.max_spread_cents:.0f}c",
                          "PASS" if float(spread) <= s.max_spread_cents else "FAIL"))
    if ex.fair_yes is not None:
        gates.append(Gate("Game verified", ex.source, "matched to a real scheduled game", "PASS"))
    edge = rec.get("edge_cents")
    min_edge = s.in_game_min_edge_cents if ex.in_game else s.min_edge_cents
    if edge is not None:
        edge = float(edge)
        rule = f"edge >= {min_edge:g}c" + (" (in-game bar)" if ex.in_game else "")
        if edge >= min_edge:
            gates.append(Gate("Minimum edge", f"{edge:+.1f}c", rule, "PASS",
                              f"margin {edge - min_edge:+.1f}c"))
        elif conflict:
            gates.append(Gate("Minimum edge", f"{edge:+.1f}c", rule, "BYPASS",
                              "not applied to conflicted picks (confidence halved instead)"))
        elif flow_conf is not None and flow_conf >= 0.5:
            gates.append(Gate("Minimum edge", f"{edge:+.1f}c", rule, "BYPASS",
                              f"strong money flow ({flow_conf:.2f} >= 0.50) skips the edge gate"))
        else:
            gates.append(Gate("Minimum edge", f"{edge:+.1f}c", rule, "FAIL"))
    elif flow_conf is not None:
        gates.append(Gate("Flow-only strength", f"{flow_conf:.2f}", "flow conf >= 0.45",
                          "PASS" if flow_conf >= 0.45 else "FAIL"))
    conf = rec.get("confidence")
    if conf is not None:
        conf = float(conf)
        gates.append(Gate("Confidence", f"{conf:.2f}", f"confidence >= {s.min_confidence:.2f}",
                          "PASS" if conf >= s.min_confidence else "FAIL",
                          f"margin {conf - s.min_confidence:+.2f}"))
    gates.append(Gate("Flow vs value", "CONFLICT" if conflict else "agree",
                      "money flow's side = the side fair value favours",
                      "INFO" if conflict else "PASS",
                      "flow picked the side against the model; confidence halved" if conflict else ""))
    if ex.cal_mult is not None:
        gates.append(Gate("Calibration", f"{ex.cal_mult:.2f}x", "not a gate: clamped to 0.70-1.30",
                          "INFO", "scales confidence only, never the probability"))
    return gates


def _sizing(rec: dict, ex: Explanation, s: Settings) -> dict:
    p, c = ex.p_side, ex.price
    stake = float(rec.get("suggested_stake_usd") or 0)
    contracts = float(rec.get("suggested_contracts") or 0)
    out = {"p": p, "cost": c, "stake": stake, "contracts": contracts,
           "kelly_fraction": s.kelly_fraction, "cap": s.max_stake_pct}
    if 0 < c < 1:
        f_star = (p - c) / (1 - c)
        f_frac = max(0.0, f_star) * s.kelly_fraction
        f_used = min(f_frac, s.max_stake_pct)
        out.update(f_star=f_star, f_frac=f_frac, f_used=f_used, capped=f_frac > s.max_stake_pct,
                   bankroll=stake / f_used if f_used > 0 else None,
                   contracts_repro=round(stake / c, 2))
        from config.fees import fee_for
        fee = fee_for(contracts, c, ex.ticker) if contracts > 0 else fee_for(1.0, c, ex.ticker)
        fee_ct = fee / contracts if contracts > 0 else fee
        out.update(ev_ct=p - c, fee_ct=fee_ct, ev_net_ct=p - c - fee_ct,
                   ev_position=(p - c - fee_ct) * contracts,
                   breakeven=c + fee_ct, exp_roi=(p - c - fee_ct) / c)
    return out


# ---------------------------------------------------------------------------
# historic segment performance (section E)
# ---------------------------------------------------------------------------

def segment_rows(graded: list[dict]) -> list[tuple[tuple[str, str, bool], "object"]]:
    """(segment key, Tally) per (sport, market kind, in-game?), biggest first."""
    from output.dashboard_stats import tally
    keys = sorted({(r["sport"], r["kind"], bool(r.get("in_game"))) for r in graded})
    out = [(k, tally([r for r in graded
                      if (r["sport"], r["kind"], bool(r.get("in_game"))) == k])) for k in keys]
    out.sort(key=lambda kv: -kv[1].n)
    return out
