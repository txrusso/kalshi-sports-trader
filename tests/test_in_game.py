"""Offline tests for in-game trading gates -- no network, no real money.

Two independent gates decide whether a started game can fire a real order:
  1. signals/recommendation.py -- is a recommendation produced at all?
  2. engine/paper.py::iter_trigger_candidates -- is it inside the trigger window?

The cases that matter are the refusals. A started game with no live model
(NFL/NBA/NHL) or an unmatched market must never reach a real order, and with
in_game_trade off nothing that has started may trigger at all.

Run:  .venv\\Scripts\\python.exe -m tests.test_in_game
"""
from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta, timezone

from config.settings import DEFAULTS
from data.fair_value import FairValue
from engine.paper import iter_trigger_candidates
from kalshi.normalize import MarketQuote
from signals.money_flow import MoneyFlow
from signals.recommendation import Recommendation, build_recommendation

results = []


def check(name, cond, detail=""):
    results.append((name, bool(cond)))
    print(f"  [{'ok' if cond else 'FAIL'}] {name}" + (f"  -- {detail}" if detail and not cond else ""))


def _quote(ticker="KXMLBGAME-26SEP222140ATLNYM-NYM"):
    return MarketQuote(
        ticker=ticker, event_ticker=ticker.rsplit("-", 1)[0], title="Atlanta vs New York M Winner?",
        yes_sub_title="New York M", status="active",
        yes_bid=0.46, yes_ask=0.50, no_bid=0.50, no_ask=0.54,
        last_price=0.48, volume=5000, volume_24h=5000, open_interest=5000,
        liquidity=5000,
        occurrence_datetime=datetime.now(timezone.utc) + timedelta(minutes=30),
        rules_primary="", close_time=None,
    )


def _mf(score=0.55, strength=0.9):
    # `direction` is a derived property of score, not a field.
    return MoneyFlow(score=score, book_imbalance=0.5, trade_flow=0.5, oi_momentum=0.3,
                     strength=strength, components={"book": 0.5, "trades": 0.5, "oi": 0.3})


def _fv(prob, state, source):
    return FairValue(prob, source, 0.80, {"yes_name": "New York M", "opp_name": "Atlanta"},
                     game_state=state)


class _Ledger:
    def already_bet(self, event_key):
        return False


class _NoGames:
    """Schedule client that matches nothing -- the trigger must still work off
    occurrence_datetime, and the matchup label degrades to the headline."""

    def find_game(self, *a, **kw):
        return None


class _Rec:
    """Minimal Recommendation stand-in for the trigger-window test."""

    def __init__(self, ticker, start_offset_min):
        self.ticker = ticker
        self.side = "YES"
        self.headline = "New York M wins vs Atlanta"
        self.yes_team = "New York M"
        self.entry_price = 0.50
        self.suggested_contracts = 2.0
        self.suggested_stake_usd = 1.0
        self.fair_prob = 0.60
        self.edge_cents = 10.0
        self.confidence = 0.6
        self.money_flow_score = 0.5
        self.game_datetime = datetime.now(timezone.utc) + timedelta(minutes=start_offset_min)


def _triggers(rec, settings):
    """Does this rec pass the trigger window?"""
    clients = {k: _NoGames() for k in ("mlb", "nfl", "nba", "nhl")}
    got = list(iter_trigger_candidates([rec], clients, _Ledger(),
                                       window_minutes=45.0, settings=settings))
    return len(got) > 0


def main() -> None:
    pregame = replace(DEFAULTS, pregame_only=True, in_game_trade=False)
    in_game = replace(DEFAULTS, pregame_only=False, in_game_trade=True)

    print("recommendation gate -- refusals that protect real money")
    q, mf = _quote(), _mf()

    r = build_recommendation(q, mf, _fv(None, None, "none"), in_game)
    check("unmatched market (game_state=None) refused even with in-game ON", r is None)

    r = build_recommendation(q, mf, _fv(None, None, "none"), replace(DEFAULTS, pregame_only=False))
    check("unmatched market refused under --allow-live too (the fixed bug)", r is None)

    r = build_recommendation(q, mf, _fv(None, "Live", "none"), in_game)
    check("started game with NO live model (NFL/NBA/NHL) refused", r is None)

    r = build_recommendation(q, mf, _fv(0.70, "Final", "none"), in_game)
    check("finished game refused", r is None)

    r = build_recommendation(q, mf, _fv(None, "Live", "skip_non_pregame"), pregame)
    check("started game refused under pregame_only", r is None)

    r = build_recommendation(q, mf, _fv(0.62, "Live", "live"), in_game)
    check("MLB live model DOES produce a rec with in-game on", r is not None)

    # Regression guard for the 2026-09-22 bug: MLB TOTALS has no live model, and
    # its pregame expected_runs() is a full-game estimate that ignores runs already
    # scored. It must refuse in-game, the way every other totals model does.
    r = build_recommendation(_quote("KXMLBTOTAL-26SEP222140ATLNYM-8"), mf,
                             _fv(0.62, "Live", "pregame_totals"), in_game)
    check("MLB totals in-game refused (pregame_totals is a full-game estimate)", r is None)

    print("in-game edge bar is stiffer than pregame")
    # edge = fair(0.56) - ask(0.50) = 6c: clears pregame's 5.5c, fails in-game's 8c.
    weak = _mf(strength=0.4, score=0.3)          # flow_conf below the 0.5 bypass
    r_pre = build_recommendation(q, weak, _fv(0.56, "Preview", "pregame_log5"),
                                 replace(DEFAULTS, pregame_only=False, in_game_trade=True))
    r_live = build_recommendation(q, weak, _fv(0.56, "Live", "live"), in_game)
    check("6c edge clears the 5.5c pregame bar", r_pre is not None)
    check("same 6c edge fails the 8c in-game bar", r_live is None)

    print("trigger window")
    pre_rec = _Rec("KXNBAGAME-26SEP22ATLNYM-NYM", 30)      # starts in 30 min
    check("pregame game triggers normally", _triggers(pre_rec, pregame))

    started = _Rec("KXNBAGAME-26SEP22ATLNYM-NYM", -20)     # started 20 min ago
    check("started game does NOT trigger with in-game OFF", not _triggers(started, pregame))
    check("started game DOES trigger with in-game ON", _triggers(started, in_game))

    stale = _Rec("KXNBAGAME-26SEP22ATLNYM-NYM", -(DEFAULTS.in_game_max_minutes + 30))
    check("game past in_game_max_minutes stops triggering",
          not _triggers(stale, in_game))

    far = _Rec("KXNBAGAME-26SEP22ATLNYM-NYM", 300)         # starts in 5 hours
    check("far-future game does not trigger", not _triggers(far, in_game))

    bad = [n for n, ok in results if not ok]
    print("\n%d/%d passed" % (len(results) - len(bad), len(results)))
    if bad:
        print("FAILED: " + ", ".join(bad))
        raise SystemExit(1)
    print("all in-game gate tests passed")


if __name__ == "__main__":
    main()
