"""Static PNG dashboard: balance, record, positions, and cumulative-return curves.

Two curves answer two different questions:
  - "every model pick"  -> the backtest's one-bet-per-game history (signal quality)
  - "your actual bets"  -> the paper ledger, real dollars, real fills (forward test)
Rendered once per invocation and opened in the OS image viewer -- a static image,
not a live page, per the user's request.
"""
from __future__ import annotations

import logging
import os
import sys
from datetime import datetime, timedelta
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

from backtest.evaluate import graded_bets, load_rows
from config.settings import DEFAULTS, EASTERN, OUTPUT_DIR, Settings
from data.games import build_clients, resolve_outcomes
from engine.paper import PaperLedger
from kalshi.client import KalshiClient, KalshiError
from kalshi.normalize import position_size

log = logging.getLogger("output.dashboard")

# --- palette (dataviz skill reference instance, light mode) ---
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"
SURFACE = "#fcfcfb"
PAGE = "#f9f9f7"
BLUE = "#2a78d6"          # categorical slot 1 -- the single series in each line chart
GOOD = "#0ca30c"          # status: positive delta
CRITICAL = "#d03b3b"      # status: negative delta

DASH_PATH = OUTPUT_DIR / "dashboard_latest.png"

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Segoe UI", "DejaVu Sans", "Arial"]


# --------------------------------------------------------------------------- #
# data
# --------------------------------------------------------------------------- #
def _account_snapshot(settings: Settings):
    client = KalshiClient(settings)
    try:
        bal = client.balance()
        balance = float(bal.get("balance_dollars") or (float(bal.get("balance") or 0) / 100))
        pos = client.get_positions().get("market_positions", []) or []
        open_positions = [p for p in pos if position_size(p) != 0]
    except KalshiError as e:
        log.warning("Account fetch failed: %s", e)
        balance, open_positions = None, []
    return balance, open_positions


def _real_bets(clients: dict):
    """Paper-ledger bets split into settled (with won/pnl) and pending."""
    bets = PaperLedger().load()
    if not bets:
        return [], []
    outcomes = resolve_outcomes({b["ticker"] for b in bets}, clients)
    settled, pending = [], []
    for b in bets:
        if b["ticker"] in outcomes:
            yes_won = outcomes[b["ticker"]]
            won = yes_won if b["side"] == "YES" else (not yes_won)
            ct = b.get("contracts") or 0
            pnl = (1 - b["entry_price"]) * ct if won else -b["entry_price"] * ct
            settled.append({**b, "won": won, "pnl_usd": round(pnl, 4)})
        else:
            pending.append(b)
    settled.sort(key=lambda b: b["first_pitch"])
    return settled, pending


def _backtest_bets(clients: dict):
    """One deduped bet per settled game the agent has ever scanned, chronological."""
    rows = load_rows(None)
    if not rows:
        return []
    outcomes = resolve_outcomes({r["ticker"] for r in rows}, clients)
    bets = graded_bets(rows, outcomes)
    bets.sort(key=lambda b: (b.get("game_date") or "", b["ticker"]))
    return bets


# --------------------------------------------------------------------------- #
# chart helpers
# --------------------------------------------------------------------------- #
def _style_axes(ax, y_fmt="${:+.2f}"):
    ax.set_facecolor(SURFACE)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color(BASELINE)
    ax.spines["bottom"].set_linewidth(1)
    ax.tick_params(colors=INK_MUTED, labelsize=9, length=0)
    ax.yaxis.grid(True, color=GRIDLINE, linewidth=1)
    ax.set_axisbelow(True)
    ax.yaxis.set_major_formatter(lambda v, _: y_fmt.format(v))


def _cum_return_chart(ax, values: list[float], title: str, subtitle: str):
    ax.text(0, 1.20, title, fontsize=12.5, weight="bold", color=INK_PRIMARY,
           ha="left", va="bottom", transform=ax.transAxes)
    ax.text(0, 1.06, subtitle, fontsize=9, color=INK_SECONDARY,
           ha="left", va="bottom", transform=ax.transAxes)

    if len(values) < 2:
        ax.set_facecolor(SURFACE)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_xticks([]); ax.set_yticks([])
        ax.text(0.5, 0.5, "Not enough settled bets yet", ha="center", va="center",
                fontsize=11, color=INK_MUTED, transform=ax.transAxes)
        return

    xs = list(range(1, len(values) + 1))
    cum = []
    running = 0.0
    for v in values:
        running += v
        cum.append(running)

    _style_axes(ax)
    ax.axhline(0, color=BASELINE, linewidth=1)
    ax.plot(xs, cum, color=BLUE, linewidth=2, solid_joinstyle="round", solid_capstyle="round")
    ax.fill_between(xs, cum, 0, color=BLUE, alpha=0.10, linewidth=0)
    end_color = GOOD if cum[-1] >= 0 else CRITICAL
    ax.scatter([xs[-1]], [cum[-1]], s=64, color=BLUE, zorder=5,
               edgecolors=SURFACE, linewidths=2)
    ax.annotate(f"{cum[-1]:+.2f}", (xs[-1], cum[-1]), xytext=(8, 0),
               textcoords="offset points", va="center", ha="left",
               fontsize=11, weight="bold", color=end_color)
    ax.set_xlim(0.5, len(xs) + max(2.0, len(xs) * 0.18))
    ax.set_xlabel("Bet # (chronological)", fontsize=9, color=INK_MUTED)


def _stat_tile(ax, label: str, value: str, value_color: str = INK_PRIMARY, sub: str = ""):
    ax.set_facecolor(SURFACE)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks([]); ax.set_yticks([])
    ax.text(0.06, 0.72, label, fontsize=10.5, color=INK_SECONDARY, transform=ax.transAxes)
    ax.text(0.06, 0.30, value, fontsize=22, weight="bold", color=value_color, transform=ax.transAxes)
    if sub:
        ax.text(0.06, 0.06, sub, fontsize=8.5, color=INK_MUTED, transform=ax.transAxes)


def _table(ax, title: str, headers: list[str], rows: list[list[str]], col_colors: Optional[list] = None):
    ax.set_facecolor(SURFACE)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title, fontsize=12, weight="bold", color=INK_PRIMARY, loc="left", pad=10)
    if not rows:
        ax.text(0.02, 0.85, "None", fontsize=10, color=INK_MUTED, transform=ax.transAxes)
        return
    n = len(rows)
    row_h = 0.80 / max(n, 1)
    top = 0.86
    ax.text(0.02, top + 0.06, headers[0], fontsize=8.5, color=INK_MUTED, transform=ax.transAxes)
    for j, h in enumerate(headers[1:], start=1):
        ax.text(0.42 + 0.19 * (j - 1), top + 0.06, h, fontsize=8.5, color=INK_MUTED,
                ha="left", transform=ax.transAxes)
    for i, row in enumerate(rows):
        y = top - i * row_h
        if i % 2 == 0:
            ax.axhspan(y - row_h * 0.42, y + row_h * 0.42, xmin=0.0, xmax=1.0,
                      color=PAGE, zorder=0)
        ax.text(0.02, y, str(row[0]), fontsize=9.5, color=INK_PRIMARY, va="center",
                transform=ax.transAxes)
        for j, cell in enumerate(row[1:], start=1):
            color = INK_PRIMARY
            if col_colors and col_colors[j - 1] is not None:
                color = col_colors[j - 1](cell)
            ax.text(0.42 + 0.19 * (j - 1), y, str(cell), fontsize=9.5, color=color, va="center",
                    ha="left", transform=ax.transAxes)


# --------------------------------------------------------------------------- #
# build
# --------------------------------------------------------------------------- #
def build_dashboard(settings: Settings = DEFAULTS, out_path=DASH_PATH, open_after: bool = True):
    clients = build_clients()
    balance, open_positions = _account_snapshot(settings)
    settled, pending = _real_bets(clients)
    bt_bets = _backtest_bets(clients)

    wins = sum(1 for b in settled if b["won"])
    losses = len(settled) - wins
    real_pnl = sum(b["pnl_usd"] for b in settled)

    now_et = datetime.now(EASTERN)
    cutoff = now_et - timedelta(days=2)
    recent_closed = [b for b in settled
                     if datetime.fromisoformat(b["first_pitch"]).astimezone(EASTERN) >= cutoff]
    recent_closed.sort(key=lambda b: b["first_pitch"], reverse=True)

    fig = plt.figure(figsize=(15, 11), dpi=160, facecolor=PAGE)
    gs = fig.add_gridspec(4, 4, height_ratios=[0.9, 2.1, 1.7, 0.15], hspace=0.55, wspace=0.35,
                          left=0.05, right=0.97, top=0.93, bottom=0.05)

    ts_label = now_et.strftime("%Y-%m-%d %I:%M %p ET")
    fig.suptitle("Kalshi Money-Flow Agent — Dashboard", fontsize=17, weight="bold",
                color=INK_PRIMARY, x=0.05, ha="left", y=0.985)
    fig.text(0.05, 0.955, f"as of {ts_label}  ·  recommend-only, no live orders placed",
             fontsize=9.5, color=INK_MUTED)

    # --- top strip: balance + record + positions + P&L ---
    ax_bal = fig.add_subplot(gs[0, 0])
    _stat_tile(ax_bal, "Kalshi account balance",
              f"${balance:,.2f}" if balance is not None else "n/a")

    ax_rec = fig.add_subplot(gs[0, 1])
    rec_str = f"{wins}-{losses}" if settled else "0-0"
    rec_sub = f"{wins/len(settled):.0%} win rate" if settled else "no settled bets yet"
    _stat_tile(ax_rec, "Overall record (real bets)", rec_str, sub=rec_sub)

    ax_pos = fig.add_subplot(gs[0, 2])
    _stat_tile(ax_pos, "Open positions", str(len(open_positions)),
              sub=f"{len(pending)} bet(s) awaiting final" if pending else "")

    ax_pnl = fig.add_subplot(gs[0, 3])
    pnl_color = GOOD if real_pnl >= 0 else CRITICAL
    _stat_tile(ax_pnl, "Real bets P&L", f"${real_pnl:+.2f}", value_color=pnl_color,
              sub=f"{len(settled)} settled bet(s)")

    # --- charts: backtest curve vs actual bets curve ---
    ax_bt = fig.add_subplot(gs[1, 0:2])
    bt_values = [(1 - b["entry"]) if b["won"] else -b["entry"] for b in bt_bets]
    _cum_return_chart(ax_bt, bt_values, "If you bet every model pick",
                      f"Backtest, $1/bet, one bet per game  ·  {len(bt_bets)} games")

    ax_real = fig.add_subplot(gs[1, 2:4])
    real_values = [b["pnl_usd"] for b in settled]
    _cum_return_chart(ax_real, real_values, "Your actual placed bets",
                      f"Real dollars, real fills  ·  {len(settled)} settled")

    # --- tables: open positions / recently closed (2 days) ---
    ax_open = fig.add_subplot(gs[2, 0:2])
    open_rows = [[p.get("ticker", ""), f"{round(position_size(p)):+d}",
                 f"${float(p.get('market_exposure_dollars') or 0):.2f}"] for p in open_positions]
    _table(ax_open, "Open positions", ["Ticker", "Pos", "Exposure"], open_rows)

    ax_closed = fig.add_subplot(gs[2, 2:4])
    closed_rows = []
    for b in recent_closed:
        result = "W" if b["won"] else "L"
        closed_rows.append([b.get("label", b["ticker"]), result, f"${b['pnl_usd']:+.2f}"])
    _table(ax_closed, "Recently closed (last 2 days)", ["Bet", "Result", "P&L"], closed_rows,
          col_colors=[lambda v: GOOD if v == "W" else CRITICAL, lambda v: GOOD if v.startswith("+") else CRITICAL])

    out_path = str(out_path)
    fig.savefig(out_path, facecolor=PAGE)
    plt.close(fig)
    log.info("Dashboard written to %s", out_path)

    if open_after and sys.platform == "win32":
        try:
            os.startfile(out_path)  # noqa: S606 (local file, user's own machine)
        except Exception as e:
            log.warning("Could not auto-open dashboard: %s", e)

    return out_path
