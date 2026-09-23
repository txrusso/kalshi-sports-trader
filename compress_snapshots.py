"""Compress old snapshot day-files in place, instead of deleting them.

WHY COMPRESS RATHER THAN PRUNE
------------------------------
`snapshots/` is the input to four backtests (`evaluate`, `in_game_backtest`,
`limit_entry`, `replay_moneyflow`) plus live confidence calibration, so an old
day deleted is a day those can never be re-run over. This project has already
been burned by exactly that: `bankroll_sim`'s cache used to overwrite on
refresh and silently amputated Jun 7-21, permanently (see CLAUDE.md) -- the
train split shrank and every "both splits" check quietly weakened with no
error raised.

Pruning would also barely help. Measured 2026-09-23: everything older than a
week was 33.8 MB of 252.5 MB (13%), because the recent days are the huge ones
-- a day went from 1.2 MB on 09-16 to 62.9 MB on 09-23 once the scan interval
dropped to 600s and `max_deep_markets` rose to 1500. Deleting a week of history
would free a eighth of the space and cost seven weeks of backtest inputs.

Compression wins on both counts. A snapshot row is mostly repeated ticker,
title, team and timestamp strings -- the top four fields are ~50% of every row
-- so the files gzip about 12x (52.0 MB -> 4.4 MB measured). The whole
directory goes from ~252 MB to ~21 MB with nothing lost, and a day file
decompresses in ~0.2s, which is noise next to a backtest's own runtime.
`backtest/evaluate.py::load_rows` reads .jsonl and .jsonl.gz transparently, and
it is the single loader every consumer goes through.

SAFETY
------
Today's file is never touched -- the loop is appending to it. Each original is
removed only after its gzip has been read back and verified byte-identical, so
an interrupted run leaves both copies rather than a truncated one (and
`load_rows` prefers the plain file when both exist, so a day can never be
double-counted).

    python compress_snapshots.py --dry-run      # show what would happen
    python compress_snapshots.py                # compress days older than 2
    python compress_snapshots.py --keep-days 7  # keep a week uncompressed
"""
from __future__ import annotations

import argparse
import gzip
import sys
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SNAPSHOTS = ROOT / "snapshots"
LEVEL = 6          # 12x on this data; 9 buys ~2% more for several times the CPU


def _mb(n: int) -> float:
    return n / 1048576


def day_of(path: Path) -> date | None:
    try:
        return date.fromisoformat(path.name.replace(".jsonl", ""))
    except ValueError:
        return None          # _last_state.json and anything else non-daily


def compress_one(path: Path, dry_run: bool = False) -> tuple[int, int]:
    """(bytes_before, bytes_after). Returns (0, 0) if it was skipped."""
    target = path.with_suffix(".jsonl.gz")
    before = path.stat().st_size
    if target.exists():
        print(f"  {path.name}: .gz already exists, skipping")
        return 0, 0
    if dry_run:
        return before, 0

    raw = path.read_bytes()
    target.write_bytes(gzip.compress(raw, LEVEL))

    # Verify before destroying the original. A truncated or corrupt .gz that
    # replaced a real day would be silent -- the backtests would just quietly
    # see fewer rows.
    if gzip.decompress(target.read_bytes()) != raw:
        target.unlink(missing_ok=True)
        print(f"  {path.name}: ROUND-TRIP MISMATCH -- original kept, .gz removed")
        return 0, 0

    after = target.stat().st_size
    path.unlink()
    print(f"  {path.name}: {_mb(before):6.1f} MB -> {_mb(after):5.1f} MB "
          f"({before / max(after, 1):4.1f}x)")
    return before, after


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--keep-days", type=int, default=2,
                    help="leave this many recent days uncompressed (default 2; "
                         "today is never touched regardless)")
    ap.add_argument("--dry-run", action="store_true",
                    help="report what would be compressed and exit")
    args = ap.parse_args(argv)

    if not SNAPSHOTS.is_dir():
        print(f"no snapshots directory at {SNAPSHOTS}")
        return 0

    cutoff = date.today() - timedelta(days=max(1, args.keep_days))
    todo = []
    for path in sorted(SNAPSHOTS.glob("*.jsonl")):
        day = day_of(path)
        if day is None or day > cutoff:
            continue
        todo.append(path)

    if not todo:
        print(f"nothing to compress (keeping days after {cutoff})")
        return 0

    print(f"compressing {len(todo)} day-file(s) on or before {cutoff}"
          f"{' [DRY RUN]' if args.dry_run else ''}:")
    before = after = 0
    for path in todo:
        b, a = compress_one(path, args.dry_run)
        before += b
        after += a

    if args.dry_run:
        print(f"would compress {_mb(before):.1f} MB "
              f"(expect roughly {_mb(before) / 12:.1f} MB after)")
    else:
        print(f"reclaimed {_mb(before - after):.1f} MB "
              f"({_mb(before):.1f} -> {_mb(after):.1f})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
