"""ESPN free MLB endpoints: scoreboard + win probability.

These are undocumented/free endpoints; treated as best-effort. Failures degrade
gracefully to "no external estimate" rather than breaking the loop.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

import requests

log = logging.getLogger("data.espn")

SCOREBOARD = "https://site.api.espn.com/apis/site/v2/sports/baseball/mlb/scoreboard"
SUMMARY = "https://site.api.espn.com/apis/site/v2/sports/baseball/mlb/summary"


@dataclass
class EspnGame:
    event_id: str
    date: datetime
    home_abbr: str
    away_abbr: str
    home_name: str
    away_name: str
    state: str                 # "pre", "in", "post"
    home_record_pct: Optional[float] = None
    away_record_pct: Optional[float] = None
    home_win_prob: Optional[float] = None   # 0..1, live if available


class EspnClient:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "Mozilla/5.0"})

    def _get(self, url: str, params: Optional[dict] = None) -> Optional[dict]:
        try:
            r = self.session.get(url, params=params, timeout=15)
            if r.status_code == 200:
                return r.json()
            log.warning("ESPN %s -> HTTP %s", url, r.status_code)
        except requests.RequestException as e:
            log.warning("ESPN request failed: %s", e)
        return None

    @staticmethod
    def _record_pct(competitor: dict) -> Optional[float]:
        for rec in competitor.get("records", []) or []:
            summ = rec.get("summary", "")
            if "-" in summ:
                try:
                    w, l = summ.split("-")[:2]
                    w, l = int(w), int(l)
                    if w + l > 0:
                        return w / (w + l)
                except ValueError:
                    continue
        return None

    def scoreboard(self, date: Optional[datetime] = None) -> list[EspnGame]:
        params = {}
        if date:
            params["dates"] = date.strftime("%Y%m%d")
        data = self._get(SCOREBOARD, params)
        if not data:
            return []
        games: list[EspnGame] = []
        for ev in data.get("events", []) or []:
            try:
                comp = ev["competitions"][0]
                competitors = comp["competitors"]
                home = next(c for c in competitors if c.get("homeAway") == "home")
                away = next(c for c in competitors if c.get("homeAway") == "away")
                state = ev.get("status", {}).get("type", {}).get("state", "")
                games.append(EspnGame(
                    event_id=str(ev.get("id")),
                    date=datetime.fromisoformat(ev["date"].replace("Z", "+00:00")),
                    home_abbr=(home["team"].get("abbreviation") or "").upper(),
                    away_abbr=(away["team"].get("abbreviation") or "").upper(),
                    home_name=home["team"].get("displayName", ""),
                    away_name=away["team"].get("displayName", ""),
                    state=state,
                    home_record_pct=self._record_pct(home),
                    away_record_pct=self._record_pct(away),
                ))
            except (KeyError, IndexError, StopIteration):
                continue
        return games

    def home_win_probability(self, event_id: str) -> Optional[float]:
        """Live win probability for the home team (0..1), if the game is in progress."""
        data = self._get(SUMMARY, {"event": event_id})
        if not data:
            return None
        wp = data.get("winprobability")
        if isinstance(wp, list) and wp:
            last = wp[-1]
            pct = last.get("homeWinPercentage")
            if pct is not None:
                try:
                    return float(pct)
                except (TypeError, ValueError):
                    return None
        return None
