"""Canonical team registry: one ID per franchise across The-Odds-API, Kalshi and ESPN.

Every external name resolves to a ``canonical_id`` such as ``NBA_BOS`` or
``NFL_KC`` via ``resolve_team``. ESPN ids/abbreviations and Kalshi ticker
codes below were verified against the live ESPN /teams and Kalshi
KXNBAGAME / KXNFLGAME feeds on 2026-11-10.

Altitude is the approximate home-venue elevation in feet; only the
HIGH_ALTITUDE_FT threshold is used by the model (DEN/UTA in the NBA, DEN in
the NFL, plus neutral-site Mexico City via NEUTRAL_VENUE_ALTITUDE_FT).
``utc_offset_std`` is the standard-time UTC offset used for travel-decay
calculations (no tz database needed; Phoenix does not observe DST).
"""

from __future__ import annotations

import difflib
import re
import sqlite3
from dataclasses import dataclass
from functools import lru_cache

HIGH_ALTITUDE_FT = 4000
NEUTRAL_VENUE_ALTITUDE_FT: dict[str, int] = {"Estadio Azteca": 7350, "Estadio Banorte": 7350}

SPORT_ALIASES = {"nba": "NBA", "basketball_nba": "NBA", "basketball": "NBA",
                 "nfl": "NFL", "americanfootball_nfl": "NFL", "football": "NFL"}


@dataclass(frozen=True)
class Team:
    canonical_id: str
    sport: str
    full_name: str              # The-Odds-API name
    kalshi_code: str
    espn_id: str
    espn_abbr: str
    arena_altitude_ft: int
    timezone: str               # IANA zone of home venue
    utc_offset_std: int
    aliases: tuple[str, ...] = ()

    @property
    def high_altitude(self) -> bool:
        return self.arena_altitude_ft >= HIGH_ALTITUDE_FT

    @property
    def abbr(self) -> str:
        return self.canonical_id.split("_", 1)[1]


_ET, _CT, _MT, _AZ, _PT = (("America/New_York", -5), ("America/Chicago", -6),
                           ("America/Denver", -7), ("America/Phoenix", -7),
                           ("America/Los_Angeles", -8))

# abbr, full name, kalshi, espn id, espn abbr, altitude ft, tz, aliases
_NBA = [
    ("ATL", "Atlanta Hawks", "ATL", "1", "ATL", 1050, _ET, ()),
    ("BOS", "Boston Celtics", "BOS", "2", "BOS", 20, _ET, ("Celts",)),
    ("BKN", "Brooklyn Nets", "BKN", "17", "BKN", 30, _ET, ("BRK",)),
    ("CHA", "Charlotte Hornets", "CHA", "30", "CHA", 750, _ET, ("CHO",)),
    ("CHI", "Chicago Bulls", "CHI", "4", "CHI", 600, _CT, ()),
    ("CLE", "Cleveland Cavaliers", "CLE", "5", "CLE", 650, _ET, ("Cavs",)),
    ("DAL", "Dallas Mavericks", "DAL", "6", "DAL", 430, _CT, ("Mavs",)),
    ("DEN", "Denver Nuggets", "DEN", "7", "DEN", 5280, _MT, ()),
    ("DET", "Detroit Pistons", "DET", "8", "DET", 600, _ET, ()),
    ("GSW", "Golden State Warriors", "GSW", "9", "GS", 50, _PT, ("Golden State", "Dubs")),
    ("HOU", "Houston Rockets", "HOU", "10", "HOU", 50, _CT, ()),
    ("IND", "Indiana Pacers", "IND", "11", "IND", 715, _ET, ()),
    ("LAC", "Los Angeles Clippers", "LAC", "12", "LAC", 300, _PT, ("LA Clippers", "Los Angeles C")),
    ("LAL", "Los Angeles Lakers", "LAL", "13", "LAL", 300, _PT, ("LA Lakers", "Los Angeles L")),
    ("MEM", "Memphis Grizzlies", "MEM", "29", "MEM", 300, _CT, ("Grizz",)),
    ("MIA", "Miami Heat", "MIA", "14", "MIA", 10, _ET, ()),
    ("MIL", "Milwaukee Bucks", "MIL", "15", "MIL", 620, _CT, ()),
    ("MIN", "Minnesota Timberwolves", "MIN", "16", "MIN", 830, _CT, ("Wolves",)),
    ("NOP", "New Orleans Pelicans", "NOP", "3", "NO", 5, _CT, ("Pels",)),
    ("NYK", "New York Knicks", "NYK", "18", "NY", 30, _ET, ()),
    ("OKC", "Oklahoma City Thunder", "OKC", "25", "OKC", 1200, _CT, ("Oklahoma City",)),
    ("ORL", "Orlando Magic", "ORL", "19", "ORL", 100, _ET, ()),
    ("PHI", "Philadelphia 76ers", "PHI", "20", "PHI", 40, _ET, ("Sixers",)),
    ("PHX", "Phoenix Suns", "PHX", "21", "PHX", 1090, _AZ, ("PHO",)),
    ("POR", "Portland Trail Blazers", "POR", "22", "POR", 50, _PT, ("Blazers", "Trail Blazers")),
    ("SAC", "Sacramento Kings", "SAC", "23", "SAC", 30, _PT, ()),
    ("SAS", "San Antonio Spurs", "SAS", "24", "SA", 650, _CT, ()),
    ("TOR", "Toronto Raptors", "TOR", "28", "TOR", 250, _ET, ()),
    ("UTA", "Utah Jazz", "UTA", "26", "UTAH", 4226, _MT, ("Utah",)),
    ("WAS", "Washington Wizards", "WAS", "27", "WSH", 50, _ET, ()),
]
_NFL = [
    ("ARI", "Arizona Cardinals", "ARI", "22", "ARI", 1100, _AZ, ("Cards",)),
    ("ATL", "Atlanta Falcons", "ATL", "1", "ATL", 1050, _ET, ()),
    ("BAL", "Baltimore Ravens", "BAL", "33", "BAL", 30, _ET, ()),
    ("BUF", "Buffalo Bills", "BUF", "2", "BUF", 650, _ET, ()),
    ("CAR", "Carolina Panthers", "CAR", "29", "CAR", 750, _ET, ()),
    ("CHI", "Chicago Bears", "CHI", "3", "CHI", 600, _CT, ()),
    ("CIN", "Cincinnati Bengals", "CIN", "4", "CIN", 500, _ET, ()),
    ("CLE", "Cleveland Browns", "CLE", "5", "CLE", 600, _ET, ()),
    ("DAL", "Dallas Cowboys", "DAL", "6", "DAL", 600, _CT, ()),
    ("DEN", "Denver Broncos", "DEN", "7", "DEN", 5280, _MT, ()),
    ("DET", "Detroit Lions", "DET", "8", "DET", 600, _ET, ()),
    ("GB", "Green Bay Packers", "GB", "9", "GB", 650, _CT, ("Green Bay", "GNB")),
    ("HOU", "Houston Texans", "HOU", "34", "HOU", 50, _CT, ()),
    ("IND", "Indianapolis Colts", "IND", "11", "IND", 715, _ET, ()),
    ("JAX", "Jacksonville Jaguars", "JAC", "30", "JAX", 15, _ET, ("Jags",)),
    ("KC", "Kansas City Chiefs", "KC", "12", "KC", 900, _CT, ("Kansas City", "KAN")),
    ("LV", "Las Vegas Raiders", "LV", "13", "LV", 2000, _PT, ("Las Vegas", "LVR")),
    ("LAC", "Los Angeles Chargers", "LAC", "24", "LAC", 100, _PT, ("LA Chargers", "Los Angeles C")),
    ("LAR", "Los Angeles Rams", "LAR", "14", "LAR", 100, _PT, ("LA Rams", "Los Angeles R", "LA")),
    ("MIA", "Miami Dolphins", "MIA", "15", "MIA", 10, _ET, ("Fins",)),
    ("MIN", "Minnesota Vikings", "MIN", "16", "MIN", 830, _CT, ("Vikes",)),
    ("NE", "New England Patriots", "NE", "17", "NE", 280, _ET, ("New England", "Pats", "NWE")),
    ("NO", "New Orleans Saints", "NO", "18", "NO", 5, _CT, ("NOR",)),
    ("NYG", "New York Giants", "NYG", "19", "NYG", 10, _ET, ("New York G",)),
    ("NYJ", "New York Jets", "NYJ", "20", "NYJ", 10, _ET, ("New York J",)),
    ("PHI", "Philadelphia Eagles", "PHI", "21", "PHI", 40, _ET, ()),
    ("PIT", "Pittsburgh Steelers", "PIT", "23", "PIT", 730, _ET, ()),
    ("SF", "San Francisco 49ers", "SF", "25", "SF", 20, _PT, ("San Francisco", "Niners", "SFO")),
    ("SEA", "Seattle Seahawks", "SEA", "26", "SEA", 20, _PT, ()),
    ("TB", "Tampa Bay Buccaneers", "TB", "27", "TB", 20, _ET, ("Tampa Bay", "Bucs", "TAM")),
    ("TEN", "Tennessee Titans", "TEN", "10", "TEN", 450, _CT, ()),
    ("WAS", "Washington Commanders", "WAS", "28", "WSH", 150, _ET, ()),
]


def _build() -> dict[str, Team]:
    reg: dict[str, Team] = {}
    for sport, rows in (("NBA", _NBA), ("NFL", _NFL)):
        for abbr, full, kalshi, espn_id, espn_abbr, alt, (tz, off), aliases in rows:
            cid = f"{sport}_{abbr}"
            reg[cid] = Team(cid, sport, full, kalshi, espn_id, espn_abbr, alt, tz, off, aliases)
    return reg


REGISTRY: dict[str, Team] = _build()
assert sum(t.sport == "NBA" for t in REGISTRY.values()) == 30
assert sum(t.sport == "NFL" for t in REGISTRY.values()) == 32


def normalize_sport(sport: str) -> str:
    s = SPORT_ALIASES.get(sport.lower(), sport.upper())
    if s not in ("NBA", "NFL"):
        raise ValueError(f"Unsupported sport {sport!r}")
    return s


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", "", text.lower())).strip()


@lru_cache(maxsize=None)
def _index(sport: str) -> dict[str, str]:
    """normalized alias -> canonical_id. Ambiguous cities/nicknames are dropped."""
    strong: dict[str, str] = {}
    weak: dict[str, set[str]] = {}
    for t in (t for t in REGISTRY.values() if t.sport == sport):
        nickname = "Trail Blazers" if t.full_name.endswith("Trail Blazers") \
            else t.full_name.split()[-1]
        city = t.full_name[: -len(nickname)].strip()
        for a in (t.full_name, t.abbr, t.kalshi_code, t.espn_abbr, t.canonical_id, *t.aliases):
            strong[_norm(a)] = t.canonical_id
        for a in (nickname, city):
            weak.setdefault(_norm(a), set()).add(t.canonical_id)
    index = {k: next(iter(v)) for k, v in weak.items() if len(v) == 1}
    index.update(strong)
    return index


@lru_cache(maxsize=None)
def _ambiguous(sport: str) -> frozenset[str]:
    """Names shared by several franchises (e.g. 'los angeles'); never fuzzy-matched."""
    seen: dict[str, set[str]] = {}
    for t in (t for t in REGISTRY.values() if t.sport == sport):
        nickname = "Trail Blazers" if t.full_name.endswith("Trail Blazers") \
            else t.full_name.split()[-1]
        for a in (nickname, t.full_name[: -len(nickname)].strip()):
            seen.setdefault(_norm(a), set()).add(t.canonical_id)
    return frozenset(k for k, v in seen.items() if len(v) > 1) - set(_index(sport))


def try_resolve_team(raw_name: str | None, sport: str) -> str | None:
    """Like ``resolve_team`` but returns None instead of raising."""
    if not raw_name:
        return None
    sport = normalize_sport(sport)
    idx = _index(sport)
    key = _norm(raw_name)
    if key in idx:
        return idx[key]
    if key in _ambiguous(sport):
        return None
    if key.upper() in REGISTRY and REGISTRY[key.upper()].sport == sport:
        return key.upper()
    # Prefixed / suffixed forms, e.g. "Boston Celtics ML", "KXNBAGAME-...-BOS"
    tail = _norm(raw_name.rsplit("-", 1)[-1])
    if tail in idx:
        return idx[tail]
    match = difflib.get_close_matches(key, list(idx), n=1, cutoff=0.84)
    return idx[match[0]] if match else None


def resolve_team(raw_name: str, sport: str) -> str:
    """Map any Odds-API / Kalshi / ESPN team string to its canonical_id.

    Exact matching first (full names, codes, aliases, unique nicknames and
    cities), then ticker-suffix parsing, then fuzzy matching (difflib, 0.84).

    Raises:
        LookupError: if the name cannot be resolved unambiguously.
    """
    cid = try_resolve_team(raw_name, sport)
    if cid is None:
        raise LookupError(f"Cannot resolve team {raw_name!r} for {sport}")
    return cid


def team(canonical_id: str) -> Team:
    return REGISTRY[canonical_id]


def teams_for(sport: str) -> list[Team]:
    s = normalize_sport(sport)
    return [t for t in REGISTRY.values() if t.sport == s]


def by_espn_id(sport: str, espn_id: str) -> Team | None:
    s = normalize_sport(sport)
    return next((t for t in REGISTRY.values() if t.sport == s and t.espn_id == str(espn_id)), None)


def sync_registry(conn: sqlite3.Connection) -> int:
    """Upsert all 62 teams into ``teams_metadata``."""
    rows = [(t.canonical_id, t.sport, t.full_name, t.kalshi_code, t.espn_id, t.espn_abbr,
             t.arena_altitude_ft, int(t.high_altitude), t.timezone, t.utc_offset_std)
            for t in REGISTRY.values()]
    with conn:
        conn.executemany("INSERT OR REPLACE INTO teams_metadata VALUES (?,?,?,?,?,?,?,?,?,?)", rows)
    return len(rows)


if __name__ == "__main__":
    probes = [("Boston Celtics", "NBA"), ("LA Clippers", "basketball_nba"), ("Los Angeles L", "NBA"),
              ("GS", "NBA"), ("UTAH", "NBA"), ("KXNBAGAME-26NOV13UTAHOU-HOU", "NBA"),
              ("Bostn Celtics", "NBA"), ("JAC", "NFL"), ("Jacksonville", "NFL"), ("WSH", "NFL"),
              ("New York J", "NFL"), ("Niners", "NFL"), ("Kansas City Chiefs", "nfl")]
    for name, sp in probes:
        print(f"{name!r:<34} {sp:<15} -> {try_resolve_team(name, sp)}")
    print("ambiguous 'Los Angeles' NBA ->", try_resolve_team("Los Angeles", "NBA"))
