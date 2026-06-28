"""Ingest one NBA season of games into a clean, leakage-safe spine.

The NBA stats API (`LeagueGameLog`) returns ONE ROW PER TEAM PER GAME, so every
game appears twice — once from the home team's side, once from the away team's.
We collapse those pairs into one row per game:

    game_id, game_date, home_team_id, away_team_id, home_pts, away_pts, home_win

Results are cached to parquet so we don't hammer the (rate-limited, flaky) API.

CLI:
    uv run python -m nba_pred.ingest.games --season 2023-24
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd
from nba_api.stats.endpoints import leaguegamelog

# Repo-root-relative paths (this file is src/nba_pred/ingest/games.py).
REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RAW_DIR = REPO_ROOT / "data" / "raw"
DEFAULT_PROCESSED_DIR = REPO_ROOT / "data" / "processed"

# Modern-era cutoff (2004-05 hand-check rule change) and latest complete season.
FIRST_SEASON_YEAR = 2004
LAST_SEASON_YEAR = 2025  # the season starting in 2025 (i.e. "2025-26")

# Disrupted seasons — kept in the data but flagged for honest handling, since
# their schedules distort rest/B2B/home-court features:
#   2011-12  lockout, 66-game season (990 games)
#   2019-20  COVID Orlando bubble — NO home court
#   2020-21  COVID 72-game season, limited fans, compressed schedule
DISRUPTED_SEASONS = frozenset({"2011-12", "2019-20", "2020-21"})

# The clean output schema — the spine everything else joins onto.
SPINE_COLUMNS = [
    "game_id",
    "game_date",
    "season",
    "home_team_id",
    "away_team_id",
    "home_pts",
    "away_pts",
    "home_win",
]


def season_str(start_year: int) -> str:
    """2004 -> '2004-05'."""
    return f"{start_year}-{str(start_year + 1)[-2:]}"


def season_range(first_year: int = FIRST_SEASON_YEAR, last_year: int = LAST_SEASON_YEAR) -> list[str]:
    """All season strings from first_year to last_year inclusive."""
    return [season_str(y) for y in range(first_year, last_year + 1)]


def fetch_raw(season: str, retries: int = 4, timeout: int = 60) -> pd.DataFrame:
    """Hit the NBA API for one regular season: one row per team per game.

    stats.nba.com is rate-limited and flaky, so we retry with exponential
    backoff rather than letting a single ReadTimeout kill a long backfill.
    """
    for attempt in range(retries):
        try:
            log = leaguegamelog.LeagueGameLog(
                season=season,
                season_type_all_star="Regular Season",
                timeout=timeout,
            )
            return log.get_data_frames()[0]
        except Exception as e:  # noqa: BLE001 — network errors vary; retry them all
            if attempt == retries - 1:
                raise
            wait = 2 ** attempt
            print(f"  ⏳ {season}: {type(e).__name__}, retrying in {wait}s ({attempt + 1}/{retries})")
            time.sleep(wait)


def collapse_to_games(raw: pd.DataFrame, season: str) -> pd.DataFrame:
    """Turn 2N team-rows into N game-rows on the clean spine schema.

    Home rows have "vs." in MATCHUP (e.g. "DEN vs. LAL"); away rows have "@"
    (e.g. "LAL @ DEN"). We split on that, merge the two halves on GAME_ID, and
    derive the home-win label.
    """
    is_away = raw["MATCHUP"].str.contains("@", regex=False)
    home = raw[~is_away]
    away = raw[is_away]

    # A normal game splits into exactly one home row ("X vs. Y") and one away row
    # ("Y @ X"). Neutral-site games (NBA Cup final, international games) mark BOTH
    # teams with "@", so they don't split cleanly — there's no real home court, so
    # we drop them rather than invent a home team. Keep only game_ids with exactly
    # one of each side, and report anything dropped so the loss is never silent.
    home_ids = home["GAME_ID"].value_counts()
    away_ids = away["GAME_ID"].value_counts()
    clean_ids = set(home_ids[home_ids == 1].index) & set(away_ids[away_ids == 1].index)
    n_dropped = raw["GAME_ID"].nunique() - len(clean_ids)
    if n_dropped:
        print(f"  ↪ {season}: dropped {n_dropped} neutral-site/anomalous game(s)")
    home = home[home["GAME_ID"].isin(clean_ids)]
    away = away[away["GAME_ID"].isin(clean_ids)]

    # validate="1:1" is a hard safety net: fail loudly if any game still maps to
    # more than one partner row (would otherwise silently duplicate or drop games).
    games = home.merge(away, on="GAME_ID", suffixes=("_home", "_away"), validate="1:1")

    # Drop games that never actually happened. A completed NBA game can't end
    # tied, so equal scores mark a canceled/forfeited game — e.g. the 2013-04-16
    # Celtics-Pacers game canceled after the Boston Marathon bombing (logged 0-0,
    # the only NBA game ever canceled and not made up).
    games = games[games["PTS_home"] != games["PTS_away"]]

    games["home_win"] = (games["PTS_home"] > games["PTS_away"]).astype(int)

    spine = games[
        [
            "GAME_ID",
            "GAME_DATE_home",
            "TEAM_ID_home",
            "TEAM_ID_away",
            "PTS_home",
            "PTS_away",
            "home_win",
        ]
    ].rename(
        columns={
            "GAME_ID": "game_id",
            "GAME_DATE_home": "game_date",
            "TEAM_ID_home": "home_team_id",
            "TEAM_ID_away": "away_team_id",
            "PTS_home": "home_pts",
            "PTS_away": "away_pts",
        }
    )

    # Parse the date to a real datetime — required for time-based splits later.
    spine["game_date"] = pd.to_datetime(spine["game_date"])

    # Tag the season so combined multi-season tables stay self-describing.
    spine["season"] = season

    return spine.sort_values("game_date").reset_index(drop=True)[SPINE_COLUMNS]


def validate(games: pd.DataFrame, season: str) -> None:
    """Loud sanity checks — garbage data is the #2 cause of broken projects.

    Hard failures (asserts) for things that are never OK; a warning for the
    game count, which legitimately varies (COVID seasons, scheduling quirks).
    """
    assert games["game_id"].is_unique, "duplicate game_id found"
    assert games[["home_pts", "away_pts"]].notna().all().all(), "missing scores"
    assert (games["home_pts"] != games["away_pts"]).all(), "tie game found (impossible in NBA)"
    assert games["game_date"].is_monotonic_increasing, "games not sorted by date"

    # A normal regular season is 82 * 30 / 2 = 1230 games. Warn, don't fail.
    n = len(games)
    if n != 1230:
        print(f"  ⚠️  {season}: got {n} games (expected ~1230 — fine for COVID seasons)")


def fetch_season(season: str, raw_dir: Path = DEFAULT_RAW_DIR) -> pd.DataFrame:
    """Idempotent, cached season ingest. Returns the clean game spine.

    If the parquet already exists, read it instead of re-hitting the API.
    """
    raw_dir.mkdir(parents=True, exist_ok=True)
    cache = raw_dir / f"games_{season}.parquet"

    if cache.exists():
        return pd.read_parquet(cache)

    raw = fetch_raw(season)
    games = collapse_to_games(raw, season)
    validate(games, season)
    games.to_parquet(cache, index=False)
    return games


def fetch_seasons(seasons: list[str], raw_dir: Path = DEFAULT_RAW_DIR) -> pd.DataFrame:
    """Ingest many seasons (idempotent per season) and return them concatenated.

    Each season is cached individually, so a re-run after a mid-backfill API
    failure resumes instantly from where it left off.
    """
    frames = []
    for season in seasons:
        games = fetch_season(season, raw_dir)
        flag = "  [DISRUPTED]" if season in DISRUPTED_SEASONS else ""
        print(f"  {season}: {len(games):>4} games{flag}")
        frames.append(games)
    return pd.concat(frames, ignore_index=True).sort_values("game_date").reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest NBA season(s) into a clean game spine.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--season", help='single season, e.g. "2023-24"')
    group.add_argument(
        "--all",
        action="store_true",
        help=f"backfill {season_str(FIRST_SEASON_YEAR)}..{season_str(LAST_SEASON_YEAR)}",
    )
    parser.add_argument("--start-year", type=int, default=FIRST_SEASON_YEAR, help="for --all")
    parser.add_argument("--end-year", type=int, default=LAST_SEASON_YEAR, help="for --all")
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--processed-dir", type=Path, default=DEFAULT_PROCESSED_DIR)
    args = parser.parse_args()

    if args.season:
        games = fetch_season(args.season, args.raw_dir)
        print(f"✅ {args.season}: {len(games)} games")
        print(games.head().to_string(index=False))
        return

    seasons = season_range(args.start_year, args.end_year)
    print(f"Backfilling {len(seasons)} seasons: {seasons[0]}..{seasons[-1]}")
    games = fetch_seasons(seasons, args.raw_dir)

    args.processed_dir.mkdir(parents=True, exist_ok=True)
    out = args.processed_dir / "games.parquet"
    games.to_parquet(out, index=False)
    n_disrupted = int(games["season"].isin(DISRUPTED_SEASONS).sum())
    print(f"✅ {len(games)} games across {games['season'].nunique()} seasons -> {out}")
    print(f"   ({n_disrupted} games in disrupted seasons, flagged via the 'season' column)")


if __name__ == "__main__":
    main()
