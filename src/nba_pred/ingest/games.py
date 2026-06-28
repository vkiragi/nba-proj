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
from pathlib import Path

import pandas as pd
from nba_api.stats.endpoints import leaguegamelog

# Repo-root-relative default cache location (this file is src/nba_pred/ingest/games.py).
DEFAULT_RAW_DIR = Path(__file__).resolve().parents[3] / "data" / "raw"

# The clean output schema — the spine everything else joins onto.
SPINE_COLUMNS = [
    "game_id",
    "game_date",
    "home_team_id",
    "away_team_id",
    "home_pts",
    "away_pts",
    "home_win",
]


def fetch_raw(season: str) -> pd.DataFrame:
    """Hit the NBA API for one regular season: one row per team per game."""
    log = leaguegamelog.LeagueGameLog(
        season=season,
        season_type_all_star="Regular Season",
    )
    return log.get_data_frames()[0]


def collapse_to_games(raw: pd.DataFrame) -> pd.DataFrame:
    """Turn 2N team-rows into N game-rows on the clean spine schema.

    Home rows have "vs." in MATCHUP (e.g. "DEN vs. LAL"); away rows have "@"
    (e.g. "LAL @ DEN"). We split on that, merge the two halves on GAME_ID, and
    derive the home-win label.
    """
    is_away = raw["MATCHUP"].str.contains("@", regex=False)
    home = raw[~is_away]
    away = raw[is_away]

    games = home.merge(away, on="GAME_ID", suffixes=("_home", "_away"))

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
    games = collapse_to_games(raw)
    validate(games, season)
    games.to_parquet(cache, index=False)
    return games


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest one NBA season into a clean game spine.")
    parser.add_argument("--season", required=True, help='e.g. "2023-24"')
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=DEFAULT_RAW_DIR,
        help="where to cache the parquet (default: data/raw)",
    )
    args = parser.parse_args()

    games = fetch_season(args.season, args.raw_dir)
    print(f"✅ {args.season}: {len(games)} games -> {args.raw_dir / f'games_{args.season}.parquet'}")
    print(games.head().to_string(index=False))


if __name__ == "__main__":
    main()
