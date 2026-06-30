"""Ingest per-player, per-game box scores via nba_api -> clean parquet.

Source: nba_api's PlayerGameLogs (one row per player per game). Unlike the
Kaggle player file (which stops in 2018), this covers the SAME seasons as our
games spine, so player-based features can help live predictions too.

Output: data/processed/player_stats.parquet with one row per player-game:
    game_id, game_date, season, team_id, player_id, player_name,
    minutes, pts, reb, ast, plus_minus

CLI:
    PYTHONPATH=src uv run python -m nba_pred.ingest.players --all
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd
from nba_api.stats.endpoints import playergamelogs

from nba_pred.ingest.games import DEFAULT_PROCESSED_DIR, REPO_ROOT, season_range

DEFAULT_RAW_DIR = REPO_ROOT / "data" / "raw" / "players"

PLAYER_COLUMNS = [
    "game_id",
    "game_date",
    "season",
    "team_id",
    "player_id",
    "player_name",
    "minutes",
    "pts",
    "reb",
    "ast",
    "plus_minus",
]


def fetch_raw(season: str, retries: int = 3, timeout: int = 60) -> pd.DataFrame:
    """All players' game logs for one regular season (one row per player-game)."""
    last_err: Exception | None = None
    for attempt in range(retries):
        try:
            return playergamelogs.PlayerGameLogs(
                season_nullable=season,
                season_type_nullable="Regular Season",
                timeout=timeout,
            ).get_data_frames()[0]
        except Exception as e:  # nba_api/stats.nba.com is flaky — back off and retry
            last_err = e
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"failed to fetch player logs for {season}: {last_err}")


def clean(raw: pd.DataFrame, season: str) -> pd.DataFrame:
    """Select + rename to the clean player-game schema."""
    df = raw.rename(
        columns={
            "GAME_ID": "game_id",
            "GAME_DATE": "game_date",
            "TEAM_ID": "team_id",
            "PLAYER_ID": "player_id",
            "PLAYER_NAME": "player_name",
            "MIN": "minutes",
            "PTS": "pts",
            "REB": "reb",
            "AST": "ast",
            "PLUS_MINUS": "plus_minus",
        }
    )
    df["game_date"] = pd.to_datetime(df["game_date"])
    df["season"] = season
    return df[PLAYER_COLUMNS].sort_values(["game_date", "game_id"]).reset_index(drop=True)


def validate(players: pd.DataFrame, season: str) -> None:
    """Loud sanity checks."""
    assert players["player_id"].notna().all(), "null player_id"
    assert players["game_id"].notna().all(), "null game_id"
    # ~10-13 players per team-game; ~20-30 per game. Warn if oddly low.
    per_game = players.groupby("game_id").size().mean()
    if per_game < 16:
        print(f"  ⚠️  {season}: only {per_game:.1f} players/game (expected ~20)")


def fetch_season(season: str, raw_dir: Path = DEFAULT_RAW_DIR) -> pd.DataFrame:
    """Idempotent, cached single-season player ingest."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    cache = raw_dir / f"players_{season}.parquet"
    if cache.exists():
        return pd.read_parquet(cache)
    players = clean(fetch_raw(season), season)
    validate(players, season)
    players.to_parquet(cache, index=False)
    return players


def fetch_seasons(seasons: list[str], raw_dir: Path = DEFAULT_RAW_DIR) -> pd.DataFrame:
    frames = []
    for season in seasons:
        p = fetch_season(season, raw_dir)
        print(f"  {season}: {len(p):>6} player-games, {p['player_id'].nunique():>4} players")
        frames.append(p)
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest per-player per-game box scores.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--season", help='single season, e.g. "2023-24"')
    group.add_argument("--all", action="store_true", help="backfill all seasons")
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--processed-dir", type=Path, default=DEFAULT_PROCESSED_DIR)
    args = parser.parse_args()

    if args.season:
        p = fetch_season(args.season, args.raw_dir)
        print(f"✅ {args.season}: {len(p)} player-games")
        return

    seasons = season_range()
    print(f"Backfilling player stats for {len(seasons)} seasons: {seasons[0]}..{seasons[-1]}")
    players = fetch_seasons(seasons, args.raw_dir)
    args.processed_dir.mkdir(parents=True, exist_ok=True)
    out = args.processed_dir / "player_stats.parquet"
    players.to_parquet(out, index=False)
    print(f"✅ {len(players)} player-games across {players['season'].nunique()} seasons -> {out}")


if __name__ == "__main__":
    main()
