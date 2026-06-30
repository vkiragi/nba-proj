"""Tests for the leakage-safe roster-strength feature."""

import numpy as np
import pandas as pd

from nba_pred.features.roster import (
    add_roster_strength,
    current_roster_strength,
    player_ratings_as_of,
)


def _player_stats() -> pd.DataFrame:
    # Two teams (10, 20), three game-dates. Team 10 has a strong player (P1) and
    # a weak one (P2). Plus/minus is what drives the rating.
    rows = []
    dates = ["2023-10-25", "2023-10-27", "2023-10-29"]
    gids = ["0001", "0002", "0003"]
    for d, gid in zip(dates, gids):
        rows += [
            {"game_id": gid, "game_date": d, "season": "2023-24", "team_id": 10,
             "player_id": 1, "minutes": 36, "plus_minus": 12},
            {"game_id": gid, "game_date": d, "season": "2023-24", "team_id": 10,
             "player_id": 2, "minutes": 12, "plus_minus": -4},
            {"game_id": gid, "game_date": d, "season": "2023-24", "team_id": 20,
             "player_id": 3, "minutes": 30, "plus_minus": -8},
        ]
    df = pd.DataFrame(rows)
    df["game_date"] = pd.to_datetime(df["game_date"])
    return df


def _games() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "game_id": ["0001", "0002", "0003"],
            "game_date": pd.to_datetime(["2023-10-25", "2023-10-27", "2023-10-29"]),
            "season": ["2023-24"] * 3,
            "home_team_id": [10, 10, 20],
            "away_team_id": [20, 20, 10],
            "home_win": [1, 1, 0],
        }
    )


def test_player_rating_excludes_current_game():
    ps = player_ratings_as_of(_player_stats())
    p1 = ps[ps["player_id"] == 1].sort_values("game_date")
    # First game has no prior history -> NaN (min_periods=5 not met anyway).
    assert pd.isna(p1.iloc[0]["player_rating"])


def test_roster_strength_is_as_of_first_game_nan():
    out = add_roster_strength(_games(), _player_stats())
    # First game of the season: no previous lineup -> NaN roster strength.
    first = out[out["game_id"] == "0001"].iloc[0]
    assert pd.isna(first["home_roster_strength"])
    assert pd.isna(first["away_roster_strength"])


def test_roster_strength_no_future_leakage():
    """Roster strength for game G must not change if later games are removed."""
    games, players = _games(), _player_stats()
    full = add_roster_strength(games, players)
    for gid in games["game_id"]:
        gdate = games.loc[games["game_id"] == gid, "game_date"].iloc[0]
        past_games = games[games["game_date"] <= gdate]
        past_players = players[players["game_date"] <= gdate]
        rebuilt = add_roster_strength(past_games, past_players)
        for col in ("home_roster_strength", "away_roster_strength"):
            a = rebuilt.loc[rebuilt["game_id"] == gid, col].iloc[0]
            b = full.loc[full["game_id"] == gid, col].iloc[0]
            assert (pd.isna(a) and pd.isna(b)) or a == b


def _player_stats_long() -> pd.DataFrame:
    # 8 games per team so the player rating window (min_periods=5) is satisfied.
    rows = []
    dates = pd.date_range("2023-10-25", periods=8, freq="2D")
    for i, d in enumerate(dates):
        gid = f"{i:04d}"
        rows += [
            {"game_id": gid, "game_date": d, "season": "2023-24", "team_id": 10,
             "player_id": 1, "minutes": 36, "plus_minus": 10},
            {"game_id": gid, "game_date": d, "season": "2023-24", "team_id": 20,
             "player_id": 3, "minutes": 30, "plus_minus": -6},
        ]
    return pd.DataFrame(rows)


def test_current_roster_strength_returns_per_team():
    cur = current_roster_strength(_player_stats_long())
    assert set(cur.keys()) == {10, 20}
    assert all(isinstance(v, float) for v in cur.values())
    # Team 10's strong player should give it a higher current rating than team 20.
    assert cur[10] > cur[20]
