"""Leakage insurance — the most important test in this repo.

The contract: for any game G, every feature value attached to G must be
computable using ONLY data dated strictly before G's tip-off. We verify it
empirically: build the feature table on the full data, then for each game
rebuild on data truncated to rows BEFORE that game plus the game itself. If any
feature value changes, a feature peeked at the future and the test fails.
"""

import pandas as pd

from nba_pred.features import build_feature_table


def _synthetic_spine() -> pd.DataFrame:
    """A tiny real-schema spine: a focal team plays several games, home & away.

    Matches the production spine (one row per game) so the leakage test
    exercises the actual explode -> as-of -> join-back transform.
    """
    return pd.DataFrame(
        {
            "game_id": ["0001", "0002", "0003", "0004"],
            "game_date": pd.to_datetime(["2023-10-25", "2023-10-27", "2023-10-29", "2023-10-30"]),
            "season": ["2023-24"] * 4,
            "home_team_id": [10, 20, 10, 30],
            "away_team_id": [20, 10, 30, 10],
            "home_pts": [120, 100, 110, 99],
            "away_pts": [105, 108, 102, 101],
            "home_win": [1, 0, 1, 0],
        }
    )


def test_no_feature_uses_future_data():
    """No feature for game G may use data dated >= G's tip-off."""
    games = _synthetic_spine()
    full = build_feature_table(games)
    feature_cols = [c for c in full.columns if c not in games.columns]

    for _, row in full.iterrows():
        past_plus_current = pd.concat(
            [
                games[games["game_date"] < row["game_date"]],
                games[games["game_id"] == row["game_id"]],
            ]
        )
        rebuilt = build_feature_table(past_plus_current)
        recomputed = rebuilt[rebuilt["game_id"] == row["game_id"]].iloc[0]
        for col in feature_cols:
            a, b = recomputed[col], row[col]
            both_nan = pd.isna(a) and pd.isna(b)
            assert both_nan or a == b, (
                f"feature {col!r} leaks future data for game {row['game_id']}: "
                f"{a} (as-of) != {b} (full)"
            )


def test_synthetic_fixture_is_chronological():
    """Sanity check on the fixture itself so the real test rests on solid ground."""
    games = _synthetic_spine()
    assert games["game_date"].is_monotonic_increasing
