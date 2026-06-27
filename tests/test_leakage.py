"""Leakage insurance — the most important test in this repo.

The contract: for any game G, every feature value attached to G must be
computable using ONLY data dated strictly before G's tip-off. If any feature
peeks at data dated >= G's tip-off, that's leakage and this test must fail.

This is written *before* the feature code on purpose: it's a constraint the
feature engineering has to satisfy, not an afterthought. As feature builders
land in `nba_pred.features`, flesh out `build_feature_table` below and remove
the skip.
"""

import pandas as pd
import pytest


def _synthetic_games() -> pd.DataFrame:
    """A tiny, fully-known game log to test the as-of join against.

    Three games for one team on known dates. A correct rolling feature for the
    last game must only reflect the first two games' results.
    """
    return pd.DataFrame(
        {
            "game_id": ["0001", "0002", "0003"],
            "game_date": pd.to_datetime(["2023-10-25", "2023-10-27", "2023-10-29"]),
            "team_id": [1610612744, 1610612744, 1610612744],
            "points": [120, 105, 110],
            "won": [1, 0, 1],
        }
    )


@pytest.mark.skip(reason="enable once nba_pred.features.build_feature_table exists")
def test_no_feature_uses_future_data():
    """No feature for game G may use data dated >= G's tip-off.

    Strategy once features exist: build the feature table, then for each game G
    rebuild features from a dataset truncated to rows with game_date < G's
    tip-off. The values must be identical. If they differ, a feature saw the
    future.
    """
    from nba_pred.features import build_feature_table  # noqa: F401  (lands in Phase 2)

    games = _synthetic_games()
    full = build_feature_table(games)

    feature_cols = [c for c in full.columns if c not in games.columns]

    for _, row in full.iterrows():
        past_only = games[games["game_date"] < row["game_date"]]
        as_of = build_feature_table(pd.concat([past_only, games[games["game_id"] == row["game_id"]]]))
        recomputed = as_of[as_of["game_id"] == row["game_id"]].iloc[0]
        for col in feature_cols:
            assert recomputed[col] == row[col], f"feature {col!r} leaks future data for game {row['game_id']}"


def test_synthetic_fixture_is_chronological():
    """Sanity check on the fixture itself so the real test rests on solid ground."""
    games = _synthetic_games()
    assert games["game_date"].is_monotonic_increasing
