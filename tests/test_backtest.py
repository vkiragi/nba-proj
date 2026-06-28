"""Tests for the walk-forward harness — especially its no-leakage guarantee."""

import numpy as np
import pandas as pd

from nba_pred.eval.backtest import elo_predict, season_splits, walk_forward


def test_season_splits_roll_forward():
    seasons = ["2018-19", "2019-20", "2020-21", "2021-22"]
    splits = list(season_splits(seasons, min_train=2))
    # test seasons start after the first min_train, each trained on all priors.
    assert [test for _, test in splits] == ["2020-21", "2021-22"]
    assert splits[0][0] == ["2018-19", "2019-20"]
    assert splits[1][0] == ["2018-19", "2019-20", "2020-21"]


def test_predict_fn_only_sees_train_and_test_no_future():
    """The harness must hand a predict_fn only past + current season — never a
    later season. We assert that by recording what each call receives."""
    seen_train_seasons = []

    def spy(train, test):
        seen_train_seasons.append(set(train["season"]))
        # every training season must be strictly before the test season
        test_season = test["season"].iloc[0]
        assert all(s < test_season for s in train["season"].unique())
        return np.full(len(test), 0.5)

    games = _synthetic_multiseason()
    walk_forward(games, spy, min_train=1)
    assert len(seen_train_seasons) >= 1


def test_elo_predict_matches_test_row_order():
    games = _synthetic_multiseason()
    train = games[games["season"] != "2020-21"]
    test = games[games["season"] == "2020-21"]
    p = elo_predict(train, test)
    assert len(p) == len(test)
    assert np.all((p > 0) & (p < 1))


def _synthetic_multiseason() -> pd.DataFrame:
    rows = []
    gid = 0
    for season, start in [("2018-19", "2018-11-01"), ("2019-20", "2019-11-01"), ("2020-21", "2020-12-01")]:
        dates = pd.date_range(start, periods=6, freq="3D")
        for i, d in enumerate(dates):
            gid += 1
            rows.append(
                {
                    "game_id": f"{gid:08d}",
                    "game_date": d,
                    "season": season,
                    "home_team_id": 1 + (i % 3),
                    "away_team_id": 1 + ((i + 1) % 3),
                    "home_win": i % 2,
                }
            )
    return pd.DataFrame(rows)
