"""Smoke tests: the model predict_fns run through walk_forward and return
valid probabilities. Uses tiny synthetic data and tiny models for speed."""

import numpy as np
import pandas as pd

from nba_pred.eval.backtest import walk_forward
from nba_pred.models.features import build_model_frame
from nba_pred.models.logistic import make_logistic_predict
from nba_pred.models.xgb import make_xgb_predict


def _synthetic_spine() -> pd.DataFrame:
    rows = []
    gid = 0
    seasons = [("2018-19", "2018-11-01"), ("2019-20", "2019-11-01"), ("2020-21", "2020-12-01")]
    for season, start in seasons:
        dates = pd.date_range(start, periods=12, freq="2D")
        for i, d in enumerate(dates):
            gid += 1
            rows.append(
                {
                    "game_id": f"{gid:08d}",
                    "game_date": d,
                    "season": season,
                    "home_team_id": 1 + (i % 4),
                    "away_team_id": 1 + ((i + 1) % 4),
                    "home_pts": 100 + (i % 5),
                    "away_pts": 100 + ((i + 2) % 5),
                    "home_win": i % 2,
                }
            )
    return pd.DataFrame(rows)


def test_logistic_runs_through_walk_forward():
    frame = build_model_frame(_synthetic_spine())
    per_season, agg = walk_forward(frame, make_logistic_predict(), min_train=1)
    assert agg.n > 0
    assert len(per_season) >= 1


def test_xgb_runs_and_returns_valid_probs():
    frame = build_model_frame(_synthetic_spine())
    # tiny model for speed
    predict = make_xgb_predict(n_estimators=10, max_depth=2)
    _, agg = walk_forward(frame, predict, min_train=1)
    assert agg.n > 0


def test_predict_fns_return_probabilities_in_range():
    frame = build_model_frame(_synthetic_spine())
    train = frame[frame["season"] != "2020-21"]
    test = frame[frame["season"] == "2020-21"]
    for predict in (make_logistic_predict(), make_xgb_predict(n_estimators=10)):
        p = predict(train, test)
        assert len(p) == len(test)
        assert np.all((p >= 0) & (p <= 1))
