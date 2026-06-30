"""Tests for the time-aware calibration wrapper."""

import numpy as np
import pandas as pd

from nba_pred.models.calibration import calibration_curve_points, make_calibrated_predict
from nba_pred.models.features import build_model_frame
from nba_pred.models.logistic import make_logistic_predict


def _synthetic_spine() -> pd.DataFrame:
    rows = []
    gid = 0
    for season, start in [("2018-19", "2018-11-01"), ("2019-20", "2019-11-01")]:
        for i, d in enumerate(pd.date_range(start, periods=20, freq="2D")):
            gid += 1
            rows.append(
                {
                    "game_id": f"{gid:08d}",
                    "game_date": d,
                    "season": season,
                    "home_team_id": 1 + (i % 4),
                    "away_team_id": 1 + ((i + 1) % 4),
                    "home_pts": 100 + (i % 6),
                    "away_pts": 100 + ((i + 3) % 6),
                    "home_win": (i + (season == "2019-20")) % 2,
                }
            )
    return pd.DataFrame(rows)


def test_calibrated_predict_returns_valid_probs():
    frame = build_model_frame(_synthetic_spine())
    train = frame[frame["season"] == "2018-19"]
    test = frame[frame["season"] == "2019-20"]
    predict = make_calibrated_predict(make_logistic_predict(), method="isotonic")
    p = predict(train, test)
    assert len(p) == len(test)
    assert np.all((p >= 0) & (p <= 1))


def test_sigmoid_method_also_runs():
    frame = build_model_frame(_synthetic_spine())
    train = frame[frame["season"] == "2018-19"]
    test = frame[frame["season"] == "2019-20"]
    p = make_calibrated_predict(make_logistic_predict(), method="sigmoid")(train, test)
    assert np.all((p >= 0) & (p <= 1))


def test_calibration_curve_points_shape():
    y = np.array([0, 0, 1, 1, 0, 1, 1, 1, 0, 1] * 5)
    p = np.linspace(0.05, 0.95, len(y))
    mean_pred, frac_pos = calibration_curve_points(y, p, n_bins=5)
    assert len(mean_pred) == len(frac_pos)
    assert len(mean_pred) <= 5
