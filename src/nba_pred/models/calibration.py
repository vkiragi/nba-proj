"""Probability calibration — make "70%" actually mean 70%.

Raw model probabilities (especially gradient boosting) are often poorly
calibrated. We wrap any base predict_fn so that, within each walk-forward fold,
it: (1) splits a TIME-LATEST slice off the training data as a calibration
holdout, (2) fits the base model on the earlier part, (3) fits an isotonic (or
sigmoid/Platt) map on the holdout's predictions vs outcomes, (4) applies that
map to the test predictions.

LEAKAGE NOTE: the calibration holdout is the time-latest slice of train, NOT a
random split — that mirrors walk-forward semantics (calibrate on the most recent
known games, predict the future). The base model never sees the holdout in
training, and nothing ever sees the test set.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def calibration_curve_points(y_true, p_pred, n_bins: int = 10):
    """Reliability-curve points: (mean predicted, observed frequency) per bin."""
    from sklearn.calibration import calibration_curve

    frac_pos, mean_pred = calibration_curve(y_true, p_pred, n_bins=n_bins, strategy="quantile")
    return mean_pred, frac_pos


def make_calibrated_predict(base_predict, method: str = "isotonic", holdout_frac: float = 0.2):
    """Wrap a base predict_fn(train, test) with time-aware calibration.

    `base_predict` must accept (train, test) and return p_home. We re-use it on
    the inner (earlier) training split, then calibrate on the latest slice.
    """

    def predict(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
        from sklearn.isotonic import IsotonicRegression
        from sklearn.linear_model import LogisticRegression

        ordered = train.sort_values("game_date")
        cut = int(len(ordered) * (1.0 - holdout_frac))
        inner_train, calib = ordered.iloc[:cut], ordered.iloc[cut:]

        # Base predictions on the calibration holdout and on the real test set.
        p_calib = base_predict(inner_train, calib)
        p_test = base_predict(inner_train, test)

        y_calib = calib["home_win"].to_numpy()
        if method == "isotonic":
            cal = IsotonicRegression(out_of_bounds="clip")
            cal.fit(p_calib, y_calib)
            return cal.predict(p_test)
        elif method == "sigmoid":  # Platt scaling
            cal = LogisticRegression()
            cal.fit(p_calib.reshape(-1, 1), y_calib)
            return cal.predict_proba(p_test.reshape(-1, 1))[:, 1]
        raise ValueError(f"unknown calibration method: {method!r}")

    return predict
