"""XGBoost — the "real" model. Handles NaN natively, no scaling needed.

Returns a predict_fn(train, test) -> p_home for the walk_forward harness.
Hyperparameters are conservative defaults; time-aware tuning happens later.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from nba_pred.models.features import MODEL_FEATURES

DEFAULT_PARAMS = dict(
    objective="binary:logistic",
    eval_metric="logloss",
    n_estimators=400,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    n_jobs=-1,
)


def make_xgb_predict(feature_cols: list[str] | None = None, **params):
    """Build an XGBoost predict_fn for walk_forward."""
    cols = feature_cols or MODEL_FEATURES
    cfg = {**DEFAULT_PARAMS, **params}

    def predict(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
        from xgboost import XGBClassifier

        clf = XGBClassifier(**cfg)
        clf.fit(train[cols], train["home_win"])  # NaNs handled natively
        return clf.predict_proba(test[cols])[:, 1]

    return predict


def fit_final_model(games_with_features: pd.DataFrame, feature_cols: list[str] | None = None, **params):
    """Fit a single XGBoost on ALL provided rows (for a deployable artifact)."""
    from xgboost import XGBClassifier

    cols = feature_cols or MODEL_FEATURES
    cfg = {**DEFAULT_PARAMS, **params}
    clf = XGBClassifier(**cfg)
    clf.fit(games_with_features[cols], games_with_features["home_win"])
    return clf
