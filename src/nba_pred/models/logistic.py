"""Logistic regression baseline — interpretable, fast, honest.

Returns a predict_fn(train, test) -> p_home for the walk_forward harness. The
sklearn Pipeline fits imputation + scaling on TRAIN ONLY (transforming test),
so there is no leakage from test statistics into preprocessing.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from nba_pred.models.features import MODEL_FEATURES


def make_logistic_predict(feature_cols: list[str] | None = None):
    """Build a logistic-regression predict_fn for walk_forward."""
    cols = feature_cols or MODEL_FEATURES

    def predict(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        pipe = Pipeline(
            [
                ("impute", SimpleImputer(strategy="median")),  # fit on train only
                ("scale", StandardScaler()),
                ("clf", LogisticRegression(max_iter=1000)),
            ]
        )
        pipe.fit(train[cols], train["home_win"])
        return pipe.predict_proba(test[cols])[:, 1]

    return predict
