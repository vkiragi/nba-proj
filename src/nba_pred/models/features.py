"""Shared model feature spec, so every model trains on the same columns.

`build_model_frame` attaches both Elo and the rolling/rest feature table to the
spine, producing the single frame all model predict_fns consume.
"""

from __future__ import annotations

import pandas as pd

from nba_pred.features import FEATURE_COLUMNS as _ROLLING_REST_COLUMNS
from nba_pred.features import build_feature_table
from nba_pred.features.elo import compute_elo

ELO_COLUMNS = ["home_elo", "away_elo"]

# The full feature set models train on: Elo strength + as-of rolling/rest form.
MODEL_FEATURES = ELO_COLUMNS + _ROLLING_REST_COLUMNS


def build_model_frame(games: pd.DataFrame) -> pd.DataFrame:
    """Spine + Elo + rolling/rest features, ready for walk_forward.

    Both feature steps are leakage-safe and order-preserving, so they compose
    cleanly: each attaches columns keyed to the same one-row-per-game spine.
    """
    with_elo = compute_elo(games)
    return build_feature_table(with_elo)
