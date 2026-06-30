"""Shared model feature spec, so every model trains on the same columns.

`build_model_frame` attaches Elo, the rolling/rest feature table, and (when
player data is available) leakage-safe roster strength to the spine, producing
the single frame all model predict_fns consume.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from nba_pred.features import FEATURE_COLUMNS as _ROLLING_REST_COLUMNS
from nba_pred.features import build_feature_table
from nba_pred.features.elo import compute_elo
from nba_pred.features.roster import ROSTER_FEATURES, add_roster_strength

ELO_COLUMNS = ["home_elo", "away_elo"]

# The full feature set models train on: Elo strength + as-of rolling/rest form +
# leakage-safe roster strength from player box scores.
MODEL_FEATURES = ELO_COLUMNS + _ROLLING_REST_COLUMNS + ROSTER_FEATURES

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_PLAYER_STATS = REPO_ROOT / "data" / "processed" / "player_stats.parquet"


def build_model_frame(
    games: pd.DataFrame, player_stats_path: Path = DEFAULT_PLAYER_STATS
) -> pd.DataFrame:
    """Spine + Elo + rolling/rest + roster features, ready for walk_forward.

    Each step is leakage-safe and order-preserving, so they compose cleanly. If
    the player-stats parquet is missing, the roster columns are added as NaN so
    the feature set stays consistent.
    """
    with_elo = compute_elo(games)
    frame = build_feature_table(with_elo)

    if player_stats_path.exists():
        players = pd.read_parquet(player_stats_path)
        frame = add_roster_strength(frame, players)
    else:
        for col in ROSTER_FEATURES:
            frame[col] = np.nan
    return frame
