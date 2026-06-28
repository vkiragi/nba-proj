"""Assemble the leakage-safe feature table via an as-of join.

The game spine is WIDE (one row per game, home_/away_ columns), but rest and
rolling-form features are naturally per-team-per-game. So:

  1. explode the spine to a per-team-long view (each game -> 2 rows),
  2. compute as-of features per team (groupby + shift(1) excludes the current
     game), then
  3. join back to the wide spine as home_/away_ feature pairs.

`build_feature_table` is a PURE function of the rows passed in (no globals, no
file I/O), so the leakage test can truncate the data to "before game G" and
rebuild, expecting identical values. NaNs (no history yet) are kept honest here;
imputation happens later, inside the linear model, fit on train only.
"""

from __future__ import annotations

import pandas as pd

from nba_pred.features.form import FORM_FEATURES, add_rolling_form
from nba_pred.features.rest import REST_FEATURES, add_rest_features

# Per-team feature columns produced on the long view (before home_/away_ prefix).
TEAM_FEATURES = REST_FEATURES + FORM_FEATURES

# Final wide feature columns attached to the spine.
FEATURE_COLUMNS = [f"home_{c}" for c in TEAM_FEATURES] + [f"away_{c}" for c in TEAM_FEATURES]


def explode_to_team_long(games: pd.DataFrame) -> pd.DataFrame:
    """Reshape the wide spine into one row per team per game (each team's view)."""
    base = ["game_id", "game_date", "season"]

    home = games[base].copy()
    home["team_id"] = games["home_team_id"].to_numpy()
    home["pts_for"] = games["home_pts"].to_numpy()
    home["pts_against"] = games["away_pts"].to_numpy()
    home["won"] = games["home_win"].to_numpy()
    home["is_home"] = True

    away = games[base].copy()
    away["team_id"] = games["away_team_id"].to_numpy()
    away["pts_for"] = games["away_pts"].to_numpy()
    away["pts_against"] = games["home_pts"].to_numpy()
    away["won"] = (1 - games["home_win"]).to_numpy()
    away["is_home"] = False

    team_long = pd.concat([home, away], ignore_index=True)
    team_long["point_diff"] = team_long["pts_for"] - team_long["pts_against"]

    # Stable chronological order per team; game_id breaks same-date ties so
    # shift(1) is deterministic. Clean RangeIndex required by the rest feature.
    team_long = team_long.sort_values(
        ["team_id", "game_date", "game_id"], kind="mergesort"
    ).reset_index(drop=True)
    return team_long


def join_back_wide(games: pd.DataFrame, team_long: pd.DataFrame) -> pd.DataFrame:
    """Attach each team's features back to the wide spine as home_/away_ pairs."""
    home_feats = team_long.loc[team_long["is_home"], ["game_id"] + TEAM_FEATURES].rename(
        columns={c: f"home_{c}" for c in TEAM_FEATURES}
    )
    away_feats = team_long.loc[~team_long["is_home"], ["game_id"] + TEAM_FEATURES].rename(
        columns={c: f"away_{c}" for c in TEAM_FEATURES}
    )
    out = games.merge(home_feats, on="game_id", how="left", validate="1:1")
    out = out.merge(away_feats, on="game_id", how="left", validate="1:1")
    return out


def build_feature_table(games: pd.DataFrame) -> pd.DataFrame:
    """Attach leakage-safe pre-game features to the game spine.

    Input: the spine (one row per game) with at least game_id, game_date,
    season, home_team_id, away_team_id, home_pts, away_pts, home_win.
    Returns a COPY of `games` with home_/away_ feature columns appended
    (see FEATURE_COLUMNS). Original columns and row order are preserved.

    Every feature for game G uses ONLY data dated strictly before G's tip-off.
    """
    team_long = explode_to_team_long(games)
    team_long = add_rest_features(team_long)
    team_long = add_rolling_form(team_long)
    return join_back_wide(games, team_long)
