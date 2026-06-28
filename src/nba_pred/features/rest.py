"""Rest / schedule features, computed as-of (strictly before tip-off).

Operates on the per-team-long view (one row per team per game, sorted by
team and date). Features are grouped by ["team_id", "season"] so rest never
crosses a season boundary — a season opener has undefined (NaN) rest, not a
nonsensical four-month gap.

LEAKAGE NOTE: games_last_7 uses rolling("7D", closed="left"). The default
closed="right" would include the current game in its own window — leakage.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

REST_FEATURES = ["rest_days", "is_b2b", "games_last_7"]


def add_rest_features(team_long: pd.DataFrame) -> pd.DataFrame:
    """Add rest_days, is_b2b, games_last_7 to a per-team-long frame.

    Expects columns: team_id, season, game_date (datetime), sorted by
    ["team_id", "game_date", "game_id"]. Returns the same frame with the
    three feature columns added.
    """
    df = team_long
    grp = df.groupby(["team_id", "season"], sort=False)

    # Days since this team's previous game (same season). Season opener -> NaT -> NaN.
    prev_date = grp["game_date"].shift(1)
    df["rest_days"] = (df["game_date"] - prev_date).dt.days

    # Back-to-back: played yesterday. Undefined for season openers (NaN, not 0).
    df["is_b2b"] = (df["rest_days"] == 1).astype("float")
    df.loc[df["rest_days"].isna(), "is_b2b"] = np.nan

    # Count of this team's games in the 7 days BEFORE the current game.
    # closed="left" => window is [t-7d, t), excluding the current game.
    # We build a per-group rolling sum on a DatetimeIndex, then realign the
    # result back to df's row index (df must have a clean RangeIndex).
    def _last7(g: pd.DataFrame) -> pd.Series:
        rolled = (
            pd.Series(1.0, index=pd.DatetimeIndex(g["game_date"]))
            .rolling("7D", closed="left")
            .sum()
        )
        return pd.Series(rolled.to_numpy(), index=g.index)

    # An empty window (a team's first game of the season) sums to NaN; a count of
    # prior games is genuinely 0 there, not "unknown", so fill it. This is a true
    # zero, not imputation — unlike rest_days, which stays NaN (undefined).
    df["games_last_7"] = (
        df.groupby(["team_id", "season"], sort=False, group_keys=False)
        .apply(_last7, include_groups=False)
        .fillna(0.0)
    )

    return df
