"""Rolling form features, computed as-of (strictly before tip-off).

Operates on the per-team-long view. All features are grouped by
["team_id", "season"] and shifted by one so the CURRENT game is excluded —
form resets each season (carrying form across a four-month offseason is noise).

LEAKAGE NOTE: the shift(1) is mandatory. An expanding/rolling mean WITHOUT it
would include the current game's own result — direct label leakage for
win_pct_std, and outcome leakage for the point-diff windows.
"""

from __future__ import annotations

import pandas as pd

FORM_FEATURES = ["roll_pdiff_5", "roll_pdiff_10", "win_pct_std"]


def add_rolling_form(team_long: pd.DataFrame) -> pd.DataFrame:
    """Add roll_pdiff_5, roll_pdiff_10, win_pct_std to a per-team-long frame.

    Expects columns: team_id, season, point_diff, won, sorted by
    ["team_id", "game_date", "game_id"] with a clean RangeIndex.
    """
    df = team_long

    # Rolling mean point differential over the last 5 / 10 games, EXCLUDING the
    # current game. Both the shift(1) AND the rolling must stay WITHIN each
    # team-season, so we wrap them in a single grouped transform — otherwise the
    # rolling window bleeds across team boundaries (a real leakage bug).
    pdiff = df.groupby(["team_id", "season"], sort=False)["point_diff"]
    df["roll_pdiff_5"] = pdiff.transform(lambda s: s.shift(1).rolling(5, min_periods=1).mean())
    df["roll_pdiff_10"] = pdiff.transform(lambda s: s.shift(1).rolling(10, min_periods=1).mean())

    # Season-to-date win pct as-of (excluding current). Shift the wins, then take
    # an expanding mean within each team-season.
    shifted_won = df.groupby(["team_id", "season"], sort=False)["won"].shift(1)
    df["win_pct_std"] = (
        shifted_won.groupby([df["team_id"], df["season"]], sort=False)
        .expanding(min_periods=1)
        .mean()
        .reset_index(level=[0, 1], drop=True)
    )

    return df
