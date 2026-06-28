"""Unit tests for the leakage-safe feature builders, with hand-computed values."""

import numpy as np
import pandas as pd

from nba_pred.features.build import build_feature_table, explode_to_team_long
from nba_pred.features.form import add_rolling_form
from nba_pred.features.rest import add_rest_features


def _one_team_spine() -> pd.DataFrame:
    """Focal team 10 plays 3 games; we know every result by hand.

    g1 10/25: team10 (home) wins 120-105  -> point_diff +15, won
    g2 10/27: team10 (away) loses 100-108 -> point_diff -8,  lost
    g3 10/29: team10 (home) wins 110-102  -> point_diff +8,  won
    """
    return pd.DataFrame(
        {
            "game_id": ["0001", "0002", "0003"],
            "game_date": pd.to_datetime(["2023-10-25", "2023-10-27", "2023-10-29"]),
            "season": ["2023-24"] * 3,
            "home_team_id": [10, 20, 10],
            "away_team_id": [20, 10, 30],
            "home_pts": [120, 108, 110],
            "away_pts": [105, 100, 102],
            # consistent with scores: g1 home(10) wins, g2 home(20) wins
            # (team10 away loses), g3 home(10) wins.
            "home_win": [1, 1, 1],
        }
    )


def _team10_long():
    long = explode_to_team_long(_one_team_spine())
    long = add_rest_features(long)
    long = add_rolling_form(long)
    return long[long["team_id"] == 10].sort_values("game_date").reset_index(drop=True)


def test_explode_doubles_rows_with_correct_perspective():
    long = explode_to_team_long(_one_team_spine())
    assert len(long) == 6  # 3 games x 2 teams
    g1_home = long[(long["game_id"] == "0001") & (long["is_home"])].iloc[0]
    assert g1_home["team_id"] == 10
    assert g1_home["point_diff"] == 15
    assert g1_home["won"] == 1


def test_rest_days_and_b2b():
    t = _team10_long()
    # First game: no prior game -> NaN rest.
    assert pd.isna(t.loc[0, "rest_days"])
    # 10/25 -> 10/27 is 2 days; 10/27 -> 10/29 is 2 days. No back-to-backs.
    assert t.loc[1, "rest_days"] == 2
    assert t.loc[2, "rest_days"] == 2
    assert t.loc[1, "is_b2b"] == 0
    assert pd.isna(t.loc[0, "is_b2b"])


def test_games_last_7_excludes_current():
    t = _team10_long()
    # Window [t-7d, t) excluding the current game.
    assert t.loc[0, "games_last_7"] == 0  # nothing before the first game
    assert t.loc[1, "games_last_7"] == 1  # only 10/25 precedes 10/27
    assert t.loc[2, "games_last_7"] == 2  # 10/25 and 10/27 precede 10/29


def test_back_to_back_detected():
    spine = _one_team_spine()
    spine.loc[1, "game_date"] = pd.Timestamp("2023-10-26")  # day after g1
    long = explode_to_team_long(spine)
    long = add_rest_features(long)
    t = long[long["team_id"] == 10].sort_values("game_date").reset_index(drop=True)
    assert t.loc[1, "rest_days"] == 1
    assert t.loc[1, "is_b2b"] == 1


def test_rolling_point_diff_excludes_current():
    t = _team10_long()
    assert pd.isna(t.loc[0, "roll_pdiff_5"])      # no history
    assert t.loc[1, "roll_pdiff_5"] == 15.0       # only g1 (+15)
    assert t.loc[2, "roll_pdiff_5"] == (15 + -8) / 2  # g1,g2 -> +3.5


def test_win_pct_is_as_of_and_excludes_current():
    t = _team10_long()
    assert pd.isna(t.loc[0, "win_pct_std"])   # no prior games
    assert t.loc[1, "win_pct_std"] == 1.0     # 1 win in 1 prior game
    assert t.loc[2, "win_pct_std"] == 0.5     # 1 win, 1 loss in 2 prior games


def test_build_feature_table_preserves_spine():
    spine = _one_team_spine()
    out = build_feature_table(spine)
    assert len(out) == len(spine)
    assert list(out["game_id"]) == list(spine["game_id"])  # order preserved
    for col in spine.columns:
        assert col in out.columns
    assert "home_rest_days" in out.columns and "away_win_pct_std" in out.columns


def test_no_fillna_nans_kept_honest():
    # build_feature_table must NOT impute; first-game features stay NaN.
    out = build_feature_table(_one_team_spine())
    first = out[out["game_id"] == "0001"].iloc[0]
    assert pd.isna(first["home_rest_days"])
    assert pd.isna(first["home_roll_pdiff_5"])
    assert np.isnan(first["home_win_pct_std"])
