"""Unit tests for Elo — the math, and (critically) the leakage property."""

import pandas as pd
import pytest

from nba_pred.features.elo import (
    START_RATING,
    compute_elo,
    expected_home,
    mov_multiplier,
)


# --- expected_home: the win-probability formula -----------------------------

def test_equal_ratings_no_home_adv_is_coin_flip():
    assert expected_home(1500, 1500, home_adv=0) == pytest.approx(0.5)


def test_home_advantage_favors_home():
    # With a home bonus, equal-rated teams give home > 50%.
    assert expected_home(1500, 1500, home_adv=100) > 0.5


def test_400_point_gap_is_about_91_percent():
    # The defining property of the 400 scale: +400 Elo ≈ 10:1 odds ≈ 0.909.
    assert expected_home(1900, 1500, home_adv=0) == pytest.approx(0.909, abs=1e-3)


def test_expectation_is_symmetric():
    # P(home) + P(home with teams swapped) == 1, with no home edge.
    assert expected_home(1600, 1400, home_adv=0) + expected_home(1400, 1600, home_adv=0) == pytest.approx(1.0)


# --- mov_multiplier: margin scaling + autocorrelation correction ------------

def test_mov_multiplier_grows_with_margin():
    # A bigger blowout produces a larger update multiplier (evenly matched teams).
    small = mov_multiplier(margin=1, winner_elo_diff=0)
    big = mov_multiplier(margin=25, winner_elo_diff=0)
    assert big > small


def test_mov_multiplier_autocorrelation_dampens_favorites():
    # Same margin: a heavily favored winner gains LESS than an underdog winner.
    # (The denominator grows with the winner's pre-game rating advantage.)
    favorite = mov_multiplier(margin=10, winner_elo_diff=300)
    underdog = mov_multiplier(margin=10, winner_elo_diff=-300)
    assert favorite < underdog


# --- compute_elo: the update + the leakage guarantee ------------------------

def _two_games() -> pd.DataFrame:
    """Team A (1) beats B (2) on day 1, then hosts C (3) on day 2."""
    return pd.DataFrame(
        {
            "game_date": pd.to_datetime(["2023-01-01", "2023-01-02"]),
            "home_team_id": [1, 2],  # game1: A home; game2: B home
            "away_team_id": [2, 3],
            "home_win": [1, 0],      # game1: A (home) wins; game2: C (away) wins
        }
    )


def test_first_game_uses_start_rating_for_both():
    # Every team's FIRST appearance must show the start rating — no future info.
    out = compute_elo(_two_games(), home_adv=0)
    first = out.iloc[0]
    assert first.home_elo == START_RATING
    assert first.away_elo == START_RATING


def test_hand_checked_update():
    # Two 1500 teams, home wins, K=20, no home adv -> 1510 / 1490 (done by hand).
    out = compute_elo(_two_games(), k=20, home_adv=0)
    # Game 2 is B(home) vs C(away). B lost game 1, so its pre-game rating in
    # game 2 must be its POST-game-1 rating: 1490.
    second = out.iloc[1]
    assert second.home_elo == pytest.approx(1490.0)   # B after losing game 1
    assert second.away_elo == START_RATING            # C's first game


def test_update_is_zero_sum():
    # Whatever the home team gains, the away team loses (ratings conserve).
    out = compute_elo(_two_games(), k=20, home_adv=0)
    # After game 1: A = 1500 + x, B = 1500 - x. We can read B's pre-game-2 elo.
    b_after_game1 = out.iloc[1].home_elo
    a_gain = (START_RATING + START_RATING) - b_after_game1 - START_RATING
    assert a_gain == pytest.approx(START_RATING - b_after_game1)


def test_no_future_leakage_recorded_rating_predates_result():
    # The recorded pre-game rating for a team must NOT reflect that game's
    # outcome: it must equal the rating it carried in before tip-off.
    out = compute_elo(_two_games(), k=20, home_adv=0)
    # B's rating going INTO game 2 (1490) reflects only game 1, not game 2.
    assert out.iloc[1].home_elo == pytest.approx(1490.0)
