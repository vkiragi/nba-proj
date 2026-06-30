"""Smoke tests for the live matchup predictor.

Skips if the processed games data isn't built yet (so CI without data passes).
"""

import pytest

from nba_pred.serve.predictor import DEFAULT_GAMES, Predictor

pytestmark = pytest.mark.skipif(
    not DEFAULT_GAMES.exists(), reason="data/processed/games.parquet not built"
)


@pytest.fixture(scope="module")
def predictor():
    return Predictor()


def test_current_teams_are_active(predictor):
    teams = predictor.current_teams()
    assert len(teams) >= 25  # ~30 active franchises
    assert all(isinstance(tid, int) and isinstance(name, str) for tid, name in teams)


def test_predict_returns_valid_probability(predictor):
    teams = predictor.current_teams()
    home_id, away_id = teams[0][0], teams[1][0]
    res = predictor.predict(home_id, away_id)
    assert 0.0 < res["p_home"] < 1.0
    assert res["p_home"] + res["p_away"] == pytest.approx(1.0)


def test_home_court_advantage_is_reflected(predictor):
    # Same two teams; whichever is home should get a boost. So team A's win prob
    # when home should exceed its win prob when away.
    teams = predictor.current_teams()
    a, b = teams[0][0], teams[1][0]
    a_home = predictor.predict(a, b)["p_home"]
    a_away = 1.0 - predictor.predict(b, a)["p_home"]
    assert a_home > a_away
