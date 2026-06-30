"""Tests for the betting math and backtest — including the bugs we hit."""

import numpy as np
import pandas as pd
import pytest

from nba_pred.eval.betting import (
    american_profit,
    american_to_prob,
    betting_backtest,
    devig_two_way,
    prob_to_american,
)


def test_american_to_prob_known_values():
    assert american_to_prob(100) == pytest.approx(0.5)
    assert american_to_prob(-110) == pytest.approx(0.5238, abs=1e-3)
    assert american_to_prob(-200) == pytest.approx(0.6667, abs=1e-3)


def test_prob_to_american_round_trips():
    # The fix for the median-of-odds bug: aggregate in prob space, convert back.
    for p in [0.3, 0.45, 0.5, 0.55, 0.7, 0.9]:
        assert american_to_prob(prob_to_american(p)) == pytest.approx(p, abs=1e-9)


def test_prob_to_american_always_valid():
    # Any probability in (0,1) maps to a valid line (|odds| >= 100).
    for p in [0.01, 0.49, 0.5, 0.51, 0.99]:
        assert abs(prob_to_american(p)) >= 100 - 1e-9


def test_devig_sums_to_one():
    h, a = devig_two_way(american_to_prob(-110), american_to_prob(-110))
    assert h + a == pytest.approx(1.0)
    assert h == pytest.approx(0.5)


def test_american_profit_and_rejects_garbage():
    assert american_profit(-110) == pytest.approx(0.909, abs=1e-3)
    assert american_profit(150) == pytest.approx(1.5)
    with pytest.raises(ValueError):
        american_profit(-8)  # the kind of garbage odds that faked an edge


def test_betting_backtest_known_outcome():
    # One game: we love the home team, market is 50/50, home wins at +100.
    preds = pd.DataFrame({"game_id": ["g1"], "p": [0.9], "home_win": [1]})
    odds = pd.DataFrame({"game_id": ["g1"], "market_p_home": [0.5], "home_ml": [100.0], "away_ml": [-110.0]})
    r = betting_backtest(preds, odds, edge_threshold=0.1)
    assert r["n_bets"] == 1
    assert r["win_rate"] == 1.0
    assert r["roi"] == pytest.approx(1.0)  # +100 odds, won -> +1 per 1 staked


def test_betting_backtest_no_bets_when_no_edge():
    preds = pd.DataFrame({"game_id": ["g1"], "p": [0.5], "home_win": [1]})
    odds = pd.DataFrame({"game_id": ["g1"], "market_p_home": [0.5], "home_ml": [-110.0], "away_ml": [-110.0]})
    r = betting_backtest(preds, odds, edge_threshold=0.05)
    assert r["n_bets"] == 0
