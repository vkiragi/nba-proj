"""Walk-forward backtesting — the honest evaluation backbone.

The rule the whole project rests on: never test on data the model could have
trained on. So we roll forward through seasons — train on seasons 1..k, test on
season k+1, advance, repeat — and score each held-out season with proper scoring
rules (log loss, Brier) against the same baselines.

A model is just a `predict_fn(train_games, test_games) -> p_home` callable, so
Elo, logistic regression, and XGBoost all run through this identical harness.
For Elo (which learns online) "training" is implicit: see `elo_predict`.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator

import numpy as np
import pandas as pd

from nba_pred.eval.metrics import Scores, baseline_scores, evaluate

PredictFn = Callable[[pd.DataFrame, pd.DataFrame], np.ndarray]


def season_splits(seasons: list[str], min_train: int = 3) -> Iterator[tuple[list[str], str]]:
    """Yield (train_seasons, test_season) rolling forward in time.

    The first `min_train` seasons are only ever used for training (warmup), so
    the earliest test season has a meaningful history behind it.
    """
    ordered = sorted(seasons)
    for i in range(min_train, len(ordered)):
        yield ordered[:i], ordered[i]


def walk_forward(
    games: pd.DataFrame,
    predict_fn: PredictFn,
    model_name: str = "model",
    min_train: int = 3,
    season_col: str = "season",
    return_predictions: bool = False,
):
    """Run a model through walk-forward validation.

    Returns (per_season_table, aggregate_scores). If return_predictions=True,
    also returns a third element: a DataFrame of out-of-fold predictions with
    columns [game_id, season, home_win, p] for plotting calibration etc.
    """
    seasons = sorted(games[season_col].unique())
    rows: list[dict] = []
    all_y: list[int] = []
    all_p: list[float] = []
    pred_frames: list[pd.DataFrame] = []

    for train_seasons, test_season in season_splits(seasons, min_train):
        train = games[games[season_col].isin(train_seasons)]
        test = games[games[season_col] == test_season]

        p = np.asarray(predict_fn(train, test), dtype=float)
        model = evaluate(test["home_win"], p)
        bases = baseline_scores(test["home_win"])

        rows.append(
            {
                "test_season": test_season,
                "n": model.n,
                "log_loss": model.log_loss,
                "brier": model.brier,
                "accuracy": model.accuracy,
                "base_rate_log_loss": bases["base rate"].log_loss,
                "beats_base_rate": model.log_loss < bases["base rate"].log_loss,
            }
        )
        all_y.extend(test["home_win"].tolist())
        all_p.extend(p.tolist())
        if return_predictions:
            pred_frames.append(
                pd.DataFrame(
                    {
                        "game_id": test["game_id"].to_numpy(),
                        "season": test_season,
                        "home_win": test["home_win"].to_numpy(),
                        "p": p,
                    }
                )
            )

    per_season = pd.DataFrame(rows)
    aggregate = evaluate(all_y, all_p)
    if return_predictions:
        return per_season, aggregate, pd.concat(pred_frames, ignore_index=True)
    return per_season, aggregate


def elo_predict(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    """Elo as a walk-forward predictor.

    Elo learns online, so the correct pre-game rating for a test game reflects
    ALL prior games (train seasons + earlier games this season). We get that by
    running compute_elo over train+test together and reading the ratings the
    test games go in with — never using a game's own result. Predictions are
    returned in `test` row order.
    """
    from nba_pred.features.elo import compute_elo, expected_home

    combined = pd.concat([train, test], ignore_index=True)
    scored = compute_elo(combined).set_index("game_id")

    test_scored = scored.loc[test["game_id"]]
    return np.array(
        [expected_home(h, a) for h, a in zip(test_scored["home_elo"], test_scored["away_elo"])]
    )
