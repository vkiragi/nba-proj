"""Generate game-by-game walk-forward predictions for the app's backtest page.

Runs the best model (logistic regression) through the SAME walk-forward harness
used for evaluation — so every prediction is genuinely out-of-sample (the model
that predicts a game trained only on prior seasons; no leakage). Joins each
prediction back to its matchup (teams, date, final score) and saves one parquet
the Streamlit app can read and filter instantly, without heavy compute at load.

    PYTHONPATH=src uv run python scripts/backtest_predictions.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from nba_pred.eval.backtest import walk_forward
from nba_pred.models.features import build_model_frame
from nba_pred.models.logistic import make_logistic_predict
from nba_pred.serve.predictor import team_lookup

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "data" / "processed" / "backtest_predictions.parquet"


def main() -> None:
    games = pd.read_parquet(REPO / "data" / "processed" / "games.parquet")
    frame = build_model_frame(games)

    # Out-of-fold predictions from the best model: [game_id, season, home_win, p].
    _, agg, preds = walk_forward(
        frame, make_logistic_predict(), "Logistic", return_predictions=True
    )
    print(f"walk-forward aggregate: {agg}")

    # Join matchup context for display (date, teams, final score).
    meta = games[
        ["game_id", "game_date", "home_team_id", "away_team_id", "home_pts", "away_pts"]
    ]
    df = preds.merge(meta, on="game_id", how="left")

    names = {tid: t["full_name"] for tid, t in team_lookup().items()}
    df["home"] = df["home_team_id"].map(names).fillna(df["home_team_id"].astype(str))
    df["away"] = df["away_team_id"].map(names).fillna(df["away_team_id"].astype(str))

    # Derived display columns.
    df["p_home"] = df["p"]
    df["pred_home_win"] = (df["p"] >= 0.5).astype(int)
    df["correct"] = (df["pred_home_win"] == df["home_win"]).astype(int)
    df["confidence"] = (df["p"] - 0.5).abs() * 2.0  # 0 = coin flip, 1 = certain

    keep = [
        "game_id", "season", "game_date", "home", "away",
        "home_pts", "away_pts", "home_win", "p_home",
        "pred_home_win", "correct", "confidence",
    ]
    df = df[keep].sort_values("game_date").reset_index(drop=True)

    df.to_parquet(OUT, index=False)
    print(f"saved {OUT}  ({len(df):,} games, {df['season'].nunique()} seasons)")


if __name__ == "__main__":
    main()
