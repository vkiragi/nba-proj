"""Generate the beat-the-baselines table — the heart of the writeup.

Runs every model through the SAME walk-forward harness and writes a comparison
table (log loss / Brier / accuracy over all held-out games) to docs/results.md.

    PYTHONPATH=src uv run python scripts/results_table.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from nba_pred.eval.backtest import elo_predict, walk_forward
from nba_pred.models.calibration import make_calibrated_predict
from nba_pred.models.features import build_model_frame
from nba_pred.models.logistic import make_logistic_predict
from nba_pred.models.xgb import make_xgb_predict

REPO = Path(__file__).resolve().parents[1]


def base_rate_predict(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    """Baseline: predict the train-set home-win rate for every game."""
    return np.full(len(test), float(train["home_win"].mean()))


def main() -> None:
    games = pd.read_parquet(REPO / "data" / "processed" / "games.parquet")
    frame = build_model_frame(games)

    specs = [
        ("Base rate (home win %)", games, base_rate_predict),
        ("Elo (home_adv=50)", games, elo_predict),
        ("Logistic (Elo + form/rest)", frame, make_logistic_predict()),
        ("XGBoost", frame, make_xgb_predict()),
        ("XGBoost (calibrated)", frame, make_calibrated_predict(make_xgb_predict(), "isotonic")),
    ]

    rows = []
    for name, data, predict in specs:
        _, agg = walk_forward(data, predict, name)
        rows.append(
            {"model": name, "log_loss": agg.log_loss, "brier": agg.brier,
             "accuracy": agg.accuracy, "n": agg.n}
        )
        print(f"{name:>30}: {agg}")

    table = pd.DataFrame(rows).sort_values("log_loss").reset_index(drop=True)

    md = ["# Results — beat-the-baselines table", "",
          "Walk-forward (train seasons 1..k, test k+1), aggregated over all "
          "held-out games (2007-08 → 2025-26). **Lower log loss is better; "
          "it is the primary metric.**", "",
          "| Model | Log loss | Brier | Accuracy | n |", "|---|---|---|---|---|"]
    for _, r in table.iterrows():
        md.append(f"| {r['model']} | {r['log_loss']:.4f} | {r['brier']:.4f} "
                  f"| {r['accuracy']:.3f} | {int(r['n'])} |")
    md += ["",
           "## Reading this",
           "- **Logistic regression (Elo + rolling form/rest) is the best model** "
           "on log loss — the as-of features add real signal on top of Elo.",
           "- **XGBoost does not beat logistic** out of the box: on a small, smooth "
           "feature set it overfits. Fancier != better (the plan's explicit lesson).",
           "- **Calibration does not help** here because the raw models are already "
           "well-calibrated (see docs/calibration.png); isotonic worsens log loss at "
           "the probability extremes.",
           "- Every model clears the base-rate baseline — but the honest headline is "
           "a *well-calibrated logistic model that beats a strong Elo baseline*, "
           "evaluated without leakage.", ""]
    out = REPO / "docs" / "results.md"
    out.write_text("\n".join(md))
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
