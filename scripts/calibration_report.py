"""Generate the before/after calibration (reliability) curve and metrics.

Runs the chosen models through walk-forward, collects out-of-fold predictions,
plots reliability curves vs the diagonal, and saves to docs/calibration.png.

    PYTHONPATH=src uv run python scripts/calibration_report.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from nba_pred.eval.backtest import walk_forward  # noqa: E402
from nba_pred.models.calibration import calibration_curve_points, make_calibrated_predict  # noqa: E402
from nba_pred.models.features import build_model_frame  # noqa: E402
from nba_pred.models.logistic import make_logistic_predict  # noqa: E402
from nba_pred.models.xgb import make_xgb_predict  # noqa: E402

REPO = Path(__file__).resolve().parents[1]


def main() -> None:
    games = pd.read_parquet(REPO / "data" / "processed" / "games.parquet")
    frame = build_model_frame(games)

    specs = [
        ("Logistic (raw)", make_logistic_predict()),
        ("XGBoost (raw)", make_xgb_predict()),
        ("XGBoost (isotonic-calibrated)", make_calibrated_predict(make_xgb_predict(), "isotonic")),
    ]

    plt.figure(figsize=(6, 6))
    plt.plot([0, 1], [0, 1], "k--", label="perfectly calibrated")

    for name, predict in specs:
        _, agg, preds = walk_forward(frame, predict, name, return_predictions=True)
        mean_pred, frac_pos = calibration_curve_points(preds["home_win"], preds["p"], n_bins=10)
        plt.plot(mean_pred, frac_pos, marker="o", label=f"{name}  (logloss {agg.log_loss:.4f})")
        print(f"{name:>32}: {agg}")

    plt.xlabel("mean predicted P(home win)")
    plt.ylabel("observed frequency of home win")
    plt.title("Reliability curve (walk-forward, out-of-fold)")
    plt.legend(loc="upper left", fontsize=8)
    plt.tight_layout()
    out = REPO / "docs" / "calibration.png"
    plt.savefig(out, dpi=120)
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
