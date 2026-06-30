"""SHAP explainability for the XGBoost model.

Trains a final XGBoost on all-but-the-last season, computes SHAP values on the
held-out last season, and saves a summary plot. SHAP also doubles as a leakage
check: if one feature dominates implausibly, investigate it.

    PYTHONPATH=src uv run python scripts/shap_report.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from nba_pred.models.features import MODEL_FEATURES, build_model_frame  # noqa: E402
from nba_pred.models.xgb import fit_final_model  # noqa: E402

REPO = Path(__file__).resolve().parents[1]


def main() -> None:
    import shap

    games = pd.read_parquet(REPO / "data" / "processed" / "games.parquet")
    frame = build_model_frame(games)

    seasons = sorted(frame["season"].unique())
    train = frame[frame["season"] != seasons[-1]]
    test = frame[frame["season"] == seasons[-1]]

    model = fit_final_model(train)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(test[MODEL_FEATURES])

    # Mean |SHAP| per feature — the global importance ranking.
    importance = (
        pd.DataFrame({"feature": MODEL_FEATURES, "mean_abs_shap": abs(shap_values).mean(axis=0)})
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )
    print("Feature importance (mean |SHAP|):")
    print(importance.to_string(index=False))

    top = importance.iloc[0]
    share = top["mean_abs_shap"] / importance["mean_abs_shap"].sum()
    print(f"\nLeakage check: top feature {top['feature']!r} holds "
          f"{share:.1%} of total importance "
          f"({'OK — no single feature dominates' if share < 0.5 else 'INVESTIGATE'}).")

    shap.summary_plot(shap_values, test[MODEL_FEATURES], show=False)
    plt.tight_layout()
    out = REPO / "docs" / "shap_summary.png"
    plt.savefig(out, dpi=120, bbox_inches="tight")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
