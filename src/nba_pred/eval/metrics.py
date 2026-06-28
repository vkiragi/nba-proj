"""Reusable evaluation for win-probability models.

Per the plan, the headline metrics are PROPER SCORING RULES — log loss
(primary) and Brier — not accuracy. Accuracy is reported as a secondary,
intuitive number. Every model is judged against the same baselines:
a coin flip and always-predict-the-base-rate.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss


@dataclass
class Scores:
    log_loss: float
    brier: float
    accuracy: float
    n: int

    def __str__(self) -> str:
        return (
            f"log_loss={self.log_loss:.4f}  "
            f"brier={self.brier:.4f}  "
            f"acc={self.accuracy:.3f}  "
            f"n={self.n}"
        )


def evaluate(y_true, p_home) -> Scores:
    """Score predicted home-win probabilities against actual outcomes."""
    y = np.asarray(y_true)
    p = np.asarray(p_home, dtype=float)
    return Scores(
        log_loss=log_loss(y, p, labels=[0, 1]),
        brier=brier_score_loss(y, p),
        accuracy=accuracy_score(y, p > 0.5),
        n=len(y),
    )


def baseline_scores(y_true) -> dict[str, Scores]:
    """The two reference points every real model must beat on log loss."""
    y = np.asarray(y_true)
    base_rate = float(y.mean())
    return {
        "coin (0.5)": evaluate(y, np.full(len(y), 0.5)),
        "base rate": evaluate(y, np.full(len(y), base_rate)),
    }


def report(y_true, p_home, model_name: str = "model") -> None:
    """Print a model's scores alongside the baselines."""
    print(f"{model_name:>16}: {evaluate(y_true, p_home)}")
    for name, s in baseline_scores(y_true).items():
        print(f"{name:>16}: {s}")
