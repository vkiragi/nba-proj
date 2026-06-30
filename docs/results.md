# Results — beat-the-baselines table

Walk-forward (train seasons 1..k, test k+1), aggregated over all held-out games (2007-08 → 2025-26). **Lower log loss is better; it is the primary metric.**

| Model | Log loss | Brier | Accuracy | n |
|---|---|---|---|---|
| Logistic (Elo + form/rest) | 0.6098 | 0.2111 | 0.664 | 22798 |
| Elo (home_adv=50) | 0.6165 | 0.2140 | 0.658 | 22798 |
| XGBoost | 0.6222 | 0.2161 | 0.656 | 22798 |
| XGBoost (calibrated) | 0.6494 | 0.2171 | 0.653 | 22798 |
| Base rate (home win %) | 0.6813 | 0.2441 | 0.577 | 22798 |

## Reading this
- **Logistic regression (Elo + rolling form/rest) is the best model** on log loss — the as-of features add real signal on top of Elo.
- **XGBoost does not beat logistic** out of the box: on a small, smooth feature set it overfits. Fancier != better (the plan's explicit lesson).
- **Calibration does not help** here because the raw models are already well-calibrated (see docs/calibration.png); isotonic worsens log loss at the probability extremes.
- Every model clears the base-rate baseline — but the honest headline is a *well-calibrated logistic model that beats a strong Elo baseline*, evaluated without leakage.
