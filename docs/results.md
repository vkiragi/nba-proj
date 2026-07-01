# Results — beat-the-baselines table

Walk-forward (train seasons 1..k, test k+1), aggregated over all held-out games (2007-08 → 2025-26). **Lower log loss is better; it is the primary metric.**

| Model | Log loss | Brier | Accuracy | n |
|---|---|---|---|---|
| Logistic (Elo + form/rest/roster) | 0.6066 | 0.2097 | 0.668 | 22798 |
| Elo (MOV, home_adv=60) | 0.6104 | 0.2113 | 0.664 | 22798 |
| XGBoost | 0.6195 | 0.2146 | 0.661 | 22798 |
| XGBoost (calibrated) | 0.6493 | 0.2157 | 0.657 | 22798 |
| Base rate (home win %) | 0.6813 | 0.2441 | 0.577 | 22798 |

## Reading this
- **Logistic regression (Elo + rolling form/rest) is the best model** on log loss — the as-of features add real signal on top of Elo.
- **XGBoost does not beat logistic** out of the box: on a small, smooth feature set it overfits. Fancier != better (the plan's explicit lesson).
- **Calibration does not help** here because the raw models are already well-calibrated (see docs/calibration.png); isotonic worsens log loss at the probability extremes.
- Every model clears the base-rate baseline — but the honest headline is a *well-calibrated logistic model that beats a strong Elo baseline*, evaluated without leakage.
