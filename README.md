# NBA Win-Probability Prediction

An end-to-end ML system that predicts win probabilities for NBA games — validated
honestly against time, calibrated, and (eventually) deployed as a live service,
benchmarked against the Vegas closing line.

See [`nba_prediction_project_plan.md`](./nba_prediction_project_plan.md) for the full plan.

> **The realistic outcome is that we *match* the market and prove it rigorously.**
> This is a portfolio/learning project. The betting comparison is an evaluation
> benchmark, not financial advice.

---

## ⚠️ Leakage checklist (the whole game)

Every feature must be computable strictly *before* tip-off. Check these constantly:

- [ ] No random k-fold — **time-based splits only** (train past, test future).
- [ ] Rolling / season-to-date features **exclude the current game** (shifted).
- [ ] No final-box-score "who played" used as a pre-game feature.
- [ ] No closing-odds feature if claiming to beat the closing line.
- [ ] Automated test: no feature for game G uses data dated ≥ G's tip-off (`tests/test_leakage.py`).
- [ ] Accuracy that looks "too good" (70%+) is **investigated, not celebrated**.

**Headline metrics are log loss + Brier + calibration curve — not accuracy.**

---

## Project layout

```
src/nba_pred/
  ingest/    nba_api -> parquet, idempotent + cached
  features/  elo, rolling form, rest — built via a leakage-safe as-of join
  models/    baselines, logistic, xgboost, probability calibration
  eval/      walk-forward backtest, log loss/brier, betting backtest (CLV)
notebooks/   exploration only (reusable logic lives in src/)
tests/       leakage test written first, on purpose
data/        gitignored: raw/ + processed/ parquet
```

## Setup

```bash
uv sync --extra dev      # create env (Python 3.12) + install deps
uv run pytest            # run tests
```

## Results (walk-forward, leakage-free)

Best model: **logistic regression on Elo + as-of form/rest features**, beating a
strong Elo baseline on log loss over 22,798 held-out games (2007-08 → 2025-26).
Full table: [`docs/results.md`](docs/results.md) · calibration:
[`docs/calibration.png`](docs/calibration.png) · explainability:
[`docs/shap_summary.png`](docs/shap_summary.png) · decisions log:
[`docs/learnings.md`](docs/learnings.md).

| Model | Log loss | Accuracy |
|---|---|---|
| Base rate | 0.6813 | 57.7% |
| Elo | 0.6165 | 65.8% |
| **Logistic** | **0.6098** | **66.4%** |
| XGBoost | 0.6222 | 65.6% |

Regenerate: `PYTHONPATH=src uv run python scripts/results_table.py`
(also `calibration_report.py`, `shap_report.py`).

## Betting backtest (vs the market)

Our calibrated probabilities vs historical moneyline odds (2006-2018): the
**market beats our model** (log loss 0.5799 vs 0.5983) and flat-stake ROI is
negative at every edge threshold — **no edge after the vig**, the correct
efficient-market result. Full report + caveats: [`docs/betting.md`](docs/betting.md).
Build: `PYTHONPATH=src uv run python -m nba_pred.ingest.odds && PYTHONPATH=src uv run python scripts/betting_report.py`
(needs the Kaggle odds CSVs in `data/raw/odds/`).

## Status

Done: ingestion, leakage-safe features, Elo/logistic/XGBoost through a
walk-forward harness, calibration, SHAP, and an honest betting backtest — all
leakage-tested (34 tests). **Next:** deployment (FastAPI inference service).
