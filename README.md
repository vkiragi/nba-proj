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

## Status

Phase 0 (scaffold) complete. Next: Phase 1 — land one season of games in parquet.
