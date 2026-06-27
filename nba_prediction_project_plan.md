# NBA Game Outcome Prediction — Project Plan

**What this project actually is:** an end-to-end ML system that predicts win probabilities for NBA games, validated honestly against time, calibrated, and deployed as a live service — benchmarked against the Vegas closing line.

**What it is NOT:** a money-printing betting bot. The realistic outcome is that you *match* the market and learn to prove it rigorously. That honesty is the selling point in interviews, not a weakness. Anyone claiming a portfolio XGBoost model reliably beats NBA closing lines is either leaking data or lying, and a sharp interviewer knows it.

**Why it's good MLE signal:** it forces you to do the four things AI-API projects never make you do — train a model, validate it under realistic constraints, calibrate probabilities, and ship + monitor it in production.

---

## The two things that make or break this

Read this section twice. Everything below serves these two points.

### 1. Data leakage is the whole game
Almost every public NBA-prediction notebook is silently broken because it lets the model see information it wouldn't have had at tip-off. If your accuracy looks great (70%+), assume leakage until proven otherwise. The single most impressive thing you can demonstrate is that you *avoided* it. Concretely:

- **Time-based splits only.** Never random k-fold across games. Train on past seasons, test on future ones. Use walk-forward (expanding-window) validation.
- **As-of features only.** Every feature must be computable strictly *before* tip-off. Rolling stats must exclude the game being predicted (shift by one). Season-to-date averages must not include the current game.
- **The injury trap.** Final box scores tell you who actually played. Using that is leakage. Official inactives post ~30 min before tip — decide your information cutoff and document it. Simplest honest v1: ignore lineups, note it as a limitation.
- **The odds trap.** Don't train on the closing line and then claim to beat the closing line — that's circular. If odds are a feature, be explicit about which line (open vs close) and when you'd actually have it.

### 2. Accuracy is the wrong headline metric
Use **proper scoring rules** for probabilities: **log loss** and **Brier score**, plus a **calibration curve**. Accuracy is secondary. A model that says "62%" and is right 62% of the time is more valuable than one that's overconfident at higher accuracy. Calibration is also what makes the betting comparison meaningful.

---

## Success criteria (define "done" before you start)

You're done when you can honestly say all of these:

1. Walk-forward backtest across ≥3 held-out seasons with log loss + Brier + calibration curve.
2. Your model beats these baselines on log loss: always-pick-home, pick-higher-Elo, and a plain logistic regression. (Beating Elo is a real bar; Elo alone is shockingly strong.)
3. An honest betting backtest vs historical closing lines, accounting for the vig, reporting ROI and closing-line value (CLV) — even if the answer is "no edge."
4. A deployed FastAPI service that produces today's predictions, Dockerized, with model versioning.
5. A README/blog writeup that states your leakage precautions, baselines, results, and limitations plainly.

---

## Phase 0 — Scope & setup (½ week)

- Pick the target: **binary home-win classification** for v1 (simplest, cleanest). Margin regression is a good v2.
- Decide metrics now: log loss (primary), Brier, calibration curve, accuracy (secondary), and the betting metrics for the backtest.
- Write down your baselines (see success criteria #2). You will compare against these constantly.
- Repo hygiene from day one: git, a clean `src/` layout, `requirements.txt`/`uv`, a `data/` dir that's gitignored, and a notebook-vs-module discipline (exploration in notebooks, reusable logic in modules). This itself is SWE signal.

**Baselines worth knowing:** home teams win ~58-60% of NBA games historically. The Vegas closing line implied probability is the hardest baseline — treat "can't beat closing" as the expected, respectable result.

---

## Phase 1 — Data acquisition (1 week)

- **Game/box-score data:** `nba_api` (wraps stats.nba.com) is the standard. Basketball-Reference is great but scrape respectfully (rate-limit, cache, check ToS). Kaggle has prebuilt historical NBA game CSVs to bootstrap.
- **Odds data (the bottleneck — plan for it):** historical odds are the hard part. Sources: sportsbookreviewsonline historical spreadsheets, OddsPortal, or a paid feed. For going-forward live odds, the-odds-api has a free tier. Be honest in the writeup about odds coverage gaps.
- **Build a reproducible ingestion layer:** scripts that pull raw data → store locally (parquet) → are idempotent and cached. Don't hammer APIs in your training loop.
- **Sanity-check everything:** game counts per season (82 × 30 / 2 ≈ 1230 regular-season games), date ranges, missing values, duplicate games. Garbage data is the #2 cause of broken projects after leakage.

---

## Phase 2 — Feature engineering (1–1.5 weeks)

This is the engineering meat. The core challenge is the **as-of join**: for each game, attach only features known before tip-off.

Start simple, add complexity only if it helps log loss:

- **Team strength:** Elo rating (implement it yourself — great interview material) and/or rolling net rating.
- **Rolling form:** offensive/defensive rating, pace, point differential over last N games (5, 10) — computed as-of the date, shifted to exclude the current game.
- **Schedule/rest:** days of rest, back-to-back flag, games in last 7 days, travel distance (optional, fun).
- **Context:** home/away, season-to-date record (as-of), head-to-head history.
- **(Advanced, optional v2):** availability-adjusted ratings using known inactives.

**Write a test** that confirms no feature for game G uses any data dated ≥ G's tip-off. This test is your leakage insurance and a great thing to mention in interviews.

---

## Phase 3 — Modeling (1 week)

Go in this order — each step is a baseline for the next:

1. **Elo-only** prediction. Surprisingly hard to beat.
2. **Logistic regression** on your features. Interpretable, fast, honest baseline.
3. **XGBoost / LightGBM.** The "real" model. Tune with **time-aware CV** (no shuffling).
4. **Probability calibration** — this is non-negotiable. Raw gradient-boosting probabilities are poorly calibrated. Apply Platt scaling or isotonic regression and show the before/after calibration curve. Calibration is what makes #4 and the betting backtest valid.
5. **SHAP** for explainability. Bonus: SHAP catches leakage — if one feature dominates suspiciously, investigate it.

Don't chase a leaderboard. A calibrated model that ties Elo on log loss, deployed and explained, beats an "85% accuracy" model that's secretly leaking.

---

## Phase 4 — Honest evaluation + betting backtest (1 week)

- **Walk-forward backtest:** train on seasons 1..k, test on k+1, roll forward. Report log loss, Brier, calibration, accuracy per season.
- **Beat-the-baselines table:** your model vs home-pick, Elo, logistic. This table is the heart of the writeup.
- **Betting backtest (the domain flex):** simulate flat-stake (and optionally fractional-Kelly) betting against historical closing lines. Account for the vig — at -110, breakeven is ~52.4%, so you must clear that to profit. Report ROI and **CLV** (did you get better numbers than the closing line?).
- **Be brutally honest about the result.** "Model is well-calibrated but does not beat the closing line after vig" is a *strong* conclusion that signals you understand efficient markets. Overclaiming a profitable system is the single fastest way to lose credibility with anyone who knows the space.

---

## Phase 5 — Deployment & MLOps (1 week) — the MLE differentiator

This is what separates you from notebook-only candidates.

- **FastAPI** inference service: endpoint that takes a game (or pulls today's slate) and returns calibrated win probabilities.
- **Dockerize** it; deploy to something cheap (Fly.io — you already use it, Railway, or a small AWS box).
- **Model versioning** with MLflow (track experiments, log metrics, register the chosen model).
- **Daily batch job:** pull today's games → build features → score → store predictions.
- **Monitoring:** track live calibration over the season — are predictions still calibrated, or is the model drifting? This is the highest-signal piece almost no one does.
- **Dashboard:** Streamlit (fast) or React (you already know it) showing today's predictions, calibration curve, and backtest performance.

---

## Phase 6 — Writeup & live tracking (ongoing)

- **README + blog post** with the baselines table, calibration plots, leakage precautions, and limitations stated plainly. This is what recruiters actually read.
- **Track live predictions** publicly for the current season. A timestamped, honest live record is far more credible than any backtest.

---

## Leakage checklist (pin this above your desk)

- [ ] No random k-fold; time-based splits only.
- [ ] Rolling/season features exclude the current game (shifted).
- [ ] No final-box-score "who played" used as a pre-game feature.
- [ ] No closing-odds feature if claiming to beat the closing line.
- [ ] Automated test: no feature for game G uses data dated ≥ G's tip-off.
- [ ] Accuracy that looks "too good" (70%+) investigated, not celebrated.

---

## Tech stack summary

`Python` · `nba_api` · `pandas`/`polars` · `scikit-learn` · `XGBoost`/`LightGBM` · `SHAP` · `MLflow` · `FastAPI` · `Docker` · `Streamlit`/`React` · deploy on `Fly.io`/`Railway`/`AWS`

---

## The interview stories this generates

You're building this to *talk about it*. By the end you should be able to tell:

1. **"How I avoided data leakage in a temporal prediction problem"** — the single best ML-rigor story you can have.
2. **"Why I optimized log loss and calibration instead of accuracy"** — shows you understand proper scoring and why it matters downstream.
3. **"How I validated against an efficient market and what I concluded"** — domain maturity + intellectual honesty.
4. **"How I deployed and monitored it for drift over a live season"** — the MLOps story most candidates lack.

That's four substantive stories from one project — exactly the "multiple interesting stories" goal instead of "I trained a model and got 92%."

---

## Honest expectations

- You will probably not beat the closing line. That's the correct, expected result and saying so is a strength.
- The hardest parts are odds-data coverage and the as-of join discipline, not the modeling.
- This is a portfolio/learning project. The betting comparison is an evaluation benchmark, not financial advice or a recommendation to bet.

---

## Defensible resume bullet (fill in your real numbers)

> Built an end-to-end NBA win-probability system: ingestion pipeline over [N] historical games, leakage-safe temporal feature engineering, and a calibrated XGBoost model evaluated by log loss/Brier against Elo and market baselines via walk-forward backtesting; deployed a Dockerized FastAPI inference service with MLflow model versioning and live calibration monitoring.

No invented accuracy figures — every claim above is something you can defend on a whiteboard.
