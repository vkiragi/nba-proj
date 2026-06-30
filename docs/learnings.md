# Learnings & decisions log

A running record of non-obvious things discovered while building this project —
the stuff worth talking about in interviews and worth not re-learning the hard
way. Newest sections at the bottom.

---

## Phase 1 — Data ingestion

### The API returns one row *per team per game*, not per game
`LeagueGameLog` gives 2,460 rows for a 1,230-game season — each game appears
twice, once from each team's perspective. The collapse: split on the `MATCHUP`
string (`"DEN vs. LAL"` = home, `"LAL @ DEN"` = away), then merge the two halves
on `GAME_ID`. The home-win label is just `home_pts > away_pts`.

### Three real-world data hazards that silently break naive notebooks
Found by sanity-checking counts season by season instead of trusting the data:

1. **Canceled games logged as 0-0.** The 2013-04-16 Celtics-Pacers game was
   canceled after the Boston Marathon bombing and never made up (the only one in
   NBA history). It sits in the data as 0-0. Fix: drop games with equal scores —
   a completed NBA game can't tie.
2. **Neutral-site games (~5/recent season).** NBA Cup finals (Las Vegas) and
   international games (Mexico City, Paris, Abu Dhabi) mark *both* teams with
   `@`, so neither is flagged home and an inner-join silently drops them. There's
   no real home court, so we drop them — but *explicitly, with a reported count*,
   never silently. `merge(validate="1:1")` is the guard that makes silent
   drop/dup impossible.
3. **Flaky API.** `stats.nba.com` times out intermittently (hit it on 2020-21).
   Fix: retry with exponential backoff + per-season parquet caching so a
   mid-backfill failure resumes instantly.

### Disrupted seasons must be flagged, not silently mixed in
- **2011-12** lockout — 66-game season (990 games)
- **2019-20** COVID bubble — *no home court at all*
- **2020-21** COVID — 72-game season, limited fans, compressed schedule

These distort rest/back-to-back/home-court features. Flagged via the `season`
column so downstream code can exclude or special-case them.

### Sanity signal that the data is clean
Overall home win rate = **58.1%**, right in the historical 58-60% band. A
too-high number here would have signaled a collapse/merge bug.

---

## Phase 2/3 — Elo (first feature AND first model)

### The leakage rule is an *ordering* rule
For each game, in strict chronological order: **(1) read** pre-game ratings →
**(2) record** them as the game's features → **(3) update** from the result.
Updating before recording would let the feature "know" the outcome. This is why
Elo must be computed one game at a time in date order — never vectorized, never
reordered.

### The textbook home-court constant is wrong for modern NBA
Everyone quotes `home_adv = 100` Elo points (FiveThirtyEight's historical
value). A sweep on 2022-25 found **~50 minimizes log loss**; 100 was actually
*worse than no home-court adjustment at all*:

| home_adv | log loss (last 3 seasons) |
|---|---|
| 0 (vanilla) | 0.6362 |
| **50** | **0.6288** ← best |
| 75 | 0.6322 |
| 100 (textbook) | 0.6402 |
| 125 | 0.6525 |

Home-court advantage has shrunk in the modern game (travel, sports science,
weaker crowd effect, COVID-era empty arenas). **Lesson: measure, don't assume.**

### Apply the home bonus consistently to prediction *and* update
The bonus lives inside `expected_home`, and the rating update calls that same
function — so home teams aren't over-rewarded for winning games they were
already expected to win at home. The bonus is never stored on a team's rating.

### A Python gotcha that produced a fake experiment result
`def expected_home(..., home_adv=HOME_ADV)` binds the default at *definition*
time. Reassigning the module-level `HOME_ADV` afterward did nothing, so an early
sweep showed identical results for every value. Fix: thread `home_adv` as an
explicit parameter through `compute_elo`. (Makes it properly tunable too.)

---

## Honest status of testing & results (as of Phase 3)

**Two different meanings of "tests" — don't conflate them:**

1. **Regression tests (`pytest`):** 9 passing, 1 skipped. `tests/test_elo.py`
   covers the formula, the hand-checked 1510/1490 update, the zero-sum property,
   and the no-future-leakage property. These guard against *code* breakage.
2. **Model evaluation (the home_adv sweep):** not pytest — a one-off scoring
   script. Proves the *modeling* result, asserts nothing.

**Results so far (Elo, home_adv=50, last 3 seasons):**
log loss **0.6288**, Brier 0.2196, accuracy **64.4%** — beats the base-rate
(0.6869) and coin-flip (0.6931) baselines on log loss.

**Caveats (important, stated up front):**
- The 0.6288 is **mildly optimistic**: `home_adv=50` was tuned on the *same*
  recent seasons it's reported on (tuning-on-test peek). Phase 4 walk-forward
  will tune on a validation period and report on a later untouched one.
- We have **not** yet beaten the full baseline set required by the plan (Elo is
  itself one of those baselines), done calibration, or the betting backtest. So:
  **promising, honest, leakage-safe results — but not yet "done" by the project's
  own success criteria.**

---

## Phase 4 — Walk-forward evaluation (the honest backbone)

### The harness is model-agnostic on purpose
`eval/backtest.py` runs any `predict_fn(train_games, test_games) -> p_home`
through the same rolling loop: train on seasons 1..k, test on k+1, advance. Elo,
logistic regression, and XGBoost all plug into the identical harness, so model
comparisons are apples-to-apples. A test asserts the harness never hands a model
a future season.

### Elo, evaluated honestly (no tuning-on-test peek)
Per-season walk-forward, 19 held-out seasons (2007-08 → 2025-26):
- **Beats the base-rate baseline on log loss in 19/19 seasons.**
- **Aggregate held-out: log loss 0.6165, Brier 0.2140, accuracy 65.8%** over
  22,798 games.

### The game is getting harder to predict
Log loss drifts UP over time: ~0.59 (2008-09) → ~0.64-0.65 (2020-23). Modern NBA
is less predictable (parity, load management, 3-pt variance). Worst seasons:
2020-21 (COVID, 0.657) and 2022-23 (0.653). A real, defensible insight.

### Online learners don't have a separate "train" step
Elo learns as it walks games in order, so its walk-forward "training" is just
running the online updates through prior games — `elo_predict` runs `compute_elo`
over train+test together and reads the ratings each test game goes in with
(never its own result). Logistic/XGBoost will use a real `fit`/`predict` split
through the same harness.

---

## Phase 2/3 — Features, models, calibration, SHAP

### The as-of join, and a real leakage bug I caught in my own code
Rest/rolling features are per-team-per-game, but the spine is per-game. The
transform: explode to per-team-long → compute features with
`groupby(["team_id","season"]).shift(1)` → join back as home_/away_ pairs.
**Bug found:** `groupby(...)["x"].shift(1).rolling(N)` — the `shift(1)` returns
an *ungrouped* Series, so the `.rolling()` bled across team boundaries (the first
game of a team averaged in the previous team's games). Fix: wrap shift+roll in a
single grouped `.transform(...)`. The empirical leakage test (truncate-and-rebuild)
plus per-feature unit tests catch exactly this class of bug.

### Leakage traps that bit or nearly bit
- `rolling("7D")` defaults to `closed="right"` → includes the current game.
  Must be `closed="left"`.
- `expanding().mean()` for win pct without `shift(1)` → includes the current
  result = direct label leakage.
- `games_last_7` empty window sums to NaN; a *count* of prior games is genuinely
  0 (not "unknown"), so fill it — distinct from rest_days, which stays NaN.

### Honest model results (walk-forward, 22,798 held-out games)
| Model | Log loss | Accuracy |
|---|---|---|
| Base rate (home %) | 0.6813 | 57.7% |
| Elo (home_adv=50) | 0.6165 | 65.8% |
| **Logistic (Elo + form/rest)** | **0.6098** | **66.4%** |
| XGBoost | 0.6222 | 65.6% |
| XGBoost (calibrated) | 0.6494 | 65.3% |

- **Logistic wins.** The as-of form/rest features add real signal on top of Elo.
- **XGBoost loses to logistic out of the box** — on a small, smooth feature set
  it overfits. Fancier != better (exactly the plan's warning).
- **Calibration didn't help**: the raw models are already well-calibrated
  (reliability curves hug the diagonal — `docs/calibration.png`); isotonic
  worsens log loss at the probability extremes. The *finding* (already
  calibrated) is the deliverable, not a fancier number.

### SHAP doubles as a leakage check
Feature importance (mean |SHAP|): home_elo & away_elo dominate (team strength),
then rolling point-diff (form), then win%, then rest; back-to-back/games-in-7
matter least. Top feature holds only **24.7%** of total importance — no single
feature dominates, which is the no-leakage signal we want (a feature at 80%+ would
scream leakage). Directions are sensible too: high home_elo pushes toward a home
win. See `docs/shap_summary.png`.
