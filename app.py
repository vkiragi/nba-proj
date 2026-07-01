"""Streamlit frontend for the NBA win-probability project.

Two tabs:
  - Predict: pick two teams -> calibrated home-win probability + the why.
  - How it works: the honest results story (baselines table, calibration, SHAP,
    betting backtest).

Run:  PYTHONPATH=src uv run streamlit run app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make `nba_pred` importable without PYTHONPATH=src (Streamlit Cloud runs
# `streamlit run app.py` from the repo root and won't set it). Local runs with
# PYTHONPATH=src still work — this insert is harmless when src is already found.
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import pandas as pd
import streamlit as st

from nba_pred.serve import Predictor

REPO = Path(__file__).resolve().parent
DOCS = REPO / "docs"

st.set_page_config(page_title="NBA Win Probability", page_icon="🏀", layout="centered")


@st.cache_resource
def load_predictor() -> Predictor:
    return Predictor()


st.title("🏀 NBA Win-Probability Predictor")
st.caption(
    "A leakage-safe, calibrated model that predicts home-win probability — "
    "and is honestly benchmarked against Elo and the betting market."
)

try:
    predictor = load_predictor()
except FileNotFoundError:
    st.error(
        "No data found. Build it first:\n\n"
        "`PYTHONPATH=src uv run python -m nba_pred.ingest.games --all`"
    )
    st.stop()

st.info(
    f"**Heads up:** predictions use team strengths frozen at the end of the data — "
    f"**{predictor.last_date.date()} ({predictor.latest_season} regular season)**. "
    "This estimates a hypothetical matchup at those strengths; it is **not** a live "
    "forecast of a scheduled future game and doesn't know about later trades, the "
    "draft, or injuries. Re-run ingestion to refresh.",
    icon="📅",
)

predict_tab, story_tab = st.tabs(["🔮 Predict a matchup", "📊 How it works & results"])

with predict_tab:
    teams = predictor.current_teams()
    names = [n for _, n in teams]
    id_by_name = {n: i for i, n in teams}

    col_home, col_away = st.columns(2)
    with col_home:
        home_name = st.selectbox("Home team", names, index=names.index("Boston Celtics") if "Boston Celtics" in names else 0)
    with col_away:
        away_name = st.selectbox("Away team", names, index=names.index("Denver Nuggets") if "Denver Nuggets" in names else 1)

    with st.expander("Rest / schedule (optional)"):
        c1, c2 = st.columns(2)
        home_rest = c1.slider("Home days rest", 0, 5, 2)
        away_rest = c2.slider("Away days rest", 0, 5, 2)

    if home_name == away_name:
        st.warning("Pick two different teams.")
    else:
        res = predictor.predict(
            id_by_name[home_name], id_by_name[away_name],
            home_rest_days=home_rest, away_rest_days=away_rest,
        )
        models = res["models"]
        ensemble = res["ensemble"]

        st.subheader(f"{home_name} (home) vs {away_name}")

        # Headline: the ensemble (average of all models).
        st.metric(f"Ensemble — P({home_name} win)", f"{ensemble:.1%}")
        st.progress(ensemble)
        st.caption(f"P({away_name} win): {1 - ensemble:.1%}  ·  average of the models below")

        # One prediction per model.
        st.markdown("**Each model's prediction** (P(home win)):")
        cols = st.columns(len(models) + 1)
        for col, (name, p) in zip(cols, models.items()):
            col.metric(name, f"{p:.1%}")
        cols[-1].metric("Ensemble", f"{ensemble:.1%}")
        st.caption(
            "Elo = team strength only · Logistic = best single model (Elo + form/rest) · "
            "XGBoost = gradient boosting · Ensemble = their average. "
            "They mostly agree — Logistic is the most accurate in backtests."
        )

        h, a = res["home"], res["away"]
        st.markdown("**Why** (the features driving this):")
        st.dataframe(
            {
                "": ["Elo rating", "Form (last-10 pt diff)", "Season win %", "Roster strength (recent +/-)"],
                home_name: [f"{h.elo:.0f}", f"{h.roll_pdiff_10:+.1f}", f"{h.win_pct_std:.0%}", f"{res['home_roster']:+.1f}"],
                away_name: [f"{a.elo:.0f}", f"{a.roll_pdiff_10:+.1f}", f"{a.win_pct_std:.0%}", f"{res['away_roster']:+.1f}"],
            },
            hide_index=True,
            width="stretch",
        )
        st.caption(
            f"Based on data through {predictor.last_date.date()} "
            f"({predictor.latest_season}). Home-court advantage is built in."
        )

with story_tab:
    st.markdown(
        "#### An NBA win-probability model, built to be *trusted* — not to look flashy."
    )
    st.markdown(
        "Most hobby sports predictors quietly leak future information and report a "
        "fake accuracy number. This one is built the opposite way: every result "
        "below is measured **walk-forward** (train on the past, test on the future "
        "it has never seen) with an automated test guarding against data leakage. "
        "The goal is a probability you can believe, benchmarked honestly against a "
        "strong baseline and the betting market."
    )

    # --- Headline numbers, up front -----------------------------------------
    m1, m2, m3 = st.columns(3)
    m1.metric("Best model — log loss", "0.607", help="Lower is better. The primary metric (a proper scoring rule), not accuracy.")
    m2.metric("Accuracy", "66.8%", help="Secondary metric. Sanity check, not the headline — accuracy is easy to game.")
    m3.metric("Held-out games", "22,798", help="Every prediction is on a season the model was never trained on (2007-08 → 2025-26).")
    st.caption(
        "Evaluated over 19 held-out seasons. Beats the base-rate baseline in "
        "**19 of 19** seasons."
    )

    st.divider()

    # --- Results table (built natively for a cleaner look) ------------------
    st.subheader("How the models stack up")
    st.markdown(
        "Every model runs through the *same* walk-forward harness, so the "
        "comparison is apples-to-apples. **Lower log loss is better.**"
    )
    results = pd.DataFrame(
        [
            ["Logistic regression (Elo + form/rest/roster)", 0.6066, 0.668, "Best — as-of features add real signal on top of Elo"],
            ["Elo (margin-of-victory)", 0.6104, 0.664, "Strong baseline; a single strength number per team"],
            ["XGBoost", 0.6195, 0.661, "Overfits a small, smooth feature set — fancier isn't better"],
            ["Base rate (always pick home)", 0.6813, 0.577, "The floor every model must clear"],
        ],
        columns=["Model", "Log loss", "Accuracy", "What it shows"],
    )
    st.dataframe(
        results,
        hide_index=True,
        width="stretch",
        column_config={
            "Log loss": st.column_config.NumberColumn(format="%.4f"),
            "Accuracy": st.column_config.NumberColumn(format="%.1f%%"),
        },
    )
    st.info(
        "**The honest headline:** a well-calibrated logistic model that beats a "
        "strong Elo baseline — evaluated without leakage. Not a 90%-accuracy claim "
        "(those are almost always a leak).",
        icon="✅",
    )

    st.divider()

    # --- Calibration --------------------------------------------------------
    st.subheader("Does “70%” actually mean 70%?")
    st.markdown(
        "A probability is only useful if it's **calibrated** — when the model says "
        "70%, the home team should win about 70% of the time. Plotting predicted vs. "
        "actual, a perfectly calibrated model sits on the diagonal. Ours already does, "
        "so extra calibration steps didn't help (a *finding*, not a failure)."
    )
    cal = DOCS / "calibration.png"
    if cal.exists():
        st.image(str(cal), caption="Reliability curve — close to the diagonal is good.")

    st.divider()

    # --- SHAP ---------------------------------------------------------------
    st.subheader("What actually drives a prediction?")
    st.markdown(
        "**SHAP** (from game theory's Shapley values) opens the black box: it splits "
        "each prediction into how much every feature pushed it up or down. Team "
        "strength (Elo) leads, then recent form, then rest."
    )
    st.markdown(
        "This doubles as a **leakage check**: the top feature holds only ~24% of the "
        "total importance. If one feature dominated (80%+), that would scream a leak. "
        "It doesn't — a healthy sign."
    )
    shap_img = DOCS / "shap_summary.png"
    if shap_img.exists():
        st.image(str(shap_img), caption="Feature importance — no single feature dominates.")

    st.divider()

    # --- Betting ------------------------------------------------------------
    st.subheader("Can it beat the betting market?")
    st.markdown(
        "The toughest benchmark there is. The **vig** is the sportsbook's built-in "
        "commission — at -110 odds you must win ~**52.4%** of bets (not 50%) just to "
        "break even. That's the edge you have to beat *before* you make a cent."
    )

    b1, b2 = st.columns(2)
    b1.metric("Our model — log loss", "0.5965")
    b2.metric("The market — log loss", "0.5799", delta="market wins", delta_color="inverse")
    st.markdown(
        "**The market wins — and that's the right answer.** Flat-stake ROI is "
        "negative across almost every edge threshold: no exploitable edge after the "
        "vig. An efficient market is *supposed* to be unbeatable by a simple model. "
        "Claiming otherwise would be the red flag."
    )

    with st.expander("🐛 The best debugging story in this project: a fake +6.3% ROI"):
        st.markdown(
            "The first betting backtest showed a **+6.3% ROI** at high edge "
            "thresholds — which should set off alarms, not celebration (beating the "
            "market usually means you're leaking or buggy until proven otherwise).\n\n"
            "The root cause: I aggregated multiple sportsbooks by taking the "
            "**median of the American odds**. American odds are *discontinuous* "
            "around ±100 — they jump from +100 to -100 with an impossible gap "
            "between — so `median(-116, +100) = -8`, a fabricated line implying a "
            "~100× payout that faked the profit.\n\n"
            "**Fix:** never average American odds. Convert to probabilities "
            "(continuous), average there, then convert back. The fake edge vanished "
            "— leaving the honest, efficient-market result above."
        )

    with st.expander("Caveats (stated up front, on purpose)"):
        st.markdown(
            "- Odds are a multi-book **consensus**, not timestamped closing lines — "
            "so this measures edge-vs-consensus, not true closing-line value.\n"
            "- Betting data covers **2006–2018** only.\n"
            "- Transaction costs and line-shopping are idealized.\n"
            "- Predictions use team strengths **frozen** at the end of the data; "
            "this is a hypothetical-matchup demo, not a live forecast."
        )
