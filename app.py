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

import streamlit as st

from nba_pred.serve import Predictor

REPO = Path(__file__).resolve().parent
DOCS = REPO / "docs"

st.set_page_config(page_title="NBA Win Probability", page_icon="🏀", layout="centered")


@st.cache_resource
def load_predictor() -> Predictor:
    return Predictor()


def _read(path: Path) -> str:
    return path.read_text() if path.exists() else f"_(missing {path.name})_"


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
        "This project's selling point is **honesty**, not a flashy accuracy number. "
        "Everything below is evaluated walk-forward (train on the past, test on the "
        "future) with no data leakage."
    )

    st.header("Beat-the-baselines")
    st.markdown(_read(DOCS / "results.md"))

    st.header("Calibration")
    st.markdown(
        "Does “70%” actually mean 70%? A reliability curve on the diagonal = yes. "
        "Our raw models are already well-calibrated."
    )
    cal = DOCS / "calibration.png"
    if cal.exists():
        st.image(str(cal))

    st.header("What drives predictions (SHAP)")
    st.caption(
        "**SHAP** (from game theory's Shapley values) splits a prediction into how "
        "much each feature pushed it up or down — opening the model's black box."
    )
    st.markdown(
        "Team strength (Elo) dominates, then recent form, then rest. No single "
        "feature dominates — a sign there's no leakage."
    )
    shap_img = DOCS / "shap_summary.png"
    if shap_img.exists():
        st.image(str(shap_img))

    st.header("Betting backtest vs the market")
    st.caption(
        "**Vig** = the sportsbook's built-in commission. At -110 odds you must win "
        "~52.4% (not 50%) of bets just to break even — the house edge you have to beat first."
    )
    st.markdown(_read(DOCS / "betting.md"))
