"""Streamlit frontend for the NBA win-probability project.

Two tabs:
  - Predict: pick two teams -> calibrated home-win probability + the why.
  - How it works: the honest results story (baselines table, calibration, SHAP,
    betting backtest).

Run:  PYTHONPATH=src uv run streamlit run app.py
"""

from __future__ import annotations

from pathlib import Path

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
        p_home = res["p_home"]

        st.subheader(f"{home_name} (home) vs {away_name}")
        st.metric(f"P({home_name} win)", f"{p_home:.1%}")
        st.progress(p_home)
        st.caption(f"P({away_name} win): {res['p_away']:.1%}")

        h, a = res["home"], res["away"]
        st.markdown("**Why** (the features driving this):")
        st.dataframe(
            {
                "": ["Elo rating", "Form (last-10 pt diff)", "Season win %"],
                home_name: [f"{h.elo:.0f}", f"{h.roll_pdiff_10:+.1f}", f"{h.win_pct_std:.0%}"],
                away_name: [f"{a.elo:.0f}", f"{a.roll_pdiff_10:+.1f}", f"{a.win_pct_std:.0%}"],
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
    st.markdown(
        "Team strength (Elo) dominates, then recent form, then rest. No single "
        "feature dominates — a sign there's no leakage."
    )
    shap_img = DOCS / "shap_summary.png"
    if shap_img.exists():
        st.image(str(shap_img))

    st.header("Betting backtest vs the market")
    st.markdown(_read(DOCS / "betting.md"))
