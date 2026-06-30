"""Live matchup predictor — turns the trained model into a "team A vs team B" call.

Serving an as-of model requires reconstructing each team's CURRENT feature state
(the values that would attach to their next game): current Elo, recent rolling
point-diff form, and season-to-date win %. Rest/schedule features aren't known
for a hypothetical game, so they default to typical values (caller can override).

The model is the project's best: logistic regression on Elo + form/rest,
fit on ALL games.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from nba_pred.features.build import explode_to_team_long
from nba_pred.features.elo import START_RATING, final_ratings
from nba_pred.models.features import MODEL_FEATURES, build_model_frame
from nba_pred.models.logistic import fit_serving_model

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_GAMES = REPO_ROOT / "data" / "processed" / "games.parquet"

# Typical defaults for the unknown schedule features of a hypothetical game.
DEFAULT_REST_DAYS = 2
DEFAULT_GAMES_LAST_7 = 3.0


@dataclass
class TeamState:
    team_id: int
    elo: float
    roll_pdiff_5: float
    roll_pdiff_10: float
    win_pct_std: float
    season: str
    last_game_date: pd.Timestamp


def team_lookup() -> dict[int, dict]:
    """team_id -> {full_name, abbreviation} via nba_api's static team list."""
    from nba_api.stats.static import teams

    return {t["id"]: t for t in teams.get_teams()}


def compute_team_states(games: pd.DataFrame) -> dict[int, TeamState]:
    """Each team's current feature state, from their most recent games."""
    ratings = final_ratings(games)
    tl = explode_to_team_long(games)

    states: dict[int, TeamState] = {}
    for tid, g in tl.groupby("team_id"):
        g = g.sort_values("game_date")
        cur_season = g["season"].iloc[-1]
        states[int(tid)] = TeamState(
            team_id=int(tid),
            elo=ratings.get(int(tid), START_RATING),
            roll_pdiff_5=float(g["point_diff"].tail(5).mean()),
            roll_pdiff_10=float(g["point_diff"].tail(10).mean()),
            win_pct_std=float(g[g["season"] == cur_season]["won"].mean()),
            season=cur_season,
            last_game_date=g["game_date"].max(),
        )
    return states


class Predictor:
    """Loads data, fits the serving model, and predicts matchups."""

    def __init__(self, games_path: Path = DEFAULT_GAMES):
        games = pd.read_parquet(games_path)
        self.last_date = games["game_date"].max()
        self.latest_season = sorted(games["season"].unique())[-1]
        self.states = compute_team_states(games)
        self.teams = team_lookup()
        self.model = fit_serving_model(build_model_frame(games))

    def current_teams(self) -> list[tuple[int, str]]:
        """(team_id, name) for teams active in the latest season, sorted by name."""
        active = [s.team_id for s in self.states.values() if s.season == self.latest_season]
        named = [(tid, self.teams.get(tid, {}).get("full_name", str(tid))) for tid in active]
        return sorted(named, key=lambda x: x[1])

    def predict(
        self,
        home_id: int,
        away_id: int,
        home_rest_days: int = DEFAULT_REST_DAYS,
        away_rest_days: int = DEFAULT_REST_DAYS,
    ) -> dict:
        """Return P(home win) and the feature context behind it."""
        h, a = self.states[home_id], self.states[away_id]
        row = {
            "home_elo": h.elo,
            "away_elo": a.elo,
            "home_rest_days": home_rest_days,
            "home_is_b2b": float(home_rest_days == 1),
            "home_games_last_7": DEFAULT_GAMES_LAST_7,
            "home_roll_pdiff_5": h.roll_pdiff_5,
            "home_roll_pdiff_10": h.roll_pdiff_10,
            "home_win_pct_std": h.win_pct_std,
            "away_rest_days": away_rest_days,
            "away_is_b2b": float(away_rest_days == 1),
            "away_games_last_7": DEFAULT_GAMES_LAST_7,
            "away_roll_pdiff_5": a.roll_pdiff_5,
            "away_roll_pdiff_10": a.roll_pdiff_10,
            "away_win_pct_std": a.win_pct_std,
        }
        X = pd.DataFrame([row])[MODEL_FEATURES]
        p_home = float(self.model.predict_proba(X)[0, 1])
        return {"p_home": p_home, "p_away": 1.0 - p_home, "home": h, "away": a}
