"""Elo ratings — our first feature AND first prediction model.

Elo assigns each team a single strength number. Two ratings produce a win
probability; after each game the winner takes points from the loser, scaled by
how surprising the result was.

LEAKAGE RULE (the whole point): for each game we attach the ratings as they
stand BEFORE tip-off, then update only AFTER recording them. Ratings are
therefore computed strictly in chronological order, one game at a time — never
vectorized, never reordered. The home_elo / away_elo columns are the ratings
GOING INTO each game, so they are safe to use as pre-game features.
"""

from __future__ import annotations

import pandas as pd

START_RATING = 1500.0  # every team begins here
K = 20.0               # update speed — how far a rating moves per game
SCALE = 400.0          # standard Elo scale (400 pts ≈ 10x odds)
# Home-court bonus in Elo points. The textbook value is ~100, but a sweep on
# 2022-25 found ~50 minimizes log loss — modern home-court advantage has shrunk.
# CAVEAT: 50 was tuned on the same recent seasons used for reporting, so it is
# mildly optimistic. Phase 4 walk-forward will tune on a validation period and
# report on a later untouched one to remove that peek.
HOME_ADV = 50.0


def expected_home(elo_home: float, elo_away: float, home_adv: float = HOME_ADV) -> float:
    """Home team's expected win probability given both pre-game ratings.

    The home-court bonus is added to the home rating ONLY for this expectation
    (it is never stored on the team's rating). Because the rating update calls
    this same function, the bonus is applied consistently to both prediction and
    update — so home teams aren't over-rewarded for winning expected home games.

    Equal ratings -> > 0.5 now (home is favored by home_adv). This is also our
    first prediction model: feed it pre-game ratings and it forecasts the game.
    """
    return 1.0 / (1.0 + 10.0 ** ((elo_away - (elo_home + home_adv)) / SCALE))


def compute_elo(games: pd.DataFrame, k: float = K, home_adv: float = HOME_ADV) -> pd.DataFrame:
    """Attach pre-game home_elo / away_elo to each game, leakage-safe.

    `games` must have: game_date, home_team_id, away_team_id, home_win.
    Returns a copy with home_elo and away_elo columns added (ratings going IN).
    """
    games = games.sort_values("game_date").reset_index(drop=True)

    # Current rating for every team, defaulting to START_RATING the first time
    # we see a team. dict.get(team, START_RATING) handles "never seen before".
    ratings: dict[int, float] = {}

    home_elos: list[float] = []
    away_elos: list[float] = []

    for row in games.itertuples(index=False):
        home_id = row.home_team_id
        away_id = row.away_team_id

        # 1. READ pre-game ratings (only past info exists here).
        elo_home = ratings.get(home_id, START_RATING)
        elo_away = ratings.get(away_id, START_RATING)

        # 2. RECORD them as this game's features — BEFORE any update.
        home_elos.append(elo_home)
        away_elos.append(elo_away)

        # 3. UPDATE both ratings from the result (now we may use the outcome).
        e_home = expected_home(elo_home, elo_away, home_adv)
        actual_home = row.home_win
        change = k * (actual_home - e_home)  # zero-sum: away gets the negative
        ratings[home_id] = elo_home + change
        ratings[away_id] = elo_away - change

    games = games.copy()
    games["home_elo"] = home_elos
    games["away_elo"] = away_elos
    return games
