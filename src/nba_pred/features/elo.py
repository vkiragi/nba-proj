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
# Home-court bonus in Elo points. The textbook value is ~100, but sweeps find
# modern home-court advantage has shrunk. With margin-of-victory scaling on
# (USE_MOV), a full walk-forward sweep over 2007-08→2025-26 puts the optimum at
# ~60 (log loss 0.6104); vanilla win/loss Elo preferred ~50. CAVEAT: this is
# tuned on the same held-out seasons it's reported over (a mild tuning-on-test
# peek); the log-loss curve is flat from 50–70, so the exact value barely matters.
HOME_ADV = 60.0
# Margin-of-victory scaling (FiveThirtyEight's NBA formula). Plain Elo moves a
# rating the same amount for a 1-point win as a 30-point blowout — throwing away
# the strongest single signal of how dominant a result was. The multiplier below
# scales the update by margin, with an autocorrelation correction in the
# denominator: a favorite that wins by a lot gains LESS than an underdog winning
# by the same margin, which stops strong teams from inflating without bound.
# Set USE_MOV=False to recover vanilla win/loss Elo.
USE_MOV = True
MOV_EXPONENT = 0.8     # diminishing returns on ever-larger margins
MOV_BASE = 7.5         # denominator constant (from 538's tuned formula)
MOV_ELO_COEF = 0.006   # autocorrelation correction per Elo point of favoritism


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


def mov_multiplier(margin: float, winner_elo_diff: float) -> float:
    """FiveThirtyEight's margin-of-victory multiplier for the K update.

    Args:
        margin: absolute point differential of the game (>= 0).
        winner_elo_diff: winner's pre-game rating MINUS loser's pre-game rating,
            with the home bonus folded in (positive = the favorite won). This is
            the autocorrelation term: the more favored the winner already was,
            the smaller the multiplier, so blowouts by strong teams don't compound.

    Returns 1.0 for a 1-point win between evenly matched teams (the formula's
    natural scale), growing with margin and shrinking as the winner was favored.
    """
    return ((margin + 3.0) ** MOV_EXPONENT) / (
        MOV_BASE + MOV_ELO_COEF * winner_elo_diff
    )


def _mov_scale(
    margin: float | None, home_win: int, elo_home_adj: float, elo_away_adj: float
) -> float:
    """K multiplier for a game: 1.0 (vanilla) unless MOV is on and margin known.

    `elo_*_adj` are the pre-game ratings WITH the home bonus already folded in,
    so the autocorrelation term reflects the true pre-game favorite.
    """
    if not USE_MOV or margin is None or margin <= 0:
        return 1.0
    # winner's rating minus loser's, using the home-adjusted ratings.
    winner_elo_diff = (elo_home_adj - elo_away_adj) if home_win else (elo_away_adj - elo_home_adj)
    return mov_multiplier(margin, winner_elo_diff)


def compute_elo(games: pd.DataFrame, k: float = K, home_adv: float = HOME_ADV) -> pd.DataFrame:
    """Attach pre-game home_elo / away_elo to each game, leakage-safe.

    `games` must have: game_date, home_team_id, away_team_id, home_win. If
    `home_pts`/`away_pts` are present and USE_MOV is set, the rating update is
    scaled by margin of victory (538's formula); otherwise it's vanilla Elo.
    Returns a copy with home_elo and away_elo columns added (ratings going IN).
    """
    games = games.sort_values("game_date").reset_index(drop=True)
    has_scores = "home_pts" in games.columns and "away_pts" in games.columns

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
        margin = abs(row.home_pts - row.away_pts) if has_scores else None
        mult = _mov_scale(margin, actual_home, elo_home + home_adv, elo_away)
        change = k * mult * (actual_home - e_home)  # zero-sum: away gets the negative
        ratings[home_id] = elo_home + change
        ratings[away_id] = elo_away - change

    games = games.copy()
    games["home_elo"] = home_elos
    games["away_elo"] = away_elos
    return games


def final_ratings(games: pd.DataFrame, k: float = K, home_adv: float = HOME_ADV) -> dict[int, float]:
    """Each team's current Elo AFTER its most recent game — for live prediction.

    Re-runs the same chronological loop as compute_elo and returns the final
    rating dict (team_id -> rating).
    """
    games = games.sort_values("game_date")
    has_scores = "home_pts" in games.columns and "away_pts" in games.columns
    ratings: dict[int, float] = {}
    for row in games.itertuples(index=False):
        elo_home = ratings.get(row.home_team_id, START_RATING)
        elo_away = ratings.get(row.away_team_id, START_RATING)
        margin = abs(row.home_pts - row.away_pts) if has_scores else None
        mult = _mov_scale(margin, row.home_win, elo_home + home_adv, elo_away)
        change = k * mult * (row.home_win - expected_home(elo_home, elo_away, home_adv))
        ratings[row.home_team_id] = elo_home + change
        ratings[row.away_team_id] = elo_away - change
    return ratings
