"""Leakage-safe roster-strength feature from per-player box scores.

The idea: estimate how strong the lineup a team is about to field is, using only
information available before tip-off.

  1. Player rating (as-of): each player's rolling average plus/minus over their
     last N games, shifted to EXCLUDE the current game.
  2. Lineup strength of a team-game: minutes-weighted mean of those as-of player
     ratings over the players who appeared.
  3. Roster strength going INTO a game: the team's PREVIOUS game's lineup
     strength (shift by one within team-season). We never read tonight's box
     score — using "who actually played tonight" would leak availability (the
     injury trap). The previous lineup is a leakage-safe proxy for who'll play.

NaN for a team's first game of a season (no prior lineup) — left honest, imputed
later inside the model.
"""

from __future__ import annotations

import pandas as pd

ROSTER_FEATURES = ["home_roster_strength", "away_roster_strength"]

PLAYER_RATING_WINDOW = 20  # games of plus/minus history per player


def player_ratings_as_of(player_stats: pd.DataFrame) -> pd.DataFrame:
    """Add `player_rating`: rolling mean plus/minus over prior games (as-of)."""
    ps = player_stats.sort_values(["player_id", "game_date", "game_id"], kind="mergesort").copy()
    ps["player_rating"] = ps.groupby("player_id", sort=False)["plus_minus"].transform(
        lambda s: s.shift(1).rolling(PLAYER_RATING_WINDOW, min_periods=5).mean()
    )
    return ps


def _lineup_strength(ps: pd.DataFrame) -> pd.DataFrame:
    """Per team-game minutes-weighted mean of as-of player ratings."""
    valid = ps["player_rating"].notna() & (ps["minutes"] > 0)
    v = ps[valid].copy()
    v["rw"] = v["player_rating"] * v["minutes"]
    grp = v.groupby(["team_id", "game_id", "game_date", "season"], sort=False)
    lineup = (grp["rw"].sum() / grp["minutes"].sum()).reset_index(name="lineup_strength")
    return lineup


def add_roster_strength(games: pd.DataFrame, player_stats: pd.DataFrame) -> pd.DataFrame:
    """Attach home_/away_roster_strength to the wide game spine, leakage-safe."""
    ps = player_ratings_as_of(player_stats)
    lineup = _lineup_strength(ps)

    # Roster strength going INTO a game = the team's PREVIOUS game's lineup
    # strength (shift within team-season so it never uses the current game).
    lineup = lineup.sort_values(["team_id", "game_date", "game_id"], kind="mergesort")
    lineup["roster_strength"] = lineup.groupby(["team_id", "season"], sort=False)[
        "lineup_strength"
    ].shift(1)

    keyed = lineup[["team_id", "game_id", "roster_strength"]]
    home = keyed.rename(columns={"team_id": "home_team_id", "roster_strength": "home_roster_strength"})
    away = keyed.rename(columns={"team_id": "away_team_id", "roster_strength": "away_roster_strength"})

    out = games.merge(home, on=["home_team_id", "game_id"], how="left")
    out = out.merge(away, on=["away_team_id", "game_id"], how="left")
    return out


def current_roster_strength(player_stats: pd.DataFrame) -> dict[int, float]:
    """Each team's latest lineup strength — what feeds its next game (for serving)."""
    ps = player_ratings_as_of(player_stats)
    lineup = _lineup_strength(ps).sort_values(["team_id", "game_date", "game_id"], kind="mergesort")
    last = lineup.groupby("team_id").tail(1)
    return {int(t): float(s) for t, s in zip(last["team_id"], last["lineup_strength"])}
