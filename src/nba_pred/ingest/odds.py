"""Ingest historical moneyline odds (ehallmar Kaggle dataset) -> clean parquet.

Source files (in data/raw/odds/):
  - nba_betting_money_line.csv : game_id, book_name, team_id, a_team_id, price1, price2
      one row per (game, sportsbook). price1 belongs to team_id, price2 to a_team_id.
  - nba_games_all.csv          : per-team-game rows; is_home == 't' marks the home team.
  - nba_teams_all.csv          : not needed — team_ids are the official NBA ids we use.

We resolve home/away via games_all, de-vig each book's two-way line, then average
the de-vigged home probability across books to a consensus market probability.

LIMITATIONS (state honestly in the writeup):
  - Coverage ~2006-11 to 2018-06 only; no odds for 2018-25.
  - These are multi-book consensus lines, NOT timestamped closing lines, so true
    open-vs-close CLV isn't possible — we use consensus as the market reference.

Output: data/processed/odds.parquet with one row per game:
    game_id (10-char, matches our spine), market_p_home, home_ml, away_ml, n_books

    PYTHONPATH=src uv run python -m nba_pred.ingest.odds
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from nba_pred.eval.betting import american_to_prob, devig_two_way, prob_to_american

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_ODDS_DIR = REPO_ROOT / "data" / "raw" / "odds"
DEFAULT_OUT = REPO_ROOT / "data" / "processed" / "odds.parquet"


def _home_team_by_game(games_csv: Path) -> pd.DataFrame:
    """game_id -> home_team_id, from games_all (is_home == 't')."""
    games = pd.read_csv(games_csv, usecols=["game_id", "team_id", "is_home"])
    home = games[games["is_home"] == "t"][["game_id", "team_id"]]
    return home.drop_duplicates("game_id").rename(columns={"team_id": "home_team_id"})


def build_odds(odds_dir: Path = DEFAULT_ODDS_DIR) -> pd.DataFrame:
    ml = pd.read_csv(odds_dir / "nba_betting_money_line.csv")
    home = _home_team_by_game(odds_dir / "nba_games_all.csv")

    ml = ml.merge(home, on="game_id", how="inner")  # need to know who's home

    # Map each book's two prices to home/away by matching team ids (robust: do not
    # assume a fixed price1/price2 ordering).
    team_is_home = ml["team_id"] == ml["home_team_id"]
    ml["home_ml"] = np.where(team_is_home, ml["price1"], ml["price2"])
    ml["away_ml"] = np.where(team_is_home, ml["price2"], ml["price1"])
    ml = ml.dropna(subset=["home_ml", "away_ml"])

    # Drop garbage book quotes: a valid American moneyline always has |odds| >= 100.
    valid = (ml["home_ml"].abs() >= 100) & (ml["away_ml"].abs() >= 100)
    ml = ml[valid]

    # Work entirely in probability space (continuous) — American odds cannot be
    # averaged directly. Per book: raw implied probs (vig included) for each side.
    ml["p_home_raw"] = ml["home_ml"].map(american_to_prob)
    ml["p_away_raw"] = ml["away_ml"].map(american_to_prob)

    agg = ml.groupby("game_id").agg(
        p_home_raw=("p_home_raw", "mean"),  # consensus, still vigged
        p_away_raw=("p_away_raw", "mean"),
        n_books=("book_name", "nunique"),
    ).reset_index()

    # De-vigged consensus prob (the market's true home-win estimate) and a
    # representative consensus American line per side (for realistic payouts).
    devig = [devig_two_way(h, a)[0] for h, a in zip(agg["p_home_raw"], agg["p_away_raw"])]
    agg["market_p_home"] = devig
    agg["home_ml"] = agg["p_home_raw"].map(prob_to_american)
    agg["away_ml"] = agg["p_away_raw"].map(prob_to_american)

    agg["game_id"] = agg["game_id"].astype(str).str.zfill(10)
    return agg[["game_id", "market_p_home", "home_ml", "away_ml", "n_books"]]


def main() -> None:
    out = build_odds()
    DEFAULT_OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(DEFAULT_OUT, index=False)

    spine = pd.read_parquet(REPO_ROOT / "data" / "processed" / "games.parquet")
    matched = spine["game_id"].isin(set(out["game_id"])).sum()
    print(f"✅ odds for {len(out)} games -> {DEFAULT_OUT}")
    print(f"   joins {matched}/{len(spine)} spine games ({matched / len(spine):.1%})")
    print(f"   market home-win prob (mean): {out['market_p_home'].mean():.3f}")


if __name__ == "__main__":
    main()
