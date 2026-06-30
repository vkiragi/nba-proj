"""Betting backtest — BLOCKED on odds data (not yet implemented).

This is the plan's "domain flex": simulate betting our calibrated probabilities
against historical closing lines, accounting for the vig, reporting ROI and
closing-line value (CLV). It is intentionally a stub: it needs a historical odds
dataset that we don't have yet, and obtaining one requires a user decision.

## What's needed to unblock
An odds source joined to our game spine on (game_date, home_team, away_team):
  - sportsbookreviewsonline historical spreadsheets (free, manual download), or
  - OddsPortal scrape (respect ToS / rate limits), or
  - the-odds-api (free tier; going-forward only, needs an API key).

Target schema for `data/processed/odds.parquet`:
    game_id, home_ml_close, away_ml_close   (American odds, e.g. -150 / +130)
optionally open lines too, to measure CLV against the close.

## The math, ready to implement once odds exist
- American odds -> implied prob: for negative o, p = -o/(-o+100); for positive
  o, p = 100/(o+100). The book's two implied probs sum to >1 — that excess is
  the vig. De-vig by normalizing so they sum to 1.
- Edge: bet when our calibrated p_home exceeds the de-vigged implied prob by a
  threshold. At -110 both sides, breakeven win rate is ~52.4%.
- Report flat-stake ROI, record, and CLV (did we beat the closing number?).

## Honest expectation
We very likely do NOT beat the closing line after vig. "Well-calibrated but no
edge vs the market" is the correct, respectable conclusion (efficient markets) —
NOT a failure. Overclaiming a profitable system is the fastest way to lose
credibility.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def american_to_prob(odds: float) -> float:
    """American moneyline odds -> implied win probability (includes vig)."""
    if odds < 0:
        return -odds / (-odds + 100.0)
    return 100.0 / (odds + 100.0)


def devig_two_way(p_home_raw: float, p_away_raw: float) -> tuple[float, float]:
    """Normalize a book's two implied probs to remove the vig (sum to 1)."""
    total = p_home_raw + p_away_raw
    return p_home_raw / total, p_away_raw / total


def prob_to_american(p: float) -> float:
    """Implied probability -> American odds. Inverse of american_to_prob.

    Always returns |odds| >= 100 for any p in (0, 1). Use this to build a
    consensus line: aggregate in probability space (continuous), then convert
    back — NEVER average American odds directly (they are discontinuous around
    +/-100 and averaging fabricates impossible values like -8).
    """
    if not 0.0 < p < 1.0:
        raise ValueError(f"probability must be in (0, 1), got {p!r}")
    if p > 0.5:
        return -100.0 * p / (1.0 - p)
    return 100.0 * (1.0 - p) / p


def american_profit(odds: float, stake: float = 1.0) -> float:
    """Profit (not including returned stake) on a winning bet at American odds.

    Valid moneylines satisfy |odds| >= 100; anything else is bad data.
    """
    if abs(odds) < 100:
        raise ValueError(f"invalid American odds {odds!r} (|odds| must be >= 100)")
    if odds > 0:
        return stake * odds / 100.0
    return stake * 100.0 / (-odds)


def betting_backtest(
    predictions: pd.DataFrame,
    odds: pd.DataFrame,
    edge_threshold: float = 0.0,
    stake: float = 1.0,
) -> dict:
    """Flat-stake betting backtest of our probabilities vs the market.

    `predictions`: columns game_id, p (model P(home win)), home_win (actual).
    `odds`: columns game_id, market_p_home (de-vigged), home_ml, away_ml.

    For each game with odds, we bet the side where our probability exceeds the
    market's de-vigged probability by more than `edge_threshold`, staking a flat
    amount at that side's American price. Returns ROI, record, and summary stats.

    NOTE: `market_p_home` is a multi-book consensus, not a timestamped close, so
    this is "edge vs the consensus market", not true closing-line value (CLV).
    """
    df = predictions.merge(odds, on="game_id", how="inner")
    if df.empty:
        return {"n_games": 0, "n_bets": 0, "note": "no overlap between predictions and odds"}

    p_home, mkt_home = df["p"].to_numpy(), df["market_p_home"].to_numpy()
    home_edge = p_home - mkt_home
    away_edge = (1.0 - p_home) - (1.0 - mkt_home)  # == -home_edge

    bet_home = home_edge > edge_threshold
    bet_away = away_edge > edge_threshold
    placed = bet_home | bet_away

    won = df["home_win"].to_numpy()
    profit = np.zeros(len(df))
    # Home bets
    hp = bet_home
    profit[hp] = np.where(
        won[hp] == 1,
        [american_profit(o, stake) for o in df["home_ml"].to_numpy()[hp]],
        -stake,
    )
    # Away bets
    ap = bet_away
    profit[ap] = np.where(
        won[ap] == 0,
        [american_profit(o, stake) for o in df["away_ml"].to_numpy()[ap]],
        -stake,
    )

    n_bets = int(placed.sum())
    total_staked = n_bets * stake
    total_profit = float(profit[placed].sum())
    bet_won = ((bet_home & (won == 1)) | (bet_away & (won == 0)))[placed]

    return {
        "n_games": len(df),
        "n_bets": n_bets,
        "bet_rate": n_bets / len(df),
        "win_rate": float(bet_won.mean()) if n_bets else float("nan"),
        "total_profit": total_profit,
        "total_staked": total_staked,
        "roi": total_profit / total_staked if total_staked else float("nan"),
        "edge_threshold": edge_threshold,
    }
