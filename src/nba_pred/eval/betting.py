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


def betting_backtest(predictions: pd.DataFrame, odds: pd.DataFrame) -> dict:
    """Flat-stake betting backtest vs closing lines. NOT YET IMPLEMENTED.

    Expects `predictions` (game_id, p, home_win) joined to `odds`
    (game_id, home_ml_close, away_ml_close). Returns ROI / record / CLV.
    """
    raise NotImplementedError(
        "Betting backtest is blocked on historical odds data. See this module's "
        "docstring for the required odds schema and sources, then implement the "
        "edge/ROI/CLV computation here."
    )
