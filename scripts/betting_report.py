"""Betting backtest: our out-of-fold probabilities vs the consensus market.

Runs the best model (logistic) through walk-forward, collects out-of-fold
predictions, joins historical odds, and reports ROI / win-rate at several edge
thresholds. Writes docs/betting.md.

    PYTHONPATH=src uv run python scripts/betting_report.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from nba_pred.eval.backtest import walk_forward
from nba_pred.eval.betting import betting_backtest
from nba_pred.eval.metrics import evaluate
from nba_pred.models.features import build_model_frame
from nba_pred.models.logistic import make_logistic_predict

REPO = Path(__file__).resolve().parents[1]


def main() -> None:
    games = pd.read_parquet(REPO / "data" / "processed" / "games.parquet")
    odds = pd.read_parquet(REPO / "data" / "processed" / "odds.parquet")
    frame = build_model_frame(games)

    # Out-of-fold predictions from the best model.
    _, agg, preds = walk_forward(frame, make_logistic_predict(), "Logistic", return_predictions=True)
    overlap = preds.merge(odds, on="game_id", how="inner")

    # How well-calibrated are WE vs the MARKET on the games we can bet?
    model_ll = evaluate(overlap["home_win"], overlap["p"]).log_loss
    market_ll = evaluate(overlap["home_win"], overlap["market_p_home"]).log_loss

    lines = [
        "# Betting backtest — model vs the market",
        "",
        f"Out-of-fold logistic predictions joined to historical odds: "
        f"**{len(overlap)} games** with odds (2006-07 → 2017-18).",
        "",
        "## Who predicts better (log loss, lower is better)",
        "",
        "| | Log loss |",
        "|---|---|",
        f"| Our model | {model_ll:.4f} |",
        f"| The market (de-vigged) | {market_ll:.4f} |",
        "",
        f"The market {'beats' if market_ll < model_ll else 'does NOT beat'} our model. "
        "This is the expected result — closing/consensus lines are extremely hard to beat.",
        "",
        "## Flat-stake betting at several edge thresholds",
        "",
        "| Min edge | Bets | Bet rate | Win rate | ROI |",
        "|---|---|---|---|---|",
    ]
    print(f"model log loss {model_ll:.4f} vs market {market_ll:.4f} on {len(overlap)} games")
    for thr in [0.0, 0.02, 0.05, 0.08, 0.10]:
        r = betting_backtest(preds, odds, edge_threshold=thr)
        lines.append(
            f"| {thr:.0%} | {r['n_bets']} | {r['bet_rate']:.1%} | "
            f"{r['win_rate']:.1%} | {r['roi']:+.1%} |"
        )
        print(f"  edge>={thr:.0%}: bets={r['n_bets']:>5} win={r['win_rate']:.1%} roi={r['roi']:+.2%}")

    lines += [
        "",
        "## Honest conclusion",
        "",
        "At -110 typical pricing, breakeven is ~52.4%. A negative ROI across "
        "thresholds means our model has **no exploitable edge against the market "
        "after the vig** — the correct, expected outcome for a portfolio model "
        "and a sign of an efficient market, not a failure.",
        "",
        "**Caveats:** odds are a multi-book consensus, not timestamped closing "
        "lines, so this measures edge-vs-consensus, not true CLV; coverage is "
        "2006-2018 only; transaction costs/line shopping are idealized.",
        "",
    ]
    out = REPO / "docs" / "betting.md"
    out.write_text("\n".join(lines))
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
