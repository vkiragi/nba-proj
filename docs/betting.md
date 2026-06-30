# Betting backtest — model vs the market

Out-of-fold logistic predictions joined to historical odds: **12751 games** with odds (2006-07 → 2017-18).

## Who predicts better (log loss, lower is better)

| | Log loss |
|---|---|
| Our model | 0.5983 |
| The market (de-vigged) | 0.5799 |

The market beats our model. This is the expected result — closing/consensus lines are extremely hard to beat.

## Flat-stake betting at several edge thresholds

| Min edge | Bets | Bet rate | Win rate | ROI |
|---|---|---|---|---|
| 0% | 12751 | 100.0% | 44.4% | -5.0% |
| 2% | 9990 | 78.3% | 43.4% | -4.4% |
| 5% | 6477 | 50.8% | 42.0% | -3.6% |
| 8% | 3911 | 30.7% | 41.1% | -0.3% |
| 10% | 2767 | 21.7% | 39.6% | -1.2% |

## Honest conclusion

At -110 typical pricing, breakeven is ~52.4%. A negative ROI across thresholds means our model has **no exploitable edge against the market after the vig** — the correct, expected outcome for a portfolio model and a sign of an efficient market, not a failure.

**Caveats:** odds are a multi-book consensus, not timestamped closing lines, so this measures edge-vs-consensus, not true CLV; coverage is 2006-2018 only; transaction costs/line shopping are idealized.
