# Betting backtest — model vs the market

Out-of-fold logistic predictions joined to historical odds: **12751 games** with odds (2006-07 → 2017-18).

## Who predicts better (log loss, lower is better)

| | Log loss |
|---|---|
| Our model | 0.5965 |
| The market (de-vigged) | 0.5799 |

The market beats our model. This is the expected result — closing/consensus lines are extremely hard to beat.

## Flat-stake betting at several edge thresholds

| Min edge | Bets | Bet rate | Win rate | ROI |
|---|---|---|---|---|
| 0% | 12751 | 100.0% | 44.5% | -5.2% |
| 2% | 9892 | 77.6% | 43.4% | -5.2% |
| 5% | 6222 | 48.8% | 42.2% | -3.5% |
| 8% | 3695 | 29.0% | 40.7% | +0.2% |
| 10% | 2539 | 19.9% | 39.1% | -1.7% |

## Honest conclusion

At -110 typical pricing, breakeven is ~52.4%. A negative ROI across thresholds means our model has **no exploitable edge against the market after the vig** — the correct, expected outcome for a portfolio model and a sign of an efficient market, not a failure.

**Caveats:** odds are a multi-book consensus, not timestamped closing lines, so this measures edge-vs-consensus, not true CLV; coverage is 2006-2018 only; transaction costs/line shopping are idealized.
