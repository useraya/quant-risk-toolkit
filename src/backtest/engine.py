"""
Backtest engine.

Takes a weekly positioning signal (from cot_signal.py) and daily price
returns (from loaders.py), turns the signal into a tradeable position
series, and computes the resulting strategy returns — ready to feed into
performance.metrics.summary_tearsheet.

Two realism details that matter for a credible backtest:

1. Publication lag: the CFTC releases Tuesday's positioning data the
   following Friday. A signal dated "Tuesday" isn't actually tradeable
   until several days later — ignoring this is a common lookahead-bias
   mistake in COT-based backtests.

2. Position lag: today's position should be based on yesterday's signal,
   not today's, since you can't react to information before it exists.
"""

import pandas as pd
import numpy as np


def align_signal_to_price(
    signal: pd.Series,
    price_dates: pd.DatetimeIndex,
    publication_lag_days: int = 3,
) -> pd.Series:
    """Shift a weekly signal forward by the CFTC's publication lag, then
    forward-fill it onto a daily price index so the position is held
    constant between signal updates.
    """
    lagged = signal.copy()
    lagged.index = lagged.index + pd.Timedelta(days=publication_lag_days)

    daily = lagged.reindex(price_dates, method="ffill")
    return daily.fillna(0)


def run_backtest(
    price_returns: pd.Series,
    signal: pd.Series,
    publication_lag_days: int = 3,
    position_mode: str = "long_short",
    transaction_cost_bps: float = 0.0,
) -> pd.DataFrame:
    """Run the backtest and return a DataFrame with position, strategy
    returns, and cumulative equity for both the strategy and a buy-and-hold
    benchmark.

    position_mode:
        "long_short" — signal +1/-1/0 maps directly to position
        "long_flat"  — signal +1 maps to long, everything else is flat
                       (no shorting)

    transaction_cost_bps: cost charged per unit of position change, in
    basis points (e.g. 5 = 0.05%). Applied whenever the position changes,
    to avoid overstating returns from a strategy that trades often.
    """
    daily_signal = align_signal_to_price(signal, price_returns.index, publication_lag_days)

    if position_mode == "long_flat":
        position = daily_signal.clip(lower=0)
    else:
        position = daily_signal

    # Position is lagged one day: today's return is earned on yesterday's
    # position, since you can't act on a signal before it's known
    position_lagged = position.shift(1).fillna(0)

    position_change = position_lagged.diff().abs().fillna(0)
    transaction_costs = position_change * (transaction_cost_bps / 10000)

    strategy_returns = position_lagged * price_returns - transaction_costs

    result = pd.DataFrame(index=price_returns.index)
    result["price_return"] = price_returns
    result["position"] = position_lagged
    result["strategy_return"] = strategy_returns
    result["strategy_equity"] = (1 + strategy_returns).cumprod()
    result["benchmark_equity"] = (1 + price_returns).cumprod()

    return result
def apply_trend_filter(
    signal: pd.Series,
    prices: pd.DataFrame,
    ma_window: int = 200,
    price_col: str = "Close",
) -> pd.Series:
    """Only keep a bullish signal when price is above its long-term moving
    average, and a bearish signal when price is below it. This tests a
    specific, pre-specified hypothesis: that COT positioning has value as
    a confirming filter alongside trend, even if it showed no standalone
    predictive edge on its own (per the IC analysis).

    This is one deliberate test, not a parameter search — the 200-day
    window is a standard, widely-used trend definition, not something
    tuned to make the result look good.
    """
    moving_avg = prices[price_col].rolling(ma_window).mean()
    trend_up = prices[price_col] > moving_avg
    trend_down = prices[price_col] < moving_avg

    trend_up_aligned = trend_up.reindex(signal.index, method="ffill").fillna(False)
    trend_down_aligned = trend_down.reindex(signal.index, method="ffill").fillna(False)

    filtered = signal.copy()
    filtered[(signal == 1) & (~trend_up_aligned)] = 0
    filtered[(signal == -1) & (~trend_down_aligned)] = 0

    return filtered

def compare_to_benchmark(backtest_result: pd.DataFrame, periods_per_year: int = 252) -> dict:
    """Quick side-by-side comparison of strategy vs. buy-and-hold, using
    the same metrics as the main tearsheet.
    """
    from src.performance.metrics import summary_tearsheet

    strategy_stats = summary_tearsheet(backtest_result["strategy_return"], periods_per_year=periods_per_year)
    benchmark_stats = summary_tearsheet(backtest_result["price_return"], periods_per_year=periods_per_year)

    comparison = pd.DataFrame({"Strategy": strategy_stats, "Buy & Hold": benchmark_stats})
    return comparison


if __name__ == "__main__":
    # Smoke test with synthetic data: weekly signal, daily prices
    dates_weekly = pd.date_range("2020-01-01", periods=100, freq="W-TUE")
    fake_signal = pd.Series(np.random.choice([-1, 0, 1], size=100), index=dates_weekly)

    dates_daily = pd.date_range("2020-01-01", periods=700, freq="D")
    fake_returns = pd.Series(np.random.normal(0.0003, 0.01, 700), index=dates_daily)

    result = run_backtest(fake_returns, fake_signal, position_mode="long_short", transaction_cost_bps=2)
    print(result.tail())

    comparison = compare_to_benchmark(result)
    print("\n", comparison)