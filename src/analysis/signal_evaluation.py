"""
Information Coefficient (IC) analysis.

The IC is the standard way quant research teams validate a signal before
ever building a trading strategy around it: it measures the correlation
between a signal's raw value and what actually happens to price
afterward, at several forward horizons.

This matters because a backtest's P&L depends on position sizing,
thresholds, transaction costs, and long/short rules — a weak backtest
result could mean the signal has no real information content, or it
could just mean the trading rules built on top of it were poorly chosen.
IC analysis answers the first question directly, independent of the
second.

Background: Grinold & Kahn's "Fundamental Law of Active Management"
uses IC as the core measure of a signal's skill.
"""

import pandas as pd
import numpy as np
from scipy.stats import pearsonr, spearmanr


def compute_forward_returns(
    prices: pd.DataFrame,
    signal_dates: pd.DatetimeIndex,
    horizon_days: int,
    price_col: str = "Close",
) -> pd.Series:
    """For each signal date, compute the cumulative price return over the
    following horizon_days. Dates without enough future price history
    (near the end of the sample) are dropped.
    """
    price_series = prices[price_col]
    results = {}

    for date in signal_dates:
        future_dates = price_series.index[price_series.index >= date]
        if len(future_dates) < horizon_days + 1:
            continue

        start_price = price_series.loc[future_dates[0]]
        end_date = future_dates[min(horizon_days, len(future_dates) - 1)]
        end_price = price_series.loc[end_date]

        results[date] = (end_price / start_price) - 1

    return pd.Series(results)


def information_coefficient(signal_values: pd.Series, forward_returns: pd.Series) -> dict:
    """Compute both Pearson and Spearman IC between a signal and its
    forward returns, with statistical significance.

    Spearman is generally preferred for IC since it only requires the
    *ranking* to be informative, not a linear relationship — signals are
    rarely linearly related to returns.
    """
    aligned = pd.DataFrame({"signal": signal_values, "forward_return": forward_returns}).dropna()

    if len(aligned) < 10:
        return {
            "n_obs": len(aligned),
            "pearson_ic": np.nan, "pearson_pvalue": np.nan,
            "spearman_ic": np.nan, "spearman_pvalue": np.nan,
            "hit_rate": np.nan,
        }

    pearson_ic, pearson_p = pearsonr(aligned["signal"], aligned["forward_return"])
    spearman_ic, spearman_p = spearmanr(aligned["signal"], aligned["forward_return"])

    # Hit rate: how often the sign of the signal (relative to its median)
    # agrees with the sign of the forward return
    signal_direction = np.sign(aligned["signal"] - aligned["signal"].median())
    return_direction = np.sign(aligned["forward_return"])
    hit_rate = (signal_direction == return_direction).mean()

    return {
        "n_obs": len(aligned),
        "pearson_ic": pearson_ic, "pearson_pvalue": pearson_p,
        "spearman_ic": spearman_ic, "spearman_pvalue": spearman_p,
        "hit_rate": hit_rate,
    }


def ic_decay_analysis(
    prices: pd.DataFrame,
    signal_values: pd.Series,
    horizons: list = (5, 10, 20, 40, 60),
) -> pd.DataFrame:
    """Run IC analysis across several forward horizons to see how quickly
    (or slowly) the signal's predictive power decays — a signal that's
    only informative 5 days out behaves very differently from one that
    holds up 60 days out, and that shapes how you'd size and hold positions.
    """
    rows = []
    for horizon in horizons:
        forward_ret = compute_forward_returns(prices, signal_values.index, horizon)
        ic_stats = information_coefficient(signal_values, forward_ret)
        ic_stats["horizon_days"] = horizon
        rows.append(ic_stats)

    return pd.DataFrame(rows).set_index("horizon_days")


if __name__ == "__main__":
    # Smoke test: a signal with genuine (synthetic) predictive power
    np.random.seed(0)
    dates = pd.date_range("2020-01-01", periods=300, freq="D")
    prices = pd.DataFrame({"Close": 100 * (1 + np.random.normal(0, 0.01, 300)).cumprod()}, index=dates)

    signal_dates = dates[::5][:-5]
    fake_signal = pd.Series(np.random.normal(0, 1, len(signal_dates)), index=signal_dates)

    decay = ic_decay_analysis(prices, fake_signal, horizons=[5, 10, 20])
    print(decay.round(4))