"""
Performance and risk metrics for a returns series.

All functions take a pandas Series of periodic returns (not prices) unless
noted otherwise. Annualization assumes daily returns by default (252
trading days) — pass a different periods_per_year for weekly/monthly data.
"""

import numpy as np
import pandas as pd


def total_return(returns: pd.Series) -> float:
    """Cumulative return over the full period."""
    return float((1 + returns).prod() - 1)


def annualized_return(returns: pd.Series, periods_per_year: int = 252) -> float:
    """Geometric average annual return."""
    n_periods = len(returns)
    if n_periods == 0:
        return np.nan
    cumulative = (1 + returns).prod()
    return float(cumulative ** (periods_per_year / n_periods) - 1)


def annualized_volatility(returns: pd.Series, periods_per_year: int = 252) -> float:
    """Standard deviation of returns, annualized."""
    return float(returns.std() * np.sqrt(periods_per_year))


def sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.0, periods_per_year: int = 252) -> float:
    """Return per unit of total volatility, in excess of the risk-free rate.

    risk_free_rate is annualized (e.g. 0.03 for 3%).
    """
    excess = returns - risk_free_rate / periods_per_year
    vol = excess.std()
    if vol == 0 or np.isnan(vol):
        return np.nan
    return float(excess.mean() / vol * np.sqrt(periods_per_year))


def downside_deviation(returns: pd.Series, mar: float = 0.0, periods_per_year: int = 252) -> float:
    """Volatility of returns falling below a minimum acceptable return (MAR),
    annualized. Unlike total volatility, this ignores upside variance —
    the basis of the Sortino ratio.
    """
    downside = returns[returns < mar] - mar
    if downside.empty:
        return 0.0
    return float(np.sqrt((downside ** 2).mean()) * np.sqrt(periods_per_year))


def sortino_ratio(returns: pd.Series, mar: float = 0.0, periods_per_year: int = 252) -> float:
    """Like Sharpe, but penalizes only downside volatility. Preferred over
    Sharpe when returns are skewed, since Sharpe treats big upside moves
    as 'risk' too.
    """
    dd = downside_deviation(returns, mar, periods_per_year)
    if dd == 0:
        return np.nan
    ann_return = annualized_return(returns, periods_per_year)
    return float((ann_return - mar) / dd)


def max_drawdown(returns: pd.Series) -> float:
    """Largest peak-to-trough decline in cumulative value, as a negative
    fraction (e.g. -0.25 for a 25% drawdown).
    """
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.cummax()
    drawdown = cumulative / running_max - 1
    return float(drawdown.min())


def drawdown_series(returns: pd.Series) -> pd.Series:
    """Full drawdown path, useful for plotting underwater curves."""
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.cummax()
    return cumulative / running_max - 1


def max_drawdown_duration(returns: pd.Series) -> int:
    """Longest number of periods spent below a previous peak (time to
    recover), not just the depth of the drawdown.
    """
    dd = drawdown_series(returns)
    is_underwater = dd < 0

    longest = 0
    current = 0
    for underwater in is_underwater:
        if underwater:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return longest


def calmar_ratio(returns: pd.Series, periods_per_year: int = 252) -> float:
    """Annualized return divided by the worst drawdown. Popular with CTAs
    and macro funds since it penalizes large drawdowns specifically,
    not just volatility in general.
    """
    mdd = max_drawdown(returns)
    if mdd == 0:
        return np.nan
    return float(annualized_return(returns, periods_per_year) / abs(mdd))


def omega_ratio(returns: pd.Series, threshold: float = 0.0) -> float:
    """Ratio of total gains above a threshold to total losses below it.
    Unlike Sharpe/Sortino, this uses the full return distribution rather
    than just mean and variance, so it captures skewness and fat tails.
    """
    excess = returns - threshold
    gains = excess[excess > 0].sum()
    losses = -excess[excess < 0].sum()
    if losses == 0:
        return np.nan
    return float(gains / losses)


def tail_ratio(returns: pd.Series, percentile: float = 0.05) -> float:
    """Ratio of the size of the right tail to the left tail (e.g. 95th
    percentile gain vs. 5th percentile loss). A ratio above 1 suggests
    the strategy has bigger winners than losers, not just more of them.
    """
    right_tail = returns.quantile(1 - percentile)
    left_tail = returns.quantile(percentile)
    if left_tail == 0:
        return np.nan
    return float(abs(right_tail / left_tail))


def skewness(returns: pd.Series) -> float:
    """Asymmetry of the return distribution. Negative skew means large
    losses are more extreme than large gains — a common feature of
    short-volatility or carry strategies, and worth flagging explicitly."""
    return float(returns.skew())


def kurtosis(returns: pd.Series) -> float:
    """Excess kurtosis (0 = normal distribution). High kurtosis means fat
    tails — more extreme moves than a normal distribution would predict,
    which standard deviation-based risk measures underestimate.
    """
    return float(returns.kurtosis())


def historical_var(returns: pd.Series, confidence: float = 0.95) -> float:
    """Historical (non-parametric) Value at Risk: the loss threshold not
    expected to be exceeded with the given confidence, based purely on
    the empirical return distribution.
    """
    return float(-returns.quantile(1 - confidence))


def parametric_var(returns: pd.Series, confidence: float = 0.95) -> float:
    """Parametric (variance-covariance) VaR, assuming returns are normally
    distributed. Faster to compute than historical VaR but understates
    risk when the true distribution has fat tails (see kurtosis above).
    """
    from scipy.stats import norm
    mu = returns.mean()
    sigma = returns.std()
    z = norm.ppf(1 - confidence)
    return float(-(mu + z * sigma))


def expected_shortfall(returns: pd.Series, confidence: float = 0.95) -> float:
    """Expected Shortfall (Conditional VaR): the average loss in the worst
    (1 - confidence) fraction of outcomes. Increasingly required alongside
    VaR by regulators since it captures tail severity, not just a threshold.
    """
    threshold = returns.quantile(1 - confidence)
    tail_losses = returns[returns <= threshold]
    if tail_losses.empty:
        return np.nan
    return float(-tail_losses.mean())
def monte_carlo_var(
    returns: pd.Series,
    confidence: float = 0.95,
    n_simulations: int = 10000,
    distribution: str = "normal",
    random_seed: int = 42,
    exclude_zero_returns: bool = True,
) -> float:
    """Monte Carlo VaR: simulate a large number of hypothetical returns
    drawn from a fitted distribution, then take the empirical quantile of
    the simulated outcomes.

    distribution="normal" fits a normal distribution to the return series
    (mean, std), matching the assumption behind parametric_var.
    distribution="t" fits a Student-t distribution instead, which allows
    fatter tails. Given how much excess kurtosis shows up in practice
    (see the kurtosis() function), the normal assumption tends to
    understate tail risk — comparing both versions makes that gap visible
    directly, rather than relying on a single VaR number.

    exclude_zero_returns: for a strategy that holds no position on many
    days (returns exactly 0.0), including those days can make the fitted
    distribution degenerate — a large point-mass at a single value can
    collapse the Student-t maximum-likelihood fit to a near-zero scale.
    Excluding them focuses the estimate on days the strategy was actually
    exposed to the market, which is also the more meaningful measure of
    risk for a signal-driven strategy.
    """
    rng = np.random.default_rng(random_seed)
    clean_returns = returns.dropna()
    if exclude_zero_returns:
        clean_returns = clean_returns[clean_returns != 0]

    if len(clean_returns) < 30:
        return np.nan

    if distribution == "t":
        from scipy.stats import t as t_dist
        params = t_dist.fit(clean_returns)
        simulated = t_dist.rvs(*params, size=n_simulations, random_state=rng)
    else:
        mu, sigma = clean_returns.mean(), clean_returns.std()
        simulated = rng.normal(mu, sigma, n_simulations)

    return float(-np.percentile(simulated, (1 - confidence) * 100))

def rolling_sharpe(returns: pd.Series, window: int = 63, periods_per_year: int = 252) -> pd.Series:
    """Sharpe ratio computed on a rolling window, to see whether
    risk-adjusted performance is stable or concentrated in a few periods.
    Default window of 63 is roughly one trading quarter.
    """
    rolling_mean = returns.rolling(window).mean()
    rolling_std = returns.rolling(window).std()
    return (rolling_mean / rolling_std) * np.sqrt(periods_per_year)


def summary_tearsheet(returns: pd.Series, risk_free_rate: float = 0.0, periods_per_year: int = 252) -> dict:
    """One-call summary of all metrics above, formatted like a fund
    tearsheet. Use this for reporting rather than calling each function
    individually.
    """
    return {
        "Total return": total_return(returns),
        "Annualized return": annualized_return(returns, periods_per_year),
        "Annualized volatility": annualized_volatility(returns, periods_per_year),
        "Sharpe ratio": sharpe_ratio(returns, risk_free_rate, periods_per_year),
        "Sortino ratio": sortino_ratio(returns, periods_per_year=periods_per_year),
        "Calmar ratio": calmar_ratio(returns, periods_per_year),
        "Omega ratio": omega_ratio(returns),
        "Max drawdown": max_drawdown(returns),
        "Max drawdown duration (periods)": max_drawdown_duration(returns),
        "Tail ratio": tail_ratio(returns),
        "Skewness": skewness(returns),
        "Kurtosis (excess)": kurtosis(returns),
        "Historical VaR (95%)": historical_var(returns, 0.95),
        "Parametric VaR (95%)": parametric_var(returns, 0.95),
        "Monte Carlo VaR (95%, normal)": monte_carlo_var(returns, 0.95, distribution="normal"),
        "Monte Carlo VaR (95%, Student-t)": monte_carlo_var(returns, 0.95, distribution="t"),
        "Expected Shortfall (95%)": expected_shortfall(returns, 0.95),
    }


if __name__ == "__main__":
    # Quick smoke test with synthetic returns
    np.random.seed(42)
    fake_returns = pd.Series(np.random.normal(0.0005, 0.01, 500))

    for label, value in summary_tearsheet(fake_returns).items():
        print(f"{label}: {value:.4f}" if isinstance(value, float) else f"{label}: {value}")