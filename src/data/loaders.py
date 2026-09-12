"""
Price data loader.

Uses yfinance, which pulls free historical price data from Yahoo Finance
(no API key required). Works for equities, FX pairs (e.g. "EURUSD=X"),
futures (e.g. "GC=F" for gold), and major indices.
"""

import pandas as pd
import yfinance as yf


def fetch_price_history(ticker: str, start: str, end: str = None, interval: str = "1d") -> pd.DataFrame:
    """Fetch OHLCV price history for a single ticker.

    ticker examples: "EURUSD=X" (EUR/USD), "GC=F" (gold futures),
    "^GSPC" (S&P 500), "AAPL" (Apple stock).

    Returns a DataFrame indexed by date with columns:
    Open, High, Low, Close, Volume
    """
    data = yf.download(ticker, start=start, end=end, interval=interval, progress=False, auto_adjust=True)

    if data.empty:
        raise ValueError(f"No data returned for ticker '{ticker}' — check the symbol is correct")

    # yfinance sometimes returns MultiIndex columns for a single ticker
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    return data


def compute_returns(prices: pd.DataFrame, price_col: str = "Close") -> pd.Series:
    """Convert a price series into simple daily returns."""
    return prices[price_col].pct_change().dropna()


def fetch_multiple(tickers: list, start: str, end: str = None, interval: str = "1d") -> dict:
    """Fetch price history for several tickers at once.

    Returns a dict mapping ticker -> price DataFrame. Tickers that fail to
    fetch are skipped and reported, rather than crashing the whole run.
    """
    results = {}
    for ticker in tickers:
        try:
            results[ticker] = fetch_price_history(ticker, start, end, interval)
        except Exception as e:
            print(f"  Skipped {ticker}: {e}")
    return results


def combined_returns(tickers: list, start: str, end: str = None) -> pd.DataFrame:
    """Fetch multiple tickers and align their returns into a single
    DataFrame (one column per ticker), useful for portfolio-level
    calculations like VaR on a multi-asset book.
    """
    price_data = fetch_multiple(tickers, start, end)
    returns = {ticker: compute_returns(df) for ticker, df in price_data.items()}
    return pd.DataFrame(returns)


if __name__ == "__main__":
    # Example: gold futures and EUR/USD over the last two years
    prices = fetch_price_history("GC=F", start="2023-01-01")
    print(prices.tail())

    returns = compute_returns(prices)
    print(f"\nFetched {len(returns)} daily returns for GC=F")