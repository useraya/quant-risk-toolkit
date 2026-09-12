"""
Run the full gold COT strategy: real positioning data + real price data
through the backtest engine, with a full performance tearsheet at the end.

Run with:
    python -m src.run_gold_strategy
"""

from src.data.loaders import fetch_price_history, compute_returns
from src.signals.cot_signal import fetch_cot_history, build_signal
from src.backtest.engine import run_backtest, compare_to_benchmark

START_DATE = "2015-01-01"


def main():
    print("Fetching gold price history...")
    prices = fetch_price_history("GC=F", start=START_DATE)
    price_returns = compute_returns(prices)
    print(f"  Got {len(price_returns)} daily returns")

    print("\nFetching gold COT positioning data...")
    cot_data = fetch_cot_history("gold", start_date=START_DATE)
    signal_df = build_signal(cot_data, trader_type="commercial")
    signal = signal_df["signal"]
    print(f"  Got {len(signal)} weekly COT reports")
    print(f"  Bullish weeks: {(signal == 1).sum()}, Bearish weeks: {(signal == -1).sum()}")

    print("\nRunning backtest...")
    result = run_backtest(
        price_returns, signal,
        publication_lag_days=3,
        position_mode="long_short",
        transaction_cost_bps=2,
    )

    print("\nPerformance comparison: strategy vs. buy-and-hold")
    comparison = compare_to_benchmark(result)
    print(comparison.round(4))

    result.to_csv("data/gold_cot_backtest.csv")
    print("\nSaved full backtest output to data/gold_cot_backtest.csv")


if __name__ == "__main__":
    main()