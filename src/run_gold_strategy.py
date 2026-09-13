"""
Run the full gold COT strategy: real positioning data + real price data
through the backtest engine, with IC validation, a trend-filter extension
test, and a full visual tearsheet at the end.

Run with:
    python -m src.run_gold_strategy
"""

from src.data.loaders import fetch_price_history, compute_returns
from src.signals.cot_signal import fetch_cot_history, build_signal, cot_index
from src.backtest.engine import run_backtest, compare_to_benchmark, apply_trend_filter
from src.analysis.signal_evaluation import ic_decay_analysis
from src.visuals.tearsheet import build_full_tearsheet

START_DATE = "2015-01-01"


def main():
    print("Fetching gold price history...")
    prices = fetch_price_history("GC=F", start=START_DATE)
    price_returns = compute_returns(prices)
    print(f"  Got {len(price_returns)} daily returns")

    print("\nFetching gold COT positioning data...")
    cot_data = fetch_cot_history("gold", start_date=START_DATE)

    print("\nRunning Information Coefficient analysis (continuous COT Index, not the discretized signal)...")
    commercial_cot_index = cot_index(cot_data["commercial_net"])
    ic_results = ic_decay_analysis(prices, commercial_cot_index, horizons=[5, 10, 20, 40, 60])
    print(ic_results.round(4))

    signal_df = build_signal(cot_data, trader_type="commercial", signal_direction="inverted")
    signal = signal_df["signal"]
    print(f"\n  Got {len(signal)} weekly COT reports")
    print(f"  Bullish weeks: {(signal == 1).sum()}, Bearish weeks: {(signal == -1).sum()}")

    print("\nRunning baseline backtest (standalone COT signal)...")
    baseline_result = run_backtest(
        price_returns, signal,
        publication_lag_days=3,
        position_mode="long_short",
        transaction_cost_bps=2,
    )
    baseline_comparison = compare_to_benchmark(baseline_result)
    print(baseline_comparison.round(4))

    print("\nRunning trend-filtered backtest (COT signal only when confirmed by 200-day trend)...")
    filtered_signal = apply_trend_filter(signal, prices, ma_window=200)
    print(f"  Bullish weeks after filter: {(filtered_signal == 1).sum()}, Bearish weeks after filter: {(filtered_signal == -1).sum()}")

    filtered_result = run_backtest(
        price_returns, filtered_signal,
        publication_lag_days=3,
        position_mode="long_short",
        transaction_cost_bps=2,
    )
    filtered_comparison = compare_to_benchmark(filtered_result)
    print(filtered_comparison.round(4))

    print("\nGenerating tearsheet visuals for the trend-filtered strategy...")
    build_full_tearsheet(filtered_result, ic_results, output_dir="data")

    baseline_result.to_csv("data/gold_cot_baseline_backtest.csv")
    filtered_result.to_csv("data/gold_cot_trend_filtered_backtest.csv")
    print("\nSaved backtest outputs and tearsheet visuals to data/")


if __name__ == "__main__":
    main()