"""
Run the COT signal validation and backtest framework across multiple
markets, to see whether the gold result generalizes or is idiosyncratic.

Signal direction per market is set by a single, pre-specified rule
(average IC sign) decided before any backtest is run — not chosen after
seeing which direction produces a better-looking result, since that would
be a subtle form of overfitting despite testing several markets.

Run with:
    python -m src.run_multi_market_strategy
"""

import pandas as pd

from src.data.loaders import fetch_price_history, compute_returns
from src.signals.cot_signal import fetch_cot_history, build_signal, cot_index
from src.backtest.engine import run_backtest, compare_to_benchmark, apply_trend_filter
from src.analysis.signal_evaluation import ic_decay_analysis
from src.visuals.tearsheet import build_full_tearsheet

START_DATE = "2015-01-01"

MARKETS = [
    {"name": "gold", "yahoo_ticker": "GC=F", "cot_key": "gold"},
    {"name": "eurusd", "yahoo_ticker": "EURUSD=X", "cot_key": "eurusd"},
    {"name": "usdjpy", "yahoo_ticker": "JPY=X", "cot_key": "usdjpy"},
]


def determine_direction(ic_df: pd.DataFrame) -> str:
    """Pre-specified rule: signal direction is set by the sign of the
    average IC across all horizons, decided before running any backtest.
    Applied identically to every market to avoid picking a direction
    per-market based on which one looks better after the fact.
    """
    return "standard" if ic_df["spearman_ic"].mean() > 0 else "inverted"


def run_market(market_config: dict) -> dict:
    name = market_config["name"]
    print(f"\nRunning {name}...")

    print(f"Fetching {name} price history...")
    prices = fetch_price_history(market_config["yahoo_ticker"], start=START_DATE)
    price_returns = compute_returns(prices)
    print(f"  Got {len(price_returns)} daily returns")

    print(f"Fetching {name} COT positioning data...")
    cot_data = fetch_cot_history(market_config["cot_key"], start_date=START_DATE)

    commercial_cot_index = cot_index(cot_data["commercial_net"])
    ic_results = ic_decay_analysis(prices, commercial_cot_index, horizons=[5, 10, 20, 40, 60])
    print("Information Coefficient by horizon:")
    print(ic_results.round(4))

    direction = determine_direction(ic_results)
    print(f"Signal direction (pre-specified rule, avg IC sign): {direction}")

    signal_df = build_signal(cot_data, trader_type="commercial", signal_direction=direction)
    signal = signal_df["signal"]

    baseline_result = run_backtest(
        price_returns, signal, publication_lag_days=3,
        position_mode="long_short", transaction_cost_bps=2,
    )
    baseline_comparison = compare_to_benchmark(baseline_result)

    filtered_signal = apply_trend_filter(signal, prices, ma_window=200)
    filtered_result = run_backtest(
        price_returns, filtered_signal, publication_lag_days=3,
        position_mode="long_short", transaction_cost_bps=2,
    )

    # Guard against degenerate statistics: with too few active trading days,
    # the return series is mostly zeros plus a handful of outliers, which
    # produces unstable/meaningless higher-moment stats (extreme kurtosis,
    # NaN tail ratios). Flag this explicitly rather than reporting numbers
    # that look precise but aren't meaningful.
    n_active_days = (filtered_result["position"] != 0).sum()
    min_active_days = 30
    filtered_reliable = n_active_days >= min_active_days

    if not filtered_reliable:
        print(f"Warning: only {n_active_days} active trading days after the trend filter "
              f"(below the {min_active_days}-day reliability threshold) — "
              f"filtered statistics below are not reliable and are excluded from the summary.")

    filtered_comparison = compare_to_benchmark(filtered_result)

    print(f"\n{name} baseline vs. buy-and-hold:")
    print(baseline_comparison.round(4))
    print(f"\n{name} trend-filtered vs. buy-and-hold:")
    print(filtered_comparison.round(4))

    output_dir = f"data/{name}"
    import os
    os.makedirs(output_dir, exist_ok=True)
    build_full_tearsheet(filtered_result, ic_results, output_dir=output_dir)
    filtered_result.to_csv(f"{output_dir}/backtest.csv")

    return {
        "market": name,
        "direction": direction,
        "baseline_sharpe": baseline_comparison.loc["Sharpe ratio", "Strategy"],
        "filtered_sharpe": filtered_comparison.loc["Sharpe ratio", "Strategy"] if filtered_reliable else None,
        "filtered_active_days": n_active_days,
        "filtered_reliable": filtered_reliable,
        "benchmark_sharpe": baseline_comparison.loc["Sharpe ratio", "Buy & Hold"],
        "avg_ic": ic_results["spearman_ic"].mean(),
        "ic_60d_pvalue": ic_results.loc[60, "spearman_pvalue"] if 60 in ic_results.index else None,
    }


def main():
    summary_rows = []
    for market_config in MARKETS:
        try:
            summary_rows.append(run_market(market_config))
        except Exception as e:
            print(f"Skipped {market_config['name']}: {e}")

    print("\nSummary across markets:")
    summary_df = pd.DataFrame(summary_rows).set_index("market")
    print(summary_df.round(4))
    summary_df.to_csv("data/multi_market_summary.csv")
    print("Saved summary to data/multi_market_summary.csv")


if __name__ == "__main__":
    main()