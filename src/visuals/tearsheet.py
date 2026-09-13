"""
Tearsheet-style visualizations for strategy performance, in the spirit of
institutional tools like pyfolio/quantstats.

Each function returns a matplotlib Figure and optionally saves it to disk.
Static matplotlib/seaborn plots are used rather than interactive charts,
since this is the standard format for research reports and CV-facing
work — the reader can drop these straight into a PDF or README.
"""

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import pandas as pd
import numpy as np

from src.performance.metrics import drawdown_series, rolling_sharpe

STYLE_PARAMS = {
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": "#333333",
    "axes.grid": True,
    "grid.color": "#e0e0e0",
    "grid.linewidth": 0.6,
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
}

COLOR_STRATEGY = "#1f77b4"
COLOR_BENCHMARK = "#7f7f7f"


def _apply_style():
    plt.rcParams.update(STYLE_PARAMS)


def plot_equity_curve(result: pd.DataFrame, log_scale: bool = True, save_path: str = None):
    """Cumulative growth of $1 for the strategy vs. buy-and-hold."""
    _apply_style()
    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(result.index, result["strategy_equity"], label="Strategy", color=COLOR_STRATEGY, linewidth=1.5)
    ax.plot(result.index, result["benchmark_equity"], label="Buy & Hold", color=COLOR_BENCHMARK,
            linewidth=1.5, linestyle="--")

    if log_scale:
        ax.set_yscale("log")
        # Default log-scale scientific notation looks garbled over a small
        # value range (e.g. 0.8x to 4x) — force plain "Nx" labels on both
        # major and minor ticks instead
        formatter = mticker.FuncFormatter(lambda x, _: f"{x:.1f}x")
        ax.yaxis.set_major_formatter(formatter)
        ax.yaxis.set_minor_formatter(formatter)
        ax.yaxis.set_minor_locator(mticker.LogLocator(subs=(2, 3, 4, 5, 6, 7, 8, 9)))

    ax.set_title("Cumulative Growth of $1")
    ax.set_ylabel("Portfolio value (log scale)" if log_scale else "Portfolio value")
    ax.legend(frameon=False)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_drawdown(result: pd.DataFrame, save_path: str = None):
    """Underwater plot: how far below the running peak each series is,
    at every point in time. Shows drawdown depth AND duration at a glance,
    which a single max-drawdown number can't convey.
    """
    _apply_style()
    fig, ax = plt.subplots(figsize=(10, 4))

    dd_strategy = drawdown_series(result["strategy_return"]) * 100
    dd_benchmark = drawdown_series(result["price_return"]) * 100

    ax.fill_between(dd_strategy.index, dd_strategy, 0, color=COLOR_STRATEGY, alpha=0.4, label="Strategy")
    ax.plot(dd_benchmark.index, dd_benchmark, color=COLOR_BENCHMARK, linewidth=1.2,
            linestyle="--", label="Buy & Hold")

    ax.set_title("Drawdown (Underwater Plot)")
    ax.set_ylabel("Drawdown (%)")
    ax.legend(frameon=False)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_return_distribution(result: pd.DataFrame, save_path: str = None):
    """Violin plot comparing the full daily return distribution of the
    strategy vs. buy-and-hold. Unlike a single skew/kurtosis number, this
    shows the actual shape — fat tails, asymmetry, and concentration
    around zero are all visible directly.
    """
    _apply_style()

    data = pd.DataFrame({
        "Strategy": result["strategy_return"],
        "Buy & Hold": result["price_return"],
    }).melt(var_name="Series", value_name="Daily return")

    fig, ax = plt.subplots(figsize=(7, 6))
    sns.violinplot(
        data=data, x="Series", y="Daily return", hue="Series", ax=ax,
        inner="quartile", palette=[COLOR_STRATEGY, COLOR_BENCHMARK], legend=False,
    )
    ax.axhline(0, color="#999999", linewidth=0.8, linestyle=":")
    ax.set_title("Daily Return Distribution")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_rolling_sharpe(returns: pd.Series, window: int = 63, save_path: str = None):
    """Rolling Sharpe ratio over time. A strategy with a decent overall
    Sharpe that's actually one lucky quarter looks very different here
    than one with genuinely stable risk-adjusted performance.
    """
    _apply_style()
    fig, ax = plt.subplots(figsize=(10, 4))

    rs = rolling_sharpe(returns, window)
    ax.plot(rs.index, rs, color=COLOR_STRATEGY, linewidth=1.2)
    ax.axhline(0, color="#999999", linewidth=0.8, linestyle=":")

    ax.set_title(f"Rolling {window}-Day Sharpe Ratio")
    ax.set_ylabel("Sharpe ratio")
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_ic_decay(ic_df: pd.DataFrame, save_path: str = None):
    """Bar chart of Spearman IC across forward horizons, with bars colored
    by statistical significance (p < 0.05) so a reader can see at a
    glance which horizons, if any, showed a real relationship.
    """
    _apply_style()
    fig, ax = plt.subplots(figsize=(8, 5))

    colors = ["#d62728" if p < 0.05 else "#c7c7c7" for p in ic_df["spearman_pvalue"]]
    ax.bar(ic_df.index.astype(str), ic_df["spearman_ic"], color=colors)
    ax.axhline(0, color="#333333", linewidth=0.8)

    ax.set_title("Information Coefficient by Forward Horizon")
    ax.set_xlabel("Horizon (trading days)")
    ax.set_ylabel("Spearman IC")

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#d62728", label="p < 0.05"),
        Patch(facecolor="#c7c7c7", label="not significant"),
    ]
    ax.legend(handles=legend_elements, frameon=False)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def build_full_tearsheet(result: pd.DataFrame, ic_df: pd.DataFrame, output_dir: str = "data"):
    """Generate and save all tearsheet visuals in one call."""
    plot_equity_curve(result, save_path=f"{output_dir}/equity_curve.png")
    plot_drawdown(result, save_path=f"{output_dir}/drawdown.png")
    plot_return_distribution(result, save_path=f"{output_dir}/return_distribution.png")
    plot_rolling_sharpe(result["strategy_return"], save_path=f"{output_dir}/rolling_sharpe.png")
    if ic_df is not None:
        plot_ic_decay(ic_df, save_path=f"{output_dir}/ic_decay.png")
    print(f"Saved tearsheet visuals to {output_dir}/")


if __name__ == "__main__":
    # Smoke test with synthetic data
    np.random.seed(1)
    dates = pd.date_range("2020-01-01", periods=1000, freq="D")
    strategy_returns = pd.Series(np.random.normal(0.0002, 0.008, 1000), index=dates)
    benchmark_returns = pd.Series(np.random.normal(0.0004, 0.012, 1000), index=dates)

    result = pd.DataFrame({
        "strategy_return": strategy_returns,
        "price_return": benchmark_returns,
        "strategy_equity": (1 + strategy_returns).cumprod(),
        "benchmark_equity": (1 + benchmark_returns).cumprod(),
    })

    fake_ic = pd.DataFrame({
        "spearman_ic": [0.05, -0.02, 0.08, 0.03, -0.01],
        "spearman_pvalue": [0.03, 0.6, 0.01, 0.2, 0.7],
    }, index=pd.Index([5, 10, 20, 40, 60], name="horizon_days"))

    build_full_tearsheet(result, fake_ic, output_dir="data")