"""
COT (Commitment of Traders) positioning signal.

Pulls weekly futures positioning data from the CFTC's free public API
(Socrata, no key required) and builds a professional-grade positioning
signal combining two complementary measures:

1. COT Index (percentile rank): where current net positioning sits within
   its own rolling lookback window, on a 0-100 scale. This is the
   standard approach used in CTA/macro research (Larry Williams' COT
   Index, popularized further by Steve Briese) — it avoids assuming
   positioning is normally distributed, which a raw z-score would.

2. Rolling z-score: a secondary confirmation measure. An extreme reading
   on the COT Index alone can be noise; agreement between both measures
   is a stronger signal than either individually.

Default trader category is "commercial" (hedgers), since Briese's original
research found commercial positioning extremes had more predictive value
than large speculator positioning, which tends to be trend-following and
lag rather than lead price moves.
"""

import pandas as pd
import requests

CFTC_LEGACY_ENDPOINT = "https://publicreporting.cftc.gov/resource/6dca-aqww.json"

# CFTC market names for common instruments (legacy futures-only report).
# Extend as needed — search https://publicreporting.cftc.gov for exact names.
MARKET_NAMES = {
    "gold": "GOLD - COMMODITY EXCHANGE INC.",
    "eurusd": "EURO FX - CHICAGO MERCANTILE EXCHANGE",
    "sp500": "S&P 500 Consolidated - CHICAGO MERCANTILE EXCHANGE",
}


def search_market_names(keyword: str, limit: int = 20) -> list:
    """Look up the exact market_and_exchange_names string for a keyword,
    since CFTC naming is inconsistent and a wrong guess silently returns
    no data. Run this once for any new instrument before calling
    fetch_cot_history.

    Example: search_market_names("gold") -> ["GOLD - COMMODITY EXCHANGE INC.", ...]
    """
    params = {
        "$where": f"upper(market_and_exchange_names) like upper('%{keyword}%')",
        "$select": "distinct market_and_exchange_names",
        "$limit": limit,
    }
    response = requests.get(CFTC_LEGACY_ENDPOINT, params=params, timeout=30)
    response.raise_for_status()
    return [row["market_and_exchange_names"] for row in response.json()]


def fetch_cot_history(market_key: str, start_date: str, limit: int = 2000) -> pd.DataFrame:
    """Fetch historical weekly COT reports for one market.

    market_key: a key from MARKET_NAMES, or the exact CFTC market name string.
    start_date: "YYYY-MM-DD"

    Returns a DataFrame with one row per report date, including commercial
    and non-commercial long/short positions.
    """
    market_name = MARKET_NAMES.get(market_key, market_key)

    params = {
        "$where": f"market_and_exchange_names='{market_name}' AND report_date_as_yyyy_mm_dd >= '{start_date}'",
        "$order": "report_date_as_yyyy_mm_dd ASC",
        "$limit": limit,
    }

    response = requests.get(CFTC_LEGACY_ENDPOINT, params=params, timeout=30)
    response.raise_for_status()
    data = response.json()

    if not data:
        raise ValueError(f"No COT data returned for '{market_name}' — check the market name is exact")

    df = pd.DataFrame(data)
    df["report_date"] = pd.to_datetime(df["report_date_as_yyyy_mm_dd"])

    numeric_cols = [
        "comm_positions_long_all", "comm_positions_short_all",
        "noncomm_positions_long_all", "noncomm_positions_short_all",
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.sort_values("report_date").set_index("report_date")
    df["commercial_net"] = df["comm_positions_long_all"] - df["comm_positions_short_all"]
    df["noncommercial_net"] = df["noncomm_positions_long_all"] - df["noncomm_positions_short_all"]

    return df[["commercial_net", "noncommercial_net"]]


def cot_index(net_position: pd.Series, lookback_weeks: int = 156) -> pd.Series:
    """Percentile rank of current net positioning within its own rolling
    lookback window (default 156 weeks = 3 years), scaled 0-100.

    100 = most net-long the position has been in the lookback window.
    0 = most net-short. This is the standard 'COT Index' used in CTA
    and macro positioning research.
    """
    def pct_rank(window):
        current = window.iloc[-1]
        return (window < current).sum() / (len(window) - 1) * 100 if len(window) > 1 else 50

    return net_position.rolling(lookback_weeks, min_periods=26).apply(pct_rank, raw=False)


def rolling_zscore(net_position: pd.Series, lookback_weeks: int = 156) -> pd.Series:
    """Rolling z-score of net positioning, as a secondary confirmation
    measure alongside the COT Index.
    """
    rolling_mean = net_position.rolling(lookback_weeks, min_periods=26).mean()
    rolling_std = net_position.rolling(lookback_weeks, min_periods=26).std()
    return (net_position - rolling_mean) / rolling_std


def build_signal(
    cot_data: pd.DataFrame,
    trader_type: str = "commercial",
    lookback_weeks: int = 156,
    index_threshold: float = 20.0,
    zscore_threshold: float = 1.5,
) -> pd.DataFrame:
    """Combine the COT Index and rolling z-score into a single signal.

    Bullish (+1): COT Index above (100 - index_threshold) AND z-score above
    zscore_threshold — commercials are unusually net-long by both measures.
    Bearish (-1): COT Index below index_threshold AND z-score below
    -zscore_threshold — unusually net-short by both measures.
    Neutral (0): otherwise, or when the two measures disagree.

    Requiring agreement between both measures is what makes this more
    robust than a single-metric signal — an extreme percentile rank with
    no z-score confirmation is more likely to be noise.
    """
    net_col = "commercial_net" if trader_type == "commercial" else "noncommercial_net"
    net_position = cot_data[net_col]

    result = pd.DataFrame(index=cot_data.index)
    result["net_position"] = net_position
    result["cot_index"] = cot_index(net_position, lookback_weeks)
    result["zscore"] = rolling_zscore(net_position, lookback_weeks)

    bullish = (result["cot_index"] >= 100 - index_threshold) & (result["zscore"] >= zscore_threshold)
    bearish = (result["cot_index"] <= index_threshold) & (result["zscore"] <= -zscore_threshold)

    result["signal"] = 0
    result.loc[bullish, "signal"] = 1
    result.loc[bearish, "signal"] = -1

    return result


if __name__ == "__main__":
    cot_data = fetch_cot_history("gold", start_date="2015-01-01")
    signal_df = build_signal(cot_data, trader_type="commercial")

    print(signal_df.tail(10))
    print(f"\nBullish weeks: {(signal_df['signal'] == 1).sum()}")
    print(f"Bearish weeks: {(signal_df['signal'] == -1).sum()}")