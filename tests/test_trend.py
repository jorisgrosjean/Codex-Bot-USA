import pandas as pd
import numpy as np

from src.strategy.trend import generate_signals


def make_ohlcv(prices: np.ndarray) -> pd.DataFrame:
    # Create a simplistic OHLCV from close prices
    close = pd.Series(prices)
    open_ = close.shift(1).fillna(close.iloc[0]) * (1 + 0.001)
    high = np.maximum(open_.values, close.values) * 1.002
    low = np.minimum(open_.values, close.values) * 0.998
    vol = np.full_like(prices, 1_000_000, dtype=float)
    df = pd.DataFrame(
        {
            "Open": open_.values,
            "High": high,
            "Low": low,
            "Close": close.values,
            "Adj Close": close.values,
            "Volume": vol,
        }
    )
    idx = pd.date_range("2020-01-01", periods=len(df), freq="B")
    df.index = idx
    df.index.name = "Date"
    return df


def test_trend_signals_basic_uptrend():
    # Build an upward trending series
    prices = np.linspace(100, 150, 300)
    df = make_ohlcv(prices)
    params = dict(short_window=10, long_window=30, momentum_window=15, atr_lookback=14, ticker="TEST")
    sig = generate_signals(df, params)
    # Should produce some buys, and score should be non-negative on last part
    assert (sig["buy"] == 1).sum() > 0
    # Last rows are in uptrend -> likely buy
    assert sig.tail(50)["buy"].mean() >= 0.2
    # next_open present and aligned
    assert sig["next_open"].isna().sum() == 0


def test_trend_crosses():
    # Construct a series with a clear cross up then down
    up = np.linspace(100, 120, 60)
    flat = np.linspace(120, 120, 20)
    down = np.linspace(120, 110, 60)
    prices = np.concatenate([up, flat, down])
    df = make_ohlcv(prices)
    params = dict(short_window=5, long_window=20, momentum_window=5)
    sig = generate_signals(df, params)
    # Ensure that at least one cross-up buy and later a cross-down sell exist
    assert (sig["buy"] == 1).any()
    assert (sig["sell"] == 1).any()

