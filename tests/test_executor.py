import pandas as pd
import numpy as np

from src.execution.executor import run_test_period, ExecConfig
from src.strategy.trend import generate_signals


def make_df(prices: np.ndarray) -> pd.DataFrame:
    close = pd.Series(prices, dtype=float)
    open_ = close.shift(1).fillna(close.iloc[0]) * (1 + 0.001)
    high = np.maximum(open_.values, close.values) * 1.01
    low = np.minimum(open_.values, close.values) * 0.99
    vol = np.full_like(prices, 1_000_000, dtype=float)
    df = pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close, "Adj Close": close, "Volume": vol}
    )
    idx = pd.date_range("2020-01-01", periods=len(df), freq="B")
    df.index = idx
    df.index.name = "Date"
    return df


def test_executor_basic_flow():
    # Uptrend followed by flat
    prices = np.concatenate([np.linspace(100, 120, 60), np.linspace(120, 120, 40)])
    df = make_df(prices)
    # Signals via TREND on same df (no warmup needed for unit test)
    sig = generate_signals(df, dict(short_window=5, long_window=20, momentum_window=5, atr_lookback=14, ticker="T"))
    # Align test_data and signals
    test_data = {"T": df}
    signals = {"T": sig}
    cfg = ExecConfig(max_positions=1, risk_per_trade=0.01, reserve_pct=0.0, notional_cap_per_name=1.0)
    eq, trades, daily = run_test_period(test_data, signals, cfg, starting_capital=100_000)

    assert len(eq) > 0
    assert daily["positions"].max() <= 1
    # There should be at least one entry recorded
    assert (trades.get("entry_reason", pd.Series([])) == "ENTRY").sum() >= 1


def test_executor_stop_first_tie_break():
    # Construct one day where both SL and TP would be hit
    prices = np.array([100, 100, 100, 100, 100], dtype=float)
    df = make_df(prices)
    # Override day with extreme high/low to simulate both thresholds
    target_day = df.index[3]
    df.loc[target_day, "High"] = 120
    df.loc[target_day, "Low"] = 80

    # Create one artificial signal that enters before target_day
    sig = pd.DataFrame(index=df.index, data={
        "buy": [0,1,0,0,0],
        "sell": [0,0,0,0,0],
        "score": [0,1,0,0,0],
        "next_open_date": df.index.shift(-1),
        "next_open": df["Open"].shift(-1),
        "atr": [2,2,2,2,2]
    })
    test_data = {"X": df}
    signals = {"X": sig}
    cfg = ExecConfig(max_positions=1, risk_per_trade=0.1, reserve_pct=0.0, notional_cap_per_name=1.0, stop_first=True)
    eq, trades, daily = run_test_period(test_data, signals, cfg, starting_capital=10_000)
    # Expect an exit with reason SL on target_day
    exit_reasons = trades.get("exit_reason", pd.Series([])).dropna().tolist()
    assert "SL" in exit_reasons


def test_executor_time_stop_next_open():
    prices = np.linspace(100, 110, 30)
    df = make_df(prices)
    sig = pd.DataFrame(index=df.index, data={
        "buy": [1] + [0]*(len(df)-1),
        "sell": [0]*len(df),
        "score": [1] + [0]*(len(df)-1),
        "next_open_date": df.index.shift(-1),
        "next_open": df["Open"].shift(-1),
        "atr": [1]*len(df)
    })
    test_data = {"Y": df}
    signals = {"Y": sig}
    cfg = ExecConfig(max_positions=1, risk_per_trade=0.05, time_stop_days=2, reserve_pct=0.0, notional_cap_per_name=1.0)
    eq, trades, daily = run_test_period(test_data, signals, cfg, starting_capital=10_000)
    # position should be exited due to time stop, execution at next open
    reasons = trades.get("exit_reason", pd.Series([])).dropna().tolist()
    assert "TIME" in reasons

