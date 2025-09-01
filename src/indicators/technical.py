from __future__ import annotations

from typing import Tuple, Optional

import numpy as np
import pandas as pd


def sma(s: pd.Series, window: int, min_periods: Optional[int] = None) -> pd.Series:
    """Simple Moving Average."""
    if min_periods is None:
        min_periods = window
    return s.rolling(window=window, min_periods=min_periods).mean()


def ema(
    s: pd.Series,
    span: Optional[int] = None,
    alpha: Optional[float] = None,
    adjust: bool = False,
) -> pd.Series:
    """Exponential Moving Average. Provide either span or alpha."""
    if span is None and alpha is None:
        raise ValueError("Provide either span or alpha for EMA")
    return s.ewm(span=span, alpha=alpha, adjust=adjust).mean()


def roc(s: pd.Series, periods: int = 1) -> pd.Series:
    """Rate of Change = s / s.shift(periods) - 1."""
    return s / s.shift(periods) - 1.0


def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    # For first row, prev_close is NaN; fallback to high-low
    tr.iloc[0] = tr1.iloc[0]
    return tr


def atr(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    length: int = 14,
    method: str = "wilder",
) -> pd.Series:
    """
    Average True Range.
    - method 'wilder' uses alpha=1/length (Wilder's smoothing)
    - method 'sma' uses simple rolling mean of TR
    """
    tr = true_range(high, low, close)
    if method == "wilder":
        return tr.ewm(alpha=1.0 / float(length), adjust=False).mean()
    elif method == "sma":
        return tr.rolling(window=length, min_periods=length).mean()
    else:
        raise ValueError("method must be 'wilder' or 'sma'")


def rsi(close: pd.Series, length: int = 14) -> pd.Series:
    """Relative Strength Index (Wilder)."""
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1.0 / float(length), adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / float(length), adjust=False).mean()
    rs = avg_gain / avg_loss
    rsi_val = 100.0 - (100.0 / (1.0 + rs))
    # Handle zero-division edge cases
    rsi_val = rsi_val.where(avg_loss != 0, 100.0)
    rsi_val = rsi_val.where(avg_gain != 0, 0.0)
    return rsi_val


def donchian(
    high: pd.Series,
    low: pd.Series,
    window: int = 20,
    include_current: bool = True,
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Donchian channel (upper, lower, middle).
    - include_current=True uses rolling over the last `window` sessions including today.
    - include_current=False uses the previous `window` sessions (shifted by 1).
    """
    if include_current:
        upper = high.rolling(window=window, min_periods=window).max()
        lower = low.rolling(window=window, min_periods=window).min()
    else:
        upper = high.shift(1).rolling(window=window, min_periods=window).max()
        lower = low.shift(1).rolling(window=window, min_periods=window).min()
    middle = (upper + lower) / 2.0
    return upper, lower, middle


def zscore(s: pd.Series, window: int = 20, ddof: int = 1) -> pd.Series:
    """Rolling z-score of a series."""
    roll_mean = s.rolling(window=window, min_periods=window).mean()
    roll_std = s.rolling(window=window, min_periods=window).std(ddof=ddof)
    z = (s - roll_mean) / roll_std
    z = z.where(roll_std != 0, 0.0)
    return z


def returns(s: pd.Series, periods: int = 1) -> pd.Series:
    """Simple returns: s/s.shift(periods)-1."""
    return s.pct_change(periods=periods)
