import numpy as np
import pandas as pd

from src.indicators.technical import sma, ema, roc, true_range, atr, rsi, donchian, zscore, returns


def test_sma_ema_basic():
    s = pd.Series([1, 2, 3, 4, 5], dtype=float)
    sma3 = sma(s, 3)
    assert np.isnan(sma3.iloc[1])
    assert sma3.iloc[2] == 2.0
    # EMA with span=3 should match pandas ewm
    ema3 = ema(s, span=3, adjust=False)
    ema3_ref = s.ewm(span=3, adjust=False).mean()
    pd.testing.assert_series_equal(ema3, ema3_ref)


def test_roc_and_returns():
    s = pd.Series([10, 11, 12.1, 12.1], dtype=float)
    r1 = roc(s, 1)
    r1_ref = s / s.shift(1) - 1
    pd.testing.assert_series_equal(r1, r1_ref)
    rret = returns(s, 1)
    pd.testing.assert_series_equal(rret, r1_ref)


def test_true_range_and_atr_sma():
    high = pd.Series([10, 12, 13, 14], dtype=float)
    low = pd.Series([8, 10, 11, 12], dtype=float)
    close = pd.Series([9, 11, 12, 13], dtype=float)
    tr = true_range(high, low, close)
    # day1 tr = high-low = 2
    assert tr.iloc[0] == 2
    # day2 max(12-10=2, |12-9|=3, |10-9|=1) = 3
    assert tr.iloc[1] == 3
    atr2 = atr(high, low, close, length=2, method="sma")
    # rolling mean of TR over 2 periods: [nan, (2+3)/2=2.5, (3+x)/2, ...]
    assert np.isnan(atr2.iloc[0])
    assert atr2.iloc[1] == 2.5


def test_rsi_extremes():
    # strictly increasing close -> RSI tends to 100
    close_up = pd.Series(np.arange(1, 51), dtype=float)
    r = rsi(close_up, length=14)
    assert r.iloc[-1] > 90
    # strictly decreasing close -> RSI tends to 0
    close_dn = pd.Series(np.arange(50, 0, -1), dtype=float)
    r2 = rsi(close_dn, length=14)
    assert r2.iloc[-1] < 10


def test_donchian_basic():
    high = pd.Series([1, 2, 3, 4, 5], dtype=float)
    low = pd.Series([1, 1, 1, 1, 1], dtype=float)
    up, lo, mid = donchian(high, low, window=3, include_current=True)
    assert np.isnan(up.iloc[1])
    assert up.iloc[2] == 3
    assert lo.iloc[2] == 1
    assert mid.iloc[2] == 2


def test_zscore_basic():
    s = pd.Series([1, 2, 3, 4, 5], dtype=float)
    z = zscore(s, window=3)
    assert np.isnan(z.iloc[1])
    # For [1,2,3] mean=2 std=1 => z for 3rd elem (3) = 1.0
    assert z.iloc[2] == 1.0

