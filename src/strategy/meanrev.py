from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any

import pandas as pd

from src.indicators.technical import rsi, sma, zscore, returns, atr


@dataclass
class MeanRevParams:
    rsi_length: int = 14
    rsi_buy: int = 30
    z_len: int = 5
    atr_lookback: int = 14


def generate_signals(df_full: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
    """
    Mean-Reversion signals: buy on oversold RSI recovering with a simple confirmation.
    - RSI length and threshold from params.
    - Confirmation: Close >= SMA(5) (indicative rebound).
    - Score: stronger when recent z-score of returns is more negative.
    """
    p = MeanRevParams(
        rsi_length=int(params.get("rsi_length", 14)),
        rsi_buy=int(params.get("rsi_buy", 30)),
        z_len=int(params.get("z_len", 5)),
        atr_lookback=int(params.get("atr_lookback", params.get("atr_length", 14))),
    )

    df = df_full.copy()
    close = df["Adj Close"].astype(float)
    r = rsi(close, length=p.rsi_length)
    sma5 = sma(close, 5)
    # Oversold then rebound confirmation
    cond_rsi_up = (r.shift(1) < p.rsi_buy) & (close >= sma5)

    # z-score of simple returns (negative = more oversold)
    ret = returns(close, 1)
    z = zscore(ret, window=p.z_len)
    score = (-z).clip(lower=0).fillna(0.0)

    a = atr(df["High"].astype(float), df["Low"].astype(float), close, length=p.atr_lookback)

    out = pd.DataFrame(index=df.index)
    out["buy"] = cond_rsi_up.astype(int)
    out["sell"] = 0
    out["score"] = score
    if "ticker" in params:
        out["ticker"] = str(params["ticker"])
    out["atr"] = a
    out["next_open_date"] = df.index.to_series().shift(-1)
    out["next_open"] = df["Open"].shift(-1)
    ready = (~out[["score", "atr", "next_open"]].isna().any(axis=1))
    out = out.loc[ready]
    return out

