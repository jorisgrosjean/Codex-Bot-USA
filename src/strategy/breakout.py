from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any

import pandas as pd

from src.indicators.technical import donchian, atr


@dataclass
class BreakoutParams:
    donchian_window: int = 55
    atr_lookback: int = 14


def generate_signals(df_full: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
    """
    Breakout signals: buy when Close breaks above Donchian upper band (previous highs).
    Score proportional to breakout magnitude over prior upper band.
    """
    wnd = int(params.get("donchian", params.get("donchian_window", 55)))
    atr_lb = int(params.get("atr_lookback", params.get("atr_length", 14)))

    df = df_full.copy()
    close = df["Adj Close"].astype(float)
    high = df["High"].astype(float)
    low = df["Low"].astype(float)

    # Use previous upper band for signal/score stability
    upper_cur, lower_cur, _ = donchian(high, low, window=wnd, include_current=True)
    upper_prev, _, _ = donchian(high, low, window=wnd, include_current=False)

    buy = (close >= upper_cur)
    # Breakout distance vs previous upper band
    prev_up = upper_prev
    score = (close / prev_up - 1.0).clip(lower=0).fillna(0.0)

    a = atr(high, low, close, length=atr_lb)

    out = pd.DataFrame(index=df.index)
    out["buy"] = buy.astype(int)
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

