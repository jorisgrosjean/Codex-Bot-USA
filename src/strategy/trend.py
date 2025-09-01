from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any

import pandas as pd

from src.indicators.technical import sma, roc, atr


@dataclass
class TrendParams:
    short_window: int = 20
    long_window: int = 100
    momentum_window: int = 63
    atr_lookback: int = 14
    ltm_window: int | None = None  # optional long-term filter (e.g., 200)


def generate_signals(df_full: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
    """
    Generate TREND-following signals.

    Input df_full columns: ['Open','High','Low','Close','Adj Close','Volume']
    Returns DataFrame indexed by date with columns:
      - buy, sell (0/1)
      - score (float)
      - ticker (str) if provided in params (optional)
      - next_open_date (Timestamp), next_open (float)
      - atr (float) auxiliary
    """
    p = TrendParams(
        short_window=int(params.get("short_window", 20)),
        long_window=int(params.get("long_window", 100)),
        momentum_window=int(params.get("momentum_window", 63)),
        atr_lookback=int(params.get("atr_lookback", params.get("atr_length", 14))),
        ltm_window=params.get("ltm_window"),
    )

    df = df_full.copy()
    adj = df["Adj Close"].astype(float)

    sma_s = sma(adj, p.short_window)
    sma_l = sma(adj, p.long_window)
    mom = roc(adj, p.momentum_window)
    a = atr(df["High"].astype(float), df["Low"].astype(float), adj, length=p.atr_lookback)

    # Basic trend conditions
    trend_up = sma_s > sma_l
    trend_down = sma_s < sma_l
    momentum_ok = mom > 0

    # Optional long-term market filter per instrument
    if p.ltm_window is not None and p.ltm_window > 0:
        ltm = sma(adj, p.ltm_window)
        ltm_ok = adj > ltm
    else:
        ltm_ok = pd.Series(True, index=df.index)

    # Signals: buy on cross-up with momentum confirmation; sell on cross-down
    cross_up = trend_up & (~trend_up.shift(1).fillna(False))
    cross_down = trend_down & (~trend_down.shift(1).fillna(False))

    buy = ((trend_up & momentum_ok) | cross_up) & ltm_ok
    sell = cross_down

    # Score: use momentum magnitude; fallback to 0 when NaN
    score = mom.fillna(0.0)

    # Execution metadata
    df_out = pd.DataFrame(index=df.index)
    df_out["buy"] = buy.astype(int)
    df_out["sell"] = sell.astype(int)
    df_out["score"] = score.astype(float)
    if "ticker" in params:
        df_out["ticker"] = str(params["ticker"])
    df_out["atr"] = a
    df_out["next_open_date"] = df.index.shift(-1, freq=None)
    df_out["next_open"] = df["Open"].shift(-1)

    # Drop last row (no next open) and rows where indicators are not ready
    ready = (~df_out[["score", "atr", "next_open"]].isna().any(axis=1))
    df_out = df_out.loc[ready]

    return df_out


def main():  # pragma: no cover - simple CLI for demo
    import typer
    from src.data.provider import get_data_with_warmup

    app = typer.Typer(add_completion=False, no_args_is_help=True)

    @app.command()
    def run(
        ticker: str = typer.Option(...),
        start: str = typer.Option(...),
        end: str = typer.Option(...),
        cache_dir: str = typer.Option("cache"),
        short_window: int = typer.Option(20),
        long_window: int = typer.Option(100),
        momentum_window: int = typer.Option(63),
        atr_lookback: int = typer.Option(14),
        ltm_window: int = typer.Option(0, help="0 disables long-term filter"),
        head: int = typer.Option(5, help="Show first N rows"),
        tail: int = typer.Option(5, help="Show last N rows"),
        save_csv: str | None = typer.Option(None, help="Path to save CSV of signals"),
    ):
        df_full, _ = get_data_with_warmup(
            ticker, start=start, end=end, cache_dir=cache_dir, retries=3, retry_backoff_sec=2, warmup_days=252
        )
        params = dict(
            short_window=short_window,
            long_window=long_window,
            momentum_window=momentum_window,
            atr_lookback=atr_lookback,
            ltm_window=(ltm_window if ltm_window > 0 else None),
            ticker=ticker,
        )
        sig = generate_signals(df_full, params)
        print("-- HEAD --")
        print(sig.head(head))
        print("-- TAIL --")
        print(sig.tail(tail))
        if save_csv:
            sig.to_csv(save_csv)
            print(f"Saved signals to {save_csv}")

    app()


if __name__ == "__main__":  # pragma: no cover
    main()

