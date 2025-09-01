from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, Optional, List

import pandas as pd
from pandas.tseries.offsets import BDay


try:
    import yfinance as yf
except Exception as e:  # pragma: no cover - yfinance may be missing in some envs
    yf = None  # type: ignore


PARQUET_COLUMNS = [
    "Open",
    "High",
    "Low",
    "Close",
    "Adj Close",
    "Volume",
]


@dataclass
class ProviderConfig:
    cache_dir: Path = Path("cache")
    retries: int = 3
    retry_backoff_sec: int = 2


def _ensure_cache_dir(cache_dir: Path) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)


def _cache_path(cache_dir: Path, ticker: str) -> Path:
    safe = ticker.replace("/", "-")
    return cache_dir / f"{safe}.parquet"


def _normalize_index(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    # Ensure datetime index, tz-naive, sorted, named 'Date'
    if not isinstance(df.index, pd.DatetimeIndex):
        if "Date" in df.columns:
            df = df.set_index("Date")
        else:
            df.index = pd.to_datetime(df.index)
    idx: pd.DatetimeIndex = pd.DatetimeIndex(df.index)
    if idx.tz is not None:
        idx = idx.tz_convert(None)
    df.index = idx
    df.index.name = "Date"
    df = df.sort_index()
    # Keep only expected columns if present
    cols = [c for c in PARQUET_COLUMNS if c in df.columns]
    if cols:
        df = df[cols]
    return df


def _download_range(
    ticker: str, start: pd.Timestamp, end: pd.Timestamp
) -> pd.DataFrame:
    if yf is None:
        raise RuntimeError(
            "yfinance is not available. Install it or run in offline mode with cache."
        )
    # yfinance end is exclusive; add 1 day to be inclusive
    end_plus = (end + pd.Timedelta(days=1)).date()
    df = yf.Ticker(ticker).history(
        start=start.date(), end=end_plus, interval="1d", auto_adjust=False
    )
    df = _normalize_index(df)
    if df.empty:
        return df
    # Ensure all expected columns exist
    for c in PARQUET_COLUMNS:
        if c not in df.columns:
            if c == "Adj Close" and "Close" in df.columns:
                df[c] = df["Close"]
            else:
                df[c] = pd.NA
    df = df[PARQUET_COLUMNS]
    return df


def _retry_download(
    ticker: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    retries: int,
    backoff_sec: int,
) -> pd.DataFrame:
    last_exc: Optional[Exception] = None
    for attempt in range(retries + 1):
        try:
            return _download_range(ticker, start, end)
        except Exception as e:  # network/transient issues
            last_exc = e
            if attempt >= retries:
                break
            time.sleep(backoff_sec * (2 ** attempt))
    assert last_exc is not None
    raise last_exc


def _merge_dedup(df_a: pd.DataFrame, df_b: pd.DataFrame) -> pd.DataFrame:
    if df_a.empty:
        return df_b.copy()
    if df_b.empty:
        return df_a.copy()
    df = pd.concat([df_a, df_b])
    df = df[~df.index.duplicated(keep="last")]
    df = df.sort_index()
    # Re-order columns
    cols = [c for c in PARQUET_COLUMNS if c in df.columns]
    return df[cols]


def get_data(
    ticker: str,
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
    cache_dir: str | Path = "cache",
    retries: int = 3,
    retry_backoff_sec: int = 2,
) -> pd.DataFrame:
    """
    Return historical OHLCV (daily) for [start, end] inclusive using local Parquet cache.
    Downloads only missing edges and updates cache.
    """
    cache_dir_p = Path(cache_dir)
    _ensure_cache_dir(cache_dir_p)
    start_ts = pd.to_datetime(start)
    end_ts = pd.to_datetime(end)

    path = _cache_path(cache_dir_p, ticker)
    if path.exists():
        try:
            df_cache = pd.read_parquet(path)
            df_cache = _normalize_index(df_cache)
        except Exception:
            df_cache = pd.DataFrame()
    else:
        df_cache = pd.DataFrame()

    # Determine missing edges
    need_downloads: List[Tuple[pd.Timestamp, pd.Timestamp]] = []
    if df_cache.empty:
        need_downloads.append((start_ts, end_ts))
    else:
        cmin, cmax = df_cache.index.min(), df_cache.index.max()
        if start_ts < cmin:
            need_downloads.append((start_ts, cmin - pd.Timedelta(days=1)))
        if end_ts > cmax:
            need_downloads.append((cmax + pd.Timedelta(days=1), end_ts))

    df_final = df_cache
    for s, e in need_downloads:
        if s <= e:
            df_new = _retry_download(ticker, s, e, retries, retry_backoff_sec)
            df_final = _merge_dedup(df_final, df_new)

    # Persist cache if updated
    if not df_final.equals(df_cache):
        df_final.to_parquet(path)

    # Slice to requested range (inclusive)
    if not df_final.empty:
        df_final = df_final.loc[(df_final.index >= start_ts) & (df_final.index <= end_ts)]

    return df_final


def get_data_with_warmup(
    ticker: str,
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
    cache_dir: str | Path = "cache",
    retries: int = 3,
    retry_backoff_sec: int = 2,
    warmup_days: int = 252,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns (df_full, df_exec):
    - df_full: [start - warmup_days (business), end]
    - df_exec: [start, end]
    """
    start_ts = pd.to_datetime(start)
    end_ts = pd.to_datetime(end)
    start_warm = (start_ts - BDay(warmup_days)).normalize()
    df_full = get_data(
        ticker,
        start=start_warm,
        end=end_ts,
        cache_dir=cache_dir,
        retries=retries,
        retry_backoff_sec=retry_backoff_sec,
    )
    df_exec = df_full.loc[(df_full.index >= start_ts) & (df_full.index <= end_ts)]
    return df_full, df_exec


# ---- CLI (prefetch) ----
def _split_tickers(tickers: str) -> List[str]:
    return [t.strip() for t in tickers.split(",") if t.strip()]


def _prefetch(
    tickers: List[str],
    start: str,
    end: str,
    cache_dir: str,
    retries: int,
    backoff: int,
) -> None:
    cache = Path(cache_dir)
    _ensure_cache_dir(cache)
    for t in tickers:
        df = get_data(t, start=start, end=end, cache_dir=cache, retries=retries, retry_backoff_sec=backoff)
        print(f"[prefetch] {t}: {len(df):,} rows cached ({df.index.min()} → {df.index.max()})")


def main():  # pragma: no cover - CLI wrapper
    import typer

    app = typer.Typer(add_completion=False, no_args_is_help=True)

    @app.command()
    def prefetch(
        tickers: str = typer.Option(..., help="Comma-separated list, e.g., AAPL,MSFT"),
        start: str = typer.Option(..., help="Start date YYYY-MM-DD"),
        end: str = typer.Option(..., help="End date YYYY-MM-DD"),
        cache_dir: str = typer.Option("cache", help="Cache directory"),
        retries: int = typer.Option(3, help="Download retries"),
        backoff: int = typer.Option(2, help="Backoff base seconds"),
    ):
        """Download and cache daily data for all tickers over the period."""
        _prefetch(_split_tickers(tickers), start, end, cache_dir, retries, backoff)

    app()


if __name__ == "__main__":  # pragma: no cover
    main()

