from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any

import pandas as pd
import typer

from .provider import get_data, PARQUET_COLUMNS


@dataclass
class QCResult:
    ticker: str
    rows: int
    start: pd.Timestamp | None
    end: pd.Timestamp | None
    issues: List[str]
    stats: Dict[str, Any]


def _sessions_range(calendar: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DatetimeIndex:
    try:
        import exchange_calendars as xcals  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError("exchange_calendars not installed; cannot perform trading-day checks") from e
    cal = xcals.get_calendar(calendar)
    return cal.sessions_in_range(start.normalize(), end.normalize())


def check_ticker(
    ticker: str,
    start: str,
    end: str,
    cache_dir: str = "cache",
    calendar: str = "XNYS",
) -> QCResult:
    df = get_data(ticker, start=start, end=end, cache_dir=cache_dir)
    issues: List[str] = []
    stats: Dict[str, Any] = {}

    rows = len(df)
    stats["rows"] = rows
    if rows == 0:
        issues.append("EMPTY_DATAFRAME")
        return QCResult(ticker, rows, None, None, issues, stats)

    # Index checks
    if not isinstance(df.index, pd.DatetimeIndex):
        issues.append("INDEX_NOT_DATETIME")
    else:
        if df.index.tz is not None:
            issues.append("INDEX_TZ_AWARE")
        if not df.index.is_monotonic_increasing:
            issues.append("INDEX_NOT_SORTED")
        dup = df.index.duplicated().sum()
        if dup:
            issues.append(f"DUPLICATE_DATES:{dup}")
        stats["start"] = df.index.min()
        stats["end"] = df.index.max()

    # Columns and nulls
    missing_cols = [c for c in PARQUET_COLUMNS if c not in df.columns]
    if missing_cols:
        issues.append(f"MISSING_COLS:{','.join(missing_cols)}")
    null_counts = df[PARQUET_COLUMNS].isna().sum().to_dict()
    null_any = sum(null_counts.values())
    if null_any:
        issues.append(f"NULLS:{null_counts}")
    stats["nulls"] = null_counts

    # Volume non-negative
    if "Volume" in df.columns:
        neg_vol = int((df["Volume"] < 0).sum())
        if neg_vol:
            issues.append(f"NEGATIVE_VOLUME:{neg_vol}")
        stats["neg_volume_count"] = neg_vol

    # Trading sessions coverage
    try:
        sessions = _sessions_range(calendar, df.index.min(), df.index.max())
        present = pd.Index(df.index.normalize().unique())
        missing_sessions = sessions.difference(present)
        stats["expected_sessions"] = int(len(sessions))
        stats["missing_sessions"] = int(len(missing_sessions))
        if len(missing_sessions) > 0:
            preview = [str(missing_sessions[0].date())]
            if len(missing_sessions) > 1:
                preview.append(str(missing_sessions[-1].date()))
            issues.append(f"MISSING_TRADING_DAYS:{len(missing_sessions)} sample={preview}")
    except Exception as e:  # pragma: no cover - calendar errors
        issues.append(f"SESSIONS_CHECK_ERROR:{e}")

    return QCResult(
        ticker=ticker,
        rows=rows,
        start=stats.get("start"),
        end=stats.get("end"),
        issues=issues,
        stats=stats,
    )


def main():  # pragma: no cover - CLI
    app = typer.Typer(add_completion=False, no_args_is_help=True)

    @app.command(help="Run data quality checks on cached data for given tickers and period.")
    def check(
        tickers: str = typer.Option(..., help="Comma-separated list, e.g., AAPL,MSFT"),
        start: str = typer.Option(..., help="Start date YYYY-MM-DD"),
        end: str = typer.Option(..., help="End date YYYY-MM-DD"),
        cache_dir: str = typer.Option("cache", help="Cache directory"),
        calendar: str = typer.Option("XNYS", help="Trading calendar for session checks"),
    ) -> None:
        tlist = [t.strip() for t in tickers.split(",") if t.strip()]
        results: List[QCResult] = []
        total_issues = 0
        for t in tlist:
            res = check_ticker(t, start, end, cache_dir, calendar)
            results.append(res)
            total_issues += len(res.issues)
            status = "OK" if not res.issues else "WARN"
            print(
                f"[qc] {t}: {status} | rows={res.rows} | range={res.start}→{res.end} | issues={';'.join(res.issues) if res.issues else 'none'}"
            )
        print(f"[qc] Completed. tickers={len(results)} total_issues={total_issues}")

    app()


if __name__ == "__main__":  # pragma: no cover
    main()

