from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Dict, List, Tuple, Optional

import pandas as pd
import typer
import yaml

from src.data.provider import get_data_with_warmup, get_data
from src.strategy.trend import generate_signals
from src.execution.executor import run_test_period, ExecConfig


def _kpis_from_equity(eq: pd.Series) -> Dict[str, float]:
    eq = eq.dropna()
    if eq.empty:
        return {"final": math.nan, "total_return": math.nan, "cagr": math.nan, "sharpe": math.nan}
    initial = float(eq.iloc[0])
    final = float(eq.iloc[-1])
    total_return = final / max(initial, 1e-9) - 1.0
    # CAGR based on calendar years between first and last index
    years = max((eq.index[-1] - eq.index[0]).days / 365.25, 1e-9)
    cagr = (final / max(initial, 1e-9)) ** (1.0 / years) - 1.0
    rets = eq.pct_change().dropna()
    sharpe = float(rets.mean() / (rets.std() + 1e-12) * math.sqrt(252)) if len(rets) > 1 else math.nan
    return {"final": final, "total_return": total_return, "cagr": cagr, "sharpe": sharpe}


def _buy_and_hold_equal_weight(data: Dict[str, pd.DataFrame], start_capital: float) -> pd.Series:
    tickers = list(data.keys())
    if not tickers:
        return pd.Series(dtype=float)
    # Build normalized index per ticker from first available date, aligned on union index
    union_idx = pd.Index([])
    for df in data.values():
        union_idx = union_idx.union(df.index)
    union_idx = union_idx.sort_values()

    norm_series = []
    for t, df in data.items():
        adj = df["Adj Close"].astype(float)
        # forward-fill to align on union index
        adj_aligned = adj.reindex(union_idx).ffill()
        base = adj_aligned.iloc[0]
        norm = adj_aligned / max(float(base), 1e-9)
        norm_series.append(norm)
    avg_norm = pd.concat(norm_series, axis=1).mean(axis=1)
    return start_capital * avg_norm


def run_backtest(config_path: str) -> None:
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    uni = cfg["universe"]
    tickers: List[str] = uni["tickers"]
    start: str = uni["start"]
    end: str = uni["end"]
    benchmarks: List[str] = uni.get("benchmarks", ["SPY"])  # ensure SPY present for comparison

    data_cfg = cfg["data"]
    cache_dir = data_cfg.get("cache_dir", "cache")
    warmup_days = int(data_cfg.get("warmup_days", 252))
    retries = int(data_cfg.get("retries", 3))
    backoff = int(data_cfg.get("retry_backoff_sec", 2))

    exec_raw = cfg["execution"]
    exec_cfg = ExecConfig(
        fee_pct=float(exec_raw.get("fee_pct", 0.0005)),
        slippage_pct=float(exec_raw.get("slippage_pct", 0.0005)),
        max_positions=int(exec_raw.get("max_positions", 5)),
        reserve_pct=float(exec_raw.get("reserve_pct", 0.05)),
        notional_cap_per_name=float(exec_raw.get("notional_cap_per_name", 0.15)),
        leverage_cap=float(exec_raw.get("leverage_cap", 1.0)),
        risk_per_trade=float(exec_raw.get("risk_per_trade", 0.003)),
        atr_lookback=int(exec_raw.get("atr_lookback", 14)),
        trailing=bool(exec_raw.get("trailing", True)),
        time_stop_days=int(exec_raw.get("time_stop_days", 40)),
        holding_min_days=int(exec_raw.get("holding_min_days", 3)),
        cooldown_days=int(exec_raw.get("cooldown_days", 3)),
        stop_first=bool(exec_raw.get("stop_first", True)),
        min_price=float(exec_raw.get("min_price", 5.0)),
        adv_cap_pct=exec_raw.get("adv_cap_pct"),
        stop_loss_atr_mult=float(exec_raw.get("stop_loss_atr_mult", 2.0)),
        take_profit_atr_mult=float(exec_raw.get("take_profit_atr_mult", 4.0)),
        vol_target_annual=exec_raw.get("vol_target_annual"),
        market_filter_enabled=bool(exec_raw.get("market_filter", {}).get("enabled", True)),
        market_filter_spy_ma_days=int(exec_raw.get("market_filter", {}).get("params", {}).get("spy_ma_days", 200)),
        market_filter_vix_threshold=float(exec_raw.get("market_filter", {}).get("params", {}).get("vix_threshold", 30)),
    )

    starting_capital = float(exec_raw.get("initial_capital", 100000))

    # Strategy params (TREND only for now)
    strat_cfg = cfg.get("strategies", {})
    trend_params = None
    for item in strat_cfg.get("mix", []):
        if item.get("name") == "TREND":
            trend_params = item.get("params", {})
            break
    if trend_params is None:
        # sensible defaults
        trend_params = dict(short_window=20, long_window=100, momentum_window=63, atr_lookback=14)

    # Load data and build signals
    test_data: Dict[str, pd.DataFrame] = {}
    signals: Dict[str, pd.DataFrame] = {}

    for t in tickers:
        df_full, df_exec = get_data_with_warmup(
            t, start=start, end=end, cache_dir=cache_dir, retries=retries, retry_backoff_sec=backoff, warmup_days=warmup_days
        )
        p = dict(trend_params)
        p["ticker"] = t
        sig = generate_signals(df_full, p)
        # Keep only execution window intersection with df_exec
        sig = sig.loc[sig.index.intersection(df_exec.index)]
        test_data[t] = df_exec
        signals[t] = sig

    # Market data for filter
    market_data: Dict[str, pd.DataFrame] = {}
    if exec_cfg.market_filter_enabled:
        try:
            spy_df = get_data("SPY", start, end, cache_dir, retries, backoff)
            vix_df = get_data("^VIX", start, end, cache_dir, retries, backoff)
            market_data["SPY"] = spy_df
            market_data["^VIX"] = vix_df
        except Exception:
            market_data = {}

    equity, trades, daily = run_test_period(test_data, signals, exec_cfg, starting_capital, market_data or None)

    # Benchmarks: SPY and B&H equal-weight
    bench_data: Dict[str, pd.DataFrame] = {}
    for b in list(set(benchmarks + ["SPY"])):
        try:
            bench_data[b] = get_data(b, start, end, cache_dir, retries, backoff)
        except Exception:
            pass

    spy_eq = None
    if "SPY" in bench_data:
        price = bench_data["SPY"]["Adj Close"].astype(float)
        price = price.reindex(equity.index).ffill()
        spy_eq = starting_capital * (price / max(float(price.iloc[0]), 1e-9))

    bh_eq = _buy_and_hold_equal_weight({t: test_data[t] for t in tickers}, starting_capital)
    bh_eq = bh_eq.reindex(equity.index).ffill()

    # KPIs
    kpi_port = _kpis_from_equity(equity)
    kpi_spy = _kpis_from_equity(spy_eq) if spy_eq is not None else {"final": math.nan, "total_return": math.nan, "cagr": math.nan, "sharpe": math.nan}
    kpi_bh = _kpis_from_equity(bh_eq)

    def fmt_pct(x: float) -> str:
        return ("{:+.2f}%".format(100 * x)) if (x == x) else "n/a"  # NaN check via x==x

    print("=== Backtest Summary (TREND) ===")
    print(f"Period: {start} → {end} | Tickers: {', '.join(tickers)} | StartCap: {starting_capital:,.0f}")
    print("Portfolio:")
    print(f"  Final: {kpi_port['final']:,.2f} | Total Return: {fmt_pct(kpi_port['total_return'])} | CAGR: {fmt_pct(kpi_port['cagr'])} | Sharpe: {kpi_port['sharpe']:.2f}")
    print("Benchmark SPY:")
    print(f"  Final: {kpi_spy['final']:,.2f} | Total Return: {fmt_pct(kpi_spy['total_return'])} | CAGR: {fmt_pct(kpi_spy['cagr'])} | Sharpe: {kpi_spy['sharpe']:.2f}")
    print("Buy & Hold Equal-Weight (universe):")
    print(f"  Final: {kpi_bh['final']:,.2f} | Total Return: {fmt_pct(kpi_bh['total_return'])} | CAGR: {fmt_pct(kpi_bh['cagr'])} | Sharpe: {kpi_bh['sharpe']:.2f}")


def main():  # pragma: no cover - CLI
    app = typer.Typer(add_completion=False, no_args_is_help=True)

    @app.command()
    def run(config: str = typer.Option("config/backtest.yml", help="Path to YAML config")):
        run_backtest(config)

    app()


if __name__ == "__main__":  # pragma: no cover
    main()

