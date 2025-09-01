from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Dict, List, Tuple, Optional

import pandas as pd
import typer
import yaml
import json
from pathlib import Path

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


def _max_drawdown(eq: pd.Series) -> float:
    eq = eq.dropna()
    if eq.empty:
        return math.nan
    running_max = eq.cummax()
    dd = eq / running_max - 1.0
    return float(dd.min())  # negative number


def _portfolio_activity(daily: pd.DataFrame, equity: pd.Series, trades: pd.DataFrame) -> Dict[str, object]:
    days = len(equity)
    # Total trades = number of exit rows
    total_trades = int(trades[trades["exit_reason"].notna()].shape[0]) if "exit_reason" in trades.columns else 0
    daily_avg_trades = (total_trades / days) if days > 0 else math.nan
    max_open_positions = int(daily["positions"].max()) if "positions" in daily.columns and not daily.empty else 0
    avg_positions = float(daily["positions"].mean()) if "positions" in daily.columns and not daily.empty else math.nan

    exposure = daily["exposure_pct"] if "exposure_pct" in daily.columns and not daily.empty else pd.Series(dtype=float)
    exposure_stats = {
        "avg": float(exposure.mean()) if not exposure.empty else math.nan,
        "med": float(exposure.median()) if not exposure.empty else math.nan,
        "min": float(exposure.min()) if not exposure.empty else math.nan,
        "max": float(exposure.max()) if not exposure.empty else math.nan,
    }

    eq = equity.sort_index()
    ret = eq.pct_change().dropna()
    best_day = float(ret.max()) if not ret.empty else math.nan
    worst_day = float(ret.min()) if not ret.empty else math.nan

    # Turnover: traded_value / equity_prev
    turnover = None
    if not daily.empty and not eq.empty:
        daily_aligned = daily.reindex(eq.index)
        eq_prev = eq.shift(1)
        turn = daily_aligned["traded_value"] / eq_prev
        turnover = float(turn.mean()) if turn.notna().any() else math.nan

    return {
        "total_trades": total_trades,
        "daily_avg_trades": daily_avg_trades,
        "max_open_positions": max_open_positions,
        "avg_positions": avg_positions,
        "exposure": exposure_stats,
        "turnover_daily_avg": turnover if turnover is not None else math.nan,
        "best_day": best_day,
        "worst_day": worst_day,
    }


def _fmt_pct(x: float) -> str:
    return ("{:+.2f}%".format(100 * x)) if (x == x) else "n/a"


def _objective_score(eq: pd.Series, lam_dd: float = 3.0, dd_cap: float = 0.2) -> float:
    k = _kpis_from_equity(eq)
    sharpe = k.get("sharpe", math.nan)
    dd = _max_drawdown(eq)
    if dd != dd:  # NaN
        dd_pen = 0.0
    else:
        dd_pen = max(0.0, abs(dd) - dd_cap)
    if sharpe != sharpe:
        return -1e9
    return float(sharpe - lam_dd * dd_pen)


def _optimize_trend(
    tickers: List[str],
    start: str,
    end: str,
    cache_dir: str,
    warmup_days: int,
    retries: int,
    backoff: int,
    exec_cfg_base: ExecConfig,
    base_trend_params: Dict[str, object],
) -> Tuple[Dict[str, object], ExecConfig, Dict[str, float]]:
    """Very small random/grid search for TREND params and risk sizing.
    Returns (best_trend_params, best_exec_cfg, best_kpis)
    """
    # Candidate grids (tiny to keep runtime small)
    short_list = [10, 20]
    long_list = [100]          # keep grid tiny for speed
    mom_list = [63]
    risk_list = [0.003, 0.005]
    slots_list = [5]
    sl_mult = [2.0]
    tp_mult = [4.0]

    best_score = -1e18
    best_params = None
    best_exec = None
    best_kpis: Dict[str, float] = {}

    # Preload data once
    test_data: Dict[str, pd.DataFrame] = {}
    signals_by_param: Dict[Tuple[int, int, int], Dict[str, pd.DataFrame]] = {}
    for t in tickers:
        df_full, df_exec = get_data_with_warmup(t, start, end, cache_dir, retries, backoff, warmup_days)
        test_data[t] = df_exec
        # We will compute signals per param triple lazily

    # No market filter to speed up opt
    market_data = None

    for sw in short_list:
        for lw in long_list:
            if sw >= lw:
                continue
            for mw in mom_list:
                key = (sw, lw, mw)
                sigs: Dict[str, pd.DataFrame] = {}
                for t in tickers:
                    df_full, _ = get_data_with_warmup(t, start, end, cache_dir, retries, backoff, warmup_days)
                    p = dict(base_trend_params)
                    p.update(dict(short_window=sw, long_window=lw, momentum_window=mw, ticker=t))
                    sigs[t] = generate_signals(df_full, p)
                # Try exec configs
                for risk in risk_list:
                    for slots in slots_list:
                        for slm in sl_mult:
                            for tpm in tp_mult:
                                exec_cfg = ExecConfig(**asdict(exec_cfg_base))
                                exec_cfg.risk_per_trade = risk
                                exec_cfg.max_positions = slots
                                exec_cfg.stop_loss_atr_mult = slm
                                exec_cfg.take_profit_atr_mult = tpm
                                exec_cfg.market_filter_enabled = False  # speed up
                                equity, trades, daily = run_test_period(test_data, sigs, exec_cfg, float(exec_cfg_base.reserve_pct * 0 + 100000))
                                score = _objective_score(equity, lam_dd=3.0, dd_cap=0.2)
                                if score > best_score:
                                    best_score = score
                                    best_params = dict(short_window=sw, long_window=lw, momentum_window=mw)
                                    best_exec = exec_cfg
                                    best_kpis = _kpis_from_equity(equity)

    assert best_params is not None and best_exec is not None
    return best_params, best_exec, best_kpis


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


def run_backtest(config_path: str, optimize: bool = False) -> None:
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

    # Optionally optimize
    if optimize:
        best_trend, best_exec, best_kpi = _optimize_trend(
            tickers, start, end, cache_dir, warmup_days, retries, backoff, exec_cfg, trend_params
        )
        print("[opt] Best TREND params:", best_trend)
        print("[opt] Updated Exec:", {
            "risk_per_trade": best_exec.risk_per_trade,
            "max_positions": best_exec.max_positions,
            "stop_loss_atr_mult": best_exec.stop_loss_atr_mult,
            "take_profit_atr_mult": best_exec.take_profit_atr_mult,
            "market_filter_enabled": best_exec.market_filter_enabled,
        })
        trend_params.update(best_trend)
        exec_cfg = best_exec

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
    activity = _portfolio_activity(daily, equity, trades)

    print("=== Backtest Summary (TREND) ===")
    print(f"Period: {start} → {end} | Tickers: {', '.join(tickers)} | StartCap: {starting_capital:,.0f}")
    print("Portfolio:")
    print(f"  Final: {kpi_port['final']:,.2f} | Total Return: {_fmt_pct(kpi_port['total_return'])} | CAGR: {_fmt_pct(kpi_port['cagr'])} | Sharpe: {kpi_port['sharpe']:.2f}")
    print("Benchmark SPY:")
    print(f"  Final: {kpi_spy['final']:,.2f} | Total Return: {_fmt_pct(kpi_spy['total_return'])} | CAGR: {_fmt_pct(kpi_spy['cagr'])} | Sharpe: {kpi_spy['sharpe']:.2f}")
    print("Buy & Hold Equal-Weight (universe):")
    print(f"  Final: {kpi_bh['final']:,.2f} | Total Return: {_fmt_pct(kpi_bh['total_return'])} | CAGR: {_fmt_pct(kpi_bh['cagr'])} | Sharpe: {kpi_bh['sharpe']:.2f}")
    print("Portfolio Activity:")
    print(
        f"  Total / Daily Avg trades {activity['total_trades']} / {activity['daily_avg_trades']:.2f}\n"
        f"  Max open positions {activity['max_open_positions']}\n"
        f"  Avg # positions {activity['avg_positions']:.3f}\n"
        f"  Exposure avg/med/min/max {_fmt_pct(activity['exposure']['avg'])} / {_fmt_pct(activity['exposure']['med'])} / {_fmt_pct(activity['exposure']['min'])} / {_fmt_pct(activity['exposure']['max'])}\n"
        f"  Turnover (daily avg) {_fmt_pct(activity['turnover_daily_avg'])}\n"
        f"  Best day / Worst day {_fmt_pct(activity['best_day'])} / {_fmt_pct(activity['worst_day'])}"
    )

    # Save outputs to disk
    run_cfg = cfg.get("run", {})
    out_dir = Path(run_cfg.get("outputs_dir", "outputs/dev"))
    out_dir.mkdir(parents=True, exist_ok=True)

    # Equity with benchmarks
    eq_index = equity.index
    eq_df = pd.DataFrame(index=eq_index)
    eq_df.index.name = "Date"
    eq_df["equity"] = equity.reindex(eq_index)
    if spy_eq is not None:
        eq_df["spy"] = spy_eq.reindex(eq_index)
    eq_df["buy_hold_eqw"] = bh_eq.reindex(eq_index)
    eq_df.to_csv(out_dir / "equity.csv")

    # Daily stats
    if not daily.empty:
        daily.to_csv(out_dir / "daily_stats.csv")

    # Trade logs: detailed and compact (exit-only with PnL)
    if not trades.empty:
        trades.to_csv(out_dir / "tradelog_detailed.csv", index=False)
        if "exit_reason" in trades.columns:
            exits = trades[trades["exit_reason"].notna()].copy()
            if not exits.empty:
                exits["pnl"] = (exits["exit_price"].astype(float) - exits["entry_price"].astype(float)) * exits["qty"].astype(int)
                exits["pnl_pct"] = exits["exit_price"].astype(float) / exits["entry_price"].astype(float) - 1.0
                if "entry_date" in exits.columns and "exit_date" in exits.columns:
                    try:
                        exits["holding_days"] = (pd.to_datetime(exits["exit_date"]) - pd.to_datetime(exits["entry_date"])) / pd.Timedelta(days=1)
                    except Exception:
                        exits["holding_days"] = float("nan")
                exits.to_csv(out_dir / "tradelog.csv", index=False)

    # Save signals if requested
    reports_cfg = cfg.get("reports", {})
    if reports_cfg.get("save_daily_signals", False):
        sig_rows: List[pd.DataFrame] = []
        for t, df in signals.items():
            df2 = df.copy()
            if "ticker" not in df2.columns:
                df2["ticker"] = t
            sig_rows.append(df2)
        if sig_rows:
            sig_all = pd.concat(sig_rows).sort_index()
            sig_all.to_csv(out_dir / "daily_signals.csv")

    # Perf summary JSON
    perf_summary = {
        "Portfolio": {
            "FinalValue": kpi_port["final"],
            "TotalReturn": kpi_port["total_return"],
            "CAGR": kpi_port["cagr"],
            "Sharpe": kpi_port["sharpe"],
            "MaxDD": _max_drawdown(equity),
        },
        "Benchmarks": {
            "SPY": (_kpis_from_equity(spy_eq) if spy_eq is not None else None),
            "B&H": _kpis_from_equity(bh_eq),
        },
        "Activity": activity,
        "Meta": {
            "period": {"start": start, "end": end},
            "tickers": tickers,
            "initial_capital": starting_capital,
            "exec": {
                "fee_pct": exec_cfg.fee_pct,
                "slippage_pct": exec_cfg.slippage_pct,
                "max_positions": exec_cfg.max_positions,
                "risk_per_trade": exec_cfg.risk_per_trade,
            },
        },
    }
    with open(out_dir / "perf_summary.json", "w") as f:
        json.dump(perf_summary, f, indent=2, default=str)


def main():  # pragma: no cover - CLI
    app = typer.Typer(add_completion=False, no_args_is_help=True)

    @app.command()
    def run(
        config: str = typer.Option("config/backtest.yml", help="Path to YAML config"),
        optimize: bool = typer.Option(False, help="Run quick param optimization before backtest"),
    ):
        run_backtest(config, optimize)

    app()


if __name__ == "__main__":  # pragma: no cover
    main()
