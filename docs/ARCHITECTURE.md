# Architecture & Decisions — Phase 0

## Scope & Objectives
- Portfolio: US equities, long-only V1, daily close signals → next open execution.
- Universe: 5 starters AAPL, MSFT, NVDA, AMZN, META (extensible to 30–50).
- Benchmarks: SPY, QQQ, and Buy & Hold equal-weight of the fixed universe.
- Targets: Portfolio CAGR ≥ B&H +5%/year; MaxDD ≤ 0.8× B&H; Information Ratio ≥ 0.5; Annual volatility ≈ 18–20%.
- Risk realism: include fees, slippage, gaps, stops with tie-break `stop_first`.

## Environment & Tooling
- Python ≥ 3.11, macOS, VS Code.
- Virtual env: `venv`. Packaging: `requirements.txt` (later).
- Data: `yfinance` (primary), Parquet cache via `pyarrow`.
- Time & calendar: naive UTC dates; trading days from NYSE calendar (e.g., `exchange_calendars` XNYS).
- CLI: `typer`. Config: YAML (`PyYAML`). Tests: `pytest`.

## Data & Cache
- Source: Yahoo Finance (Adj Close used for valuation). Ticker `^VIX` optionally for market filter.
- Cache: per-ticker Parquet under `cache/{TICKER}.parquet` with columns [Date index, Open, High, Low, Close, Adj Close, Volume].
- Warmup: default 252 trading days to initialize indicators.
- Incremental updates: fill missing ranges only, merge, de-duplicate, rewrite file.
- Retries: transient download retries with exponential backoff.
- Prefetch CLI: batch download for an interval to prime cache; offline mode supported when cache covers requests.

## Signals & Execution Semantics
- Timing: strategies emit signals at D (close). Orders for entries execute at next open D+1.
- Stops (intraday):
  - SL/TP checked against daily High/Low. If both breached same day, `stop_first: true` → SL prevails.
  - Gap modeling: if opening gap crosses SL/TP, execute at the opening price, not the stop level.
- Price for entries: next session Open (actual historical open during backtests).
- Fees & slippage: per-side percentage. Apply to notional; entry/exit prices optionally adjusted or accounted in cash.
- One position per ticker. Cooldown after exit to avoid churn. Minimum holding days before discretionary exits (stops override).

## Risk & Portfolio Management
- Position sizing: risk-per-trade on ATR distance.
  - Risk budget = `risk_per_trade * current_equity`.
  - Stop distance = `ATR * stop_loss_atr_mult`; Qty ≈ `risk_budget / stop_distance` (floored to integer); notional = Qty * entry_price.
- Constraints: `max_positions`, `reserve_pct` cash, `notional_cap_per_name`, `min_price`, optional `adv_cap_pct` and `sector_caps`.
- Volatility targeting: reduce new sizes when realized portfolio vol > target (no leverage-up in V1).
- Drawdown brakes: stepwise reductions of `risk_per_trade` and `max_positions` with deeper DD; freeze new entries at severe DD.
- Leverage: 1.0 in V1 (no margin; can extend later).

## Strategies & Signal Interface
- Modules: `src/strategy/{trend,meanrev,breakout}.py`.
- Signature: `generate_signals(df_full: pd.DataFrame, params: dict) -> pd.DataFrame` with columns: `buy`, `sell`, `score`, `ticker`, optionally `next_open`, `next_open_date`, `atr`.
- Combination: merge daily candidates across strategies and tickers.
  - Default: weighted score aggregation, de-dup per ticker (sum or max of weighted scores).
  - Recommended robust alternative: rank/quantile per day per strategy, then average ranks (less scale-sensitive cross-asset).

## Backtesting & Walk-Forward
- Simple backtest: one fixed period for fast iteration.
- WFA: rolling windows (default: train 24m, test 6m, step 6m); capital chained across windows.
- Optimizer: random/grid; per-ticker optimization optional (guard with `min_trades_train`).
- Objective example: `score = Sharpe - lambda_dd * max(0, MaxDD - dd_cap)` with seed for reproducibility.
- Outputs per window: equity series, trade logs, KPIs, and aggregate summaries across windows.

## Reporting & Outputs
- Equity: `outputs/equity.csv` (portfolio), plus benchmarks aligned.
- Trades: `outputs/tradelog_detailed.csv` and compact `outputs/tradelog.csv`.
- Daily stats: positions count, exposure, traded value, equity.
- WFA: `windows_summary.csv`, `opt_results_window_{i}.csv`, `best_params_per_ticker.csv` (optional).
- Perf summary: `perf_summary.json`. Report: `report.html` (plots, KPIs, activity, top trades/days, WFA table).

## Testing & Determinism (minimum set)
- Unit tests: ATR sizing, next open resolution, SL/TP tie-break, FORCED_EOW neutrality on equity, invariants (#pos ≤ slots, exposure ≤ leverage).
- Reproducibility: fixed seed; persist run metadata (seed, config hash, package versions) in outputs.
- Sanity: WFA with `n_candidates=1` ≈ single backtest result (modulo EOW closures).

## Phase 0 Decisions — Defaults
- Initial capital: 100,000.
- Fees & slippage: 0.05% each (0.0005).
- Warmup: 252 trading days.
- Universe starters: AAPL, MSFT, NVDA, AMZN, META.
- Benchmarks: SPY (primary), QQQ, B&H equal-weight (fixed universe at t0).
- Market filter: enable simple regime filter (SPY > MA200 AND VIX < threshold) in V1.

## Notes & Next Steps
- Next (Phase 1): scaffold repo structure, CLI skeleton, and data provider cache with Parquet + prefetch command, plus basic tests.
- Challenge we will implement early: gap-aware stops (execute at open when stop is jumped), and a rank-based multi-strategy combiner option.
