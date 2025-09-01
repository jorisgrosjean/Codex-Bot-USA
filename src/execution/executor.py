from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import pandas as pd

from .position import Position
from .sizing import compute_stop_levels, compute_risk_qty, apply_cash_and_caps


@dataclass
class ExecConfig:
    fee_pct: float = 0.0005
    slippage_pct: float = 0.0005
    max_positions: int = 5
    reserve_pct: float = 0.05
    notional_cap_per_name: float = 0.15
    leverage_cap: float = 1.0

    risk_per_trade: float = 0.003
    atr_lookback: int = 14
    trailing: bool = True
    time_stop_days: int = 40
    holding_min_days: int = 3
    cooldown_days: int = 3
    stop_first: bool = True
    min_price: float = 5.0
    adv_cap_pct: Optional[float] = None

    # Defaults for stops if strategy doesn't provide
    stop_loss_atr_mult: float = 2.0
    take_profit_atr_mult: float = 4.0

    vol_target_annual: Optional[float] = 0.18  # reduce-only

    market_filter_enabled: bool = True
    market_filter_spy_ma_days: int = 200
    market_filter_vix_threshold: float = 30.0


def _union_dates(dfs: Dict[str, pd.DataFrame]) -> List[pd.Timestamp]:
    idx = pd.Index([])
    for df in dfs.values():
        idx = idx.union(df.index)
    idx = idx.sort_values()
    return list(idx)


def _realized_vol_annualized(equity_series: List[Tuple[pd.Timestamp, float]], window: int = 20) -> Optional[float]:
    if len(equity_series) <= window:
        return None
    df = pd.DataFrame(equity_series, columns=["Date", "Equity"]).set_index("Date").sort_index()
    ret = df["Equity"].pct_change().dropna()
    if len(ret) < window:
        return None
    vol = ret.tail(window).std() * (252 ** 0.5)
    return float(vol)


def _adv_limit_shares(df: pd.DataFrame, date: pd.Timestamp, adv_cap_pct: float, lookback: int = 20) -> Optional[int]:
    if "Volume" not in df.columns:
        return None
    # compute ADV up to previous session
    prev = df.loc[:date]
    if prev.empty:
        return None
    adv = prev["Volume"].tail(lookback).mean()
    if pd.isna(adv):
        return None
    return int(adv_cap_pct * adv)


def run_test_period(
    test_data: Dict[str, pd.DataFrame],
    signals: Dict[str, pd.DataFrame],
    exec_cfg: ExecConfig,
    starting_capital: float,
    market_data: Optional[Dict[str, pd.DataFrame]] = None,
) -> Tuple[pd.Series, pd.DataFrame, pd.DataFrame]:
    """
    Execute a backtest on provided period.
    - test_data: per-ticker OHLCV df covering execution dates
    - signals: per-ticker df with buy/sell/score/next_open/atr
    - market_data: optional dict with SPY, ^VIX for market filter
    Returns (equity_series, tradelog_detailed, daily_stats)
    """
    cash = float(starting_capital)
    positions: Dict[str, Position] = {}
    last_exit_date: Dict[str, pd.Timestamp] = {}
    equity_track: List[Tuple[pd.Timestamp, float]] = []

    trade_rows: List[Dict] = []
    daily_rows: List[Dict] = []

    union_dates = _union_dates(test_data)
    max_equity = cash

    for date in union_dates:
        traded_value_today = 0.0

        # 1) Update positions at open
        for pos in positions.values():
            pos.update_on_open()

        # 1b) Evaluate time-stop pending exits (exit at today's open)
        tickers_to_remove: List[str] = []
        for tkr, pos in positions.items():
            if pos.exit_pending and pos.exit_reason_pending == "TIME":
                # exit at today's open if available
                df = test_data.get(tkr)
                if df is None or date not in df.index:
                    continue
                exit_price = float(df.loc[date, "Open"]) * (1 - exec_cfg.slippage_pct)
                notional = pos.qty * exit_price
                fee = notional * exec_cfg.fee_pct
                cash += notional - fee
                traded_value_today += notional
                trade_rows.append(
                    dict(
                        ticker=tkr,
                        entry_date=pos.entry_date,
                        entry_price=pos.entry_price,
                        exit_date=date,
                        exit_price=exit_price,
                        qty=pos.qty,
                        exit_reason="TIME",
                        entry_reason="ENTRY",
                    )
                )
                last_exit_date[tkr] = date
                tickers_to_remove.append(tkr)
        for t in tickers_to_remove:
            positions.pop(t, None)

        # 1c) Update trailing and intraday stops
        tickers_to_remove = []
        for tkr, pos in positions.items():
            df = test_data.get(tkr)
            if df is None or date not in df.index:
                continue
            high = float(df.loc[date, "High"])
            low = float(df.loc[date, "Low"])
            close = float(df.loc[date, "Adj Close"]) if "Adj Close" in df.columns else float(df.loc[date, "Close"]) 
            atr_today = None
            # If ATR column exists, use it (precomputed elsewhere), else None
            if "ATR" in df.columns:
                atr_today = float(df.loc[date, "ATR"]) if pd.notna(df.loc[date, "ATR"]) else None
            pos.update_after_close(high, low, atr_today)

            # check intraday SL/TP
            hit_sl = low <= pos.stop_loss
            hit_tp = high >= pos.take_profit
            if hit_sl and hit_tp:
                reason = "SL" if exec_cfg.stop_first else "TP"
            elif hit_sl:
                reason = "SL"
            elif hit_tp:
                reason = "TP"
            else:
                reason = None

            if reason is not None:
                # execute intraday at the stop level
                if reason == "SL":
                    exit_price = pos.stop_loss * (1 - exec_cfg.slippage_pct)
                else:
                    exit_price = pos.take_profit * (1 - exec_cfg.slippage_pct)
                notional = pos.qty * exit_price
                fee = notional * exec_cfg.fee_pct
                cash += notional - fee
                traded_value_today += notional
                trade_rows.append(
                    dict(
                        ticker=tkr,
                        entry_date=pos.entry_date,
                        entry_price=pos.entry_price,
                        exit_date=date,
                        exit_price=exit_price,
                        qty=pos.qty,
                        exit_reason=reason,
                        entry_reason="ENTRY",
                    )
                )
                last_exit_date[tkr] = date
                tickers_to_remove.append(tkr)
                continue

            # Time stop marking (exit at next open)
            if pos.days_held >= exec_cfg.time_stop_days and not pos.exit_pending:
                pos.exit_pending = True
                pos.exit_reason_pending = "TIME"

        for t in tickers_to_remove:
            positions.pop(t, None)

        # 2) Market filter (optional)
        allow_new_entries = True
        if exec_cfg.market_filter_enabled and market_data is not None:
            spy = market_data.get("SPY")
            vix = market_data.get("^VIX")
            if spy is not None and date in spy.index:
                ma = spy["Adj Close"].rolling(exec_cfg.market_filter_spy_ma_days).mean()
                spy_ok = spy.loc[date, "Adj Close"] >= ma.loc[date]
            else:
                spy_ok = True
            if vix is not None and date in vix.index:
                vix_ok = vix.loc[date, "Adj Close"] < exec_cfg.market_filter_vix_threshold
            else:
                vix_ok = True
            allow_new_entries = bool(spy_ok and vix_ok)

        # 3) New entries
        if allow_new_entries:
            # Collect today's buy signals
            candidates: List[Tuple[str, float, float, float, float]] = []
            # tuple: (ticker, score, next_open_price, atr, next_open_date)
            for tkr, sig in signals.items():
                if date not in sig.index:
                    continue
                row = sig.loc[date]
                if int(row.get("buy", 0)) != 1:
                    continue
                next_open_date = row.get("next_open_date")
                next_open = float(row.get("next_open")) if pd.notna(row.get("next_open")) else None
                if pd.isna(next_open) or next_open is None:
                    continue
                score = float(row.get("score", 0.0))
                atr_val = float(row.get("atr")) if pd.notna(row.get("atr")) else None
                candidates.append((tkr, score, next_open, atr_val, next_open_date))

            # sort by score desc
            candidates.sort(key=lambda x: x[1], reverse=True)

            # current constraints (drawdown brakes could scale these; omitted here for brevity)
            current_max_positions = exec_cfg.max_positions
            risk_per_trade = exec_cfg.risk_per_trade

            for tkr, score, entry_price, atr_val, next_open_date in candidates:
                # slots check
                if len(positions) >= current_max_positions:
                    break
                # already in position
                if tkr in positions:
                    continue
                # cooldown
                last_exit = last_exit_date.get(tkr)
                if last_exit is not None and (date - last_exit).days <= exec_cfg.cooldown_days:
                    continue
                # price floor
                if entry_price < exec_cfg.min_price:
                    continue

                # size via risk
                stop_distance, stop_loss, take_profit = compute_stop_levels(
                    entry_price=entry_price,
                    atr=atr_val,
                    stop_loss_atr_mult=exec_cfg.stop_loss_atr_mult,
                    take_profit_atr_mult=exec_cfg.take_profit_atr_mult,
                )
                qty = compute_risk_qty(
                    equity=max(cash, 1e-9) + 0.0,  # approximate using cash; improved by using equity later
                    risk_per_trade=risk_per_trade,
                    stop_distance=stop_distance,
                )

                # apply cash/leverage/name caps
                # approximate gross exposure as equity - cash (we compute equity later in the loop)
                # here we use last tracked equity if available, else starting_capital
                last_equity = equity_track[-1][1] if equity_track else starting_capital
                gross_exposure = max(last_equity - cash, 0.0)
                qty = apply_cash_and_caps(
                    qty=qty,
                    entry_price=entry_price,
                    equity=last_equity,
                    cash=cash,
                    reserve_pct=exec_cfg.reserve_pct,
                    notional_cap_per_name=exec_cfg.notional_cap_per_name,
                    leverage_cap=exec_cfg.leverage_cap,
                    gross_exposure=gross_exposure,
                )
                if qty <= 0:
                    continue

                # ADV cap (optional)
                if exec_cfg.adv_cap_pct is not None:
                    df = test_data.get(tkr)
                    if df is not None:
                        adv_limit = _adv_limit_shares(df, date, exec_cfg.adv_cap_pct)
                        if adv_limit is not None:
                            qty = min(qty, adv_limit)
                            if qty <= 0:
                                continue

                # fees & slippage (adjust price upward for buys)
                entry_price_eff = entry_price * (1 + exec_cfg.slippage_pct)
                notional = qty * entry_price_eff
                fee = notional * exec_cfg.fee_pct
                cash_after = cash - notional - fee
                if cash_after < exec_cfg.reserve_pct * (equity_track[-1][1] if equity_track else starting_capital):
                    # reserve violated after fee; try reduce qty once
                    max_afford = int(max((cash - fee - exec_cfg.reserve_pct * (equity_track[-1][1] if equity_track else starting_capital)) // entry_price_eff, 0))
                    qty = min(qty, max_afford)
                    if qty <= 0:
                        continue
                    notional = qty * entry_price_eff
                    fee = notional * exec_cfg.fee_pct
                    cash_after = cash - notional - fee

                cash = cash_after
                traded_value_today += notional

                pos = Position(
                    ticker=tkr,
                    entry_date=next_open_date if isinstance(next_open_date, pd.Timestamp) else date,
                    entry_price=entry_price_eff,
                    qty=qty,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    trailing=exec_cfg.trailing,
                    trail_mult_atr=exec_cfg.stop_loss_atr_mult,
                    atr=atr_val,
                    days_held=0,
                    max_price=entry_price_eff,
                )
                positions[tkr] = pos
                trade_rows.append(
                    dict(
                        ticker=tkr,
                        signal_date=date,
                        exec_date=pos.entry_date,
                        entry_price=entry_price_eff,
                        qty=qty,
                        entry_reason="ENTRY",
                    )
                )

        # 4) End-of-day portfolio valuation
        # equity = cash + sum(qty * adj close)
        equity_val = cash
        for tkr, pos in positions.items():
            df = test_data.get(tkr)
            if df is None or date not in df.index:
                continue
            px = float(df.loc[date, "Adj Close"]) if "Adj Close" in df.columns else float(df.loc[date, "Close"]) 
            equity_val += pos.qty * px

        equity_track.append((date, equity_val))
        max_equity = max(max_equity, equity_val)
        exposure_value = max(equity_val - cash, 0.0)
        daily_rows.append(
            dict(
                date=date,
                equity=equity_val,
                positions=len(positions),
                exposure_value=exposure_value,
                exposure_pct=(exposure_value / equity_val if equity_val > 0 else 0.0),
                traded_value=traded_value_today,
            )
        )

    equity_series = pd.Series({d: v for d, v in equity_track}).sort_index()
    tradelog_detailed = pd.DataFrame(trade_rows)
    daily_stats = pd.DataFrame(daily_rows).set_index("date").sort_index()
    return equity_series, tradelog_detailed, daily_stats

