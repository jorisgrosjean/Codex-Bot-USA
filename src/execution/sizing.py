from __future__ import annotations

from math import floor
from typing import Optional, Tuple


def compute_stop_levels(
    entry_price: float,
    atr: Optional[float],
    stop_loss_atr_mult: float = 2.0,
    take_profit_atr_mult: float = 4.0,
    fallback_stop_pct: float = 0.07,
    fallback_tp_pct: float = 0.15,
) -> Tuple[float, float, float]:
    """Return (stop_distance, stop_loss_abs, take_profit_abs)."""
    if atr is not None and atr > 0:
        stop_distance = atr * stop_loss_atr_mult
        tp_distance = atr * take_profit_atr_mult
    else:
        stop_distance = entry_price * fallback_stop_pct
        tp_distance = entry_price * fallback_tp_pct
    stop_loss = entry_price - stop_distance
    take_profit = entry_price + tp_distance
    return stop_distance, stop_loss, take_profit


def compute_risk_qty(
    equity: float,
    risk_per_trade: float,
    stop_distance: float,
) -> int:
    if stop_distance <= 0:
        return 0
    risk_budget = equity * risk_per_trade
    qty = floor(risk_budget / stop_distance)
    return max(qty, 0)


def apply_cash_and_caps(
    qty: int,
    entry_price: float,
    equity: float,
    cash: float,
    reserve_pct: float,
    notional_cap_per_name: float,
    leverage_cap: float,
    gross_exposure: float,
) -> int:
    if qty <= 0:
        return 0
    notional = qty * entry_price

    # Cap per name
    max_notional = notional_cap_per_name * equity
    if notional > max_notional:
        qty = int(max_notional // entry_price)
        notional = qty * entry_price

    # Cash reserve
    min_cash = reserve_pct * equity
    max_affordable_qty = int(max((cash - min_cash) // entry_price, 0))
    qty = min(qty, max_affordable_qty)

    # Leverage cap on gross exposure
    max_additional = max(leverage_cap * equity - gross_exposure, 0)
    max_qty_by_leverage = int(max_additional // entry_price)
    qty = min(qty, max_qty_by_leverage)

    return max(qty, 0)

