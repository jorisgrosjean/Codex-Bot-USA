from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional
import pandas as pd


@dataclass
class Position:
    ticker: str
    entry_date: pd.Timestamp
    entry_price: float
    qty: int

    stop_loss: float
    take_profit: float
    trailing: bool = False
    trail_mult_atr: Optional[float] = None
    atr: Optional[float] = None

    days_held: int = 0
    max_price: float = 0.0
    exit_pending: bool = False
    exit_reason_pending: Optional[str] = None

    def update_on_open(self):
        self.days_held += 1

    def update_after_close(self, high: float, low: float, atr: Optional[float]):
        # Update trailing stop based on new highs
        if self.trailing:
            if high > self.max_price:
                self.max_price = high
                if atr is not None and self.trail_mult_atr is not None:
                    new_stop = self.max_price - self.trail_mult_atr * atr
                    self.stop_loss = max(self.stop_loss, new_stop)

