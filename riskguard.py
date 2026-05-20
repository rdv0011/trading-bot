from dataclasses import dataclass, field
from datetime import date


@dataclass
class RiskGuard:
    max_daily_loss_frac: float = 0.05
    max_drawdown_frac: float = 0.15
    max_leverage: float = 10.0

    _start_of_day_equity: float = field(default=0.0, init=False)
    _peak_equity: float = field(default=0.0, init=False)
    _last_date: date = field(default=None, init=False)
    _halted: bool = field(default=False, init=False)

    def update(self, current_equity: float) -> bool:
        today = date.today()

        if self._last_date != today:
            self._start_of_day_equity = current_equity
            self._last_date = today

        if current_equity > self._peak_equity:
            self._peak_equity = current_equity

        if self._start_of_day_equity > 0:
            daily_loss = (self._start_of_day_equity - current_equity) / self._start_of_day_equity
            if daily_loss >= self.max_daily_loss_frac:
                self._halted = True

        if self._peak_equity > 0:
            drawdown = (self._peak_equity - current_equity) / self._peak_equity
            if drawdown >= self.max_drawdown_frac:
                self._halted = True

        return not self._halted

    def clamp_leverage(self, recommended: float) -> float:
        return min(recommended, self.max_leverage)

    @property
    def is_halted(self) -> bool:
        return self._halted
