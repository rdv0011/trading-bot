import logging
from dataclasses import dataclass, field
from datetime import date

logger = logging.getLogger(__name__)


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

        # --- New day reset ---
        if self._last_date != today:
            if self._halted:
                logger.info(
                    "🔄 New day — RiskGuard halt lifted"
                    " (start_of_day_equity=%s)",
                    f"{current_equity:.2f}",
                )
            self._halted = False
            self._start_of_day_equity = current_equity
            self._peak_equity = current_equity
            self._last_date = today

        # --- Track peak equity ---
        if current_equity > self._peak_equity:
            self._peak_equity = current_equity

        halt_triggered_this_call = False

        # --- Daily loss check ---
        if self._start_of_day_equity > 0:
            daily_loss = (
                self._start_of_day_equity - current_equity
            ) / self._start_of_day_equity
            if daily_loss >= self.max_daily_loss_frac:
                self._halted = True
                halt_triggered_this_call = True
                logger.warning(
                    "🛑 RiskGuard halted — daily loss %.2f%% >= %.2f%%"
                    " (equity=%s start_of_day=%s)",
                    daily_loss * 100,
                    self.max_daily_loss_frac * 100,
                    f"{current_equity:.2f}",
                    f"{self._start_of_day_equity:.2f}",
                )
        elif self._last_date is not None:
            logger.warning(
                "⚠️ RiskGuard daily loss check skipped — start_of_day_equity=0"
                " (current_equity=%s)",
                f"{current_equity:.2f}",
            )

        # --- Drawdown check ---
        if self._peak_equity > 0:
            drawdown = (
                self._peak_equity - current_equity
            ) / self._peak_equity
            if drawdown >= self.max_drawdown_frac:
                self._halted = True
                if not halt_triggered_this_call:
                    logger.warning(
                        "🛑 RiskGuard halted — drawdown %.2f%% >= %.2f%%"
                        " (equity=%s peak=%s)",
                        drawdown * 100,
                        self.max_drawdown_frac * 100,
                        f"{current_equity:.2f}",
                        f"{self._peak_equity:.2f}",
                    )

        return not self._halted

    def clamp_leverage(self, recommended: float) -> float:
        return min(recommended, self.max_leverage)

    @property
    def is_halted(self) -> bool:
        return self._halted

    def reset(self, current_equity: float):
        """Manually reset RiskGuard state — e.g. after restart or manual override."""
        logger.info(
            "🔄 RiskGuard manually reset (equity=%s)",
            f"{current_equity:.2f}",
        )
        self._halted = False
        self._start_of_day_equity = current_equity
        self._peak_equity = current_equity
        self._last_date = date.today()
