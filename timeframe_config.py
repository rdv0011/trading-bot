# timeframe_config.py

from dataclasses import dataclass

@dataclass(frozen=True)
class TimeframeConfig:
    name: str
    minutes: int

    # ===== Time-based intents =====
    label_horizon_minutes: int = 100
    adaptive_history_hours: int = 50
    label_window_hours: int = 12
    max_history_hours: int = 50
    min_feature_hours: int = 20

    # EMA durations in HOURS (NOT candles)
    ema_hours = (0.5, 1.25, 5, 20)

    def candles_per_hour(self) -> int:
        return max(1, int(60 / self.minutes))

    def candles(self, minutes: int) -> int:
        return max(1, int(round(minutes / self.minutes)))

    # ===== Derived candle counts =====
    @property
    def label_horizon_candles(self):
        return self.candles(self.label_horizon_minutes)

    @property
    def adaptive_history_candles(self):
        return int(self.adaptive_history_hours * self.candles_per_hour())

    @property
    def label_window_candles(self):
        return int(self.label_window_hours * self.candles_per_hour())

    @property
    def max_history_candles(self):
        return int(self.max_history_hours * self.candles_per_hour())

    @property
    def min_feature_candles(self):
        return int(self.min_feature_hours * self.candles_per_hour())

    @property
    def ema_spans(self):
        return tuple(
            max(1, int(h * self.candles_per_hour()))
            for h in self.ema_hours
        )


# Presets
TIMEFRAMES = {
    "5m":  TimeframeConfig("5m", 5),
    "15m": TimeframeConfig("15m", 15),
    "1h":  TimeframeConfig("1h", 60),
    "4h":  TimeframeConfig("4h", 240),
}