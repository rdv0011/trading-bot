import pytest

from timeframe_config import TimeframeConfig, TIMEFRAMES


TF_5M = TIMEFRAMES["5m"]
TF_1H = TIMEFRAMES["1h"]


def test_tc01_candles_per_hour_5m():
    assert TF_5M.candles_per_hour() == 12


def test_tc02_candles_per_hour_1h():
    assert TF_1H.candles_per_hour() == 1


def test_tc03_label_horizon_candles_5m():
    assert TF_5M.label_horizon_candles == 20


def test_tc04_adaptive_history_candles_5m():
    assert TF_5M.adaptive_history_candles == 600


def test_tc05_ema_spans_all_positive_ints():
    spans = TF_5M.ema_spans
    assert all(isinstance(s, int) and s >= 1 for s in spans)


def test_tc06_candles_10min_on_5m():
    assert TF_5M.candles(10) == 2


def test_tc07_candles_3min_on_5m_clamped_to_1():
    assert TF_5M.candles(3) == 1


def test_tc08_timeframes_5m_minutes():
    assert TIMEFRAMES["5m"].minutes == 5


def test_tc09_timeframes_1h_minutes():
    assert TIMEFRAMES["1h"].minutes == 60
