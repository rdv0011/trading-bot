import math

import numpy as np
import pandas as pd
import pytest

from mltrainingcore import (
    adaptive_thresholding,
    calculate_metrics,
    detect_regime,
    get_features,
    get_param_row,
    make_features,
    make_labels,
    time_to_candles,
)
from timeframe_config import TIMEFRAMES

TF_5M = TIMEFRAMES["5m"]


def _make_row(ema_20, ema_100, atr14, vol_12, vol_48):
    return {
        "ema_20": ema_20,
        "ema_100": ema_100,
        "atr14": atr14,
        "vol_12": vol_12,
        "vol_48": vol_48,
    }


def test_mc01_detect_regime_weak_trend_is_chop():
    row = _make_row(ema_20=100.0, ema_100=100.3, atr14=1.0, vol_12=0.01, vol_48=0.01)
    assert detect_regime(row) == "chop"


def test_mc02_detect_regime_strong_trend_low_vol_is_trend():
    row = _make_row(ema_20=100.0, ema_100=99.0, atr14=1.0, vol_12=0.01, vol_48=0.01)
    assert detect_regime(row) == "trend"


def test_mc03_detect_regime_strong_trend_high_vol_is_high_vol():
    row = _make_row(ema_20=100.0, ema_100=99.0, atr14=1.0, vol_12=0.03, vol_48=0.01)
    assert detect_regime(row) == "high_vol"


def test_mc04_detect_regime_zero_atr_does_not_raise():
    row = _make_row(ema_20=100.0, ema_100=100.0, atr14=0.0, vol_12=0.01, vol_48=0.01)
    result = detect_regime(row)
    assert result in ("chop", "trend", "high_vol")


def test_mc05_adaptive_thresholding_short_series_returns_nan():
    series = pd.Series([1.0, 2.0, 3.0])
    max_th, min_th = adaptive_thresholding(series, TF_5M)
    assert math.isnan(max_th)
    assert math.isnan(min_th)


def test_mc06_adaptive_thresholding_sufficient_series_max_gt_min():
    rng = np.random.default_rng(0)
    series = pd.Series(rng.normal(0, 1, TF_5M.adaptive_history_candles + 10))
    max_th, min_th = adaptive_thresholding(series, TF_5M)
    assert max_th > min_th


def test_mc07_adaptive_thresholding_constant_series_max_equals_min():
    series = pd.Series([1.0] * (TF_5M.adaptive_history_candles + 10))
    max_th, min_th = adaptive_thresholding(series, TF_5M)
    assert abs(max_th - min_th) < 1e-9


def test_mc08_calculate_metrics_empty_trades():
    score, metrics = calculate_metrics([], wallet=1.0)
    assert score == pytest.approx(-0.1)
    assert metrics["trades_count"] == 0


def test_mc09_calculate_metrics_all_winning_trades():
    trades = [{"return": 0.05}, {"return": 0.03}, {"return": 0.02}]
    _, metrics = calculate_metrics(trades, wallet=1.1)
    assert metrics["win_rate"] == pytest.approx(1.0)


def test_mc10_calculate_metrics_all_losing_trades():
    trades = [{"return": -0.05}, {"return": -0.03}]
    _, metrics = calculate_metrics(trades, wallet=0.92)
    assert metrics["downside"] > 0.0


def test_mc11_calculate_metrics_composite_formula():
    trades = [{"return": 0.04}, {"return": -0.02}, {"return": 0.03}]
    score, metrics = calculate_metrics(trades, wallet=1.05, expected_trades=3)
    mean_ret = metrics["mean_return"]
    win_rate = metrics["win_rate"]
    downside = metrics["downside"]
    activity = metrics["activity"]
    expected = 3.0 * mean_ret + 2.0 * win_rate - 1.0 * downside + 0.5 * activity
    assert score == pytest.approx(expected, rel=1e-6)


def test_mc12_calculate_metrics_activity_capped_at_one():
    trades = [{"return": 0.01}] * 100
    _, metrics = calculate_metrics(trades, wallet=2.0, expected_trades=10)
    assert metrics["activity"] == pytest.approx(1.0)


def test_mc13_get_param_row_single_dict_returns_same():
    d = {"a": 1, "b": 2}
    assert get_param_row(d, 0) is d


def test_mc14_get_param_row_list_valid_index():
    lst = [{"x": 0}, {"x": 1}, {"x": 2}]
    assert get_param_row(lst, 2) == {"x": 2}


def test_mc15_get_param_row_list_out_of_range_returns_first():
    lst = [{"x": 0}, {"x": 1}]
    assert get_param_row(lst, 99) == {"x": 0}


def test_mc16_get_param_row_empty_list_returns_none():
    assert get_param_row([], 0) is None


def test_mc17_get_param_row_invalid_type_raises():
    with pytest.raises(TypeError):
        get_param_row("not_a_dict_or_list", 0)


def test_mc18_time_to_candles_minutes():
    assert time_to_candles(minutes=10, timeframe_minutes=5) == 2


def test_mc19_time_to_candles_hours():
    assert time_to_candles(hours=1, timeframe_minutes=5) == 12


def test_mc20_time_to_candles_below_min_clamped():
    assert time_to_candles(minutes=1, timeframe_minutes=5, min_candles=3) == 3


def test_mc21_time_to_candles_no_input_raises():
    with pytest.raises(ValueError):
        time_to_candles(timeframe_minutes=5)


def test_mc22_make_features_expected_columns(sample_ohlcv):
    df = make_features(sample_ohlcv, TF_5M)
    for col in ("ret1", "atr14", "regime", "hour_sin", "hour_cos"):
        assert col in df.columns, f"Missing column: {col}"


def test_mc23_make_features_no_nan(sample_ohlcv):
    df = make_features(sample_ohlcv, TF_5M)
    assert df.isna().sum().sum() == 0


def test_mc24_make_labels_adds_future_ret(sample_ohlcv):
    df = make_features(sample_ohlcv, TF_5M)
    df_labeled = make_labels(df, TF_5M)
    assert "future_ret" in df_labeled.columns


def test_mc25_make_labels_shorter_than_input(sample_ohlcv):
    df = make_features(sample_ohlcv, TF_5M)
    df_labeled = make_labels(df, TF_5M)
    assert len(df_labeled) < len(df)


def test_mc26_get_features_excludes_target_columns(sample_ohlcv):
    df = make_features(sample_ohlcv, TF_5M)
    df = make_labels(df, TF_5M)
    features = get_features(df)
    for excluded in ("future_ret", "future_close", "regime"):
        assert excluded not in features
