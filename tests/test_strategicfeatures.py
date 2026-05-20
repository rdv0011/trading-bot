import pytest

from strategic.strategicfeatures import (
    _classify_vol_state,
    make_strategic_features,
)
from timeframe_config import TIMEFRAMES

TF_1H = TIMEFRAMES["1h"]


def test_sf01_classify_vol_state_below_1_is_low():
    assert _classify_vol_state(0.8) == 0.0


def test_sf02_classify_vol_state_between_1_and_1_6_is_normal():
    assert _classify_vol_state(1.2) == 1.0


def test_sf03_classify_vol_state_between_1_6_and_2_5_is_high():
    assert _classify_vol_state(2.0) == 2.0


def test_sf04_classify_vol_state_at_or_above_2_5_is_extreme():
    assert _classify_vol_state(2.5) == 3.0


def test_sf05_classify_vol_state_exactly_at_1_6_boundary_is_high():
    assert _classify_vol_state(1.6) == 2.0


def test_sf06_make_strategic_features_has_vol_state_column(sample_ohlcv):
    df = make_strategic_features(sample_ohlcv, TF_1H)
    assert "vol_state" in df.columns


def test_sf07_make_strategic_features_vol_state_valid_values(sample_ohlcv):
    df = make_strategic_features(sample_ohlcv, TF_1H)
    valid = {0.0, 1.0, 2.0, 3.0}
    assert set(df["vol_state"].unique()).issubset(valid)


def test_sf08_make_strategic_features_drawdown_non_positive(sample_ohlcv):
    df = make_strategic_features(sample_ohlcv, TF_1H)
    assert (df["drawdown"] <= 0).all()


def test_sf09_make_strategic_features_no_nan(sample_ohlcv):
    df = make_strategic_features(sample_ohlcv, TF_1H)
    assert df.isna().sum().sum() == 0
