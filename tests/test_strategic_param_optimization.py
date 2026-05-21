import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch, MagicMock

from mltraining import build_param_grid
from mltrainingcore import SIGNAL_COLUMN
from strategic.strategictraining import (
    _build_strategic_labels,
    _build_strategic_labels_from_simulation,
)
from strategic.strategicfeatures import make_strategic_features
from timeframe_config import TIMEFRAMES


def _make_ohlcv_1h(periods=500):
    idx = pd.date_range("2024-01-01", periods=periods, freq="1h")
    rng = np.random.default_rng(0)
    close = 40000 + np.cumsum(rng.normal(0, 80, periods))
    return pd.DataFrame(
        {
            "open": close * 0.999,
            "high": close * 1.002,
            "low": close * 0.998,
            "close": close,
            "volume": rng.uniform(1, 10, periods),
        },
        index=idx,
    )


def _make_ohlcv_5m(periods=2000):
    idx = pd.date_range("2024-01-01", periods=periods, freq="5min")
    rng = np.random.default_rng(1)
    close = 40000 + np.cumsum(rng.normal(0, 20, periods))
    return pd.DataFrame(
        {
            "open": close * 0.999,
            "high": close * 1.002,
            "low": close * 0.998,
            "close": close,
            "volume": rng.uniform(1, 10, periods),
        },
        index=idx,
    )


def _make_5m_predictions(periods=2000):
    df = _make_ohlcv_5m(periods)
    rng = np.random.default_rng(2)
    df[SIGNAL_COLUMN] = rng.normal(0, 0.01, periods)
    return df


class TestParamGridStructure:
    REQUIRED_KEYS = {
        "stake_long_frac",
        "stake_short_frac",
        "stop_loss_frac",
        "take_profit_frac",
        "max_hold_hours",
    }

    def test_all_required_keys_present(self):
        grid = build_param_grid(
            stake_short=[0.05, 0.10],
            stake_long=[0.10, 0.15],
            stop_loss=[0.02],
            max_hold_hours=[4, 8],
        )
        for entry in grid:
            assert set(entry.keys()) == self.REQUIRED_KEYS

    def test_grid_size_is_cartesian_product(self):
        grid = build_param_grid(
            stake_short=[0.05, 0.10],
            stake_long=[0.10, 0.15],
            stop_loss=[0.01, 0.02],
            max_hold_hours=[4, 8],
        )
        assert len(grid) == 2 * 2 * 2 * 2

    def test_take_profit_is_double_stop_loss(self):
        grid = build_param_grid(
            stake_short=0.05,
            stake_long=0.10,
            stop_loss=[0.01, 0.03],
            max_hold_hours=4,
            take_profit_mult=2.0,
        )
        for entry in grid:
            assert abs(entry["take_profit_frac"] - entry["stop_loss_frac"] * 2.0) < 1e-9


class TestPred1hAlignment:
    def test_resample_last_value_per_hour(self):
        idx = pd.date_range("2024-01-01", periods=24, freq="5min")
        vals = np.arange(24, dtype=float)
        df = pd.DataFrame({SIGNAL_COLUMN: vals}, index=idx)

        result = df[SIGNAL_COLUMN].resample("1h").last()

        assert result.index[0] == pd.Timestamp("2024-01-01 00:00:00")
        assert result.iloc[0] == 11.0
        assert result.iloc[1] == 23.0

    def test_resample_row_count(self):
        df = _make_5m_predictions(periods=720)
        result = df[SIGNAL_COLUMN].resample("1h").last().dropna()
        assert len(result) == 720 // 12

    def test_resample_preserves_last_5m_value_in_each_bucket(self):
        idx = pd.date_range("2024-01-01", periods=36, freq="5min")
        vals = np.arange(36, dtype=float)
        df = pd.DataFrame({SIGNAL_COLUMN: vals}, index=idx)

        result = df[SIGNAL_COLUMN].resample("1h").last()

        assert result.iloc[0] == 11.0
        assert result.iloc[1] == 23.0
        assert result.iloc[2] == 35.0


class TestBuildStrategicLabelsFallback:
    def test_stake_values_are_rule_based_only(self):
        tf_cfg = TIMEFRAMES["1h"]
        df_raw = _make_ohlcv_1h(500)
        df_feat = make_strategic_features(df_raw, tf_cfg)

        df_labeled = _build_strategic_labels(df_feat, tf_cfg)

        allowed = {0.1, 0.2}
        actual = set(df_labeled["stake_long_frac"].round(6).unique())
        assert actual.issubset(allowed), f"Unexpected stake_long_frac values: {actual - allowed}"

    def test_all_required_columns_present(self):
        tf_cfg = TIMEFRAMES["1h"]
        df_raw = _make_ohlcv_1h(500)
        df_feat = make_strategic_features(df_raw, tf_cfg)

        df_labeled = _build_strategic_labels(df_feat, tf_cfg)

        for col in ["stake_long_frac", "stake_short_frac", "stop_loss_frac",
                    "take_profit_frac", "max_hold_hours",
                    "allow_trading", "recommended_leverage", "max_exposure_frac"]:
            assert col in df_labeled.columns, f"Missing column: {col}"


class TestBuildStrategicLabelsFromSimulation:
    PARAM_GRID_LONG_VALUES = {0.10, 0.15, 0.25}

    def _make_inputs(self):
        tf_cfg = TIMEFRAMES["1h"]
        df_raw_1h = _make_ohlcv_1h(400)
        df_feat = make_strategic_features(df_raw_1h, tf_cfg)
        df_5m_preds = _make_5m_predictions(periods=4800)
        return df_feat, df_5m_preds, tf_cfg

    def test_stake_values_come_from_param_grid(self):
        df_feat, df_5m_preds, tf_cfg = self._make_inputs()

        df_labeled = _build_strategic_labels_from_simulation(df_feat, df_5m_preds, tf_cfg)

        actual = set(df_labeled["stake_long_frac"].round(6).unique())
        assert actual.issubset(self.PARAM_GRID_LONG_VALUES), (
            f"Values not from param_grid: {actual - self.PARAM_GRID_LONG_VALUES}"
        )

    def test_rule_based_value_0_2_not_exclusively_present(self):
        df_feat, df_5m_preds, tf_cfg = self._make_inputs()

        df_labeled = _build_strategic_labels_from_simulation(df_feat, df_5m_preds, tf_cfg)

        actual = set(df_labeled["stake_long_frac"].round(6).unique())
        assert actual != {0.2}, (
            "stake_long_frac contains only 0.2 — simulation had no effect"
        )

    def test_returns_nonempty_dataframe(self):
        df_feat, df_5m_preds, tf_cfg = self._make_inputs()

        df_labeled = _build_strategic_labels_from_simulation(df_feat, df_5m_preds, tf_cfg)

        assert len(df_labeled) > 0


class TestNonOptimizedColumnsPreserved:
    def _make_inputs(self):
        tf_cfg = TIMEFRAMES["1h"]
        df_raw_1h = _make_ohlcv_1h(400)
        df_feat = make_strategic_features(df_raw_1h, tf_cfg)
        df_5m_preds = _make_5m_predictions(periods=4800)
        return df_feat, df_5m_preds, tf_cfg

    def test_risk_gate_columns_present_after_rule_based_labeling(self):
        tf_cfg = TIMEFRAMES["1h"]
        df_raw = _make_ohlcv_1h(500)
        df_feat = make_strategic_features(df_raw, tf_cfg)

        df_labeled = _build_strategic_labels(df_feat, tf_cfg)

        for col in ["allow_trading", "recommended_leverage", "max_exposure_frac"]:
            assert col in df_labeled.columns

    def test_risk_gate_columns_present_after_simulation_labeling(self):
        df_feat, df_5m_preds, tf_cfg = self._make_inputs()

        df_labeled = _build_strategic_labels_from_simulation(df_feat, df_5m_preds, tf_cfg)

        for col in ["allow_trading", "recommended_leverage", "max_exposure_frac"]:
            assert col in df_labeled.columns

    def test_allow_trading_values_are_binary(self):
        df_feat, df_5m_preds, tf_cfg = self._make_inputs()

        df_labeled = _build_strategic_labels_from_simulation(df_feat, df_5m_preds, tf_cfg)

        assert set(df_labeled["allow_trading"].unique()).issubset({0.0, 1.0})
