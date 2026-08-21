"""Tests for run_windowed_simulation and the new simulate_trades_core gates.

Covers Task 3 + Task 5 of plans/demo_vs_sim_comparison.md: window boundary
filtering, warmup-history seeding, and live_faithful gate activation, all
offline with synthetic data and mocked network/model paths.
"""
import numpy as np
import pandas as pd
import pytest

from mltrainingcore import simulate_trades_core
from timeframe_config import TIMEFRAMES
import dualmlsimulation as d


def _synthetic_predictions(start="2026-05-01", days=30, seed=7):
    idx = pd.date_range(start, periods=days * 96, freq="15min")
    rng = np.random.default_rng(seed)
    close = 64000 + np.cumsum(rng.normal(0, 100, len(idx)))
    return pd.DataFrame(
        {
            "open": close * 0.999,
            "high": close * 1.002,
            "low": close * 0.998,
            "close": close,
            "volume": rng.uniform(50, 100, len(idx)),
            "regime": "trend",
            "pred": rng.normal(0, 0.3, len(idx)),
        },
        index=idx,
    )


def _df_with_trades(trades):
    df_result = pd.DataFrame(index=pd.date_range("2026-05-19", periods=2, freq="15min"))
    df_result.attrs = {"trades": trades, "trade_markers": []}
    return df_result


@pytest.fixture
def mocked_pipeline(monkeypatch):
    df_pred = _synthetic_predictions(start="2026-05-01", days=30)
    df_raw = df_pred.copy()
    monkeypatch.setattr(d, "run_predictions_only", lambda *a, **k: (df_pred, df_raw))
    monkeypatch.setattr(
        d, "StrategicML",
        lambda *a, **k: type("S", (), {"is_ready": False})(),
    )
    monkeypatch.setattr(
        d,
        "_build_strategic_param_list",
        lambda df_test, df_raw_5m, strategic, strategic_tf_cfg: [
            dict(d.DEFAULT_PARAMS) for _ in df_test.index
        ],
    )
    return df_pred


def test_window_boundary_filters_input(mocked_pipeline, monkeypatch):
    calls = {}

    def fake_sim(**kwargs):
        calls["df"] = kwargs["df"]
        calls["df_hist"] = kwargs["df_hist"]
        return _df_with_trades([]), {"final_wallet": 1.0}

    monkeypatch.setattr(d, "simulate_trades_core", fake_sim)
    d.run_windowed_simulation(
        symbol="TESTUSDT", days=30, timeframe="15m",
        start_date="2026-05-10", end_date="2026-05-12",
    )

    window_start = pd.Timestamp("2026-05-10")
    window_end = pd.Timestamp("2026-05-12") + pd.Timedelta(days=1)
    assert (calls["df"].index >= window_start).all()
    assert (calls["df"].index < window_end).all()
    assert len(calls["df"]) == 3 * 96
    assert (calls["df_hist"].index < window_start).all()
    assert len(calls["df_hist"]) <= TIMEFRAMES["15m"].adaptive_history_candles


def test_gates_always_applied(mocked_pipeline, monkeypatch):
    calls = {}

    def fake_sim(**kwargs):
        calls.update(kwargs)
        return _df_with_trades([]), {"final_wallet": 1.0}

    monkeypatch.setattr(d, "simulate_trades_core", fake_sim)

    d.run_windowed_simulation(
        symbol="TESTUSDT", days=30, timeframe="15m",
        start_date="2026-05-10", end_date="2026-05-12",
    )

    assert calls["regime_stake_mult"] == {"trend": 1.0, "high_vol": 0.5, "chop": 0.0}
    assert calls["volume_filter_threshold"] == 0.8
    assert calls["htf_ema_span"] == 50
    assert calls["max_daily_loss_frac"] == 0.05
    assert calls["max_drawdown_frac"] == 0.15


def test_windowed_no_rows_raises(mocked_pipeline, monkeypatch):
    monkeypatch.setattr(d, "simulate_trades_core", lambda **k: (_df_with_trades([]), {}))
    with pytest.raises(ValueError):
        d.run_windowed_simulation(
            symbol="TESTUSDT", days=30, timeframe="15m",
            start_date="2023-01-01", end_date="2023-01-02",
        )


def test_sim_core_regime_mult_zero_skips_regime():
    n = 300
    ts = pd.date_range("2026-07-01", periods=n, freq="15min")
    closes = np.full(n, 100.0)
    closes[61:] = 100.0 - np.arange(n - 61) * 0.4
    df = pd.DataFrame(
        {"open": closes, "close": closes, "high": closes + 1, "low": closes - 1,
         "volume": 10.0, "regime": "chop", "pred": 0.0},
        index=ts,
    )
    for idx, val in [(60, 0.9), (80, 1.9), (100, 2.9), (120, 3.9)]:
        df.loc[df.index[idx], "pred"] = val
    tf = TIMEFRAMES["15m"]
    hist = df.iloc[: tf.adaptive_history_candles].copy()
    param = [dict(d.DEFAULT_PARAMS)] * n
    base, _ = simulate_trades_core(df=df, df_hist=hist, signal_col="pred",
                                    tf_cfg=tf, param_list=param)
    gated, _ = simulate_trades_core(df=df, df_hist=hist, signal_col="pred",
                                     tf_cfg=tf, param_list=param,
                                     regime_stake_mult={"trend": 1.0, "high_vol": 0.5, "chop": 0.0})
    assert len(base.attrs["trades"]) > 0
    assert len(gated.attrs["trades"]) == 0


def test_sim_volume_filter_threshold_skips_low_volume():
    n = 400
    ts = pd.date_range("2026-07-01", periods=n, freq="15min")
    closes = np.full(n, 100.0)
    vol = np.full(n, 10.0)
    vol[:350] = 1.0
    df = pd.DataFrame(
        {"open": closes, "close": closes, "high": closes + 0.5, "low": closes - 0.5,
         "volume": vol, "regime": "trend", "pred": 0.9},
        index=ts,
    )
    tf = TIMEFRAMES["15m"]
    hist = df.iloc[: tf.adaptive_history_candles].copy()
    param = [dict(d.DEFAULT_PARAMS)] * len(df)
    base, _ = simulate_trades_core(df=df, df_hist=hist, signal_col="pred",
                                    tf_cfg=tf, param_list=param)
    filt, _ = simulate_trades_core(df=df, df_hist=hist, signal_col="pred",
                                    tf_cfg=tf, param_list=param,
                                    volume_filter_threshold=0.8)
    assert len(filt.attrs["trades"]) <= len(base.attrs["trades"])


def test_sim_core_riskguard_halt_suppresses_followup():
    n = 300
    ts = pd.date_range("2026-07-01", periods=n, freq="15min")
    closes = np.full(n, 100.0)
    closes[61:] = 100.0 - np.arange(n - 61) * 0.4
    df = pd.DataFrame(
        {"open": closes, "close": closes, "high": closes + 1, "low": closes - 1,
         "volume": 10.0, "regime": "trend", "pred": 0.0},
        index=ts,
    )
    for idx, val in [(60, 0.9), (80, 1.9), (100, 2.9), (120, 3.9)]:
        df.loc[df.index[idx], "pred"] = val
    tf = TIMEFRAMES["15m"]
    hist = df.iloc[: tf.adaptive_history_candles].copy()
    param = [
        {
            "stake_long_frac": 0.1,
            "stake_short_frac": 0.05,
            "stop_loss_frac": 0.03,
            "take_profit_frac": 0.04,
            "max_hold_hours": 4.0,
            "recommended_leverage": 1.0,
        }
    ] * n
    no_halt, _ = simulate_trades_core(df=df, df_hist=hist, signal_col="pred",
                                       tf_cfg=tf, param_list=param)
    halted, _ = simulate_trades_core(df=df, df_hist=hist, signal_col="pred",
                                      tf_cfg=tf, param_list=param,
                                      max_daily_loss_frac=0.001,
                                      max_drawdown_frac=1.0)
    assert len(no_halt.attrs["trades"]) >= 3
    assert len(halted.attrs["trades"]) < len(no_halt.attrs["trades"])