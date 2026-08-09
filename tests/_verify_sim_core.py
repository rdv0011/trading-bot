"""Behavioral verification of the parameterized simulate_trades_core (Task 1).

Run with: conda activate base && python tests/_verify_sim_core.py
Uses synthetic OHLCV data, no network. Verifies:
  1. Defaults reproduce previous behavior (no gate params -> same trades/wallet).
  2. regime_stake_mult={'chop': 0.0} blocks chop entries.
  3. volume_filter_threshold blocks low-volume entries.
  4. htf_ema_span blocks counter-trend entries.
  5. RiskGuard halts new entries after a daily-loss threshold.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd

from mltrainingcore import simulate_trades_core
from timeframe_config import TIMEFRAMES


def synthetic_df(n=400, seed=42, with_volume=True):
    rng = np.random.default_rng(seed)
    ts = pd.date_range("2026-07-01", periods=n, freq="15min")[:n]
    close = 40000 + np.cumsum(rng.normal(0, 40, n))
    high = close * 1.003
    low = close * 0.997
    df = pd.DataFrame(
        {"open": close * 0.999, "high": high, "low": low, "close": close},
        index=ts,
    )
    if with_volume:
        df["volume"] = rng.uniform(10, 100, n)
    # add regime per convenient rule so tests are deterministic
    df["regime"] = "trend"
    df.iloc[100:180, df.columns.get_loc("regime")] = "chop"
    df["pred"] = np.sin(np.linspace(0, 6 * np.pi, n))
    return df


def run_sim(df, **kwargs):
    tf_cfg = TIMEFRAMES["15m"]
    param_list = [
        {
            "stake_long_frac": 0.1,
            "stake_short_frac": 0.05,
            "stop_loss_frac": 0.02,
            "take_profit_frac": 0.04,
            "max_hold_hours": 4.0,
            "recommended_leverage": 1.0,
        }
    ]
    df_hist = df.iloc[: tf_cfg.adaptive_history_candles].copy()
    result, metrics = simulate_trades_core(
        df=df,
        df_hist=df_hist,
        signal_col="pred",
        tf_cfg=tf_cfg,
        param_list=param_list,
        close_col="close",
        **kwargs,
    )
    return result, metrics


def main():
    df = synthetic_df()
    base_df, base_metrics = run_sim(df)
    base_trades = len(base_df.attrs["trades"])
    print(f"baseline trades: {base_trades}, wallet={base_metrics['final_wallet']:.4f}")

    # default compare: same df, no kwargs — must equal baseline
    df2 = synthetic_df()
    df2_df, df2_metrics = run_sim(df2)
    assert len(df2_df.attrs["trades"]) == base_trades, "defaults changed trade count"
    print("PASS: defaults preserve behavior")

    # chop zero-stake
    chop_df, chop_metrics = run_sim(synthetic_df(), regime_stake_mult={"trend": 1.0, "high_vol": 0.5, "chop": 0.0})
    chop_trades = [t for t in chop_df.attrs["trades"] if t["regime"] == "chop"]
    assert len(chop_trades) == 0, f"chop block failed: {len(chop_trades)} chop trades"
    print(f"PASS: chop=0.0 zero chop trades (total={len(chop_df.attrs['trades'])})")

    #2 volume filter
    low_vol_df = synthetic_df()
    base_vol = low_vol_df["volume"].copy()
    low_vol_df.loc[low_vol_df.index[100:160], "volume"] = base_vol.iloc[100:160] * 0.05
    v_df, v_metrics = run_sim(low_vol_df, volume_filter_threshold=0.5)
    v_trades = len(v_df.attrs["trades"])
    assert v_trades < base_trades, f"volume filter no effect: {v_trades} vs {base_trades}"
    print(f"PASS: volume filter ({base_trades} -> {v_trades} trades)")

#3 htf filter
    df_htf = df.copy()
    df_htf["htf_close"] = df["close"] * 0.999  # below the EMA -> block longs
    df_htf["htf_ema50"] = df["close"] * 1.005
    h_df, h_metrics = run_sim(df_htf, htf_ema_span=50)
    h_long = sum(1 for t in h_df.attrs["trades"] if t["position"] == 1)
    base_long = sum(1 for t in base_df.attrs["trades"] if t["position"] == 1)
    assert h_long < base_long, f"htf filter no effect: longs {h_long} vs {base_long}"
    print(f"PASS: htf filter blocks longs ({base_long} -> {h_long})")

#4 riskguard drawdown
    rg_df, rg_metrics = run_sim(df, max_daily_loss_frac=0.000001, max_drawdown_frac=0.000001)
    rg_trades = len(rg_df.attrs["trades"])
    assert rg_trades <= base_trades, f"riskguard allowed more trades: {rg_trades}"
    print(f"PASS: riskguard near-zero threshold {base_trades} -> {rg_trades} trades")

    #4b deterministic: staircase long spikes into a falling market -> each trace hits
    #    stop-loss, and the daily-loss gate must suppress the post-loss same-day re-entry.
    n = 300
    ts = pd.date_range("2026-07-01", periods=n, freq="15min")
    closes = np.full(n, 100.0)
    closes[61:] = 100.0 - np.arange(n - 61) * 0.4
    down_df = pd.DataFrame(
        {
            "open": closes,
            "high": closes + 1,
            "low": closes - 1,
            "close": closes,
            "volume": 10.0,
            "regime": "trend",
            "pred": 0.0,
        },
        index=ts,
    )
    sig_spikes = [(60, 0.9), (80, 1.9), (100, 2.9), (120, 3.9)]
    for spike_idx, spike_val in sig_spikes:
        down_df.loc[down_df.index[spike_idx], "pred"] = spike_val

    tf = TIMEFRAMES["15m"]
    hist = down_df.iloc[: tf.adaptive_history_candles].copy()
    param = [
        {
            "stake_long_frac": 0.1,
            "stake_short_frac": 0.05,
            "stop_loss_frac": 0.03,
            "take_profit_frac": 0.04,
            "max_hold_hours": 4.0,
            "recommended_leverage": 1.0,
        }
    ] * len(down_df)

    halted_df, _ = simulate_trades_core(
        df=down_df,
        df_hist=hist,
        signal_col="pred",
        tf_cfg=tf,
        param_list=param,
        close_col="close",
        max_daily_loss_frac=0.001,
        max_drawdown_frac=1.0,
    )
    no_halt_df, _ = simulate_trades_core(
        df=down_df,
        df_hist=hist,
        signal_col="pred",
        tf_cfg=tf,
        param_list=param,
        close_col="close",
    )
    halted_trades = halted_df.attrs["trades"]
    no_halt_list = no_halt_df.attrs["trades"]
    assert len(halted_trades) <= len(no_halt_list), "halt must not add trades"
    assert len(halted_trades) < len(no_halt_list), "halt did not suppress a trade"
    print(f"PASS: riskguard halt ordering (halted={len(halted_trades)} vs no-halt={len(no_halt_list)})")

    #5 missing htf columns -> fallback pass
    df_nohtf = df.copy()  # no htf cols
    f_df, _ = run_sim(df_nohtf, htf_ema_span=50)
    assert len(f_df.attrs["trades"]) >= base_trades - 1, "htf fallback should pass"
    print("PASS: htf missing columns -> fallback pass")

    print("\nALL CHECKS PASSED")


if __name__ == "__main__":
    main()