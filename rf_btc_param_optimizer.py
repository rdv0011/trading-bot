import gc
import math
import numpy as np
import pandas as pd
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from sklearn.ensemble import RandomForestRegressor
from typing import Dict, Any, List

# Use same timezone as original
TIMEZONE = "UTC"

# Parameter tuning config (kept from your original)
PARAM_CONFIG = [
    {"name": "lookback_period", "coarse_grid": [100, 200, 300], "fine_range": (100, 400), "step": 25, "priority": 1},
    {"name": "price_change_threshold_up", "coarse_grid": [0.003, 0.005, 0.007], "fine_range": (0.002, 0.01), "step": 0.001, "priority": 1},
    {"name": "price_change_threshold_down", "coarse_grid": [-0.003, -0.005, -0.007], "fine_range": (-0.01, -0.002), "step": 0.001, "priority": 1},
    {"name": "fraction_portfolio_per_trade", "coarse_grid": [0.8, 0.9, 1.0], "fine_range": (0.5, 1.0), "step": 0.05, "priority": 2},
    {"name": "take_profit_multiplier", "coarse_grid": [0.8, 1.0, 1.2], "fine_range": (0.5, 1.5), "step": 0.1, "priority": 2},
    {"name": "stop_loss_multiplier", "coarse_grid": [0.3, 0.5, 0.7], "fine_range": (0.2, 0.8), "step": 0.1, "priority": 2},
]

# -----------------------------
# Phase 1: Train RF in-memory & precompute predictions
# -----------------------------
def train_model_and_precompute_predictions(
    hist_df: pd.DataFrame,
    compute_frequency: int = 15,
    horizon: int = 15,
    rf_kwargs: Dict[str, Any] = None,
    feature_fn = None,
    current_datetime: datetime = None,
):
    """
    1) Compute features using feature_fn
    2) Train RandomForestRegressor on the full hist_df_features (horizon)
    3) Precompute predictions at decision timestamps (every compute_frequency minutes)
    Returns: (rf_model, hist_df_features, decision_index_array, pred_array)
    """
    if feature_fn is None:
        raise RuntimeError("feature_fn not provided")

    # compute features once
    hist_df_features = feature_fn(hist_df, current_datetime=current_datetime)

    # prepare training target and X
    df_train = hist_df_features.copy()
    df_train["target"] = df_train["close"].shift(-horizon)
    df_train = df_train.dropna(inplace=False)
    if df_train.empty:
        raise ValueError("Not enough data to train RF with the given horizon")

    X = df_train.drop(columns=["target"])
    y = df_train["target"]

    rf_kwargs = rf_kwargs or {}
    rf = RandomForestRegressor(**rf_kwargs)
    rf.fit(X, y)

    # Determine decision indices (timestamps) for generating signals:
    timestamps = hist_df_features.index
    minute_index = (timestamps.view(np.int64) // 10**9) // 60
    start_min = int(minute_index[0])
    decision_mask = ((minute_index - start_min) % compute_frequency) == 0
    decision_idx = np.nonzero(decision_mask)[0]  # integer positions into hist_df_features

    # Precompute predictions at decision timestamps (vectorized)
    X_all = hist_df_features.drop(columns=[c for c in hist_df_features.columns if c == "target"], errors="ignore")
    if len(X_all) == 0:
        raise ValueError("No feature columns available for prediction")
    X_decisions = X_all.iloc[decision_idx]
    preds = rf.predict(X_decisions)  # numpy array

    return rf, hist_df_features, decision_idx, preds

# -----------------------------
# Simulate trades allowing multi-candle durations
# -----------------------------
def simulate_trades_multi_candle(
    prices_df: pd.DataFrame,
    decision_idx: np.ndarray,
    preds: np.ndarray,
    params: Dict[str, Any],
    compute_frequency: int,
    initial_capital: float = 1.0,
) -> Dict[str, Any]:
    """
    Simulate trades on minute bars.
    - decisions occur at indices provided by `decision_idx`, and predictions given by `preds` align with decision_idx.
    - positions persist across multiple candles until TP/SL, opposite signal (configurable), or end of window.
    Returns metrics dict: total_return, equity_curve (pd.Series), num_trades, sharpe, max_drawdown
    """

    # Unpack params with defaults
    fraction = float(params.get("fraction_portfolio_per_trade", 0.1))
    take_mult = float(params.get("take_profit_multiplier", 1.2))
    stop_mult = float(params.get("stop_loss_multiplier", 0.8))
    thr_up = float(params.get("price_change_threshold_up", 0.006))
    thr_down = float(params.get("price_change_threshold_down", -0.005))
    max_long_exposure = float(params.get("max_portfolio_exposure_long", 1.0))
    max_short_exposure = float(params.get("max_portfolio_exposure_short", 0.5))

    close_prices = prices_df["close"].values
    n = len(close_prices)

    # Prepare arrays for quick access
    decision_set = set(decision_idx.tolist())

    # Map from decision index to expected change / predicted price
    pred_map = dict(zip(decision_idx.tolist(), preds.tolist()))

    # State variables
    capital = float(initial_capital)
    equity = capital
    equity_curve = np.zeros(n, dtype=float)
    current_position = None  # dict with keys: side, entry_price, tp, sl, quantity, entry_idx
    num_trades = 0

    # track position value for mark-to-market
    position_value = 0.0

    # We'll allow new entries only at decision times.
    # If a position is open, it remains open and is checked every minute for TP/SL.
    # If opposite signal occurs at a decision time, we will close existing position at close price and (optionally) open new one in same minute.

    # Precompute per-minute returns later from equity_curve.

    for t in range(n):
        price = close_prices[t]

        # 1) If there is an open position, check TP/SL at the current minute price
        if current_position is not None:
            side = current_position["side"]
            entry_price = current_position["entry_price"]
            tp = current_position["tp"]
            sl = current_position["sl"]
            quantity = current_position["quantity"]

            # Check TP/SL
            exited = False
            if side == "long":
                if price >= tp:
                    exit_price = tp
                    pnl = (exit_price - entry_price) / entry_price
                    exited = True
                elif price <= sl:
                    exit_price = sl
                    pnl = (exit_price - entry_price) / entry_price
                    exited = True
            else:  # short
                if price <= tp:
                    exit_price = tp
                    pnl = (entry_price - exit_price) / entry_price
                    exited = True
                elif price >= sl:
                    exit_price = sl
                    pnl = (entry_price - exit_price) / entry_price
                    exited = True

            if exited:
                # realize pnl proportionally to capital fraction used
                trade_return = pnl * (current_position["value_traded"] / capital)
                equity += trade_return
                current_position = None
                num_trades += 1
                position_value = 0.0
                # proceed to next minute (but still allow new entry in same minute at decision)

        # 2) At decision times, optionally open/close positions according to signal
        if t in decision_set:
            pred_price = pred_map.get(t, None)
            if pred_price is not None:
                expected_change = (pred_price - price) / price
                # Determine desired direction
                if expected_change > thr_up:
                    desired = "long"
                elif expected_change < thr_down:
                    desired = "short"
                else:
                    desired = "flat"

                # If we have an existing position and desired is opposite, close existing and possibly open new
                if current_position is not None:
                    if desired == "flat":
                        # do nothing (hold or wait for TP/SL)
                        pass
                    elif desired != current_position["side"]:
                        # Close current at current price
                        entry_price = current_position["entry_price"]
                        side = current_position["side"]
                        if side == "long":
                            pnl = (price - entry_price) / entry_price
                        else:
                            pnl = (entry_price - price) / entry_price
                        trade_return = pnl * (current_position["value_traded"] / capital)
                        equity += trade_return
                        current_position = None
                        num_trades += 1
                        position_value = 0.0
                        # We'll allow opening below

                # If no position, and desired is long/short, attempt to open
                if current_position is None and desired in ("long", "short"):
                    value_to_trade = capital * fraction
                    expected_move = abs(price * expected_change)
                    if expected_move == 0:
                        expected_move = price * 0.001
                    if desired == "long":
                        tp = price + (expected_move * take_mult)
                        sl = price - (expected_move * stop_mult)
                        # check exposure cap
                        if (value_to_trade) <= (max_long_exposure * capital):
                            quantity = value_to_trade / price
                            current_position = {
                                "side": "long",
                                "entry_price": price,
                                "tp": tp,
                                "sl": sl,
                                "quantity": quantity,
                                "entry_idx": t,
                                "value_traded": value_to_trade,
                            }
                            position_value = value_to_trade
                    else:  # short
                        tp = price - (expected_move * take_mult)
                        sl = price + (expected_move * stop_mult)
                        if (value_to_trade) <= (max_short_exposure * capital):
                            quantity = value_to_trade / price
                            current_position = {
                                "side": "short",
                                "entry_price": price,
                                "tp": tp,
                                "sl": sl,
                                "quantity": quantity,
                                "entry_idx": t,
                                "value_traded": value_to_trade,
                            }
                            position_value = value_to_trade

        # 3) Update mark-to-market position_value and equity_curve
        if current_position is not None:
            # compute unrealized pnl on mark-to-market basis
            if current_position["side"] == "long":
                unreal_pnl = (price - current_position["entry_price"]) / current_position["entry_price"]
            else:
                unreal_pnl = (current_position["entry_price"] - price) / current_position["entry_price"]
            # unrealized change applied to fraction of capital used
            equity_mt = equity + unreal_pnl * (current_position["value_traded"] / capital)
            equity_curve[t] = equity_mt
        else:
            equity_curve[t] = equity

    # End-of-window: close any open position at last price
    if current_position is not None:
        price = close_prices[-1]
        entry_price = current_position["entry_price"]
        side = current_position["side"]
        if side == "long":
            pnl = (price - entry_price) / entry_price
        else:
            pnl = (entry_price - price) / entry_price
        trade_return = pnl * (current_position["value_traded"] / capital)
        equity += trade_return
        # write final equity to curve end
        equity_curve[-1] = equity
        num_trades += 1

    # Fill any zeros at start (e.g., before first index) with initial capital
    first_nonzero = np.nonzero(equity_curve)[0]
    if len(first_nonzero) == 0:
        equity_curve[:] = capital
    else:
        idx0 = first_nonzero[0]
        equity_curve[:idx0] = equity_curve[idx0]

    # Convert to pandas Series with same index as prices_df
    equity_series = pd.Series(equity_curve, index=prices_df.index)

    # Compute per-minute returns of equity (simple pct change)
    returns = equity_series.pct_change().fillna(0.0).values

    # Total return (fraction)
    total_return = float(equity_series.iloc[-1] / capital - 1.0)

    # Sharpe (annualized). For crypto, use 365*24*60 minutes/year
    minutes_per_year = 365.0 * 24.0 * 60.0
    # If returns are all zero -> sharpe zero
    if returns.std() == 0 or np.isclose(returns.std(), 0.0):
        sharpe = 0.0
    else:
        sharpe = (np.mean(returns) / np.std(returns)) * math.sqrt(minutes_per_year)

    # Max drawdown: based on equity_series
    rolling_max = equity_series.cummax()
    drawdown = (equity_series / rolling_max) - 1.0
    max_drawdown = float(drawdown.min())

    metrics = {
        "total_return": total_return,
        "num_trades": int(num_trades),
        "sharpe": float(sharpe),
        "max_drawdown": float(max_drawdown),
        "equity_series": equity_series,  # included for debugging / more metrics
    }
    return metrics

# -----------------------------
# Candidate evaluation (thread-friendly)
# -----------------------------
def evaluate_candidate_threadsafe(
    hist_df_features: pd.DataFrame,
    prices_df: pd.DataFrame,
    decision_idx: np.ndarray,
    preds: np.ndarray,
    params: Dict[str, Any],
    compute_frequency: int,
) -> Dict[str, Any]:
    """
    Evaluate a single params dict using precomputed preds & decision indices.
    Returns metrics dict (see simulate_trades_multi_candle).
    """
    try:
        metrics = simulate_trades_multi_candle(
            prices_df=prices_df,
            decision_idx=decision_idx,
            preds=preds,
            params=params,
            compute_frequency=compute_frequency,
            initial_capital=1.0,
        )
        return {"status": "ok", "params": params, "metrics": metrics}
    except Exception as e:
        return {"status": "error", "params": params, "error": str(e)}

# -----------------------------
# Top-level optimizer (coarse grid + binary refinement) using threads & precomputed preds
# -----------------------------
def optimize_parameters_two_phase(
    hist_df: pd.DataFrame,
    start_dt: datetime,
    end_dt: datetime,
    initial_params: Dict[str, Any] = None,
    binary_search_iters: int = 4,
    opt_workers: int = 6,
    logger=print,
    rf_kwargs: Dict[str, Any] = None,
    feature_fn = None,
):
    """
    Two-phase optimizer:
      - Phase 1 (sequential): train RF in memory and precompute predictions on decision timestamps
      - Phase 2 (parallel threads): run coarse + binary parameter search where each candidate uses only precomputed preds
    """

    base = {
        "compute_frequency": 15,
        "lookback_period": 200,
        "fraction_portfolio_per_trade": 0.1,
        "price_change_threshold_up": 0.006,
        "price_change_threshold_down": -0.005,
        "max_portfolio_exposure_long": 1.0,
        "max_portfolio_exposure_short": 0.5,
        "take_profit_multiplier": 1.2,
        "stop_loss_multiplier": 0.8,
    }
    if initial_params:
        base.update(initial_params)

    # slice hist_df for requested [start_dt, end_dt]
    df_slice = hist_df.copy()
    if df_slice.index.tz is None:
        df_slice.index = df_slice.index.tz_localize("UTC")
    df_slice = df_slice.sort_index()
    df_slice = df_slice[(df_slice.index >= pd.to_datetime(start_dt).tz_localize("UTC")) &
                        (df_slice.index <= pd.to_datetime(end_dt).tz_localize("UTC"))]
    if df_slice.empty:
        raise ValueError("No data in requested [start_dt, end_dt] window")

    compute_frequency = int(base.get("compute_frequency", 15))
    horizon = compute_frequency  # keep parity with earlier logic (predict horizon = compute_frequency)

    # Phase 1: train RF & precompute preds
    logger("Phase 1: training RF and precomputing predictions...")
    rf_model, hist_df_features, decision_idx, preds = train_model_and_precompute_predictions(
        hist_df=df_slice,
        compute_frequency=compute_frequency,
        horizon=horizon,
        rf_kwargs=rf_kwargs or {"n_jobs": -1, "n_estimators": 100},
        feature_fn=feature_fn,
        current_datetime=end_dt, # ✅ Use end_dt as default reference datetime if not provided
    )
    logger(f"Trained RF; decisions count = {len(decision_idx)}")

    # prices_df for simulation: use hist_df_features (includes 'close' etc.)
    prices_df = hist_df_features[["close", "high", "low", "volume"]].copy()

    # Helper to run candidates in parallel
    def run_candidate_batch(candidate_params: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        results = []
        if opt_workers <= 1:
            # serial
            for p in candidate_params:
                res = evaluate_candidate_threadsafe(hist_df_features, prices_df, decision_idx, preds, p, compute_frequency)
                results.append(res)
            return results

        with ThreadPoolExecutor(max_workers=opt_workers) as exe:
            futures = {exe.submit(evaluate_candidate_threadsafe, hist_df_features, prices_df, decision_idx, preds, p, compute_frequency): p for p in candidate_params}
            for fut in as_completed(futures):
                try:
                    results.append(fut.result())
                except Exception as e:
                    results.append({"status": "error", "params": futures[fut], "error": str(e)})
        return results

    # Parameter search (coarse + binary refinement) similar to original
    best_params = dict(base)

    for priority in sorted(set(cfg["priority"] for cfg in PARAM_CONFIG)):
        for cfg in [c for c in PARAM_CONFIG if c["priority"] == priority]:
            pname = cfg["name"]

            logger(f"\n=== Coarse search for {pname} ===")
            # build candidate params
            candidate_params = []
            for val in cfg["coarse_grid"]:
                ptest = dict(best_params)
                ptest[pname] = int(val) if pname == "lookback_period" else float(val)
                candidate_params.append(ptest)

            results = run_candidate_batch(candidate_params)
            # Parse total_return or fallback -inf
            best_val = None
            best_profit = -np.inf
            for r in results:
                if r.get("status") == "ok":
                    profit = float(r["metrics"]["total_return"])
                else:
                    profit = -np.inf
                val = r["params"][pname]
                logger(f"  Candidate {pname}={val} -> profit={profit:.6f}")
                if profit > best_profit:
                    best_profit = profit
                    best_val = val

            best_params[pname] = best_val
            logger(f"✔ Best coarse {pname}={best_val} profit={best_profit:.6f}")

            # Binary refinement
            low, high = cfg["fine_range"]
            step = cfg["step"]
            logger(f"\n--- Refining {pname} in range ({low}, {high}), step={step} ---")
            for it in range(binary_search_iters):
                mid = (low + high) / 2.0
                candidate_vals = [low, mid, high]
                candidate_params = []
                for c in candidate_vals:
                    val = int(round(c)) if pname == "lookback_period" else round(c, 6)
                    ptest = dict(best_params)
                    ptest[pname] = val
                    candidate_params.append(ptest)

                results = run_candidate_batch(candidate_params)

                # evaluate
                profits = []
                for r in results:
                    if r.get("status") == "ok":
                        profits.append(float(r["metrics"]["total_return"]))
                    else:
                        profits.append(-np.inf)

                for ptest, profit in zip(candidate_params, profits):
                    logger(f"   Candidate {pname}={ptest[pname]} -> profit={profit:.6f}")

                # pick best
                best_idx = int(np.nanargmax(profits))
                best_val = candidate_params[best_idx][pname]
                best_profit = profits[best_idx]
                best_params[pname] = best_val
                logger(f"✔ Best so far {pname}={best_val}, profit={best_profit:.6f}")

                # Narrow range
                if best_idx == 0:
                    high = mid
                elif best_idx == 2:
                    low = mid
                else:
                    low = max(low, mid - step)
                    high = min(high, mid + step)

                logger(f" -> New search range ({low}, {high})")

    # Final cleanup
    gc.collect()

    # Return best_params and optionally diagnostics: retrain RF on final choice and compute its metrics
    logger("\nOptimization finished. Recomputing final metrics for best_params...")
    final_metrics = evaluate_candidate_threadsafe(hist_df_features, prices_df, decision_idx, preds, best_params, compute_frequency)
    return {
        "best_params": best_params,
        "final_metrics": final_metrics,
        "rf_model": rf_model,
        "decision_idx": decision_idx,
        "preds": preds,
        "hist_df_features": hist_df_features,
    }