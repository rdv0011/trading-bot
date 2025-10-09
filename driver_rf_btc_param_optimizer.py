import os
import argparse
import pandas as pd
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

from rf_btc_param_optimizer import optimize_parameters_two_phase
from historical_data import get_historical_data

from rf_btc_lumibot_strategy import RandomForestBTCStrategy  # when run inside your original script
feature_fn = RandomForestBTCStrategy._compute_features

def generate_training_dataset(
    start_dt: datetime,
    end_dt: datetime,
    loopback_days: int,
    step_days: int,
    historical_folder: str,
    params_log: str,
    asset_symbol="BTC",
    quote_asset_symbol="USDT",
    benchmark_asset_symbol="SPY",
):
    """Iterate over rolling windows and run optimizer to generate dataset of tuned parameters."""
    os.makedirs(os.path.dirname(params_log), exist_ok=True)

    # CSV logging setup
    if not os.path.exists(params_log):
        pd.DataFrame(
            columns=[
                "window_start",
                "window_end",
                "total_return",
                "sharpe",
                "max_drawdown",
                "num_trades",
                "optimized_params",
            ]
        ).to_csv(params_log, index=False)

    binance_symbol = f"{asset_symbol}{quote_asset_symbol}"

    print(f"📡 Fetching historical data for {binance_symbol} from {start_dt:%Y-%m-%d} to {end_dt:%Y-%m-%d}...")
    # ✅ Use the original data loader
    full_data = get_historical_data(binance_symbol, "1m", start_dt, end_dt, cache_dir=historical_folder)

    if full_data.empty:
        raise ValueError("No historical data retrieved for the given range.")

    full_data = full_data.sort_index()

    current_start = start_dt
    loopback_delta = timedelta(days=loopback_days)
    step_delta = timedelta(days=step_days)

    while current_start + loopback_delta <= end_dt:
        current_end = current_start + loopback_delta
        print(f"\n=== Optimizing window {current_start:%Y-%m-%d} → {current_end:%Y-%m-%d} ===")

        try:
            # Convert naive datetimes to UTC-aware before slicing
            current_start_aware = pd.to_datetime(current_start).tz_localize("UTC")
            current_end_aware = pd.to_datetime(current_end).tz_localize("UTC")
            # Slice window data
            hist_df = full_data[(full_data.index >= current_start_aware) & (full_data.index <= current_end_aware)]
            if hist_df.empty:
                print(f"⚠️ No data in window {current_start:%Y-%m-%d} – {current_end:%Y-%m-%d}, skipping.")
                current_start += step_delta
                continue

            # Run two-phase optimization (training + parameter search)
            result = optimize_parameters_two_phase(
                hist_df=hist_df,
                start_dt=current_start,
                end_dt=current_end,
                initial_params=None,
                binary_search_iters=3,
                opt_workers=8,
                logger=print,
                rf_kwargs={"n_estimators": 200, "max_depth": 10, "n_jobs": -1, "random_state": 42},
                feature_fn=feature_fn,
            )

            metrics = result["final_metrics"]["metrics"]
            best_params = result["best_params"]

            # Log results
            log_entry = {
                "window_start": current_start.strftime("%Y-%m-%d"),
                "window_end": current_end.strftime("%Y-%m-%d"),
                "total_return": metrics["total_return"],
                "sharpe": metrics["sharpe"],
                "max_drawdown": metrics["max_drawdown"],
                "num_trades": metrics["num_trades"],
                "optimized_params": best_params,
            }

            print(f"✅ Best params: {best_params}")
            print(f"📈 Return={metrics['total_return']:.3f}, Sharpe={metrics['sharpe']:.3f}, Drawdown={metrics['max_drawdown']:.3f}")

            pd.DataFrame([log_entry]).to_csv(params_log, mode="a", header=False, index=False)

        except Exception as e:
            print(f"❌ Error optimizing window {current_start:%Y-%m-%d}: {e}")

        current_start += step_delta

    print(f"\n✅ Optimization dataset generation complete. Logged results to {params_log}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate dataset of optimized parameters over rolling windows")
    parser.add_argument("--start", required=True, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", required=True, help="End date (YYYY-MM-DD)")
    parser.add_argument("--loopback-days", type=int, default=10, help="Window length in days")
    parser.add_argument("--step-days", type=int, default=1, help="Sliding step size in days")
    parser.add_argument("--historical-folder", required=True, default="./historical_data", help="Folder with cached historical data")
    parser.add_argument("--params-log", required=True, default="./optimized_params.csv", help="Path to CSV log of optimized parameters")

    args = parser.parse_args()

    start_dt = datetime.strptime(args.start, "%Y-%m-%d")
    end_dt = datetime.strptime(args.end, "%Y-%m-%d")

    generate_training_dataset(
        start_dt,
        end_dt,
        args.loopback_days,
        args.step_days,
        args.historical_folder,
        args.params_log,
        asset_symbol="BTC",
        quote_asset_symbol="USDT",
        benchmark_asset_symbol="SPY",
    )