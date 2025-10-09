import os
from duckdb import df
import requests
import zipfile
import io
import pandas as pd
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import logging
import time
from typing import List, Optional, Tuple

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CACHE_DIR = "historical_data"
BINANCE_KLINES_MONTHLY = "monthly"
BINANCE_KLINES_DAILY = "daily"
BINANCE_URL_TEMPLATE = "https://data.binance.vision/data/spot/{period}/klines/{symbol}/{interval}/{file_stem}.zip"
COLUMN_NAMES = [
    "open_time", "open", "high", "low", "close", "volume",
    "close_time", "quote_asset_volume", "number_of_trades",
    "taker_buy_base_volume", "taker_buy_quote_volume", "ignore"
]
TIMEZONE = "UTC"
MAX_RETRIES = 3
RETRY_DELAY = 5  # seconds

def _normalize_timestamps(df: pd.DataFrame, timestamp_col: str = "date") -> pd.DataFrame:
    """
    Normalize timestamps of different magnitudes:
    - 13-digit (ms) → multiply by 1000 to convert to microseconds
    - 16-digit (us) → keep as is
    Converts everything to datetime64[ns, UTC] and sets as index.
    """
    ts = df[timestamp_col]

    # Multiply milliseconds to match microseconds
    df["timestamp_us"] = ts.apply(lambda x: x if x > 1e15 else x * 1000)

    # Convert to datetime
    df[timestamp_col] = pd.to_datetime(df["timestamp_us"], unit="us", utc=True, errors="coerce")

    # Set index and drop helper column
    df = df.set_index(timestamp_col)
    df = df.drop(columns=["timestamp_us"])

    # Drop any NaT that remain
    df = df[~df.index.isna()]

    return df

def _generate_day_range(start_date: datetime, end_date: datetime) -> List[datetime]:
    """Generate day range between start and end dates."""
    current = start_date.replace(hour=0, minute=0, second=0, microsecond=0)
    end = end_date.replace(hour=0, minute=0, second=0, microsecond=0)
    days = []
    while current <= end:
        days.append(current)
        current += relativedelta(days=1)
    return days

def _download_data_for_months(
    symbol: str,
    interval: str,
    months: List[Tuple[int, int]]
) -> List[pd.DataFrame]:
    """Download and merge data for each month."""
    try:
        dfs = []
        for year, month in months:
            df_month = _download_binance_klines(
                period=BINANCE_KLINES_MONTHLY,
                symbol=symbol,
                interval=interval,
                year=year,
                month=month
                )
            dfs.append(df_month)
            logger.info(f"Successfully loaded daily data for {symbol} {interval} {year}-{month:02d}")
        return dfs
    except Exception as e:
        logger.error(f"Error downloading daily data for {symbol} {interval} {year}-{month:02d}: {e}")
        raise

def _download_data_for_days(
    symbol: str,
    interval: str,
    start_date: datetime,
    end_date: datetime,
    cache_dir: str = None
) -> List[pd.DataFrame]:
    """Download data for each specified day from Binance's daily archives."""
    dfs = []
    days = _generate_day_range(start_date, end_date)
    for day in days:
        try:
            df_day = _download_binance_klines(
                period=BINANCE_KLINES_DAILY,
                symbol=symbol,
                interval=interval,
                year=day.year,
                month=day.month,
                day=day.day,
                cache_dir=cache_dir
                )
            dfs.append(df_day)
            logger.info(f"Successfully loaded daily data for {symbol} {interval} {day.year}-{day.month:02d}-{day.day:02d}")
        except FileNotFoundError:
            logger.warning(f"Daily data not found for {symbol} {interval} {day.year}-{day.month:02d}-{day.day:02d}. Skipping.")
        except Exception as e:
            logger.error(f"Error downloading daily data for {symbol} {interval} {day.year}-{day.month:02d}-{day.day:02d}: {e}")
    return dfs

def _download_file(url: str, max_retries: int = MAX_RETRIES) -> bytes:
    """Download a file with retry mechanism."""
    for attempt in range(max_retries):
        response = requests.get(url)
        if response.status_code == 200:
            return response.content
        elif response.status_code == 404:
            raise FileNotFoundError(f"Resource not found (404) for {url}")
        else:
            logger.warning(f"Attempt {attempt + 1}/{max_retries} failed to download {url}. Status code: {response.status_code}")
            if attempt < max_retries - 1:
                time.sleep(RETRY_DELAY)
    raise Exception(f"Failed to download file after {max_retries} attempts: {url}")

def _extract_csv_from_zip(zip_content: bytes) -> pd.DataFrame:
    """Extract CSV from a zip file and return as DataFrame."""
    with zipfile.ZipFile(io.BytesIO(zip_content)) as z:
        csv_file = z.namelist()[0]
        with z.open(csv_file) as f:
            df = pd.read_csv(f, header=None, names=COLUMN_NAMES)
    return df

def _get_file_stem(period: str, symbol: str, interval: str, year: int, month: int, day: Optional[int] = None) -> str:
    """Generate file stem for Binance data based on period."""
    month_str = f"{month:02d}"
    if period == BINANCE_KLINES_MONTHLY:
        return f"{symbol}-{interval}-{year}-{month_str}"
    elif period == BINANCE_KLINES_DAILY and day is not None:
        day_str = f"{day:02d}"
        return f"{symbol}-{interval}-{year}-{month_str}-{day_str}"
    else:
        raise ValueError("Invalid period or missing day for daily data.")

def _get_cache_path(file_stem: str, cache_dir: str = None) -> str:
    """Generate cache path for Binance data."""
    cache_dir = cache_dir or CACHE_DIR
    return os.path.join(cache_dir, f"{file_stem}.csv")

def _generate_month_range(start_date: datetime, end_date: datetime) -> List[Tuple[int, int]]:
    """Generate month range between start and end dates."""
    current = datetime(start_date.year, start_date.month, 1)
    end = datetime(end_date.year, end_date.month, 1)
    months = []
    while current <= end:
        months.append((current.year, current.month))
        current += relativedelta(months=1)
    return months

def _ensure_cache_directory_exists(cache_dir: str = None) -> None:
    """Ensure the cache directory exists."""
    cache_dir = cache_dir or CACHE_DIR
    os.makedirs(cache_dir, exist_ok=True)


def _process_ohlcv_data(df: pd.DataFrame) -> pd.DataFrame:
    """Process and keep only date (open_time) + OHLCV columns without converting timestamps."""
    # Select relevant columns and make a copy to avoid SettingWithCopyWarning
    df = df[["open_time", "open", "high", "low", "close", "volume"]].copy()

    # Rename open_time → date (keeping as original integer timestamp)
    df.rename(columns={"open_time": "date"}, inplace=True)

    # Ensure numeric columns are correct type
    numeric_cols = ["open", "high", "low", "close", "volume"]
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")

    return df

def _download_binance_klines(
    period: str,
    symbol: str,
    interval: str,
    year: int,
    month: int,
    day: Optional[int] = None,
    cache_dir: str = None
) -> pd.DataFrame:
    """Download and cache Binance klines data for a specific period (monthly or daily)."""
    file_stem = _get_file_stem(period, symbol, interval, year, month, day)
    csv_path = _get_cache_path(file_stem, cache_dir=cache_dir)

    if os.path.exists(csv_path):
        logger.info(f"Using cached file: {csv_path}")
        return read_ohlcv_df_from_file(csv_path, convert_dates=False)

    url = BINANCE_URL_TEMPLATE.format(period=period, symbol=symbol, interval=interval, file_stem=file_stem)
    logger.info(f"Attempting to download {file_stem} from Binance ({period} data)...")

    zip_content = _download_file(url)
    df = _extract_csv_from_zip(zip_content)
    df = _process_ohlcv_data(df)

    _ensure_cache_directory_exists(cache_dir)

    # Save cache file with original timestamp column intact
    df.to_csv(csv_path, index=False)
    logger.info(f"Saved to cache: {csv_path}")

    return df


def _read_df_from_file(file_path: str, convert_dates: bool = True) -> pd.DataFrame:
    """Read cached OHLCV data; optionally convert 'date' column to datetime."""
    df = pd.read_csv(file_path)

    if convert_dates:
        df["date"] = pd.to_datetime(df["date"], unit="ms", errors="coerce")
        df.set_index("date", inplace=True)
        df.index = df.index.tz_localize(TIMEZONE)

    return df


def read_ohlcv_df_from_file(file_path: str, convert_dates: bool = True) -> pd.DataFrame:
    """Read cached OHLCV DataFrame."""
    df = _read_df_from_file(file_path, convert_dates=convert_dates)
    if convert_dates:
        return df[["open", "high", "low", "close", "volume"]]
    else:
        return df[["date", "open", "high", "low", "close", "volume"]]


def get_historical_data(
    symbol: str,
    interval: str,
    backtesting_start: datetime,
    backtesting_end: datetime,
    cache_dir: str = None
) -> pd.DataFrame:
    """Download, merge, and return Binance historical OHLCV data."""
    all_dfs = []

    # Get current date
    today = datetime.now()

    # Determine the range for monthly data download
    monthly_end_date = min(backtesting_end, datetime(today.year, today.month, 1) - timedelta(days=1))
    
    if backtesting_start <= monthly_end_date:
        monthly_months = _generate_month_range(backtesting_start, monthly_end_date)
        if monthly_months:
            logger.info(f"Downloading monthly data from {monthly_months[0]} to {monthly_months[-1]}")
            monthly_dfs = _download_data_for_months(symbol, interval, monthly_months)
            all_dfs.extend(monthly_dfs)

    # --- Only try daily data if backtesting_end overlaps with current month ---
    current_month_start = datetime(today.year, today.month, 1)
    if backtesting_end >= current_month_start:
        daily_start_date = max(backtesting_start, datetime(today.year, today.month, 1))
        original_daily_end_date = backtesting_end

        retries_daily_data = 2
        current_daily_end_date = original_daily_end_date

        for attempt in range(retries_daily_data + 1):
            if daily_start_date <= current_daily_end_date:
                logger.info(f"Attempt {attempt + 1}: Downloading daily data from {daily_start_date.date()} to {current_daily_end_date.date()}")
                try:
                    daily_dfs = _download_data_for_days(symbol, interval, daily_start_date, current_daily_end_date, cache_dir=cache_dir)
                    all_dfs.extend(daily_dfs)
                    break
                except FileNotFoundError as e:
                    logger.warning(f"Failed to download recent daily data: {e}")
                    if attempt < retries_daily_data:
                        current_daily_end_date -= timedelta(days=1)
                    else:
                        logger.error(f"Failed after {retries_daily_data} adjustments.")
            else:
                break

    # Merge all
    final_df = pd.concat(all_dfs, ignore_index=True)

    # Normalize timestamps (handle both ms and us variants)
    final_df = _normalize_timestamps(final_df, timestamp_col="date")

    return final_df[["open", "high", "low", "close", "volume"]].astype(float)
