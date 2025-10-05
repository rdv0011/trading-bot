import os
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
TIMESTAMP_MIN = 1262304000000000  # January 1, 2010
TIMESTAMP_MAX = 4102444800000000  # January 1, 2099
MAX_RETRIES = 3
RETRY_DELAY = 5  # seconds

def get_historical_data(
    symbol: str,
    interval: str,
    backtesting_start: datetime,
    backtesting_end: datetime,
    cache_dir: str = None
) -> pd.DataFrame:
    """
    Download and merge Binance kline data for a specified date range,
    handling both monthly and daily data and adjusting for missing recent daily data.
    """
    
    all_dfs = []

    # Get current date
    today = datetime.now()

    # Determine the range for monthly data download
    # Monthly data is available for full past months.
    # So, monthly_end_date is the last day of the month before the current month.
    monthly_end_date = min(backtesting_end, datetime(today.year, today.month, 1) - timedelta(days=1))
    
    if backtesting_start <= monthly_end_date:
        monthly_months = _generate_month_range(backtesting_start, monthly_end_date)
        if monthly_months:
            logger.info(f"Downloading monthly data from {monthly_months[0]} to {monthly_months[-1]}")
            monthly_dfs = _download_data_for_months(symbol, interval, monthly_months)
            all_dfs.extend(monthly_dfs)

    # --- FIX: Only try daily data if the backtesting_end overlaps with current month ---
    current_month_start = datetime(today.year, today.month, 1)
    if backtesting_end >= current_month_start:
        # Determine the range for daily data download
        # This covers the current month up to backtesting_end.
        daily_start_date = max(backtesting_start, datetime(today.year, today.month, 1))
        original_daily_end_date = backtesting_end
        
        # Attempt to download daily data, adjusting end date if recent data is missing
        retries_daily_data = 2 # Allow adjusting end date by 1 or 2 days
        current_daily_end_date = original_daily_end_date

        for attempt in range(retries_daily_data + 1):
            if daily_start_date <= current_daily_end_date:
                logger.info(f"Attempt {attempt + 1}: Downloading daily data from {daily_start_date.date()} to {current_daily_end_date.date()}")
                try:
                    daily_dfs = _download_data_for_days(symbol, interval, daily_start_date, current_daily_end_date, cache_dir=cache_dir)
                    all_dfs.extend(daily_dfs)
                    # If successful, break the loop
                    break 
                except FileNotFoundError as e:
                    logger.warning(f"Failed to download recent daily data up to {current_daily_end_date.date()}: {e}")
                    if attempt < retries_daily_data:
                        current_daily_end_date -= timedelta(days=1)
                        logger.info(f"Adjusting daily end date to {current_daily_end_date.date()} and retrying.")
                    else:
                        logger.error(f"Failed to download recent daily data after {retries_daily_data} adjustments.")
                        
            else:
                logger.info(f"Daily data range ({daily_start_date.date()} to {current_daily_end_date.date()}) is invalid for this attempt. Skipping daily download.")
                break # No valid range left for daily data

    return pd.concat(all_dfs)

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

def _process_ohlcv_data(df: pd.DataFrame) -> pd.DataFrame:
    """Process OHLCV data: convert timestamps, set timezone, and ensure correct data types."""
    df["open_time"] = pd.to_numeric(df["open_time"], errors="coerce")
    df = df[(df["open_time"] >= TIMESTAMP_MIN) & (df["open_time"] <= TIMESTAMP_MAX)].copy()

    df["date"] = pd.to_datetime(df["open_time"], unit="us")
    df.set_index("date", inplace=True)
    df.index = df.index.tz_localize(TIMEZONE) if df.index.tz is None else df.index.tz_convert(TIMEZONE)

    return df[["open", "high", "low", "close", "volume"]].astype(float)

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
        return read_ohlcv_df_from_file(csv_path)

    url = BINANCE_URL_TEMPLATE.format(period=period, symbol=symbol, interval=interval, file_stem=file_stem)
    logger.info(f"Attempting to download {file_stem} from Binance ({period} data)...")
    
    zip_content = _download_file(url)

    df = _extract_csv_from_zip(zip_content)
    df = _process_ohlcv_data(df)
    
    _ensure_cache_directory_exists()
    
    df.reset_index().to_csv(csv_path, index=False)
    
    logger.info(f"Saved to cache: {csv_path}")

    return df

def _read_df_from_file(file_path: str) -> pd.DataFrame:
    """Read DataFrame from file with timezone handling."""
    df = pd.read_csv(file_path, parse_dates=["date"])
    df.set_index("date", inplace=True)
    df.index = df.index.tz_localize(TIMEZONE) if df.index.tz is None else df.index.tz_convert(TIMEZONE)
    return df

def read_ohlcv_df_from_file(file_path: str) -> pd.DataFrame:
    """Read OHLCV DataFrame from file."""
    df = _read_df_from_file(file_path)
    return df[["open", "high", "low", "close", "volume"]].astype(float)

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