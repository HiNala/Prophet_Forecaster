"""
Data ingestion module for the Prophet Forecaster application.
Handles data loading from various sources and initial preprocessing.
"""

import pandas as pd
import yfinance as yf
from typing import Dict, Any, Optional
import os
from datetime import datetime, timedelta
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests.exceptions

from ..utils import logger, config_loader

class DataIngestion:
    def __init__(self, config: Dict[str, Any]):
        """Initialize the data ingestion module with configuration."""
        self.config = config
        self.data_config = config.get('data', {})
        
        # Data parameters
        self.min_data_points = self.data_config.get('min_data_points', 100)
        self.allowed_intervals = self.data_config.get('allowed_intervals', ['1d', '1wk', '1mo'])
        self.max_retries = self.data_config.get('max_retries', 3)
        self.retry_delay = self.data_config.get('retry_delay', 5)  # seconds
        
        # Rate limiting
        self.requests_per_minute = self.data_config.get('requests_per_minute', 30)
        self.last_request_time = 0
        
        # Paths
        self.data_dir = "data"
        os.makedirs(self.data_dir, exist_ok=True)

    def validate_dates(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> tuple[Optional[datetime], Optional[datetime]]:
        """Validate and parse date strings."""
        try:
            # Parse dates if provided
            start = pd.to_datetime(start_date) if start_date else None
            end = pd.to_datetime(end_date) if end_date else datetime.now()
            
            # Validate date range
            if start and end and start >= end:
                raise ValueError("Start date must be before end date")
            
            # If no start date, use default lookback period
            if not start:
                lookback_days = self.data_config.get('default_lookback_days', 365)
                start = end - timedelta(days=lookback_days)
            
            return start, end
            
        except Exception as e:
            logger.error(f"Error validating dates: {str(e)}")
            raise

    def validate_interval(self, interval: str) -> None:
        """Validate the data interval."""
        if interval not in self.allowed_intervals:
            raise ValueError(f"Invalid interval: {interval}. Allowed values: {self.allowed_intervals}")

    def rate_limit(self) -> None:
        """Implement rate limiting for API requests."""
        if self.requests_per_minute > 0:
            current_time = time.time()
            elapsed = current_time - self.last_request_time
            min_interval = 60.0 / self.requests_per_minute
            
            if elapsed < min_interval:
                time.sleep(min_interval - elapsed)
            
            self.last_request_time = time.time()

    def fetch_data_with_retry(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        interval: str
    ) -> pd.DataFrame:
        """Fetch data with retry logic."""
        for attempt in range(self.max_retries):
            try:
                self.rate_limit()  # Apply rate limiting
                
                # Fetch data
                ticker = yf.Ticker(symbol)
                df = ticker.history(
                    start=start,
                    end=end,
                    interval=interval
                )
                
                if df.empty:
                    raise ValueError(f"No data returned for symbol {symbol}")
                
                if len(df) < self.min_data_points:
                    raise ValueError(f"Insufficient data points ({len(df)}) for symbol {symbol}")
                
                return df
                
            except requests.exceptions.RequestException as e:
                logger.warning(f"Network error on attempt {attempt + 1}: {str(e)}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (attempt + 1))  # Exponential backoff
                else:
                    raise
            except Exception as e:
                logger.error(f"Error fetching data: {str(e)}")
                raise

    def preprocess_data(
        self,
        df: pd.DataFrame,
        target_col: str = 'Adj Close'
    ) -> pd.DataFrame:
        """Perform initial data preprocessing."""
        try:
            # Reset index to make date a column
            df = df.reset_index()
            
            # Ensure datetime type for Date column
            df['Date'] = pd.to_datetime(df['Date'])
            
            # Sort by date
            df = df.sort_values('Date').reset_index(drop=True)
            
            # Handle different column name variations
            if target_col not in df.columns:
                # Try alternative column names
                alt_names = ['Adj_Close', 'Adjusted_Close', 'Close']
                for alt_name in alt_names:
                    if alt_name in df.columns:
                        target_col = alt_name
                        break
                else:
                    raise KeyError(f"Could not find target column. Tried: {[target_col] + alt_names}")
            
            # Basic validation
            if df[target_col].isnull().any():
                logger.warning(f"Found {df[target_col].isnull().sum()} missing values in target column")
            
            # Rename columns to Prophet requirements
            df = df.rename(columns={
                'Date': 'ds',
                target_col: 'y'
            })
            
            return df
            
        except Exception as e:
            logger.error(f"Error preprocessing data: {str(e)}")
            raise

    def process(
        self,
        symbol: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        interval: str = '1d',
        target_col: str = 'Adj Close'
    ) -> pd.DataFrame:
        """
        Main processing function that orchestrates data ingestion.
        
        Args:
            symbol (str): Stock symbol
            start_date (str, optional): Start date (YYYY-MM-DD)
            end_date (str, optional): End date (YYYY-MM-DD)
            interval (str): Data interval (1d, 1wk, 1mo)
            target_col (str): Target column for forecasting
            
        Returns:
            pd.DataFrame: Loaded and preprocessed data
        """
        try:
            # Validate inputs
            self.validate_interval(interval)
            start, end = self.validate_dates(start_date, end_date)
            
            # Fetch data
            logger.info(f"Fetching data for {symbol} from {start} to {end}")
            df = self.fetch_data_with_retry(symbol, start, end, interval)
            
            # Preprocess data
            df = self.preprocess_data(df, target_col)
            
            # Save raw data
            output_path = os.path.join(self.data_dir, f"raw_{symbol}_{interval}.csv")
            df.to_csv(output_path, index=False)
            logger.info(f"Raw data saved to {output_path}")
            
            return df
            
        except Exception as e:
            logger.error(f"Error in data ingestion process: {str(e)}")
            raise

def initialize() -> DataIngestion:
    """Initialize the data ingestion module with configuration."""
    config = config_loader.load_config()
    return DataIngestion(config)

# Convenience function for direct usage
def process(**kwargs) -> pd.DataFrame:
    """Convenience function to process data ingestion."""
    ingestion = initialize()
    return ingestion.process(**kwargs) 