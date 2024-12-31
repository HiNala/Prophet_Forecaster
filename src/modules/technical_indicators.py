"""
Technical indicators module for the Prophet Forecaster application.
Handles calculation and visualization of technical indicators using the ta library.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
import os
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats

import ta
from ta.trend import SMAIndicator, EMAIndicator
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands
from ta.volume import VolumeWeightedAveragePrice

from ..utils import logger, config_loader

class TechnicalIndicators:
    def __init__(self, config: Dict[str, Any]):
        """Initialize the technical indicators module with configuration."""
        self.config = config
        self.indicators_config = config.get('indicators', {})
        
        # Default parameters
        self.default_params = {
            'sma': {'window': 20},
            'ema': {'window': 20},
            'rsi': {'window': 14},
            'bb': {'window': 20, 'window_dev': 2},
            'vwap': {'window': 14}
        }
        
        # Data quality thresholds
        self.min_data_points = self.indicators_config.get('min_data_points', 100)
        self.max_missing_pct = self.indicators_config.get('max_missing_pct', 0.1)
        
        # Paths
        self.data_dir = "data"
        os.makedirs(self.data_dir, exist_ok=True)

    def validate_data(self, df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """
        Validate input data for technical analysis.
        
        Returns:
            Tuple[bool, List[str]]: (is_valid, list of issues found)
        """
        issues = []
        
        required_columns = ['ds', 'Open', 'High', 'Low', 'Close', 'Volume']
        
        # Check minimum data points
        if len(df) < self.min_data_points:
            issues.append(f"Insufficient data points ({len(df)}) for reliable technical analysis")
        
        # Check for required columns
        for col in required_columns:
            if col not in df.columns:
                issues.append(f"Missing required column '{col}'")
        
        # Check data types
        if 'ds' in df.columns and not pd.api.types.is_datetime64_any_dtype(df['ds']):
            issues.append("Column 'ds' is not datetime type")
            
        numeric_columns = [col for col in required_columns if col != 'ds']
        for col in numeric_columns:
            if col in df.columns and not pd.api.types.is_numeric_dtype(df[col]):
                issues.append(f"Column '{col}' is not numeric type")
        
        # Check for missing values
        for col in numeric_columns:
            if col in df.columns:
                missing_pct = df[col].isnull().mean()
                if missing_pct > self.max_missing_pct:
                    issues.append(f"Column '{col}' has {missing_pct:.1%} missing values")
        
        # Check for monotonic timestamps
        if 'ds' in df.columns and len(df) > 1:
            if not df['ds'].is_monotonic_increasing:
                issues.append("Timestamps are not monotonically increasing")
        
        return len(issues) == 0, issues

    def calculate_indicators(self, df: pd.DataFrame, categories: List[str]) -> pd.DataFrame:
        """
        Calculate technical indicators based on selected categories.
        
        Args:
            df: DataFrame with OHLCV data
            categories: List of indicator categories to calculate
        
        Returns:
            DataFrame with added technical indicators
        """
        try:
            # Create a copy to avoid modifying the original
            result = df.copy()
            
            # Map Yahoo Finance column names to standard names
            column_mapping = {
                'Open': 'Open',
                'High': 'High',
                'Low': 'Low',
                'Close': 'Close',
                'Volume': 'Volume',
                'y': 'Close'  # Map prophet target column to Close
            }
            
            # Rename columns that exist in the DataFrame
            for old_name, new_name in column_mapping.items():
                if old_name in result.columns and old_name != new_name:
                    result[new_name] = result[old_name]
            
            # If no categories selected, use default categories
            if not categories:
                categories = ['trend', 'momentum']  # Default categories
                logger.info("No categories selected, using defaults: trend, momentum")
            
            # Validate input data
            is_valid, issues = self.validate_data(result)
            if not is_valid:
                for issue in issues:
                    logger.warning(issue)
                logger.warning("Proceeding with calculations despite validation issues")
            
            # Calculate indicators based on categories
            for category in categories:
                if category == 'trend':
                    # SMA
                    sma = SMAIndicator(
                        close=result['Close'],
                        window=self.default_params['sma']['window']
                    )
                    result['SMA'] = sma.sma_indicator()
                    
                    # EMA
                    ema = EMAIndicator(
                        close=result['Close'],
                        window=self.default_params['ema']['window']
                    )
                    result['EMA'] = ema.ema_indicator()
                    
                elif category == 'momentum':
                    # RSI
                    rsi = RSIIndicator(
                        close=result['Close'],
                        window=self.default_params['rsi']['window']
                    )
                    result['RSI'] = rsi.rsi()
                    
                elif category == 'volatility':
                    # Bollinger Bands
                    bb = BollingerBands(
                        close=result['Close'],
                        window=self.default_params['bb']['window'],
                        window_dev=self.default_params['bb']['window_dev']
                    )
                    result['BB_upper'] = bb.bollinger_hband()
                    result['BB_middle'] = bb.bollinger_mavg()
                    result['BB_lower'] = bb.bollinger_lband()
                    
                elif category == 'volume':
                    # VWAP
                    vwap = VolumeWeightedAveragePrice(
                        high=result['High'],
                        low=result['Low'],
                        close=result['Close'],
                        volume=result['Volume'],
                        window=self.default_params['vwap']['window']
                    )
                    result['VWAP'] = vwap.volume_weighted_average_price()
            
            # Handle NaN values
            result = result.fillna(method='bfill').fillna(method='ffill')
            
            return result
            
        except Exception as e:
            logger.error(f"Error calculating indicators: {str(e)}")
            raise

    def plot_indicators(
        self,
        df: pd.DataFrame,
        categories: List[str],
        output_dir: str = "data/visualizations"
    ) -> None:
        """Create interactive plots for technical indicators."""
        try:
            os.makedirs(output_dir, exist_ok=True)
            
            # If no categories selected, use default categories
            if not categories:
                categories = ['trend', 'momentum']  # Default categories
                logger.info("No categories selected, using defaults: trend, momentum")
            
            # Create subplots based on categories
            n_rows = len(categories)
            if n_rows == 0:
                logger.warning("No indicators to plot")
                return
            
            fig = make_subplots(rows=n_rows, cols=1, shared_xaxes=True)
            
            row = 1
            for category in categories:
                if category == 'trend':
                    # Price and moving averages
                    fig.add_trace(
                        go.Scatter(x=df['ds'], y=df['Close'], name='Close'),
                        row=row, col=1
                    )
                    if 'SMA' in df.columns:
                        fig.add_trace(
                            go.Scatter(x=df['ds'], y=df['SMA'], name='SMA'),
                            row=row, col=1
                        )
                    if 'EMA' in df.columns:
                        fig.add_trace(
                            go.Scatter(x=df['ds'], y=df['EMA'], name='EMA'),
                            row=row, col=1
                        )
                        
                elif category == 'momentum':
                    # RSI
                    if 'RSI' in df.columns:
                        fig.add_trace(
                            go.Scatter(x=df['ds'], y=df['RSI'], name='RSI'),
                            row=row, col=1
                        )
                        # Add RSI levels
                        fig.add_hline(y=70, line_dash="dash", line_color="red", row=row, col=1)
                        fig.add_hline(y=30, line_dash="dash", line_color="green", row=row, col=1)
                        
                elif category == 'volatility':
                    # Bollinger Bands
                    if 'BB_upper' in df.columns:
                        fig.add_trace(
                            go.Scatter(x=df['ds'], y=df['Close'], name='Close'),
                            row=row, col=1
                        )
                        fig.add_trace(
                            go.Scatter(x=df['ds'], y=df['BB_upper'], name='BB Upper'),
                            row=row, col=1
                        )
                        fig.add_trace(
                            go.Scatter(x=df['ds'], y=df['BB_middle'], name='BB Middle'),
                            row=row, col=1
                        )
                        fig.add_trace(
                            go.Scatter(x=df['ds'], y=df['BB_lower'], name='BB Lower'),
                            row=row, col=1
                        )
                        
                elif category == 'volume':
                    # Volume and VWAP
                    fig.add_trace(
                        go.Bar(x=df['ds'], y=df['Volume'], name='Volume'),
                        row=row, col=1
                    )
                    if 'VWAP' in df.columns:
                        fig.add_trace(
                            go.Scatter(x=df['ds'], y=df['VWAP'], name='VWAP'),
                            row=row, col=1
                        )
                
                row += 1
            
            # Update layout
            fig.update_layout(
                title='Technical Indicators',
                height=300 * n_rows,
                showlegend=True,
                template='plotly_dark'
            )
            
            # Save plot
            fig.write_html(os.path.join(output_dir, 'technical_indicators.html'))
            
        except Exception as e:
            logger.error(f"Error plotting indicators: {str(e)}")
            raise

def initialize() -> TechnicalIndicators:
    """Initialize the technical indicators module with configuration."""
    config = config_loader.load_config()
    return TechnicalIndicators(config)

# Convenience function for direct usage
def process(**kwargs) -> pd.DataFrame:
    """Convenience function to process technical indicators."""
    indicators = initialize()
    return indicators.process(**kwargs) 