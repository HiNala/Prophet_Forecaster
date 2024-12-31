"""
Test script to verify the Prophet Forecaster workflow.
"""

import os
import sys
import shutil
import pytest
from datetime import datetime, timedelta
import pandas as pd

# Add parent directory to path to import main module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.modules import (
    data_ingestion,
    data_cleaning,
    technical_indicators,
    model_preparation,
    model_training,
    forecasting
)
from src.utils import config_loader, logger

@pytest.fixture(scope="session", autouse=True)
def setup_test_environment():
    """Set up test environment before all tests."""
    # Load config
    config = config_loader.load_config()
    paths = config.get('paths', {})
    
    # Create necessary directories
    for path in paths.values():
        os.makedirs(path, exist_ok=True)
    
    # Setup logging
    logger.setup_logger(config)
    
    yield  # This is where the testing happens
    
    # Cleanup after all tests (optional)
    # Uncomment if you want to clean up after tests
    # for path in paths.values():
    #     if os.path.exists(path):
    #         shutil.rmtree(path)

@pytest.fixture(scope="function")
def sample_data():
    """Provide sample data for tests."""
    return {
        'symbol': 'AAPL',
        'end_date': datetime.now(),
        'start_date': datetime.now() - timedelta(days=365),
        'interval': '1d'
    }

def test_config_loading():
    """Test configuration loading."""
    config = config_loader.load_config()
    assert config is not None
    assert 'data' in config
    assert 'model' in config
    assert 'paths' in config
    
    # Validate critical settings
    assert config['data'].get('min_data_points') > 0
    assert config['model']['prophet'].get('seasonality_mode') in ['additive', 'multiplicative']

def test_directory_structure():
    """Test that all required directories exist."""
    config = config_loader.load_config()
    paths = config.get('paths', {})
    for path in paths.values():
        assert os.path.exists(path), f"Directory {path} does not exist"
        assert os.access(path, os.W_OK), f"Directory {path} is not writable"

def test_data_ingestion(sample_data):
    """Test data ingestion from Yahoo Finance."""
    df = data_ingestion.process(
        symbol=sample_data['symbol'],
        start_date=sample_data['start_date'].strftime("%Y-%m-%d"),
        end_date=sample_data['end_date'].strftime("%Y-%m-%d"),
        interval=sample_data['interval']
    )
    
    assert df is not None
    assert not df.empty
    assert 'ds' in df.columns
    assert 'y' in df.columns
    assert len(df) > 0
    assert df['y'].dtype in ['float64', 'float32']
    assert pd.to_datetime(df['ds']).is_monotonic_increasing

def test_data_cleaning():
    """Test data cleaning process."""
    df = data_cleaning.process(
        missing_value_method="impute",
        outlier_method="zscore"
    )
    
    assert df is not None
    assert not df.empty
    assert df.isnull().sum().sum() == 0  # No missing values
    assert len(df) > 0
    assert all(df['y'].notna())  # No NaN values in target column

def test_technical_indicators():
    """Test technical indicators calculation."""
    df = technical_indicators.process(
        indicators=["SMA", "EMA", "RSI"]
    )
    
    assert df is not None
    assert not df.empty
    assert any(col.startswith('SMA_') for col in df.columns)
    assert any(col.startswith('EMA_') for col in df.columns)
    assert any(col.startswith('RSI_') for col in df.columns)
    
    # Validate indicator values
    sma_cols = [col for col in df.columns if col.startswith('SMA_')]
    assert all(df[sma_cols].min() >= df['y'].min())
    assert all(df[sma_cols].max() <= df['y'].max())
    
    rsi_cols = [col for col in df.columns if col.startswith('RSI_')]
    assert all(df[rsi_cols].min() >= 0)
    assert all(df[rsi_cols].max() <= 100)

def test_model_preparation():
    """Test model preparation process."""
    train_df, test_df = model_preparation.process()
    
    assert train_df is not None and test_df is not None
    assert not train_df.empty and not test_df.empty
    assert len(train_df) > len(test_df)  # Train set should be larger
    
    # Validate data splits
    assert train_df.index.intersection(test_df.index).empty  # No overlap
    assert pd.to_datetime(train_df['ds']).max() <= pd.to_datetime(test_df['ds']).min()  # Time-based split

def test_model_training():
    """Test model training process."""
    model, cv_results, metrics = model_training.process()
    
    assert model is not None
    assert cv_results is not None
    assert metrics is not None
    assert 'mape' in metrics.columns
    
    # Validate metrics
    assert all(metrics['mape'] >= 0)  # MAPE should be non-negative
    assert all(metrics['rmse'] >= 0)  # RMSE should be non-negative

def test_forecasting():
    """Test forecast generation."""
    forecast, metrics = forecasting.process(
        periods=30,
        include_history=True
    )
    
    assert forecast is not None
    assert not forecast.empty
    assert metrics is not None
    assert 'mape' in metrics
    assert 'rmse' in metrics
    
    # Validate forecast
    assert len(forecast) >= 30  # At least requested periods
    assert all(forecast['yhat'].notna())  # No NaN in predictions
    assert all(forecast['yhat_lower'] <= forecast['yhat'])  # Lower bound check
    assert all(forecast['yhat_upper'] >= forecast['yhat'])  # Upper bound check

@pytest.mark.slow
def test_full_workflow(sample_data):
    """Test the entire workflow end-to-end."""
    try:
        # 1. Data Ingestion
        df = data_ingestion.process(
            symbol=sample_data['symbol'],
            start_date=sample_data['start_date'].strftime("%Y-%m-%d"),
            end_date=sample_data['end_date'].strftime("%Y-%m-%d"),
            interval=sample_data['interval']
        )
        assert not df.empty
        
        # 2. Data Cleaning
        df = data_cleaning.process()
        assert not df.empty
        
        # 3. Technical Indicators
        df = technical_indicators.process(["SMA", "EMA", "RSI"])
        assert not df.empty
        
        # 4. Model Preparation
        train_df, test_df = model_preparation.process()
        assert not train_df.empty and not test_df.empty
        
        # 5. Model Training
        model, cv_results, metrics = model_training.process()
        assert model is not None
        
        # 6. Forecasting
        forecast, metrics = forecasting.process(periods=30)
        assert not forecast.empty
        
        # Check output files exist
        config = config_loader.load_config()
        paths = config.get('paths', {})
        
        expected_files = [
            (paths['data'], f"{sample_data['symbol']}_{sample_data['interval']}.csv"),
            (paths['data'], "data_with_indicators.csv"),
            (paths['data'], "prophet_train.csv"),
            (paths['data'], "prophet_test.csv"),
            (paths['models'], "prophet_model_latest.pkl"),
            (paths['output'], "forecast_latest.csv"),
            (paths['output'], "metrics_latest.json")
        ]
        
        for directory, filename in expected_files:
            filepath = os.path.join(directory, filename)
            assert os.path.exists(filepath), f"Expected file not found: {filepath}"
            assert os.path.getsize(filepath) > 0, f"File is empty: {filepath}"
            
    except Exception as e:
        pytest.fail(f"Full workflow test failed: {str(e)}")

if __name__ == "__main__":
    pytest.main([__file__, "-v"])