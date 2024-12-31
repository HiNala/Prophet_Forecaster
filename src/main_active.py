"""
Interactive CLI for the Prophet Forecaster application.
"""

import questionary
import pandas as pd
from typing import Dict, Any, List
import os
import logging
from tqdm import tqdm
import time

from .modules.data_ingestion import DataIngestion
from .modules.data_cleaning import DataCleaning
from .modules.technical_indicators import TechnicalIndicators
from .modules.model_preparation import ModelPreparation
from .modules.model_training import ModelTraining
from .modules.forecasting import Forecasting
from .utils.config_loader import load_config
from .utils.logger import setup_logger

# Initialize logger
logger = setup_logger()

TOTAL_STEPS = 6  # Total number of main processing steps

def log_module_transition(module_name: str, step: int) -> None:
    """Log transition to a new module with progress information."""
    logger.info(f"\n{'='*50}")
    logger.info(f"Step {step}/{TOTAL_STEPS}: Entering {module_name} Module")
    logger.info(f"{'='*50}\n")

def prompt_data_source() -> Dict[str, Any]:
    """Prompt user for data source configuration."""
    questions = [
        {
            'type': 'select',
            'name': 'source_type',
            'message': 'Select data source:',
            'choices': ['Yahoo Finance', 'Local CSV', 'Custom API'],
            'default': 'Yahoo Finance'
        }
    ]
    
    source_type = questionary.prompt(questions)['source_type']
    
    if source_type == 'Yahoo Finance':
        symbol = questionary.text(
            'Enter stock symbol (e.g., AAPL):',
            default='AAPL'
        ).ask()
        
        interval = questionary.select(
            'Select data interval:',
            choices=['1d', '1h', '15m'],
            default='1d'
        ).ask()
        
        use_date_range = questionary.confirm(
            'Do you want to specify a date range?',
            default=False
        ).ask()
        
        if use_date_range:
            from datetime import datetime, timedelta
            default_end = datetime.now().strftime('%Y-%m-%d')
            default_start = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
            
            start_date = questionary.text(
                'Enter start date (YYYY-MM-DD):',
                default=default_start
            ).ask()
            end_date = questionary.text(
                'Enter end date (YYYY-MM-DD):',
                default=default_end
            ).ask()
        else:
            start_date = end_date = None
            
        return {
            'source_type': source_type,
            'symbol': symbol,
            'interval': interval,
            'start_date': start_date,
            'end_date': end_date
        }
        
    elif source_type == 'Local CSV':
        file_path = questionary.text(
            'Enter path to CSV file:',
            default='data/input.csv'
        ).ask()
        return {
            'source_type': source_type,
            'file_path': file_path
        }
        
    else:  # Custom API
        api_url = questionary.text(
            'Enter API URL:',
            default='https://api.example.com/data'
        ).ask()
        api_key = questionary.password(
            'Enter API key:',
            default=''
        ).ask()
        return {
            'source_type': source_type,
            'api_url': api_url,
            'api_key': api_key
        }

def prompt_data_cleaning() -> Dict[str, Any]:
    """Prompt user for data cleaning configuration."""
    questions = [
        {
            'type': 'select',
            'name': 'missing_method',
            'message': 'Select method for handling missing values:',
            'choices': ['impute', 'drop', 'interpolate', 'forward_fill'],
            'default': 'impute'
        },
        {
            'type': 'select',
            'name': 'outlier_method',
            'message': 'Select method for outlier detection:',
            'choices': ['zscore', 'iqr', 'isolation_forest', 'none'],
            'default': 'zscore'
        }
    ]
    
    return questionary.prompt(questions)

def prompt_technical_indicators() -> Dict[str, Any]:
    """Prompt user for technical indicators configuration."""
    choices = [
        questionary.Choice('trend', checked=True),      # SMA, EMA
        questionary.Choice('momentum', checked=True),   # RSI
        questionary.Choice('volatility'),              # Bollinger Bands
        questionary.Choice('volume')                   # VWAP
    ]
    
    categories = questionary.checkbox(
        'Select indicator categories to calculate:',
        choices=choices
    ).ask()
    
    return {'categories': categories}

def prompt_model_configuration() -> Dict[str, Any]:
    """Prompt user for model configuration."""
    questions = [
        {
            'type': 'select',
            'name': 'seasonality_mode',
            'message': 'Select seasonality mode:',
            'choices': ['additive', 'multiplicative'],
            'default': 'multiplicative'
        },
        {
            'type': 'confirm',
            'name': 'use_hyperparameter_tuning',
            'message': 'Use hyperparameter tuning?',
            'default': True
        },
        {
            'type': 'confirm',
            'name': 'use_cross_validation',
            'message': 'Use cross-validation?',
            'default': True
        },
        {
            'type': 'select',
            'name': 'ensemble_method',
            'message': 'Select ensemble method:',
            'choices': ['none', 'stacking', 'weighted_average', 'voting'],
            'default': 'weighted_average'
        },
        {
            'type': 'text',
            'name': 'forecast_periods',
            'message': 'Enter number of periods to forecast:',
            'default': '30',
            'validate': lambda text: text.isdigit() and int(text) > 0,
            'filter': lambda text: int(text)
        }
    ]
    
    return questionary.prompt(questions)

def main():
    """Main function to run the interactive CLI."""
    try:
        # Load configuration
        config = load_config()
        
        # Welcome message
        logger.info("Starting Prophet Forecaster interactive CLI")
        print("\nWelcome to Prophet Forecaster!")
        print("This interactive CLI will guide you through the forecasting process.\n")
        
        # Step 1: Data Source Configuration
        log_module_transition("Data Source", 1)
        data_config = prompt_data_source()
        logger.info(f"Data source configuration: {data_config}")
        
        # Initialize data ingestion
        data_ingestion = DataIngestion(config)
        
        # Fetch data with progress
        logger.info("Fetching data...")
        if data_config['source_type'] == 'Yahoo Finance':
            df = data_ingestion.process(
                symbol=data_config['symbol'],
                interval=data_config['interval'],
                start_date=data_config.get('start_date'),
                end_date=data_config.get('end_date')
            )
        elif data_config['source_type'] == 'Local CSV':
            df = data_ingestion.load_csv_data(data_config['file_path'])
        else:
            df = data_ingestion.fetch_api_data(
                api_url=data_config['api_url'],
                api_key=data_config['api_key']
            )
        logger.info("Data fetching completed")
        
        # Step 2: Data Cleaning
        log_module_transition("Data Cleaning", 2)
        cleaning_config = prompt_data_cleaning()
        logger.info(f"Data cleaning configuration: {cleaning_config}")
        
        # Initialize data cleaning
        data_cleaning = DataCleaning(config)
        
        # Clean data with progress
        logger.info("Cleaning data...")
        df_cleaned = data_cleaning.process(
            df,
            missing_method=cleaning_config['missing_method'],
            outlier_method=cleaning_config['outlier_method']
        )
        logger.info("Data cleaning completed")
        
        # Step 3: Technical Indicators
        log_module_transition("Technical Indicators", 3)
        indicators_config = prompt_technical_indicators()
        logger.info(f"Technical indicators configuration: {indicators_config}")
        
        # Initialize technical indicators
        tech_indicators = TechnicalIndicators(config)
        
        # Calculate indicators with progress
        logger.info("Calculating technical indicators...")
        df_with_indicators = tech_indicators.calculate_indicators(
            df_cleaned,
            categories=indicators_config['categories']
        )
        logger.info("Technical indicators calculation completed")
        
        # Generate indicator plots
        logger.info("Generating technical indicator plots...")
        tech_indicators.plot_indicators(
            df_with_indicators,
            categories=indicators_config['categories']
        )
        logger.info("Technical indicator plots generated")
        
        # Step 4: Model Configuration
        log_module_transition("Model Configuration", 4)
        model_config = prompt_model_configuration()
        logger.info(f"Model configuration: {model_config}")
        
        # Initialize model preparation
        model_prep = ModelPreparation(config)
        
        # Prepare data for training
        logger.info("Preparing data for model training...")
        train_data, test_data = model_prep.prepare_data(df_with_indicators)
        logger.info("Data preparation completed")
        
        # Step 5: Model Training
        log_module_transition("Model Training", 5)
        model_training = ModelTraining(config)
        
        # Train model with progress tracking
        if model_config['use_hyperparameter_tuning']:
            logger.info("Starting hyperparameter tuning...")
            model = model_training.train_with_tuning(
                train_data,
                seasonality_mode=model_config['seasonality_mode']
            )
            logger.info("Hyperparameter tuning completed")
        else:
            logger.info("Training model with default parameters...")
            model = model_training.train(
                train_data,
                seasonality_mode=model_config['seasonality_mode']
            )
            logger.info("Model training completed")
        
        # Cross-validation if requested
        if model_config['use_cross_validation']:
            logger.info("Performing cross-validation...")
            cv_metrics = model_training.cross_validate(model, train_data)
            logger.info("Cross-validation completed")
            print("\nCross-validation metrics:")
            print(cv_metrics)
        
        # Step 6: Forecasting
        log_module_transition("Forecasting", 6)
        forecasting = Forecasting(config)
        
        # Generate forecasts
        logger.info(f"Generating forecasts for {model_config['forecast_periods']} periods...")
        forecasts = forecasting.generate_forecast(
            model,
            periods=model_config['forecast_periods']
        )
        logger.info("Forecast generation completed")
        
        # Plot forecasts
        logger.info("Generating forecast plots...")
        forecasting.plot_forecast(forecasts)
        logger.info("Forecast plots generated")
        
        logger.info("\n=== Forecasting process completed successfully! ===")
        print("\nForecasting process completed successfully!")
        print("Check the output directories for results and visualizations.")
        
    except Exception as e:
        logger.error(f"Error in main process: {str(e)}")
        raise

if __name__ == "__main__":
    main() 