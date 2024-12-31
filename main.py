#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Main entry point for the Prophet Forecaster application.
This module provides both CLI and programmatic interfaces for the forecasting process.
"""

import argparse
import sys
import os
from typing import Dict, Any, Optional
from datetime import datetime
import logging
from tqdm import tqdm

from src.modules.data_ingestion import DataIngestion
from src.modules.data_cleaning import DataCleaning
from src.modules.technical_indicators import TechnicalIndicators
from src.modules.model_preparation import ModelPreparation
from src.modules.model_training import ModelTraining
from src.modules.forecasting import Forecasting
from src.utils import config_loader, logger
from src.utils.logger import setup_logger

# Initialize logger
logger = setup_logger()

def setup_directories() -> None:
    """Create necessary directories if they don't exist."""
    paths = [
        "data/raw",
        "data/processed",
        "data/interim",
        "data/features",
        "data/visualizations",
        "models",
        "logs",
        "config"
    ]
    for path in paths:
        os.makedirs(path, exist_ok=True)

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Prophet Forecaster - Advanced Time Series Forecasting Tool"
    )
    
    parser.add_argument(
        "--source",
        default="yahoo",
        choices=["yahoo", "csv", "api"],
        help="Data source type"
    )
    parser.add_argument(
        "--symbol",
        help="Stock symbol (e.g., AAPL) or file path for CSV/API"
    )
    parser.add_argument(
        "--interval",
        default="1d",
        choices=["1d", "1h", "15m"],
        help="Data interval"
    )
    parser.add_argument(
        "--start_date",
        help="Start date (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--end_date",
        help="End date (YYYY-MM-DD)"
    )
    
    # Data cleaning options
    parser.add_argument(
        "--missing_method",
        default="impute",
        choices=["impute", "drop", "interpolate", "forward_fill"],
        help="Missing value handling method"
    )
    parser.add_argument(
        "--outlier_method",
        default="zscore",
        choices=["zscore", "iqr", "isolation_forest", "none"],
        help="Outlier detection method"
    )
    
    # Technical indicators options
    parser.add_argument(
        "--indicators",
        nargs="+",
        choices=["trend", "momentum", "volatility", "volume"],
        default=["trend", "momentum"],
        help="Technical indicator categories to calculate"
    )
    
    # Model options
    parser.add_argument(
        "--seasonality_mode",
        default="multiplicative",
        choices=["additive", "multiplicative"],
        help="Prophet seasonality mode"
    )
    parser.add_argument(
        "--tune",
        action="store_true",
        help="Use hyperparameter tuning"
    )
    parser.add_argument(
        "--cv",
        action="store_true",
        help="Use cross-validation"
    )
    parser.add_argument(
        "--ensemble_method",
        default="weighted_average",
        choices=["none", "stacking", "weighted_average", "voting"],
        help="Ensemble method for combining models"
    )
    parser.add_argument(
        "--forecast_periods",
        type=int,
        default=30,
        help="Number of periods to forecast"
    )
    
    return parser.parse_args()

def run_pipeline(args: argparse.Namespace) -> None:
    """Run the complete forecasting pipeline."""
    try:
        # Load configuration
        config = config_loader.load_config()
        
        # Setup directories
        setup_directories()
        
        logger.info("Starting Prophet Forecaster pipeline")
        
        # Step 1: Data Ingestion
        logger.info("\n=== Step 1/6: Data Ingestion ===")
        data_ingestion = DataIngestion(config)
        
        if args.source == "yahoo":
            df = data_ingestion.process(
                symbol=args.symbol,
                interval=args.interval,
                start_date=args.start_date,
                end_date=args.end_date
            )
        elif args.source == "csv":
            df = data_ingestion.load_csv_data(args.symbol)
        else:
            df = data_ingestion.fetch_api_data(args.symbol)
        
        # Step 2: Data Cleaning
        logger.info("\n=== Step 2/6: Data Cleaning ===")
        data_cleaning = DataCleaning(config)
        df_cleaned = data_cleaning.process(
            df,
            missing_method=args.missing_method,
            outlier_method=args.outlier_method
        )
        
        # Step 3: Technical Indicators
        logger.info("\n=== Step 3/6: Technical Indicators ===")
        tech_indicators = TechnicalIndicators(config)
        df_with_indicators = tech_indicators.calculate_indicators(
            df_cleaned,
            categories=args.indicators
        )
        tech_indicators.plot_indicators(df_with_indicators, categories=args.indicators)
        
        # Step 4: Model Preparation
        logger.info("\n=== Step 4/6: Model Preparation ===")
        model_prep = ModelPreparation(config)
        train_data, test_data = model_prep.prepare_data(df_with_indicators)
        
        # Step 5: Model Training
        logger.info("\n=== Step 5/6: Model Training ===")
        model_training = ModelTraining(config)
        
        if args.tune:
            logger.info("Starting hyperparameter tuning...")
            model = model_training.train_with_tuning(
                train_data,
                seasonality_mode=args.seasonality_mode
            )
        else:
            logger.info("Training model with default parameters...")
            model = model_training.train(
                train_data,
                seasonality_mode=args.seasonality_mode
            )
        
        if args.cv:
            logger.info("Performing cross-validation...")
            cv_metrics = model_training.cross_validate(model, train_data)
            print("\nCross-validation metrics:")
            print(cv_metrics)
        
        # Step 6: Forecasting
        logger.info("\n=== Step 6/6: Forecasting ===")
        forecasting = Forecasting(config)
        
        # Generate forecast with historical data
        forecast_results = forecasting.generate_forecast(
            model,
            periods=args.forecast_periods,
            include_history=True  # Include historical data
        )
        
        # Create visualization with all components
        forecasting.plot_forecast(
            forecast_results,
            output_dir="forecasts"
        )
        
        # Evaluate forecast if we have test data
        if test_data is not None:
            metrics = forecasting.evaluate_forecast(
                forecast_results['forecast'],
                test_data
            )
            print("\nForecast Evaluation Metrics:")
            for metric, value in metrics.items():
                print(f"{metric.upper()}: {value:.2f}")
        
        logger.info("\n=== Forecasting process completed successfully! ===")
        print("\nForecasting process completed successfully!")
        print("Check the forecasts directory for visualizations and predictions.")
        
    except Exception as e:
        logger.error(f"Error in pipeline: {str(e)}")
        raise

def main():
    """Main entry point for the application."""
    try:
        args = parse_arguments()
        run_pipeline(args)
    except KeyboardInterrupt:
        logger.info("Operation cancelled by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
        sys.exit(1)
    finally:
        logger.info("=== Prophet Forecaster Session Ended ===")

if __name__ == "__main__":
    main() 