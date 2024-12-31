#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Main entry point for the Prophet Forecaster application.
This module orchestrates the entire forecasting process by coordinating all other modules.
"""

import argparse
import sys
import os
from typing import Dict, Any, Optional
from datetime import datetime

from src.modules import (
    data_ingestion,
    data_cleaning,
    technical_indicators,
    model_preparation,
    model_training,
    model_tuning,
    model_ensemble,
    forecasting,
    visualization
)
from src.utils import config_loader, logger, data_manager

def setup_directories(config: Dict[str, Any]) -> None:
    """Create necessary directories if they don't exist."""
    paths = [
        "data/raw",
        "data/processed",
        "data/interim",
        "data/features",
        "data/visualizations",
        "models/trained",
        "models/metadata",
        "models/evaluation",
        "forecasts/predictions",
        "forecasts/visualizations",
        "forecasts/evaluation",
        "logs",
        "config",
        "docs"
    ]
    for path in paths:
        os.makedirs(path, exist_ok=True)
        
def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Prophet Forecaster - Advanced Time Series Forecasting Tool",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Main command subparser
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Data command
    data_parser = subparsers.add_parser("data", help="Load and visualize data")
    data_parser.add_argument("--source", default="yahoo", choices=["yahoo", "csv", "api"], help="Data source type")
    data_parser.add_argument("--symbol", help="Stock symbol (e.g., AAPL) or file path for CSV/API")
    data_parser.add_argument("--start_date", help="Start date (YYYY-MM-DD)")
    data_parser.add_argument("--end_date", help="End date (YYYY-MM-DD)")
    data_parser.add_argument("--interval", default="1d", choices=["1d", "1h", "1wk", "1mo"], help="Data interval")
    data_parser.add_argument("--target_col", default="Adj Close", help="Target column for forecasting")
    
    # Clean command
    clean_parser = subparsers.add_parser("clean", help="Clean and validate data")
    clean_parser.add_argument("--validation", nargs="+", default=["missing", "duplicates", "outliers"],
                            choices=["missing", "duplicates", "outliers", "gaps", "consistency"],
                            help="Data validation checks to perform")
    clean_parser.add_argument("--missing_method", default="impute",
                            choices=["impute", "interpolate", "forward_fill", "backward_fill", "drop"],
                            help="Missing value handling method")
    clean_parser.add_argument("--impute_method", default="mean",
                            choices=["mean", "median", "mode", "knn"],
                            help="Imputation method if using impute")
    clean_parser.add_argument("--outlier_method", default="zscore",
                            choices=["zscore", "iqr", "isolation_forest", "local_outlier_factor", "dbscan"],
                            help="Outlier detection method")
    clean_parser.add_argument("--remove_duplicates", action="store_true", help="Remove duplicate rows")
    clean_parser.add_argument("--fill_gaps", action="store_true", help="Fill time series gaps")
    
    # Indicators command
    indicators_parser = subparsers.add_parser("indicators", help="Calculate technical indicators")
    indicators_parser.add_argument("--categories", nargs="+", default=["trend"],
                                choices=["trend", "momentum", "volatility", "volume", "custom"],
                                help="Indicator categories to include")
    indicators_parser.add_argument("--indicators", nargs="+", help="Specific indicators to calculate")
    indicators_parser.add_argument("--window_size", type=int, default=20, help="Window size for applicable indicators")
    indicators_parser.add_argument("--custom_file", help="Path to custom indicators file")
    
    # Model command
    model_parser = subparsers.add_parser("model", help="Configure and train models")
    model_parser.add_argument("--models", nargs="+",
                            default=["prophet", "xgboost", "random_forest"],
                            choices=["prophet", "xgboost", "lightgbm", "catboost", "random_forest",
                                   "exp_smoothing", "arima", "lstm"],
                            help="Models to include in ensemble")
    model_parser.add_argument("--tune", action="store_true", help="Perform hyperparameter tuning")
    model_parser.add_argument("--tune_method", default="random",
                            choices=["grid", "random", "bayesian"],
                            help="Hyperparameter tuning method")
    model_parser.add_argument("--cv_folds", type=int, default=5, help="Number of cross-validation folds")
    model_parser.add_argument("--max_trials", type=int, default=50, help="Maximum tuning trials")
    
    # Train command
    train_parser = subparsers.add_parser("train", help="Train the model ensemble")
    train_parser.add_argument("--ensemble_method", default="stacking",
                            choices=["stacking", "dynamic_weighted", "simple_average", "bagging", "boosting"],
                            help="Ensemble method")
    train_parser.add_argument("--meta_learner", default="ridge",
                            choices=["ridge", "lasso", "elastic_net", "random_forest", "xgboost"],
                            help="Meta-learner for stacking ensemble")
    train_parser.add_argument("--cv_method", default="time",
                            choices=["time", "expanding", "sliding"],
                            help="Cross-validation method")
    train_parser.add_argument("--n_splits", type=int, default=5, help="Number of CV splits")
    
    # Forecast command
    forecast_parser = subparsers.add_parser("forecast", help="Generate forecasts")
    forecast_parser.add_argument("--periods", type=int, default=30, help="Number of periods to forecast")
    forecast_parser.add_argument("--confidence_level", type=int, default=95,
                               help="Confidence level for prediction intervals")
    forecast_parser.add_argument("--include_history", action="store_true", help="Include historical data")
    forecast_parser.add_argument("--output", help="Output file path")
    
    # Evaluate command
    evaluate_parser = subparsers.add_parser("evaluate", help="Evaluate model performance")
    evaluate_parser.add_argument("--metrics", nargs="+",
                               default=["rmse", "mae", "mape", "r2"],
                               help="Metrics to calculate")
    evaluate_parser.add_argument("--cross_validation", action="store_true", help="Perform cross-validation")
    evaluate_parser.add_argument("--output", help="Output file path")
    
    # Visualize command
    viz_parser = subparsers.add_parser("visualize", help="Generate visualizations")
    viz_parser.add_argument("--plots", nargs="+",
                          default=["forecast", "components", "metrics"],
                          choices=["forecast", "components", "residuals",
                                 "feature_importance", "cross_validation", "metrics"],
                          help="Plots to generate")
    viz_parser.add_argument("--format", default="interactive",
                          choices=["interactive", "static", "both"],
                          help="Output format")
    viz_parser.add_argument("--style", default="default",
                          choices=["default", "dark", "light", "minimal"],
                          help="Visualization style")
    
    return parser.parse_args()

def validate_inputs(args: argparse.Namespace, config: Dict[str, Any]) -> bool:
    """Validate command line arguments against configuration."""
    try:
        if args.command == "data":
            if args.source == "yahoo" and not args.symbol:
                logger.error("Stock symbol is required for Yahoo Finance data source")
                return False
            if args.source in ["csv", "api"] and not args.symbol:
                logger.error("File path or API endpoint is required")
                return False
                
        if args.command == "forecast":
            if args.periods <= 0:
                logger.error("Forecast periods must be positive")
                return False
            if not 0 <= args.confidence_level <= 100:
                logger.error("Confidence level must be between 0 and 100")
                return False
                
        return True
        
    except Exception as e:
        logger.error(f"Validation error: {str(e)}")
        return False

def run_data_pipeline(args: argparse.Namespace, config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Run the data ingestion and preprocessing pipeline."""
    try:
        # Initialize components
        dm = data_manager.DataManager(config)
        
        # Fetch data
        raw_data = data_ingestion.fetch_data(
            source=args.source,
            symbol=args.symbol,
            interval=args.interval,
            start_date=args.start_date,
            end_date=args.end_date
        )
        if raw_data is None:
            return None
            
        # Save raw data
        dm.save_data(raw_data, "raw", f"{args.symbol}_{args.interval}")
        
        # Clean data
        clean_data = data_cleaning.process(
            data=raw_data,
            validation=args.validation,
            missing_method=args.missing_method,
            impute_method=args.impute_method,
            outlier_method=args.outlier_method,
            remove_duplicates=args.remove_duplicates,
            fill_gaps=args.fill_gaps
        )
        
        # Calculate indicators
        data_with_indicators = technical_indicators.calculate_indicators(
            data=clean_data,
            categories=args.categories,
            indicators=args.indicators,
            window_size=args.window_size,
            custom_file=args.custom_file
        )
        
        # Prepare data for modeling
        prepared_data = model_preparation.prepare_data(
            data=data_with_indicators,
            config=config
        )
        
        return {
            "raw_data": raw_data,
            "clean_data": clean_data,
            "data_with_indicators": data_with_indicators,
            "prepared_data": prepared_data
        }
        
    except Exception as e:
        logger.error(f"Error in data pipeline: {str(e)}")
        return None

def run_model_pipeline(
    args: argparse.Namespace,
    config: Dict[str, Any],
    data: Dict[str, Any]
) -> Optional[Dict[str, Any]]:
    """Run the model training and forecasting pipeline."""
    try:
        # Initialize components
        dm = data_manager.DataManager(config)
        
        # Tune hyperparameters if requested
        if args.tune:
            best_params = model_tuning.tune_hyperparameters(
                data=data["prepared_data"],
                method=args.tune_method,
                cv_folds=args.cv_folds,
                max_trials=args.max_trials
            )
            config["model"]["params"].update(best_params)
        
        # Train ensemble
        ensemble = model_ensemble.train_ensemble(
            data=data["prepared_data"],
            models=args.models,
            method=args.ensemble_method,
            meta_learner=args.meta_learner,
            cv_method=args.cv_method,
            n_splits=args.n_splits
        )
        
        # Generate forecasts
        forecast_results = forecasting.generate_forecast(
            ensemble=ensemble,
            data=data["prepared_data"],
            periods=args.periods,
            confidence_level=args.confidence_level,
            include_history=args.include_history
        )
        
        # Evaluate performance
        evaluation = forecasting.evaluate_forecast(
            forecast=forecast_results["forecast"],
            actual=data["prepared_data"]["test"],
            metrics=args.metrics,
            cross_validation=args.cross_validation
        )
        
        # Save results
        if args.output:
            dm.save_forecast(forecast_results, args.output)
            
        return {
            "forecast": forecast_results,
            "evaluation": evaluation,
            "ensemble": ensemble
        }
        
    except Exception as e:
        logger.error(f"Error in model pipeline: {str(e)}")
        return None

def main():
    """Main function to orchestrate the forecasting process."""
    try:
        # Load configuration
        config = config_loader.load_config()
        
        # Setup directories
        setup_directories(config)
        
        # Parse and validate arguments
        args = parse_arguments()
        if not args.command:
            logger.error("No command specified. Use --help for usage information.")
            sys.exit(1)
            
        if not validate_inputs(args, config):
            sys.exit(1)
        
        # Process commands
        if args.command == "data":
            data = run_data_pipeline(args, config)
            if data is None:
                sys.exit(1)
            logger.info("Data pipeline completed successfully")
            
        elif args.command in ["model", "train"]:
            data = run_data_pipeline(args, config)
            if data is None:
                sys.exit(1)
                
            results = run_model_pipeline(args, config, data)
            if results is None:
                sys.exit(1)
            logger.info("Model pipeline completed successfully")
            
        elif args.command == "forecast":
            data = run_data_pipeline(args, config)
            if data is None:
                sys.exit(1)
                
            results = run_model_pipeline(args, config, data)
            if results is None:
                sys.exit(1)
            logger.info("Forecast generated successfully")
            
        elif args.command == "evaluate":
            data = run_data_pipeline(args, config)
            if data is None:
                sys.exit(1)
                
            results = run_model_pipeline(args, config, data)
            if results is None:
                sys.exit(1)
            logger.info("Evaluation completed successfully")
            
        elif args.command == "visualize":
            # Generate visualizations
            visualization.generate_plots(
                plots=args.plots,
                format=args.format,
                style=args.style,
                data=data,
                results=results,
                output_dir="visualizations"
            )
            logger.info("Visualizations generated successfully")
            
    except KeyboardInterrupt:
        logger.info("Operation cancelled by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        logger.exception(e)
        sys.exit(1)

if __name__ == "__main__":
    main() 