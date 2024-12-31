"""
Model training module for the Prophet Forecaster application.
Handles Prophet model configuration, training, and cross-validation.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
import os
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
import json
import pickle
import itertools

from ..utils import logger, config_loader

class ModelTraining:
    def __init__(self, config: Dict[str, Any]):
        """Initialize the model training module with configuration."""
        self.config = config
        self.model_config = config.get('model', {})
        
        # Prophet configuration
        self.prophet_config = self.model_config.get('prophet', {})
        self.seasonality_mode = self.prophet_config.get('seasonality_mode', 'additive')
        self.yearly_seasonality = self.prophet_config.get('yearly_seasonality', 'auto')
        self.weekly_seasonality = self.prophet_config.get('weekly_seasonality', 'auto')
        self.daily_seasonality = self.prophet_config.get('daily_seasonality', 'auto')
        
        # Cross-validation configuration
        self.cv_config = self.model_config.get('cross_validation', {})
        self.initial = self.cv_config.get('initial', '60 days')
        self.period = self.cv_config.get('period', '7 days')
        self.horizon = self.cv_config.get('horizon', '7 days')
        
        # Model save path
        self.model_dir = "models"
        os.makedirs(self.model_dir, exist_ok=True)

    def load_training_data(self) -> pd.DataFrame:
        """Load the prepared training data."""
        try:
            train_path = "data/prophet_train.csv"
            if not os.path.exists(train_path):
                raise FileNotFoundError("Training data not found. Run model preparation first.")
            
            return pd.read_csv(train_path)
            
        except Exception as e:
            logger.error(f"Error loading training data: {str(e)}")
            raise

    def configure_prophet(
        self,
        regressors: Optional[List[str]] = None
    ) -> Prophet:
        """Configure and return a Prophet model instance."""
        try:
            # Initialize Prophet with basic configuration
            model = Prophet(
                seasonality_mode=self.seasonality_mode,
                yearly_seasonality=self.yearly_seasonality,
                weekly_seasonality=self.weekly_seasonality,
                daily_seasonality=self.daily_seasonality
            )
            
            # Add regressors if specified
            if regressors:
                for regressor in regressors:
                    model.add_regressor(regressor)
                    logger.info(f"Added regressor: {regressor}")
            
            return model
            
        except Exception as e:
            logger.error(f"Error configuring Prophet model: {str(e)}")
            raise

    def identify_regressors(self, df: pd.DataFrame) -> List[str]:
        """Identify potential regressors from the dataframe."""
        # Exclude Prophet's required columns and common datetime features
        exclude_cols = {'ds', 'y', 'day_of_week', 'day_of_month', 'month', 'quarter'}
        
        # Get all numeric columns that aren't in exclude_cols
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        regressors = [col for col in numeric_cols if col not in exclude_cols]
        
        logger.info(f"Identified {len(regressors)} potential regressors")
        return regressors

    def train_model(
        self,
        df: pd.DataFrame,
        regressors: Optional[List[str]] = None
    ) -> Prophet:
        """Train the Prophet model."""
        try:
            # Configure model
            model = self.configure_prophet(regressors)
            
            # Fit model
            logger.info("Starting model training...")
            model.fit(df)
            logger.info("Model training completed")
            
            return model
            
        except Exception as e:
            logger.error(f"Error training model: {str(e)}")
            raise

    def perform_cross_validation(
        self,
        model: Prophet,
        df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Perform cross-validation and calculate performance metrics."""
        try:
            logger.info("Starting cross-validation...")
            
            # Perform cross-validation
            cv_results = cross_validation(
                model,
                initial=self.initial,
                period=self.period,
                horizon=self.horizon,
                parallel="processes"
            )
            
            # Calculate performance metrics
            metrics = performance_metrics(cv_results)
            
            logger.info("Cross-validation completed")
            return cv_results, metrics
            
        except Exception as e:
            logger.error(f"Error in cross-validation: {str(e)}")
            raise

    def save_model(
        self,
        model: Prophet,
        metrics: Optional[pd.DataFrame] = None,
        regressors: Optional[List[str]] = None
    ) -> None:
        """Save the trained model and its metadata."""
        try:
            timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
            
            # Save model
            model_path = os.path.join(self.model_dir, f"prophet_model_{timestamp}.pkl")
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            
            # Save metadata
            metadata = {
                'timestamp': timestamp,
                'seasonality_mode': self.seasonality_mode,
                'yearly_seasonality': self.yearly_seasonality,
                'weekly_seasonality': self.weekly_seasonality,
                'daily_seasonality': self.daily_seasonality,
                'regressors': regressors if regressors else [],
                'cv_config': {
                    'initial': self.initial,
                    'period': self.period,
                    'horizon': self.horizon
                }
            }
            
            # Add metrics if available
            if metrics is not None:
                metadata['metrics'] = metrics.to_dict()
            
            metadata_path = os.path.join(self.model_dir, f"model_metadata_{timestamp}.json")
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=4)
            
            logger.info(f"Model and metadata saved with timestamp {timestamp}")
            
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            raise

    def process(
        self,
        skip_cv: bool = False
    ) -> Tuple[Prophet, Optional[pd.DataFrame], Optional[pd.DataFrame]]:
        """
        Main processing function that orchestrates model training and evaluation.
        
        Args:
            skip_cv (bool): Whether to skip cross-validation
            
        Returns:
            Tuple[Prophet, Optional[pd.DataFrame], Optional[pd.DataFrame]]: (model, cv_results, metrics)
        """
        try:
            # Load training data
            df = self.load_training_data()
            
            # Check if we're working with future data or skipping CV
            max_date = pd.to_datetime(df['ds']).max()
            if skip_cv or max_date > pd.Timestamp.now():
                if skip_cv:
                    logger.info("Skipping cross-validation as requested")
                else:
                    logger.info("Working with future data, skipping cross-validation")
                
                # Identify regressors
                regressors = self.identify_regressors(df)
                
                # Train model
                model = self.train_model(df, regressors)
                
                # Save model and metadata without metrics
                self.save_model(model, regressors=regressors)
                
                return model, None, None
            else:
                # Regular training with cross-validation
                # Identify regressors
                regressors = self.identify_regressors(df)
                
                # Train model
                model = self.train_model(df, regressors)
                
                # Perform cross-validation
                cv_results, metrics = self.perform_cross_validation(model, df)
                
                # Save model and metadata
                self.save_model(model, metrics, regressors)
                
                return model, cv_results, metrics
            
        except Exception as e:
            logger.error(f"Error in model training process: {str(e)}")
            raise

    def train_with_tuning(
        self,
        df: pd.DataFrame,
        seasonality_mode: str = 'multiplicative'
    ) -> Prophet:
        """
        Train Prophet model with hyperparameter tuning.
        
        Args:
            df: Training data
            seasonality_mode: Seasonality mode ('additive' or 'multiplicative')
            
        Returns:
            Prophet: Best trained model
        """
        try:
            logger.info("Starting hyperparameter tuning")
            
            # Define hyperparameter grid
            param_grid = {
                'changepoint_prior_scale': [0.001, 0.01, 0.1, 0.5],
                'seasonality_prior_scale': [0.01, 0.1, 1.0, 10.0],
                'holidays_prior_scale': [0.01, 0.1, 1.0, 10.0],
                'seasonality_mode': [seasonality_mode]
            }
            
            # Generate all combinations of parameters
            all_params = [
                dict(zip(param_grid.keys(), v))
                for v in itertools.product(*param_grid.values())
            ]
            
            # Storage for results
            rmses = []  # Store the RMSEs for each params
            params = []  # Store the parameters for each run
            
            logger.info(f"Testing {len(all_params)} parameter combinations")
            
            # Calculate minimum periods for cross-validation based on data size
            n_points = len(df)
            horizon_days = '3 days'  # Very short horizon for CV
            initial_days = '7 days'  # Minimum initial training period
            period_days = '1 days'   # Evaluate at every day
            
            # Use cross validation to evaluate all parameters
            for params_dict in all_params:
                # Configure model with current parameters
                model = Prophet(**params_dict)
                
                # Add regressors if available
                regressors = self.identify_regressors(df)
                for regressor in regressors:
                    model.add_regressor(regressor)
                
                # Fit model
                model.fit(df)
                
                try:
                    # Cross validate with minimal periods
                    cv_results = cross_validation(
                        model,
                        initial=initial_days,
                        period=period_days,
                        horizon=horizon_days,
                        parallel="processes"
                    )
                    
                    # Calculate metrics
                    metrics = performance_metrics(cv_results)
                    rmse = metrics['rmse'].mean()
                    
                    rmses.append(rmse)
                    params.append(params_dict)
                    
                    logger.info(f"RMSE: {rmse:.3f} with parameters {params_dict}")
                except Exception as cv_error:
                    logger.warning(f"Cross-validation failed for parameters {params_dict}: {str(cv_error)}")
                    continue
            
            if not rmses:
                logger.warning("No successful cross-validation runs. Using default parameters.")
                final_model = Prophet(seasonality_mode=seasonality_mode)
            else:
                # Find the best parameters
                best_idx = np.argmin(rmses)
                best_params = params[best_idx]
                best_rmse = rmses[best_idx]
                
                logger.info(f"Best RMSE: {best_rmse:.3f}")
                logger.info(f"Best parameters: {best_params}")
                
                # Train final model with best parameters
                final_model = Prophet(**best_params)
            
            # Add regressors to final model
            for regressor in regressors:
                final_model.add_regressor(regressor)
            
            # Fit final model
            final_model.fit(df)
            
            # Save model metadata
            self.save_model(
                final_model,
                metrics=pd.DataFrame({
                    'rmse': rmses,
                    'params': [str(p) for p in params]
                }),
                regressors=regressors
            )
            
            return final_model
            
        except Exception as e:
            logger.error(f"Error in hyperparameter tuning: {str(e)}")
            raise

def initialize() -> ModelTraining:
    """Initialize the model training module with configuration."""
    config = config_loader.load_config()
    return ModelTraining(config)

# Convenience function for direct usage
def process(skip_cv: bool = False) -> Tuple[Prophet, Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """Convenience function to process model training."""
    training = initialize()
    return training.process(skip_cv=skip_cv) 