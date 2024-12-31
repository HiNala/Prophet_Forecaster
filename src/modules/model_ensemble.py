"""
Ensemble module for combining multiple forecasting models.
Implements stacking, weighted averaging, and dynamic model selection.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple, Optional, Union
from datetime import datetime
from prophet import Prophet
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import xgboost as xgb
from tqdm import tqdm

from ..utils import logger, config_loader, data_manager

class ModelEnsemble:
    def __init__(self, config: Dict[str, Any]):
        """Initialize model ensemble with configuration."""
        self.config = config
        self.model_config = config.get('model', {})
        self.ensemble_config = self.model_config.get('ensemble', {})
        
        # Initialize data manager
        self.data_manager = data_manager.initialize()
        
        # Define base models
        self.base_models = {
            'prophet': None,  # Will be loaded from saved model
            'xgboost': xgb.XGBRegressor(
                objective='reg:squarederror',
                n_estimators=100,
                learning_rate=0.1
            ),
            'random_forest': RandomForestRegressor(
                n_estimators=100,
                random_state=42
            ),
            'exp_smoothing': None  # Will be initialized per series
        }
        
        # Meta-model for stacking
        self.meta_model = Ridge(alpha=1.0)
        
        # Dynamic weights
        self.model_weights = None
        
        # Performance tracking
        self.model_performances = {}

    def prepare_features_for_ml(
        self,
        df: pd.DataFrame,
        target_col: str = 'y',
        datetime_col: str = 'ds'
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare features for ML models from Prophet-formatted data."""
        try:
            # Extract datetime features
            X = pd.DataFrame()
            X['year'] = df[datetime_col].dt.year
            X['month'] = df[datetime_col].dt.month
            X['day'] = df[datetime_col].dt.day
            X['dayofweek'] = df[datetime_col].dt.dayofweek
            X['quarter'] = df[datetime_col].dt.quarter
            
            # Add any additional features present in the data
            for col in df.columns:
                if col not in [datetime_col, target_col]:
                    X[col] = df[col]
            
            y = df[target_col]
            return X, y
            
        except Exception as e:
            logger.error(f"Error preparing features: {str(e)}")
            raise

    def train_base_models(
        self,
        train_df: pd.DataFrame,
        target_col: str = 'y',
        datetime_col: str = 'ds'
    ):
        """Train all base models."""
        try:
            # Load saved Prophet model
            self.base_models['prophet'] = self.data_manager.load_data(
                category='models',
                subcategory='trained',
                prefix='prophet_model',
                extension='pkl'
            )
            
            # Prepare features for ML models
            X_train, y_train = self.prepare_features_for_ml(train_df, target_col, datetime_col)
            
            # Train ML models
            for name, model in self.base_models.items():
                if name == 'prophet':
                    continue  # Already loaded
                elif name == 'exp_smoothing':
                    # Initialize and fit exponential smoothing
                    self.base_models[name] = ExponentialSmoothing(
                        y_train,
                        seasonal_periods=30,
                        trend='add',
                        seasonal='add'
                    ).fit()
                else:
                    model.fit(X_train, y_train)
                logger.info(f"Trained {name} model")
                
        except Exception as e:
            logger.error(f"Error training base models: {str(e)}")
            raise

    def get_model_predictions(
        self,
        df: pd.DataFrame,
        target_col: str = 'y',
        datetime_col: str = 'ds'
    ) -> pd.DataFrame:
        """Get predictions from all base models."""
        try:
            predictions = pd.DataFrame()
            predictions[datetime_col] = df[datetime_col]
            
            # Get Prophet predictions
            prophet_forecast = self.base_models['prophet'].predict(df)
            predictions['prophet'] = prophet_forecast['yhat']
            
            # Prepare features for ML models
            X, _ = self.prepare_features_for_ml(df, target_col, datetime_col)
            
            # Get ML model predictions
            for name, model in self.base_models.items():
                if name == 'prophet':
                    continue  # Already done
                elif name == 'exp_smoothing':
                    predictions[name] = model.forecast(len(df))
                else:
                    predictions[name] = model.predict(X)
            
            return predictions
            
        except Exception as e:
            logger.error(f"Error getting model predictions: {str(e)}")
            raise

    def calculate_dynamic_weights(
        self,
        predictions: pd.DataFrame,
        actual: pd.Series,
        window_size: int = 30
    ) -> pd.DataFrame:
        """Calculate dynamic weights based on recent performance."""
        try:
            weights = pd.DataFrame(index=predictions.index)
            model_cols = [col for col in predictions.columns if col != 'ds']
            
            for i in range(len(predictions)):
                if i < window_size:
                    # Use equal weights for initial period
                    weights.loc[i, model_cols] = 1.0 / len(model_cols)
                else:
                    # Calculate weights based on recent performance
                    start_idx = max(0, i - window_size)
                    recent_errors = pd.DataFrame()
                    
                    for model in model_cols:
                        error = np.mean(np.abs(
                            predictions[model].iloc[start_idx:i] - 
                            actual.iloc[start_idx:i]
                        ))
                        recent_errors[model] = [error]
                    
                    # Convert errors to weights (inverse of error)
                    model_weights = (1.0 / recent_errors).values[0]
                    model_weights = model_weights / np.sum(model_weights)
                    weights.loc[i, model_cols] = model_weights
            
            return weights
            
        except Exception as e:
            logger.error(f"Error calculating dynamic weights: {str(e)}")
            raise

    def train_stacking_model(
        self,
        predictions: pd.DataFrame,
        actual: pd.Series
    ):
        """Train meta-model for stacking ensemble."""
        try:
            # Prepare features (base model predictions)
            X_meta = predictions.drop('ds', axis=1)
            
            # Train meta-model
            self.meta_model.fit(X_meta, actual)
            logger.info("Trained stacking meta-model")
            
            # Save meta-model
            self.data_manager.save_data(
                data=self.meta_model,
                category='models',
                subcategory='trained',
                prefix='stacking_metamodel',
                extension='pkl'
            )
            
        except Exception as e:
            logger.error(f"Error training stacking model: {str(e)}")
            raise

    def predict_ensemble(
        self,
        df: pd.DataFrame,
        method: str = 'stacking',
        window_size: int = 30
    ) -> pd.DataFrame:
        """
        Generate ensemble predictions using specified method.
        
        Args:
            df: Input data
            method: Ensemble method ('stacking', 'dynamic_weighted', 'simple_average')
            window_size: Window size for dynamic weighting
            
        Returns:
            DataFrame with predictions and confidence intervals
        """
        try:
            # Get base model predictions
            predictions = self.get_model_predictions(df)
            
            if method == 'stacking':
                # Use meta-model for final predictions
                X_meta = predictions.drop('ds', axis=1)
                ensemble_pred = self.meta_model.predict(X_meta)
            
            elif method == 'dynamic_weighted':
                # Use dynamic weights if available
                if self.model_weights is not None:
                    weights = self.model_weights
                else:
                    # Use equal weights
                    model_cols = [col for col in predictions.columns if col != 'ds']
                    weights = pd.DataFrame(
                        1.0 / len(model_cols),
                        index=predictions.index,
                        columns=model_cols
                    )
                
                # Calculate weighted predictions
                pred_cols = [col for col in predictions.columns if col != 'ds']
                ensemble_pred = np.sum(
                    predictions[pred_cols].values * weights[pred_cols].values,
                    axis=1
                )
            
            else:  # simple_average
                # Simple average of all models
                pred_cols = [col for col in predictions.columns if col != 'ds']
                ensemble_pred = predictions[pred_cols].mean(axis=1)
            
            # Prepare output DataFrame
            result = pd.DataFrame({
                'ds': df['ds'],
                'yhat': ensemble_pred
            })
            
            # Add uncertainty estimates
            pred_std = predictions.drop('ds', axis=1).std(axis=1)
            result['yhat_lower'] = ensemble_pred - 1.96 * pred_std
            result['yhat_upper'] = ensemble_pred + 1.96 * pred_std
            
            # Save predictions
            self.data_manager.save_data(
                data=result,
                category='forecasts',
                subcategory='predictions',
                prefix='ensemble_forecast',
                metadata={
                    'method': method,
                    'window_size': window_size,
                    'timestamp': datetime.now().isoformat()
                }
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Error generating ensemble predictions: {str(e)}")
            raise

    def evaluate_ensemble(
        self,
        predictions: pd.DataFrame,
        actual: pd.Series
    ) -> Dict[str, float]:
        """Calculate ensemble performance metrics."""
        try:
            metrics = {}
            
            # Calculate various error metrics
            errors = predictions['yhat'] - actual
            abs_errors = np.abs(errors)
            
            metrics['mae'] = np.mean(abs_errors)
            metrics['rmse'] = np.sqrt(np.mean(errors ** 2))
            metrics['mape'] = np.mean(np.abs(errors / actual)) * 100
            
            # Calculate prediction interval coverage
            in_interval = (
                (actual >= predictions['yhat_lower']) &
                (actual <= predictions['yhat_upper'])
            )
            metrics['coverage'] = np.mean(in_interval) * 100
            
            # Save evaluation results
            self.data_manager.save_data(
                data=metrics,
                category='forecasts',
                subcategory='evaluation',
                prefix='ensemble_evaluation',
                extension='json'
            )
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error evaluating ensemble: {str(e)}")
            raise

def initialize() -> ModelEnsemble:
    """Initialize the model ensemble."""
    config = config_loader.load_config()
    return ModelEnsemble(config) 