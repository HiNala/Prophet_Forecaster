"""
Model tuning module for hyperparameter optimization and model selection.
Implements grid search, cross-validation, and model evaluation for Prophet models.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from datetime import datetime
import itertools
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
import json
from tqdm import tqdm

from ..utils import logger, config_loader, data_manager

class ModelTuner:
    def __init__(self, config: Dict[str, Any]):
        """Initialize model tuner with configuration."""
        self.config = config
        self.model_config = config.get('model', {})
        
        # Hyperparameter search spaces
        self.param_grid = {
            'changepoint_prior_scale': [0.001, 0.01, 0.1, 0.5],
            'seasonality_prior_scale': [0.01, 0.1, 1.0, 10.0],
            'holidays_prior_scale': [0.01, 0.1, 1.0, 10.0],
            'seasonality_mode': ['additive', 'multiplicative'],
            'changepoint_range': [0.8, 0.85, 0.9, 0.95]
        }
        
        # Cross-validation parameters
        self.cv_params = {
            'initial': '365 days',
            'period': '30 days',
            'horizon': '60 days',
            'parallel': 'processes'
        }
        
        # Evaluation metrics
        self.metrics = ['mse', 'rmse', 'mae', 'mape', 'coverage']
        
        # Initialize data manager
        self.data_manager = data_manager.initialize()

    def generate_parameter_combinations(self) -> List[Dict[str, Any]]:
        """Generate all combinations of hyperparameters for grid search."""
        param_combinations = [
            dict(zip(self.param_grid.keys(), v))
            for v in itertools.product(*self.param_grid.values())
        ]
        logger.info(f"Generated {len(param_combinations)} parameter combinations")
        return param_combinations

    def train_and_evaluate_model(
        self,
        params: Dict[str, Any],
        train_df: pd.DataFrame,
        cv_params: Optional[Dict[str, Any]] = None
    ) -> Tuple[Prophet, Dict[str, Any]]:
        """
        Train a Prophet model with given parameters and evaluate using cross-validation.
        
        Args:
            params: Model hyperparameters
            train_df: Training data
            cv_params: Cross-validation parameters (optional)
            
        Returns:
            Tuple[Prophet, Dict[str, Any]]: (trained model, evaluation metrics)
        """
        try:
            # Initialize model with parameters
            model = Prophet(**params)
            
            # Add regressors if present in training data
            for column in train_df.columns:
                if column not in ['ds', 'y']:
                    model.add_regressor(column)
            
            # Fit model
            model.fit(train_df)
            
            # Perform cross-validation
            if cv_params is None:
                cv_params = self.cv_params
            
            cv_results = cross_validation(
                model,
                initial=cv_params['initial'],
                period=cv_params['period'],
                horizon=cv_params['horizon'],
                parallel=cv_params['parallel']
            )
            
            # Calculate performance metrics
            metrics = performance_metrics(cv_results)
            
            # Add parameter information to metrics
            metrics_dict = metrics.to_dict('records')[0]
            metrics_dict.update(params)
            metrics_dict['timestamp'] = datetime.now().isoformat()
            
            return model, metrics_dict
            
        except Exception as e:
            logger.error(f"Error in model training and evaluation: {str(e)}")
            raise

    def perform_grid_search(
        self,
        train_df: pd.DataFrame,
        cv_params: Optional[Dict[str, Any]] = None
    ) -> Tuple[Prophet, Dict[str, Any], List[Dict[str, Any]]]:
        """
        Perform grid search over hyperparameter space.
        
        Args:
            train_df: Training data
            cv_params: Cross-validation parameters (optional)
            
        Returns:
            Tuple[Prophet, Dict[str, Any], List[Dict[str, Any]]]: 
                (best model, best parameters, all results)
        """
        try:
            param_combinations = self.generate_parameter_combinations()
            all_results = []
            best_rmse = float('inf')
            best_model = None
            best_params = None
            
            # Track tuning metadata
            tuning_metadata = {
                'param_grid': self.param_grid,
                'cv_params': cv_params or self.cv_params,
                'start_time': datetime.now().isoformat(),
                'total_combinations': len(param_combinations)
            }
            
            # Perform grid search
            for params in tqdm(param_combinations, desc="Tuning hyperparameters"):
                model, metrics = self.train_and_evaluate_model(params, train_df, cv_params)
                all_results.append(metrics)
                
                # Update best model if current one is better
                if metrics['rmse'] < best_rmse:
                    best_rmse = metrics['rmse']
                    best_model = model
                    best_params = params
                    
                # Save intermediate results
                self._save_tuning_results(all_results, tuning_metadata)
            
            # Update and save final results
            tuning_metadata['end_time'] = datetime.now().isoformat()
            tuning_metadata['best_params'] = best_params
            tuning_metadata['best_rmse'] = best_rmse
            self._save_tuning_results(all_results, tuning_metadata)
            
            # Save best model
            self._save_best_model(best_model, best_params)
            
            return best_model, best_params, all_results
            
        except Exception as e:
            logger.error(f"Error in grid search: {str(e)}")
            raise

    def _save_tuning_results(
        self,
        results: List[Dict[str, Any]],
        metadata: Dict[str, Any]
    ):
        """Save tuning results and metadata."""
        try:
            # Save results
            self.data_manager.save_data(
                data=results,
                category='models',
                subcategory='evaluation',
                prefix='tuning_results',
                extension='json',
                metadata=metadata
            )
        except Exception as e:
            logger.error(f"Error saving tuning results: {str(e)}")
            raise

    def _save_best_model(
        self,
        model: Prophet,
        params: Dict[str, Any]
    ):
        """Save best model and its parameters."""
        try:
            # Save model
            self.data_manager.save_data(
                data=model,
                category='models',
                subcategory='trained',
                prefix='prophet_model',
                extension='pkl',
                metadata=params
            )
            
            logger.info("Saved best model and parameters")
        except Exception as e:
            logger.error(f"Error saving best model: {str(e)}")
            raise

    def load_best_model(self) -> Tuple[Prophet, Dict[str, Any]]:
        """Load the best performing model and its parameters."""
        try:
            # Load latest model
            model = self.data_manager.load_data(
                category='models',
                subcategory='trained',
                prefix='prophet_model',
                extension='pkl'
            )
            
            # Load model parameters
            version_id = self.data_manager.get_latest_version(
                category='models',
                subcategory='trained',
                prefix='prophet_model',
                extension='pkl'
            )
            
            params = self.data_manager.load_data(
                category='models',
                subcategory='trained',
                prefix=f"{version_id}_metadata",
                extension='json'
            )
            
            return model, params
            
        except Exception as e:
            logger.error(f"Error loading best model: {str(e)}")
            raise

    def plot_parameter_importance(
        self,
        results: List[Dict[str, Any]],
        metric: str = 'rmse'
    ) -> Dict[str, float]:
        """
        Analyze parameter importance based on performance metrics.
        
        Args:
            results: List of tuning results
            metric: Metric to use for importance calculation
            
        Returns:
            Dict[str, float]: Parameter importance scores
        """
        try:
            df = pd.DataFrame(results)
            importance_scores = {}
            
            # Calculate importance for each parameter
            for param in self.param_grid.keys():
                if param in df.columns:
                    # Calculate variance of metric for each parameter value
                    param_importance = df.groupby(param)[metric].mean().var()
                    importance_scores[param] = param_importance
            
            # Normalize scores
            total = sum(importance_scores.values())
            importance_scores = {k: v/total for k, v in importance_scores.items()}
            
            # Save importance analysis
            self.data_manager.save_data(
                data=importance_scores,
                category='models',
                subcategory='evaluation',
                prefix='parameter_importance',
                extension='json',
                metadata={'metric': metric}
            )
            
            return importance_scores
            
        except Exception as e:
            logger.error(f"Error analyzing parameter importance: {str(e)}")
            raise

def initialize() -> ModelTuner:
    """Initialize the model tuner."""
    config = config_loader.load_config()
    return ModelTuner(config) 