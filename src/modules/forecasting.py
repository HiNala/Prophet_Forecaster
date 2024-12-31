"""
Forecasting module for the Prophet Forecaster application.
Handles forecast generation, visualization, and evaluation.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
import os
from prophet import Prophet
from prophet.plot import plot_cross_validation_metric
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pickle
import json
from datetime import datetime

from ..utils import logger, config_loader

class Forecasting:
    def __init__(self, config: Dict[str, Any]):
        """Initialize the forecasting module with configuration."""
        self.config = config
        self.model_config = config.get('model', {})
        
        # Forecasting configuration
        self.forecast_config = self.model_config.get('forecast', {})
        self.default_periods = self.forecast_config.get('default_periods', 60)
        self.confidence_interval = self.forecast_config.get('confidence_interval', 0.95)
        
        # Visualization configuration
        self.viz_config = config.get('visualization', {})
        self.figure_size = self.viz_config.get('figure_size', [1200, 800])
        self.line_colors = self.viz_config.get('colors', {
            'actual': 'blue',
            'forecast': 'red',
            'ci': 'rgba(255, 0, 0, 0.2)'
        })
        
        # Paths
        self.model_dir = "models"
        self.output_dir = "forecasts"
        os.makedirs(self.output_dir, exist_ok=True)

    def load_latest_model(self) -> Tuple[Prophet, Dict[str, Any]]:
        """Load the most recent trained model and its metadata."""
        try:
            # Find latest model file
            model_files = [f for f in os.listdir(self.model_dir) if f.endswith('.pkl')]
            if not model_files:
                raise FileNotFoundError("No trained models found")
            
            latest_model = max(model_files, key=lambda x: os.path.getctime(os.path.join(self.model_dir, x)))
            model_path = os.path.join(self.model_dir, latest_model)
            
            # Load model
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            
            # Load corresponding metadata
            # Get the timestamp from the model filename (format: prophet_model_YYYYMMDD_HHMMSS.pkl)
            timestamp = latest_model.replace('prophet_model_', '').replace('.pkl', '')
            metadata_path = os.path.join(self.model_dir, f"model_metadata_{timestamp}.json")
            
            if not os.path.exists(metadata_path):
                logger.warning(f"Metadata file not found at {metadata_path}")
                # Try to find metadata file by matching timestamp pattern
                metadata_files = [f for f in os.listdir(self.model_dir) if f.startswith('model_metadata_')]
                if metadata_files:
                    metadata_path = os.path.join(self.model_dir, max(metadata_files, key=lambda x: os.path.getctime(os.path.join(self.model_dir, x))))
                    logger.info(f"Using metadata file: {metadata_path}")
            
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            logger.info(f"Loaded model from {latest_model}")
            return model, metadata
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise

    def load_test_data(self) -> pd.DataFrame:
        """Load the test dataset."""
        try:
            test_path = "data/prophet_test.csv"
            if not os.path.exists(test_path):
                raise FileNotFoundError("Test data not found")
            
            return pd.read_csv(test_path)
            
        except Exception as e:
            logger.error(f"Error loading test data: {str(e)}")
            raise

    def generate_forecast(
        self,
        model: Prophet,
        periods: int = None,
        include_history: bool = True
    ) -> pd.DataFrame:
        """Generate forecast using the trained model."""
        try:
            if periods is None:
                periods = self.default_periods
            
            # Create future dataframe
            future = model.make_future_dataframe(
                periods=periods,
                include_history=include_history
            )
            
            # Add any required regressors for future dates
            for regressor in model.extra_regressors:
                if regressor in future.columns:
                    continue
                # For now, use the mean of historical values
                historical_mean = model.history[regressor].mean()
                future[regressor] = historical_mean
            
            # Generate forecast
            forecast = model.predict(future)
            logger.info(f"Generated forecast for {periods} periods")
            
            return forecast
            
        except Exception as e:
            logger.error(f"Error generating forecast: {str(e)}")
            raise

    def calculate_forecast_metrics(
        self,
        forecast: pd.DataFrame,
        actuals: pd.DataFrame
    ) -> Dict[str, float]:
        """Calculate forecast accuracy metrics."""
        try:
            # Ensure datetime columns are in the same format
            forecast['ds'] = pd.to_datetime(forecast['ds'])
            actuals['ds'] = pd.to_datetime(actuals['ds'])
            
            # Merge forecast with actuals
            merged = forecast.merge(
                actuals[['ds', 'y']],
                on='ds',
                how='inner',
                suffixes=('_pred', '')
            )
            
            # Calculate metrics
            metrics = {
                'mape': np.mean(np.abs((merged['y'] - merged['yhat']) / merged['y'])) * 100,
                'mae': np.mean(np.abs(merged['y'] - merged['yhat'])),
                'rmse': np.sqrt(np.mean((merged['y'] - merged['yhat'])**2)),
                'coverage': np.mean((merged['y'] >= merged['yhat_lower']) & 
                                 (merged['y'] <= merged['yhat_upper'])) * 100
            }
            
            logger.info("Calculated forecast metrics")
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating metrics: {str(e)}")
            raise

    def plot_forecast(
        self,
        forecast: pd.DataFrame,
        history: pd.DataFrame = None,
        show_components: bool = True,
        output_path: Optional[str] = None
    ) -> None:
        """Create interactive forecast visualization."""
        try:
            # Create figure with secondary y-axis
            fig = make_subplots(
                rows=3 if show_components else 1,
                cols=1,
                subplot_titles=['Forecast', 'Trend', 'Seasonalities'] if show_components else ['Forecast'],
                row_heights=[0.5, 0.25, 0.25] if show_components else [1],
                vertical_spacing=0.1
            )
            
            # Plot actual values if available
            if history is not None:
                fig.add_trace(
                    go.Scatter(
                        x=history['ds'],
                        y=history['y'],
                        name='Actual',
                        line=dict(color=self.line_colors['actual'])
                    ),
                    row=1, col=1
                )
            
            # Plot forecast
            fig.add_trace(
                go.Scatter(
                    x=forecast['ds'],
                    y=forecast['yhat'],
                    name='Forecast',
                    line=dict(color=self.line_colors['forecast'])
                ),
                row=1, col=1
            )
            
            # Add confidence intervals
            fig.add_trace(
                go.Scatter(
                    x=forecast['ds'].tolist() + forecast['ds'].tolist()[::-1],
                    y=forecast['yhat_upper'].tolist() + forecast['yhat_lower'].tolist()[::-1],
                    fill='toself',
                    fillcolor=self.line_colors['ci'],
                    line=dict(color='rgba(255,255,255,0)'),
                    name='Confidence Interval'
                ),
                row=1, col=1
            )
            
            if show_components:
                # Plot trend
                fig.add_trace(
                    go.Scatter(
                        x=forecast['ds'],
                        y=forecast['trend'],
                        name='Trend',
                        line=dict(color='green')
                    ),
                    row=2, col=1
                )
                
                # Plot yearly seasonality if available
                if 'yearly' in forecast.columns:
                    fig.add_trace(
                        go.Scatter(
                            x=forecast['ds'],
                            y=forecast['yearly'],
                            name='Yearly Seasonality',
                            line=dict(color='purple')
                        ),
                        row=3, col=1
                    )
                
                # Plot weekly seasonality if available
                if 'weekly' in forecast.columns:
                    fig.add_trace(
                        go.Scatter(
                            x=forecast['ds'],
                            y=forecast['weekly'],
                            name='Weekly Seasonality',
                            line=dict(color='orange')
                        ),
                        row=3, col=1
                    )
            
            # Update layout
            fig.update_layout(
                title='Prophet Forecast with Components',
                showlegend=True,
                width=self.figure_size[0],
                height=self.figure_size[1]
            )
            
            # Save plot
            if output_path is None:
                output_path = os.path.join(self.output_dir, f"forecast_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html")
            
            fig.write_html(output_path)
            logger.info(f"Saved forecast plot to {output_path}")
            
        except Exception as e:
            logger.error(f"Error plotting forecast: {str(e)}")
            raise

    def process(
        self,
        periods: Optional[int] = None,
        include_history: bool = True,
        output_file: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Main processing function that orchestrates forecast generation and visualization.
        
        Args:
            periods (int, optional): Number of periods to forecast
            include_history (bool): Whether to include historical data in the forecast
            output_file (str, optional): Path to save the forecast data
            
        Returns:
            pd.DataFrame: Generated forecast
        """
        try:
            # Load model and metadata
            model, metadata = self.load_latest_model()
            
            # Generate forecast
            forecast = self.generate_forecast(
                model,
                periods=periods,
                include_history=include_history
            )
            
            # Save forecast data if output path is provided
            if output_file:
                forecast.to_csv(output_file, index=False)
                logger.info(f"Saved forecast data to {output_file}")
            else:
                # Save to default location
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_path = os.path.join(self.output_dir, f"forecast_{timestamp}.csv")
                forecast.to_csv(output_path, index=False)
                logger.info(f"Saved forecast data to {output_path}")
            
            # Create visualization
            plot_path = os.path.join(self.output_dir, f"forecast_plot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html")
            self.plot_forecast(
                forecast,
                history=model.history if include_history else None,
                output_path=plot_path
            )
            
            return forecast
            
        except Exception as e:
            logger.error(f"Error in forecasting process: {str(e)}")
            raise

    def evaluate(
        self,
        cross_validation: bool = False,
        output_file: Optional[str] = None
    ) -> Dict[str, float]:
        """
        Evaluate forecast performance using test data or cross-validation.
        
        Args:
            cross_validation (bool): Whether to use cross-validation for evaluation
            output_file (str, optional): Path to save the evaluation results
            
        Returns:
            Dict[str, float]: Evaluation metrics
        """
        try:
            # Load model and metadata
            model, metadata = self.load_latest_model()
            
            if cross_validation:
                # Use cross-validation results if available
                if 'metrics' not in metadata:
                    raise ValueError("No cross-validation metrics available in model metadata")
                metrics = metadata['metrics']
            else:
                # Load test data and generate forecast
                test_data = self.load_test_data()
                forecast = self.generate_forecast(
                    model,
                    periods=len(test_data),
                    include_history=False
                )
                metrics = self.calculate_forecast_metrics(forecast, test_data)
            
            # Save metrics if output path is provided
            if output_file:
                with open(output_file, 'w') as f:
                    json.dump(metrics, f, indent=4)
                logger.info(f"Saved evaluation metrics to {output_file}")
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error in evaluation process: {str(e)}")
            raise

def initialize() -> Forecasting:
    """Initialize the forecasting module with configuration."""
    config = config_loader.load_config()
    return Forecasting(config)

# Convenience functions for direct usage
def process(**kwargs) -> pd.DataFrame:
    """Convenience function to process forecasting."""
    forecasting = initialize()
    return forecasting.process(**kwargs)

def evaluate(**kwargs) -> Dict[str, float]:
    """Convenience function to evaluate forecasts."""
    forecasting = initialize()
    return forecasting.evaluate(**kwargs) 