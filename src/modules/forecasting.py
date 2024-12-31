"""
Forecasting module for generating and visualizing Prophet predictions.
"""

import pandas as pd
import numpy as np
from prophet import Prophet
from typing import Dict, Any, Optional, List, Union
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
from datetime import datetime, timedelta
import json
import pickle

from ..utils.logger import setup_logger

logger = setup_logger()

class Forecasting:
    """Handles forecast generation and visualization."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize Forecasting with configuration."""
        self.config = config
        self.forecast_dir = "forecasts/predictions"
        os.makedirs(self.forecast_dir, exist_ok=True)
        self._model = None
        
    @property
    def model(self) -> Optional[Prophet]:
        """Get the current Prophet model."""
        if self._model is None:
            # Try to load the latest model
            try:
                model_files = [f for f in os.listdir("models") if f.endswith('.pkl')]
                if model_files:
                    latest_model = max(model_files, key=lambda x: os.path.getctime(os.path.join("models", x)))
                    model_path = os.path.join("models", latest_model)
                    with open(model_path, 'rb') as f:
                        self._model = pickle.load(f)
                    logger.info(f"Loaded model from {model_path}")
            except Exception as e:
                logger.warning(f"Could not load model: {str(e)}")
        return self._model
    
    @model.setter
    def model(self, value: Prophet) -> None:
        """Set the current Prophet model."""
        self._model = value
    
    def generate_forecast(
        self,
        model: Prophet,
        periods: int = 30,
        include_history: bool = True
    ) -> Dict[str, pd.DataFrame]:
        """
        Generate forecasts using the trained Prophet model.
        
        Args:
            model: Trained Prophet model
            periods: Number of periods to forecast
            include_history: Whether to include historical data in the forecast
            
        Returns:
            Dict containing forecast DataFrame and components
        """
        logger.info(f"Generating forecast for {periods} periods")
        
        # Get the last date in the training data
        last_date = model.history['ds'].max()
        logger.info(f"Last date in training data: {last_date.strftime('%Y-%m-%d')}")
        
        # Create future dataframe starting from the last date
        future = model.make_future_dataframe(
            periods=periods,
            freq='D',
            include_history=include_history
        )
        
        # Generate forecast
        forecast = model.predict(future)
        
        # Get historical data from the model
        history = model.history
        
        # Save predictions to file
        self._save_predictions(forecast, history, periods, last_date)
        
        return {
            'forecast': forecast,
            'history': history
        }
    
    def _save_predictions(
        self,
        forecast: pd.DataFrame,
        history: pd.DataFrame,
        periods: int,
        last_training_date: pd.Timestamp
    ) -> None:
        """Save predictions to a text file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"forecast_predictions_{timestamp}.txt"
        filepath = os.path.join(self.forecast_dir, filename)
        
        # Get future predictions (dates after the last training date)
        future_predictions = forecast[forecast['ds'] > last_training_date].copy()
        future_predictions = future_predictions.tail(periods)  # Get only the requested number of periods
        
        with open(filepath, 'w') as f:
            f.write("Prophet Forecast Predictions\n")
            f.write(f"Generated at: {timestamp}\n")
            f.write("=" * 50 + "\n\n")
            
            f.write("Forecast Details:\n")
            f.write(f"Last training date: {last_training_date.strftime('%Y-%m-%d')}\n")
            f.write(f"Number of forecast periods: {periods}\n")
            f.write(f"Forecast start date: {future_predictions['ds'].min().strftime('%Y-%m-%d')}\n")
            f.write(f"Forecast end date: {future_predictions['ds'].max().strftime('%Y-%m-%d')}\n\n")
            
            f.write("Daily Predictions:\n")
            f.write("-" * 80 + "\n")
            f.write("Date            | Predicted Value | Lower Bound (95%) | Upper Bound (95%) |\n")
            f.write("-" * 80 + "\n")
            
            for _, row in future_predictions.iterrows():
                f.write(f"{row['ds'].strftime('%Y-%m-%d')} | {row['yhat']:14.2f} | {row['yhat_lower']:15.2f} | {row['yhat_upper']:15.2f} |\n")
            
            # Add summary statistics
            f.write("\nSummary Statistics:\n")
            f.write("-" * 40 + "\n")
            f.write(f"Mean prediction: {future_predictions['yhat'].mean():.2f}\n")
            f.write(f"Min prediction:  {future_predictions['yhat'].min():.2f}\n")
            f.write(f"Max prediction:  {future_predictions['yhat'].max():.2f}\n")
            f.write(f"Std deviation:   {future_predictions['yhat'].std():.2f}\n")
        
        logger.info(f"Predictions saved to {filepath}")
        
        # Also save as CSV for further analysis
        csv_filepath = os.path.join(self.forecast_dir, f"forecast_predictions_{timestamp}.csv")
        future_predictions.to_csv(csv_filepath, index=False)
        logger.info(f"Predictions also saved as CSV to {csv_filepath}")
    
    def plot_forecast(
        self,
        results: Dict[str, pd.DataFrame],
        output_dir: str = "forecasts"
    ) -> None:
        """
        Create an interactive plot of the forecast with components.
        
        Args:
            results: Dictionary containing forecast and components
            output_dir: Directory to save the plot
        """
        forecast = results['forecast']
        history = results['history']
        
        # Get the last historical date
        last_historical_date = history['ds'].max()
        
        # Create figure with secondary y-axis
        fig = make_subplots(
            rows=3,
            cols=1,
            subplot_titles=('Forecast', 'Trend', 'Seasonalities'),
            vertical_spacing=0.1,
            row_heights=[0.5, 0.25, 0.25]
        )
        
        # Add historical data
        fig.add_trace(
            go.Scatter(
                name='Historical',
                x=history['ds'],
                y=history['y'],
                mode='lines',
                line=dict(color='black', width=2),
                showlegend=True
            ),
            row=1, col=1
        )
        
        # Split forecast into historical and future periods
        historical_forecast = forecast[forecast['ds'] <= last_historical_date]
        future_forecast = forecast[forecast['ds'] > last_historical_date]
        
        # Add historical forecast (dashed line)
        fig.add_trace(
            go.Scatter(
                name='Historical Forecast',
                x=historical_forecast['ds'],
                y=historical_forecast['yhat'],
                mode='lines',
                line=dict(color='blue', width=1, dash='dash'),
                showlegend=True
            ),
            row=1, col=1
        )
        
        # Add future forecast (solid line)
        fig.add_trace(
            go.Scatter(
                name='Future Forecast',
                x=future_forecast['ds'],
                y=future_forecast['yhat'],
                mode='lines',
                line=dict(color='red', width=2),
                showlegend=True
            ),
            row=1, col=1
        )
        
        # Add confidence interval for future forecast only
        fig.add_trace(
            go.Scatter(
                name='Confidence Interval',
                x=pd.concat([future_forecast['ds'], future_forecast['ds'][::-1]]),
                y=pd.concat([future_forecast['yhat_upper'], future_forecast['yhat_lower'][::-1]]),
                fill='toself',
                fillcolor='rgba(255,0,0,0.2)',
                line=dict(color='rgba(255,0,0,0)'),
                showlegend=True
            ),
            row=1, col=1
        )
        
        # Add vertical line at the last historical date using shapes
        fig.add_shape(
            type="line",
            x0=last_historical_date,
            x1=last_historical_date,
            y0=0,
            y1=1,
            yref="paper",
            line=dict(color="gray", width=1, dash="dash"),
            row=1,
            col=1
        )
        
        # Add annotation for the vertical line
        fig.add_annotation(
            x=last_historical_date,
            y=1,
            yref="paper",
            text="Last Historical Date",
            showarrow=False,
            textangle=-90,
            xanchor="right",
            yanchor="bottom",
            row=1,
            col=1
        )
        
        # Add trend if available
        if 'trend' in forecast.columns:
            fig.add_trace(
                go.Scatter(
                    name='Trend',
                    x=forecast['ds'],
                    y=forecast['trend'],
                    mode='lines',
                    line=dict(color='green', width=2),
                    showlegend=True
                ),
                row=2, col=1
            )
        
        # Add seasonalities if available
        seasonality_added = False
        if 'yearly' in forecast.columns:
            fig.add_trace(
                go.Scatter(
                    name='Yearly Seasonality',
                    x=forecast['ds'],
                    y=forecast['yearly'],
                    mode='lines',
                    line=dict(color='purple', width=1),
                    showlegend=True
                ),
                row=3, col=1
            )
            seasonality_added = True
        
        if 'weekly' in forecast.columns:
            fig.add_trace(
                go.Scatter(
                    name='Weekly Seasonality',
                    x=forecast['ds'],
                    y=forecast['weekly'],
                    mode='lines',
                    line=dict(color='orange', width=1),
                    showlegend=True
                ),
                row=3, col=1
            )
            seasonality_added = True
            
        if not seasonality_added:
            # If no seasonalities are available, adjust the layout
            fig.update_layout(
                height=800  # Reduce height since we don't need the seasonalities plot
            )
        else:
            fig.update_layout(
                height=1200
            )
        
        # Update layout
        symbol_name = history['y'].name if hasattr(history['y'], 'name') else 'Value'
        fig.update_layout(
            title={
                'text': f'Prophet Forecast for {symbol_name}',
                'y':0.95,
                'x':0.5,
                'xanchor': 'center',
                'yanchor': 'top'
            },
            xaxis_title='Date',
            yaxis_title=symbol_name,
            showlegend=True,
            template='plotly_white',
            hovermode='x unified'
        )
        
        # Update y-axes titles
        fig.update_yaxes(title_text=symbol_name, row=1, col=1)
        if 'trend' in forecast.columns:
            fig.update_yaxes(title_text="Trend", row=2, col=1)
        if seasonality_added:
            fig.update_yaxes(title_text="Seasonality", row=3, col=1)
        
        # Save plot
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = os.path.join(output_dir, f"forecast_{timestamp}.html")
        fig.write_html(filepath)
        
        logger.info(f"Forecast plot saved to {filepath}")
    
    def evaluate_forecast(
        self,
        forecast: pd.DataFrame,
        actual: pd.DataFrame,
        metrics: Optional[list[str]] = None
    ) -> Dict[str, float]:
        """
        Calculate forecast evaluation metrics.
        
        Args:
            forecast: Forecast DataFrame
            actual: Actual values DataFrame
            metrics: List of metrics to calculate
            
        Returns:
            Dictionary of metric names and values
        """
        if metrics is None:
            metrics = ['rmse', 'mae', 'mape']
            
        results = {}
        merged = pd.merge(
            forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']],
            actual[['ds', 'y']],
            on='ds',
            how='inner'
        )
        
        if 'rmse' in metrics:
            results['rmse'] = np.sqrt(((merged['y'] - merged['yhat']) ** 2).mean())
        if 'mae' in metrics:
            results['mae'] = abs(merged['y'] - merged['yhat']).mean()
        if 'mape' in metrics:
            results['mape'] = (abs(merged['y'] - merged['yhat']) / merged['y']).mean() * 100
            
        return results 

def initialize() -> Forecasting:
    """Initialize the forecasting module with configuration."""
    from ..utils.config_loader import load_config
    config = load_config()
    return Forecasting(config)

def process(periods: int = 30, include_history: bool = True) -> Dict[str, Any]:
    """
    Convenience function for direct forecasting usage.
    
    Args:
        periods: Number of periods to forecast
        include_history: Whether to include historical data
        
    Returns:
        Dictionary containing forecast results
    """
    forecasting = initialize()
    model = forecasting.model
    
    # Generate forecast
    results = forecasting.generate_forecast(
        model=model,
        periods=periods,
        include_history=include_history
    )
    
    return results 