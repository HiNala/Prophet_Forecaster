"""
Initialize the modules package.
"""

from .data_ingestion import DataIngestion
from .data_cleaning import DataCleaning
from .technical_indicators import TechnicalIndicators
from .model_preparation import ModelPreparation
from .model_training import ModelTraining
from .forecasting import Forecasting

# Import process functions for backward compatibility
from .data_ingestion import process as process_ingestion
from .data_cleaning import process as process_cleaning
from .technical_indicators import process as process_indicators
from .model_preparation import process as process_preparation
from .model_training import process as process_training
from .forecasting import process as process_forecasting

__all__ = [
    'DataIngestion',
    'DataCleaning',
    'TechnicalIndicators',
    'ModelPreparation',
    'ModelTraining',
    'Forecasting',
    'process_ingestion',
    'process_cleaning',
    'process_indicators',
    'process_preparation',
    'process_training',
    'process_forecasting'
] 