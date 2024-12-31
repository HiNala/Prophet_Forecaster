"""
Model preparation module for the Prophet Forecaster application.
Handles synthetic feature generation and final data shaping for Prophet.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
import os
from datetime import datetime
import holidays
from scipy import stats
from sklearn.preprocessing import StandardScaler

from ..utils import logger, config_loader, data_manager

class ModelPreparation:
    def __init__(self, config: Dict[str, Any]):
        """Initialize the model preparation module with configuration."""
        self.config = config
        self.model_config = config.get('model', {})
        
        # Training configuration
        self.training_config = self.model_config.get('training', {})
        self.default_periods = self.training_config.get('default_periods', 60)
        self.max_periods = self.training_config.get('max_periods', 365)
        
        # Feature generation settings
        self.default_lags = [1, 7, 14, 30]  # Default lag periods
        self.default_rolling_windows = [7, 14, 30]  # Default rolling windows
        
        # Calendar features
        self.us_holidays = holidays.US()  # US holiday calendar
        
        # Data quality thresholds
        self.min_data_points = self.model_config.get('min_data_points', 100)
        self.max_missing_pct = self.model_config.get('max_missing_pct', 0.1)
        self.correlation_threshold = self.model_config.get('correlation_threshold', 0.95)
        
        # Initialize data manager
        self.data_manager = data_manager.initialize()

    def validate_data(self, df: pd.DataFrame, target_col: str, datetime_col: str) -> Tuple[bool, List[str]]:
        """
        Validate input data for model preparation.
        
        Returns:
            Tuple[bool, List[str]]: (is_valid, list of issues found)
        """
        issues = []
        
        # Check minimum data points
        if len(df) < self.min_data_points:
            issues.append(f"Insufficient data points ({len(df)}) for reliable modeling")
        
        # Check for required columns
        if datetime_col not in df.columns:
            issues.append(f"Missing datetime column '{datetime_col}'")
        if target_col not in df.columns:
            issues.append(f"Missing target column '{target_col}'")
        
        # Check data types
        if datetime_col in df.columns:
            if not pd.api.types.is_datetime64_any_dtype(df[datetime_col]):
                issues.append(f"Column '{datetime_col}' is not datetime type")
        if target_col in df.columns:
            if not pd.api.types.is_numeric_dtype(df[target_col]):
                issues.append(f"Column '{target_col}' is not numeric type")
        
        # Check for missing values
        missing_cols = df.isnull().sum()
        missing_cols = missing_cols[missing_cols > 0]
        if not missing_cols.empty:
            for col, count in missing_cols.items():
                pct = count / len(df)
                if pct > self.max_missing_pct:
                    issues.append(f"Column {col} has {pct:.1%} missing values")
        
        # Check for constant or near-constant columns
        for col in df.select_dtypes(include=[np.number]).columns:
            unique_ratio = df[col].nunique() / len(df)
            if unique_ratio < 0.01:  # Arbitrary threshold
                issues.append(f"Column {col} has low variance (unique ratio: {unique_ratio:.3f})")
        
        # Check for monotonic timestamps
        if datetime_col in df.columns:
            if not df[datetime_col].is_monotonic_increasing:
                issues.append("Timestamps are not monotonically increasing")
        
        return len(issues) == 0, issues

    def check_feature_quality(self, df: pd.DataFrame, feature_name: str) -> Tuple[bool, List[str]]:
        """
        Check quality of generated feature.
        
        Returns:
            Tuple[bool, List[str]]: (is_valid, list of issues found)
        """
        issues = []
        
        if feature_name not in df.columns:
            issues.append(f"Feature {feature_name} not found in DataFrame")
            return False, issues
        
        # Check for missing values
        missing_pct = df[feature_name].isnull().mean()
        if missing_pct > 0:
            issues.append(f"Feature {feature_name} has {missing_pct:.1%} missing values")
        
        # Check for infinite values
        inf_count = np.isinf(df[feature_name]).sum()
        if inf_count > 0:
            issues.append(f"Feature {feature_name} has {inf_count} infinite values")
        
        # Check for constant values
        if df[feature_name].nunique() == 1:
            issues.append(f"Feature {feature_name} is constant")
        
        return len(issues) == 0, issues

    def load_latest_data(self) -> pd.DataFrame:
        """Load the most recent data file with indicators."""
        try:
            # Try to load latest data with indicators
            df = self.data_manager.load_data(
                category='data',
                subcategory='interim',
                prefix='data_with_indicators'
            )
            
            # If not found, try cleaned data
            if df is None:
                df = self.data_manager.load_data(
                    category='data',
                    subcategory='interim',
                    prefix='cleaned'
                )
            
            # Ensure datetime type
            if 'ds' in df.columns:
                df['ds'] = pd.to_datetime(df['ds'])
            
            return df
            
        except FileNotFoundError:
            logger.error("No suitable data files found")
            raise
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise

    def save_features(self, df: pd.DataFrame, feature_type: str, metadata: Optional[Dict] = None):
        """Save generated features with metadata."""
        try:
            self.data_manager.save_data(
                data=df,
                category='data',
                subcategory='features',
                prefix=f'{feature_type}_features',
                metadata=metadata
            )
        except Exception as e:
            logger.error(f"Error saving features: {str(e)}")
            raise

    def add_lagged_features(
        self,
        df: pd.DataFrame,
        target_col: str = 'y',
        lags: Optional[List[int]] = None
    ) -> pd.DataFrame:
        """Add lagged versions of the target variable and handle NaN values."""
        try:
            if lags is None:
                lags = self.default_lags
            
            # Create a copy to avoid modifying the original dataframe
            df_copy = df.copy()
            
            # Track feature metadata
            feature_metadata = {
                'feature_type': 'lag',
                'target_column': target_col,
                'lag_periods': lags,
                'timestamp': datetime.now().isoformat()
            }
            
            # Add lag features
            for lag in lags:
                feature_name = f'{target_col}_lag_{lag}'
                df_copy[feature_name] = df_copy[target_col].shift(lag)
                
                # Validate feature
                is_valid, issues = self.check_feature_quality(df_copy, feature_name)
                if not is_valid:
                    for issue in issues:
                        logger.warning(issue)
                else:
                    logger.info(f"Added lag {lag} feature")
            
            # Drop rows with NaN values
            original_len = len(df_copy)
            df_copy = df_copy.dropna()
            dropped_rows = original_len - len(df_copy)
            
            if dropped_rows > 0:
                logger.info(f"Dropped {dropped_rows} rows with NaN values from lag features")
            
            # Save features
            feature_metadata['dropped_rows'] = dropped_rows
            self.save_features(df_copy, 'lag', feature_metadata)
            
            return df_copy
            
        except Exception as e:
            logger.error(f"Error adding lagged features: {str(e)}")
            raise

    def add_rolling_features(
        self,
        df: pd.DataFrame,
        target_col: str = 'y',
        windows: Optional[List[int]] = None
    ) -> pd.DataFrame:
        """Add rolling statistics features."""
        try:
            if windows is None:
                windows = self.default_rolling_windows
            
            df_copy = df.copy()
            
            # Track feature metadata
            feature_metadata = {
                'feature_type': 'rolling',
                'target_column': target_col,
                'window_sizes': windows,
                'timestamp': datetime.now().isoformat()
            }
            
            for window in windows:
                # Rolling mean
                feature_name = f'rolling_mean_{window}'
                df_copy[feature_name] = df_copy[target_col].rolling(window=window, min_periods=1).mean()
                df_copy[feature_name] = df_copy[feature_name].bfill().fillna(df_copy[target_col].mean())
                self.check_feature_quality(df_copy, feature_name)
                
                # Rolling standard deviation
                feature_name = f'rolling_std_{window}'
                df_copy[feature_name] = df_copy[target_col].rolling(window=window, min_periods=1).std()
                df_copy[feature_name] = df_copy[feature_name].bfill().fillna(df_copy[target_col].std())
                self.check_feature_quality(df_copy, feature_name)
                
                # Rolling min/max
                feature_name = f'rolling_min_{window}'
                df_copy[feature_name] = df_copy[target_col].rolling(window=window, min_periods=1).min()
                df_copy[feature_name] = df_copy[feature_name].bfill().fillna(df_copy[target_col].min())
                self.check_feature_quality(df_copy, feature_name)
                
                feature_name = f'rolling_max_{window}'
                df_copy[feature_name] = df_copy[target_col].rolling(window=window, min_periods=1).max()
                df_copy[feature_name] = df_copy[feature_name].bfill().fillna(df_copy[target_col].max())
                self.check_feature_quality(df_copy, feature_name)
                
                # Additional rolling statistics
                feature_name = f'rolling_skew_{window}'
                df_copy[feature_name] = df_copy[target_col].rolling(window=window, min_periods=1).skew()
                df_copy[feature_name] = df_copy[feature_name].bfill().fillna(0)  # Fill skew with 0
                self.check_feature_quality(df_copy, feature_name)
                
                feature_name = f'rolling_kurt_{window}'
                df_copy[feature_name] = df_copy[target_col].rolling(window=window, min_periods=1).kurt()
                df_copy[feature_name] = df_copy[feature_name].bfill().fillna(0)  # Fill kurtosis with 0
                self.check_feature_quality(df_copy, feature_name)
                
                logger.info(f"Added rolling window {window} features")
            
            # Save features
            self.save_features(df_copy, 'rolling', feature_metadata)
            
            return df_copy
            
        except Exception as e:
            logger.error(f"Error adding rolling features: {str(e)}")
            raise

    def add_calendar_features(
        self,
        df: pd.DataFrame,
        datetime_col: str = 'ds'
    ) -> pd.DataFrame:
        """Add calendar-based features."""
        try:
            # Track feature metadata
            feature_metadata = {
                'feature_type': 'calendar',
                'datetime_column': datetime_col,
                'timestamp': datetime.now().isoformat()
            }
            
            # Ensure datetime column is datetime type
            df[datetime_col] = pd.to_datetime(df[datetime_col])
            
            # Basic time features
            df['day_of_week'] = df[datetime_col].dt.dayofweek
            df['day_of_month'] = df[datetime_col].dt.day
            df['month'] = df[datetime_col].dt.month
            df['quarter'] = df[datetime_col].dt.quarter
            df['year'] = df[datetime_col].dt.year
            df['week_of_year'] = df[datetime_col].dt.isocalendar().week
            
            # Cyclical encoding of time features
            df['day_of_week_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
            df['day_of_week_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
            df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
            df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
            
            # Is weekend
            df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
            
            # Is holiday
            df['is_holiday'] = df[datetime_col].map(lambda x: x in self.us_holidays).astype(int)
            
            # Month start/end
            df['is_month_start'] = df[datetime_col].dt.is_month_start.astype(int)
            df['is_month_end'] = df[datetime_col].dt.is_month_end.astype(int)
            df['is_quarter_start'] = df[datetime_col].dt.is_quarter_start.astype(int)
            df['is_quarter_end'] = df[datetime_col].dt.is_quarter_end.astype(int)
            df['is_year_start'] = df[datetime_col].dt.is_year_start.astype(int)
            df['is_year_end'] = df[datetime_col].dt.is_year_end.astype(int)
            
            # Save features
            self.save_features(df, 'calendar', feature_metadata)
            
            logger.info("Added calendar features")
            return df
            
        except Exception as e:
            logger.error(f"Error adding calendar features: {str(e)}")
            raise

    def remove_highly_correlated_features(
        self,
        df: pd.DataFrame,
        threshold: float = None,
        preserve_cols: List[str] = None
    ) -> pd.DataFrame:
        """Remove highly correlated features to reduce multicollinearity."""
        try:
            if threshold is None:
                threshold = self.correlation_threshold
            
            if preserve_cols is None:
                preserve_cols = ['ds', 'y']  # Always preserve Prophet required columns
            
            # Calculate correlation matrix for numeric columns except preserved ones
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            cols_to_check = [col for col in numeric_cols if col not in preserve_cols]
            
            if not cols_to_check:
                return df
            
            corr_matrix = df[cols_to_check].corr().abs()
            
            # Select upper triangle of correlation matrix
            upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
            
            # Find features with correlation greater than threshold
            to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
            
            if to_drop:
                logger.info(f"Removing {len(to_drop)} highly correlated features: {to_drop}")
                df = df.drop(columns=to_drop)
            
            return df
            
        except Exception as e:
            logger.error(f"Error removing correlated features: {str(e)}")
            raise

    def prepare_prophet_features(
        self,
        df: pd.DataFrame,
        target_col: str = 'y',
        datetime_col: str = 'ds'
    ) -> pd.DataFrame:
        """Ensure data is in Prophet format and handle any necessary column renaming."""
        try:
            # Track feature metadata
            feature_metadata = {
                'feature_type': 'prophet',
                'target_column': target_col,
                'datetime_column': datetime_col,
                'timestamp': datetime.now().isoformat()
            }
            
            prophet_df = pd.DataFrame()
            
            # Ensure required columns are present and handle datetime
            if datetime_col != 'ds':
                # Convert to datetime and remove timezone
                prophet_df['ds'] = pd.to_datetime(df[datetime_col]).dt.tz_localize(None)
            else:
                # Remove timezone if present
                prophet_df['ds'] = pd.to_datetime(df['ds']).dt.tz_localize(None)
                
            if target_col != 'y':
                prophet_df['y'] = df[target_col]
            else:
                prophet_df['y'] = df['y']
            
            # Scale numeric features
            scaler = StandardScaler()
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            numeric_cols = [col for col in numeric_cols if col not in [target_col, 'ds', 'y']]
            
            if numeric_cols:
                scaled_features = scaler.fit_transform(df[numeric_cols])
                scaled_df = pd.DataFrame(scaled_features, columns=numeric_cols, index=df.index)
                
                # Add scaled features to prophet_df
                for col in numeric_cols:
                    prophet_df[col] = scaled_df[col]
                
                # Update metadata
                feature_metadata['scaled_features'] = numeric_cols
            
            # Add categorical features
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns
            for col in categorical_cols:
                if col not in [datetime_col, target_col, 'ds', 'y']:
                    prophet_df[col] = df[col]
            
            # Save features
            self.save_features(prophet_df, 'prophet', feature_metadata)
            
            return prophet_df
            
        except Exception as e:
            logger.error(f"Error preparing Prophet features: {str(e)}")
            raise

    def split_train_test(
        self,
        df: pd.DataFrame,
        test_ratio: float = 0.2
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Split data into training and test sets."""
        try:
            if test_ratio <= 0 or test_ratio >= 1:
                raise ValueError("test_ratio must be between 0 and 1")
            
            df = df.sort_values('ds').reset_index(drop=True)
            split_idx = int(len(df) * (1 - test_ratio))
            
            if split_idx <= 0 or split_idx >= len(df):
                raise ValueError("Invalid split index calculated")
            
            train_df = df.iloc[:split_idx].copy()
            test_df = df.iloc[split_idx:].copy()
            
            # Validate split
            if len(train_df) == 0 or len(test_df) == 0:
                raise ValueError("Invalid split: empty train or test set")
            
            logger.info(f"Split data into train ({len(train_df)} rows) and test ({len(test_df)} rows) sets")
            return train_df, test_df
            
        except Exception as e:
            logger.error(f"Error splitting data: {str(e)}")
            raise

    def process(
        self,
        target_col: str = 'y',
        datetime_col: str = 'ds',
        test_ratio: float = 0.2
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Main processing function that orchestrates feature generation and data preparation.
        
        Args:
            target_col (str): Target column name
            datetime_col (str): Datetime column name
            test_ratio (float): Ratio of data to use for testing
            
        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: (train_df, test_df)
        """
        try:
            # Load the data
            df = self.load_latest_data()
            
            # Validate input data
            is_valid, issues = self.validate_data(df, target_col, datetime_col)
            if not is_valid:
                for issue in issues:
                    logger.warning(f"Data validation issue: {issue}")
            
            # Add synthetic features
            df = self.add_lagged_features(df, target_col)
            df = self.add_rolling_features(df, target_col)
            df = self.add_calendar_features(df, datetime_col)
            
            # Remove highly correlated features
            df = self.remove_highly_correlated_features(df)
            
            # Prepare for Prophet
            df = self.prepare_prophet_features(df, target_col, datetime_col)
            
            # Drop any remaining NaN values
            original_len = len(df)
            df = df.dropna()
            dropped_rows = original_len - len(df)
            if dropped_rows > 0:
                logger.warning(f"Dropped {dropped_rows} rows with NaN values")
            
            # Split into train/test sets
            train_df, test_df = self.split_train_test(df, test_ratio)
            
            # Save processed datasets
            train_metadata = {
                'type': 'train',
                'target_column': target_col,
                'datetime_column': datetime_col,
                'test_ratio': test_ratio,
                'timestamp': datetime.now().isoformat()
            }
            
            test_metadata = {
                'type': 'test',
                'target_column': target_col,
                'datetime_column': datetime_col,
                'test_ratio': test_ratio,
                'timestamp': datetime.now().isoformat()
            }
            
            self.data_manager.save_data(
                data=train_df,
                category='data',
                subcategory='processed',
                prefix='prophet_train',
                metadata=train_metadata
            )
            
            self.data_manager.save_data(
                data=test_df,
                category='data',
                subcategory='processed',
                prefix='prophet_test',
                metadata=test_metadata
            )
            
            logger.info("Saved prepared train and test datasets")
            
            return train_df, test_df
            
        except Exception as e:
            logger.error(f"Error in model preparation process: {str(e)}")
            raise

    def prepare_data(
        self,
        df: pd.DataFrame,
        test_size: float = 0.2,
        target_col: str = 'y',
        datetime_col: str = 'ds'
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Prepare data for model training by adding features and splitting into train/test sets.
        
        Args:
            df: Input DataFrame
            test_size: Proportion of data to use for testing
            target_col: Name of target column
            datetime_col: Name of datetime column
            
        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: (train_data, test_data)
        """
        try:
            # Validate input data
            is_valid, issues = self.validate_data(df, target_col, datetime_col)
            if not is_valid:
                for issue in issues:
                    logger.warning(f"Data validation issue: {issue}")
            
            # Add features
            df_features = df.copy()
            
            # Add lagged features
            df_features = self.add_lagged_features(df_features, target_col=target_col)
            
            # Add rolling features
            df_features = self.add_rolling_features(df_features, target_col=target_col)
            
            # Add calendar features
            df_features = self.add_calendar_features(df_features, datetime_col=datetime_col)
            
            # Remove highly correlated features while preserving essential ones
            preserve_cols = [datetime_col, target_col]  # Essential columns to keep
            df_features = self.remove_highly_correlated_features(
                df_features,
                preserve_cols=preserve_cols
            )
            
            # Prepare final features for Prophet
            df_features = self.prepare_prophet_features(
                df_features,
                target_col=target_col,
                datetime_col=datetime_col
            )
            
            # Split into train and test sets
            train_data, test_data = self.split_train_test(df_features, test_ratio=test_size)
            
            # Save the prepared data
            self.save_features(
                df_features,
                feature_type='prepared',
                metadata={
                    'target_column': target_col,
                    'datetime_column': datetime_col,
                    'test_size': test_size,
                    'n_features': len(df_features.columns),
                    'timestamp': datetime.now().isoformat()
                }
            )
            
            return train_data, test_data
            
        except Exception as e:
            logger.error(f"Error preparing data: {str(e)}")
            raise

def initialize() -> ModelPreparation:
    """Initialize the model preparation module with configuration."""
    config = config_loader.load_config()
    return ModelPreparation(config)

# Convenience function for direct usage
def process(**kwargs) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Convenience function to process model preparation."""
    preparation = initialize()
    return preparation.process(**kwargs) 