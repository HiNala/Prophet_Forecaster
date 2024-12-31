"""
Data cleaning module for the Prophet Forecaster application.
Handles data validation, missing value imputation, and outlier detection/handling.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
import os
from scipy import stats
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
from sklearn.covariance import EllipticEnvelope
import matplotlib.pyplot as plt
import seaborn as sns

from ..utils import logger, config_loader

class DataCleaning:
    def __init__(self, config: Dict[str, Any]):
        """Initialize the data cleaning module with configuration."""
        self.config = config
        self.cleaning_config = config.get('cleaning', {})
        
        # Missing value handling
        self.missing_methods = {
            'drop': self.handle_missing_drop,
            'impute': self.handle_missing_impute,
            'interpolate': self.handle_missing_interpolate,
            'forward': self.handle_missing_forward,
            'backward': self.handle_missing_backward
        }
        
        # Outlier detection methods
        self.outlier_methods = {
            'zscore': self.detect_outliers_zscore,
            'iqr': self.detect_outliers_iqr,
            'isolation_forest': self.detect_outliers_isolation_forest,
            'elliptic_envelope': self.detect_outliers_elliptic_envelope
        }
        
        # Configuration parameters
        self.zscore_threshold = self.cleaning_config.get('zscore_threshold', 3.0)
        self.iqr_multiplier = self.cleaning_config.get('iqr_multiplier', 1.5)
        self.contamination = self.cleaning_config.get('contamination', 0.1)
        
        # Data quality thresholds
        self.max_missing_pct = self.cleaning_config.get('max_missing_pct', 0.2)
        self.min_unique_ratio = self.cleaning_config.get('min_unique_ratio', 0.01)
        self.max_constant_pct = self.cleaning_config.get('max_constant_pct', 0.95)
        
        # Paths
        self.data_dir = "data"
        os.makedirs(self.data_dir, exist_ok=True)

    def check_data_quality(self, df: pd.DataFrame, stage: str = "") -> Tuple[bool, List[str]]:
        """
        Perform comprehensive data quality checks.
        
        Args:
            df: DataFrame to check
            stage: Current processing stage for logging
            
        Returns:
            Tuple[bool, List[str]]: (is_valid, list of issues found)
        """
        issues = []
        prefix = f"[{stage}] " if stage else ""
        
        # Check for NaN values
        nan_cols = df.isna().sum()
        nan_cols = nan_cols[nan_cols > 0]
        if not nan_cols.empty:
            for col, count in nan_cols.items():
                pct = count / len(df)
                if pct > self.max_missing_pct:
                    issues.append(f"{prefix}Column {col} has {pct:.1%} missing values (threshold: {self.max_missing_pct:.1%})")
        
        # Check for constant or near-constant columns
        for col in df.select_dtypes(include=[np.number]).columns:
            unique_ratio = df[col].nunique() / len(df)
            if unique_ratio < self.min_unique_ratio:
                issues.append(f"{prefix}Column {col} has low variance (unique ratio: {unique_ratio:.3f})")
            
            # Check for constant values
            value_counts = df[col].value_counts(normalize=True)
            if value_counts.iloc[0] > self.max_constant_pct:
                issues.append(f"{prefix}Column {col} is nearly constant ({value_counts.iloc[0]:.1%} same value)")
        
        # Check for infinite values
        inf_cols = df.select_dtypes(include=[np.number]).isin([np.inf, -np.inf]).sum()
        inf_cols = inf_cols[inf_cols > 0]
        if not inf_cols.empty:
            for col, count in inf_cols.items():
                issues.append(f"{prefix}Column {col} has {count} infinite values")
        
        # Check data distribution
        if 'y' in df.columns:
            skew = stats.skew(df['y'].dropna())
            kurt = stats.kurtosis(df['y'].dropna())
            if abs(skew) > 3:
                issues.append(f"{prefix}Target variable is highly skewed (skewness: {skew:.2f})")
            if abs(kurt) > 10:
                issues.append(f"{prefix}Target variable has extreme kurtosis (kurtosis: {kurt:.2f})")
        
        return len(issues) == 0, issues

    def validate_data(self, df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """
        Validate the input data for common issues.
        
        Returns:
            Tuple[bool, List[str]]: (is_valid, list of issues found)
        """
        issues = []
        
        # Check for empty DataFrame
        if df.empty:
            issues.append("DataFrame is empty")
            return False, issues
        
        # Check required columns
        required_cols = {'ds', 'y'}
        missing_cols = required_cols - set(df.columns)
        if missing_cols:
            issues.append(f"Missing required columns: {missing_cols}")
        
        # Check data types
        if not pd.api.types.is_datetime64_any_dtype(df['ds']):
            issues.append("Column 'ds' is not datetime type")
        if not pd.api.types.is_numeric_dtype(df['y']):
            issues.append("Column 'y' is not numeric type")
        
        # Check for duplicate dates
        duplicates = df['ds'].duplicated()
        if duplicates.any():
            issues.append(f"Found {duplicates.sum()} duplicate timestamps")
        
        # Check for negative values in target variable (if applicable)
        if self.cleaning_config.get('check_negative_values', True):
            if (df['y'] < 0).any():
                issues.append("Found negative values in target variable")
        
        # Check for large gaps in time series
        if len(df) > 1:
            time_diff = df['ds'].diff()
            median_diff = time_diff.median()
            max_diff = time_diff.max()
            if max_diff > median_diff * 5:  # Arbitrary threshold
                issues.append(f"Found large gap in time series: {max_diff}")
        
        # Additional time series specific checks
        if len(df) > 1:
            # Check for non-monotonic timestamps
            if not df['ds'].is_monotonic_increasing:
                issues.append("Timestamps are not monotonically increasing")
            
            # Check for future timestamps - handle timezone-aware comparison
            current_time = pd.Timestamp.now(tz=df['ds'].dt.tz)
            if df['ds'].max() > current_time:
                issues.append("Found timestamps in the future")
            
            # Check for reasonable date range
            date_range = (df['ds'].max() - df['ds'].min()).days
            if date_range < 7:  # Arbitrary minimum
                issues.append(f"Date range too short: {date_range} days")
        
        return len(issues) == 0, issues

    def plot_diagnostics(self, df: pd.DataFrame, stage: str = "") -> None:
        """Generate diagnostic plots for data quality assessment."""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle(f'Data Diagnostics - {stage}')
            
            # Time series plot
            axes[0, 0].plot(df['ds'], df['y'])
            axes[0, 0].set_title('Time Series Plot')
            axes[0, 0].tick_params(axis='x', rotation=45)
            
            # Distribution plot
            sns.histplot(data=df, x='y', ax=axes[0, 1])
            axes[0, 1].set_title('Distribution Plot')
            
            # QQ plot
            stats.probplot(df['y'].dropna(), dist="norm", plot=axes[1, 0])
            axes[1, 0].set_title('Q-Q Plot')
            
            # Box plot
            sns.boxplot(data=df, y='y', ax=axes[1, 1])
            axes[1, 1].set_title('Box Plot')
            
            plt.tight_layout()
            
            # Save plot
            output_path = os.path.join(self.data_dir, f"diagnostics_{stage}.png")
            plt.savefig(output_path)
            plt.close()
            
            logger.info(f"Diagnostic plots saved to {output_path}")
            
        except Exception as e:
            logger.warning(f"Error generating diagnostic plots: {str(e)}")

    def handle_missing_drop(self, df: pd.DataFrame) -> pd.DataFrame:
        """Drop rows with missing values."""
        before_count = len(df)
        df = df.dropna()
        dropped_count = before_count - len(df)
        if dropped_count > 0:
            logger.info(f"Dropped {dropped_count} rows with missing values")
        return df

    def handle_missing_impute(self, df: pd.DataFrame) -> pd.DataFrame:
        """Impute missing values using various strategies."""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        # Track imputation statistics
        imputation_stats = {}
        
        # Use different imputation strategies for different columns
        for col in numeric_cols:
            missing_count = df[col].isnull().sum()
            if missing_count > 0:
                if col == 'y':
                    # For target variable, use more sophisticated imputation
                    imputer = SimpleImputer(strategy='mean')
                    df[col] = imputer.fit_transform(df[[col]])
                    imputation_stats[col] = {
                        'method': 'mean',
                        'count': missing_count,
                        'fill_value': imputer.statistics_[0]
                    }
                else:
                    # For other numeric columns, use median
                    imputer = SimpleImputer(strategy='median')
                    df[col] = imputer.fit_transform(df[[col]])
                    imputation_stats[col] = {
                        'method': 'median',
                        'count': missing_count,
                        'fill_value': imputer.statistics_[0]
                    }
        
        # Log imputation statistics
        for col, stats in imputation_stats.items():
            logger.info(f"Imputed {stats['count']} missing values in {col} using {stats['method']} ({stats['fill_value']:.2f})")
        
        return df

    def handle_missing_interpolate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values using interpolation."""
        before_missing = df.isnull().sum()
        df = df.interpolate(method='time', limit_direction='both')
        after_missing = df.isnull().sum()
        
        # Log interpolation results
        for col in df.columns:
            filled = before_missing[col] - after_missing[col]
            if filled > 0:
                logger.info(f"Interpolated {filled} missing values in {col}")
        
        return df

    def handle_missing_forward(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values using forward fill."""
        before_missing = df.isnull().sum()
        df = df.fillna(method='ffill')
        after_missing = df.isnull().sum()
        
        # Log fill results
        for col in df.columns:
            filled = before_missing[col] - after_missing[col]
            if filled > 0:
                logger.info(f"Forward filled {filled} missing values in {col}")
        
        return df

    def handle_missing_backward(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values using backward fill."""
        before_missing = df.isnull().sum()
        df = df.fillna(method='bfill')
        after_missing = df.isnull().sum()
        
        # Log fill results
        for col in df.columns:
            filled = before_missing[col] - after_missing[col]
            if filled > 0:
                logger.info(f"Backward filled {filled} missing values in {col}")
        
        return df

    def detect_outliers_zscore(self, series: pd.Series) -> pd.Series:
        """Detect outliers using Z-score method."""
        z_scores = np.abs(stats.zscore(series))
        return z_scores > self.zscore_threshold

    def detect_outliers_iqr(self, series: pd.Series) -> pd.Series:
        """Detect outliers using IQR method."""
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - self.iqr_multiplier * IQR
        upper_bound = Q3 + self.iqr_multiplier * IQR
        return (series < lower_bound) | (series > upper_bound)

    def detect_outliers_isolation_forest(self, series: pd.Series) -> pd.Series:
        """Detect outliers using Isolation Forest."""
        from sklearn.ensemble import IsolationForest
        iso_forest = IsolationForest(contamination=self.contamination, random_state=42)
        yhat = iso_forest.fit_predict(series.values.reshape(-1, 1))
        return yhat == -1

    def detect_outliers_elliptic_envelope(self, series: pd.Series) -> pd.Series:
        """Detect outliers using Elliptic Envelope."""
        envelope = EllipticEnvelope(contamination=self.contamination, random_state=42)
        yhat = envelope.fit_predict(series.values.reshape(-1, 1))
        return yhat == -1

    def handle_outliers(
        self,
        df: pd.DataFrame,
        outlier_method: str = 'zscore'
    ) -> pd.DataFrame:
        """Handle outliers in the target variable."""
        try:
            # Get outlier detection function
            detect_func = self.outlier_methods.get(outlier_method)
            if not detect_func:
                raise ValueError(f"Unknown outlier method: {outlier_method}")
            
            # Detect outliers
            outliers = detect_func(df['y'])
            outlier_count = outliers.sum()
            
            if outlier_count > 0:
                logger.info(f"Found {outlier_count} outliers using {outlier_method} method")
                
                # Log outlier statistics
                outlier_values = df.loc[outliers, 'y']
                logger.info(f"Outlier statistics: min={outlier_values.min():.2f}, "
                          f"max={outlier_values.max():.2f}, "
                          f"mean={outlier_values.mean():.2f}")
                
                # Handle outliers based on configuration
                handling_strategy = self.cleaning_config.get('outlier_handling', 'clip')
                
                if handling_strategy == 'remove':
                    df = df[~outliers].copy()
                    logger.info(f"Removed {outlier_count} outliers")
                elif handling_strategy == 'clip':
                    # Clip to percentile values
                    lower = df['y'].quantile(0.01)
                    upper = df['y'].quantile(0.99)
                    df.loc[outliers, 'y'] = df.loc[outliers, 'y'].clip(lower, upper)
                    logger.info(f"Clipped {outlier_count} outliers to [{lower:.2f}, {upper:.2f}]")
                elif handling_strategy == 'winsorize':
                    # Winsorize the data
                    df['y'] = stats.mstats.winsorize(df['y'], limits=[0.01, 0.01])
                    logger.info(f"Winsorized {outlier_count} outliers")
            
            return df
            
        except Exception as e:
            logger.error(f"Error handling outliers: {str(e)}")
            raise

    def process(
        self,
        df: pd.DataFrame,
        missing_method: str = 'impute',
        outlier_method: str = 'zscore'
    ) -> pd.DataFrame:
        """
        Main processing function that orchestrates data cleaning.
        
        Args:
            df (pd.DataFrame): Input DataFrame to clean
            missing_method (str): Method to handle missing values
            outlier_method (str): Method to handle outliers
            
        Returns:
            pd.DataFrame: Cleaned DataFrame
        """
        try:
            # Validate input data
            is_valid, issues = self.validate_data(df)
            if not is_valid:
                for issue in issues:
                    logger.warning(f"Data validation issue: {issue}")
            
            # Generate diagnostic plots for raw data
            self.plot_diagnostics(df, "raw")
            
            # Check data quality
            is_quality_ok, quality_issues = self.check_data_quality(df, "raw")
            if not is_quality_ok:
                for issue in quality_issues:
                    logger.warning(f"Data quality issue: {issue}")
            
            # Handle missing values
            if missing_method not in self.missing_methods:
                raise ValueError(f"Invalid missing value method: {missing_method}")
            df_clean = self.missing_methods[missing_method](df)
            
            # Handle outliers
            if outlier_method != 'none':
                if outlier_method not in self.outlier_methods:
                    raise ValueError(f"Invalid outlier method: {outlier_method}")
                df_clean = self.handle_outliers(df_clean, outlier_method)
            
            # Generate diagnostic plots for cleaned data
            self.plot_diagnostics(df_clean, "cleaned")
            
            # Final data quality check
            is_quality_ok, quality_issues = self.check_data_quality(df_clean, "cleaned")
            if not is_quality_ok:
                for issue in quality_issues:
                    logger.warning(f"Post-cleaning data quality issue: {issue}")
            
            # Save cleaned data
            output_path = os.path.join(self.data_dir, "cleaned_data.csv")
            df_clean.to_csv(output_path, index=False)
            logger.info(f"Cleaned data saved to {output_path}")
            
            return df_clean
            
        except Exception as e:
            logger.error(f"Error in data cleaning process: {str(e)}")
            raise

def initialize() -> DataCleaning:
    """Initialize the data cleaning module with configuration."""
    config = config_loader.load_config()
    return DataCleaning(config)

# Convenience function for direct usage
def process(**kwargs) -> pd.DataFrame:
    """Convenience function to process data cleaning."""
    cleaning = initialize()
    return cleaning.process(**kwargs) 