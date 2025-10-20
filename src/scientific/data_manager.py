"""
Scientific Trading Strategy Framework - Data Manager

This module implements the DataManager class for data preparation,
quality control, and train/test splitting for scientific strategy validation.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple, Optional
import logging
from pathlib import Path
from datetime import datetime, timedelta

class DataManager:
    """
    Data preparation and quality control for scientific strategy validation
    
    This class handles:
    - Data loading and validation
    - Quality checks and preprocessing
    - Train/test temporal splitting
    - Data integrity verification
    """
    
    def __init__(self, data_path: str, min_years: float = 3.0, 
                 train_ratio: float = 0.7):
        """
        Initialize DataManager
        
        Args:
            data_path: Path to data file
            min_years: Minimum years of data required
            train_ratio: Ratio of data to use for training
        """
        self.data_path = Path(data_path)
        self.min_years = min_years
        self.train_ratio = train_ratio
        self.data = None
        self.train_data = None
        self.test_data = None
        self.logger = logging.getLogger(__name__)
        
        # Quality check results
        self.quality_results = {}
    
    def prepare_data(self) -> pd.DataFrame:
        """
        Prepare data for testing
        
        Returns:
            Prepared DataFrame
            
        Raises:
            ValueError: If data doesn't meet minimum requirements
        """
        try:
            # Load data
            self.data = self.load_data()
            
            # Perform quality checks
            self.quality_checks()
            
            # Preprocess data
            self.preprocess_data()
            
            # Split data
            self.split_data()
            
            self.logger.info(f"Data preparation completed. Total records: {len(self.data)}")
            return self.data
            
        except Exception as e:
            self.logger.error(f"Error in data preparation: {e}")
            raise
    
    def load_data(self) -> pd.DataFrame:
        """
        Load data from file
        
        Returns:
            Loaded DataFrame
            
        Raises:
            FileNotFoundError: If data file doesn't exist
            ValueError: If data format is invalid
        """
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data file not found: {self.data_path}")
        
        # Determine file format and load accordingly
        if self.data_path.suffix.lower() == '.csv':
            data = pd.read_csv(self.data_path, index_col=0, parse_dates=True)
        elif self.data_path.suffix.lower() == '.feather':
            data = pd.read_feather(self.data_path)
            data.set_index('date', inplace=True)
        elif self.data_path.suffix.lower() == '.parquet':
            data = pd.read_parquet(self.data_path)
            data.set_index('date', inplace=True)
        else:
            raise ValueError(f"Unsupported file format: {self.data_path.suffix}")
        
        # Validate basic structure
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Ensure index is datetime
        if not isinstance(data.index, pd.DatetimeIndex):
            data.index = pd.to_datetime(data.index)
        
        self.logger.info(f"Data loaded successfully. Shape: {data.shape}")
        return data
    
    def quality_checks(self) -> Dict[str, Any]:
        """
        Perform comprehensive data quality checks
        
        Returns:
            Dictionary of quality check results
        """
        self.logger.info("Performing data quality checks...")
        
        quality_results = {}
        
        # Check completeness
        quality_results['completeness'] = self.check_completeness()
        
        # Check accuracy
        quality_results['accuracy'] = self.check_accuracy()
        
        # Check outliers
        quality_results['outliers'] = self.check_outliers()
        
        # Check survivorship bias
        quality_results['survivorship_bias'] = self.check_survivorship_bias()
        
        # Check look-ahead bias
        quality_results['look_ahead_bias'] = self.check_look_ahead_bias()
        
        # Check temporal consistency
        quality_results['temporal_consistency'] = self.check_temporal_consistency()
        
        # Calculate overall quality score
        quality_results['overall_score'] = self.calculate_quality_score(quality_results)
        
        self.quality_results = quality_results
        self.logger.info(f"Quality checks completed. Overall score: {quality_results['overall_score']:.2f}")
        
        return quality_results
    
    def check_completeness(self) -> Dict[str, Any]:
        """
        Check data completeness
        
        Returns:
            Completeness check results
        """
        total_records = len(self.data)
        missing_values = self.data.isnull().sum()
        missing_ratio = missing_values.sum() / (total_records * len(self.data.columns))
        
        # Check for gaps in time series
        time_gaps = self.data.index.to_series().diff().dropna()
        expected_freq = time_gaps.mode().iloc[0] if len(time_gaps) > 0 else None
        actual_gaps = time_gaps[time_gaps > expected_freq * 2] if expected_freq else pd.Series()
        
        return {
            'total_records': total_records,
            'missing_values': missing_values.to_dict(),
            'missing_ratio': missing_ratio,
            'time_gaps': len(actual_gaps),
            'completeness_score': max(0, 1 - missing_ratio - len(actual_gaps) / total_records)
        }
    
    def check_accuracy(self) -> Dict[str, Any]:
        """
        Check data accuracy
        
        Returns:
            Accuracy check results
        """
        accuracy_issues = []
        
        # Check for negative prices
        price_columns = ['open', 'high', 'low', 'close']
        for col in price_columns:
            if col in self.data.columns:
                negative_prices = (self.data[col] <= 0).sum()
                if negative_prices > 0:
                    accuracy_issues.append(f"Negative prices in {col}: {negative_prices}")
        
        # Check for negative volume
        if 'volume' in self.data.columns:
            negative_volume = (self.data['volume'] < 0).sum()
            if negative_volume > 0:
                accuracy_issues.append(f"Negative volume: {negative_volume}")
        
        # Check OHLC relationships
        if all(col in self.data.columns for col in price_columns):
            invalid_ohlc = (
                (self.data['high'] < self.data['low']) |
                (self.data['high'] < self.data['open']) |
                (self.data['high'] < self.data['close']) |
                (self.data['low'] > self.data['open']) |
                (self.data['low'] > self.data['close'])
            ).sum()
            if invalid_ohlc > 0:
                accuracy_issues.append(f"Invalid OHLC relationships: {invalid_ohlc}")
        
        return {
            'issues': accuracy_issues,
            'accuracy_score': max(0, 1 - len(accuracy_issues) * 0.2)
        }
    
    def check_outliers(self) -> Dict[str, Any]:
        """
        Check for outliers
        
        Returns:
            Outlier check results
        """
        outlier_results = {}
        
        # Check price outliers using IQR method
        price_columns = ['open', 'high', 'low', 'close']
        for col in price_columns:
            if col in self.data.columns:
                Q1 = self.data[col].quantile(0.25)
                Q3 = self.data[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outliers = ((self.data[col] < lower_bound) | (self.data[col] > upper_bound)).sum()
                outlier_results[col] = {
                    'count': outliers,
                    'ratio': outliers / len(self.data),
                    'bounds': (lower_bound, upper_bound)
                }
        
        # Check volume outliers
        if 'volume' in self.data.columns:
            Q1 = self.data['volume'].quantile(0.25)
            Q3 = self.data['volume'].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = ((self.data['volume'] < lower_bound) | (self.data['volume'] > upper_bound)).sum()
            outlier_results['volume'] = {
                'count': outliers,
                'ratio': outliers / len(self.data),
                'bounds': (lower_bound, upper_bound)
            }
        
        # Calculate overall outlier score
        total_outliers = sum(result['count'] for result in outlier_results.values())
        outlier_ratio = total_outliers / (len(self.data) * len(outlier_results))
        
        return {
            'outlier_details': outlier_results,
            'total_outliers': total_outliers,
            'outlier_ratio': outlier_ratio,
            'outlier_score': max(0, 1 - outlier_ratio * 2)
        }
    
    def check_survivorship_bias(self) -> Dict[str, Any]:
        """
        Check for survivorship bias
        
        Returns:
            Survivorship bias check results
        """
        # This is a placeholder for survivorship bias check
        # In practice, this would check if the asset was delisted during the period
        return {
            'survivorship_bias_detected': False,
            'survivorship_score': 1.0,
            'notes': 'Survivorship bias check not implemented'
        }
    
    def check_look_ahead_bias(self) -> Dict[str, Any]:
        """
        Check for look-ahead bias
        
        Returns:
            Look-ahead bias check results
        """
        # Check if data is properly ordered temporally
        is_sorted = self.data.index.is_monotonic_increasing
        
        # Check for future information leakage (placeholder)
        future_leakage_detected = False
        
        return {
            'temporally_sorted': is_sorted,
            'future_leakage_detected': future_leakage_detected,
            'look_ahead_score': 1.0 if is_sorted and not future_leakage_detected else 0.5
        }
    
    def check_temporal_consistency(self) -> Dict[str, Any]:
        """
        Check temporal consistency
        
        Returns:
            Temporal consistency check results
        """
        # Check for consistent time intervals
        time_diffs = self.data.index.to_series().diff().dropna()
        
        if len(time_diffs) > 0:
            most_common_interval = time_diffs.mode().iloc[0]
            inconsistent_intervals = (time_diffs != most_common_interval).sum()
            consistency_ratio = 1 - (inconsistent_intervals / len(time_diffs))
        else:
            consistency_ratio = 1.0
        
        return {
            'most_common_interval': most_common_interval,
            'inconsistent_intervals': inconsistent_intervals,
            'consistency_ratio': consistency_ratio,
            'temporal_score': consistency_ratio
        }
    
    def calculate_quality_score(self, quality_results: Dict[str, Any]) -> float:
        """
        Calculate overall quality score
        
        Args:
            quality_results: Results from quality checks
            
        Returns:
            Overall quality score between 0 and 1
        """
        scores = []
        
        # Weight different quality aspects
        weights = {
            'completeness': 0.3,
            'accuracy': 0.25,
            'outliers': 0.2,
            'survivorship_bias': 0.1,
            'look_ahead_bias': 0.1,
            'temporal_consistency': 0.05
        }
        
        for aspect, weight in weights.items():
            if aspect in quality_results:
                if aspect == 'completeness':
                    score = quality_results[aspect]['completeness_score']
                elif aspect == 'accuracy':
                    score = quality_results[aspect]['accuracy_score']
                elif aspect == 'outliers':
                    score = quality_results[aspect]['outlier_score']
                elif aspect == 'survivorship_bias':
                    score = quality_results[aspect]['survivorship_score']
                elif aspect == 'look_ahead_bias':
                    score = quality_results[aspect]['look_ahead_score']
                elif aspect == 'temporal_consistency':
                    score = quality_results[aspect]['temporal_score']
                else:
                    score = 1.0
                
                scores.append(score * weight)
        
        return sum(scores) if scores else 0.0
    
    def preprocess_data(self) -> None:
        """
        Preprocess data for analysis
        """
        self.logger.info("Preprocessing data...")
        
        # Handle missing values
        self.data = self.data.fillna(method='ffill').fillna(method='bfill')
        
        # Remove extreme outliers (beyond 5 standard deviations)
        numeric_columns = self.data.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            mean = self.data[col].mean()
            std = self.data[col].std()
            self.data[col] = self.data[col].clip(
                lower=mean - 5 * std,
                upper=mean + 5 * std
            )
        
        # Ensure positive values for prices and volume
        price_columns = ['open', 'high', 'low', 'close']
        for col in price_columns:
            if col in self.data.columns:
                self.data[col] = self.data[col].abs()
        
        if 'volume' in self.data.columns:
            self.data['volume'] = self.data['volume'].abs()
        
        self.logger.info("Data preprocessing completed")
    
    def split_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split data into training and testing sets
        
        Returns:
            Tuple of (train_data, test_data)
        """
        if self.data is None:
            raise ValueError("Data must be loaded before splitting")
        
        # Check minimum data requirements
        total_days = (self.data.index.max() - self.data.index.min()).days
        total_years = total_days / 365.25
        
        if total_years < self.min_years:
            raise ValueError(f"Insufficient data: {total_years:.2f} years < {self.min_years} years required")
        
        # Calculate split point
        split_point = int(len(self.data) * self.train_ratio)
        
        # Split data temporally
        self.train_data = self.data.iloc[:split_point].copy()
        self.test_data = self.data.iloc[split_point:].copy()
        
        # Verify split
        train_years = (self.train_data.index.max() - self.train_data.index.min()).days / 365.25
        test_years = (self.test_data.index.max() - self.test_data.index.min()).days / 365.25
        
        self.logger.info(f"Data split completed:")
        self.logger.info(f"  Training: {len(self.train_data)} records ({train_years:.2f} years)")
        self.logger.info(f"  Testing: {len(self.test_data)} records ({test_years:.2f} years)")
        
        return self.train_data, self.test_data
    
    def get_data_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive data summary
        
        Returns:
            Data summary dictionary
        """
        if self.data is None:
            return {}
        
        return {
            'total_records': len(self.data),
            'date_range': (self.data.index.min(), self.data.index.max()),
            'duration_years': (self.data.index.max() - self.data.index.min()).days / 365.25,
            'columns': list(self.data.columns),
            'data_types': self.data.dtypes.to_dict(),
            'missing_values': self.data.isnull().sum().to_dict(),
            'quality_score': self.quality_results.get('overall_score', 0.0),
            'train_records': len(self.train_data) if self.train_data is not None else 0,
            'test_records': len(self.test_data) if self.test_data is not None else 0
        }