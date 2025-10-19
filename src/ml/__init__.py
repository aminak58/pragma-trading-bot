"""
Machine Learning Module for Pragma Trading Bot

This module provides FreqAI integration and ML model management including:
- FreqAI model training and prediction
- Auto-retraining mechanisms
- Model performance monitoring
- Feature engineering for ML models
"""

from .freqai_helper import FreqAIHelper
from .model_manager import ModelManager
from .feature_engineering import FeatureEngineer

__all__ = [
    'FreqAIHelper',
    'ModelManager', 
    'FeatureEngineer'
]
