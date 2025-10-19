"""
FreqAI Integration Helper

Provides a simplified interface for FreqAI integration with:
- Model training and prediction
- Auto-retraining mechanisms
- Performance monitoring
- Regime-specific model management
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import logging
from pathlib import Path
import joblib
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class FreqAIHelper:
    """
    Helper class for FreqAI integration.
    
    Provides a simplified interface for working with FreqAI models
    and managing auto-retraining processes.
    """
    
    def __init__(
        self,
        model_path: str = "user_data/models",
        retrain_frequency_days: int = 15,
        max_models_per_regime: int = 3,
        min_training_samples: int = 1000
    ):
        """
        Initialize FreqAI helper.
        
        Args:
            model_path: Path to store trained models
            retrain_frequency_days: Days between retraining
            max_models_per_regime: Maximum models to keep per regime
            min_training_samples: Minimum samples required for training
        """
        self.model_path = Path(model_path)
        self.retrain_frequency_days = retrain_frequency_days
        self.max_models_per_regime = max_models_per_regime
        self.min_training_samples = min_training_samples
        
        # Create model directory
        self.model_path.mkdir(parents=True, exist_ok=True)
        
        # Model tracking
        self.models = {}
        self.last_training = {}
        self.model_performance = {}
        
        # Training configuration
        self.training_config = {
            'model_type': 'XGBoost',
            'hyperopt_epochs': 100,
            'cv_folds': 3,
            'test_size': 0.2,
            'random_state': 42
        }
        
        logger.info(f"FreqAIHelper initialized with model path: {self.model_path}")
    
    def prepare_features(
        self,
        dataframe: pd.DataFrame,
        regime: str = 'unknown',
        include_regime_features: bool = True
    ) -> pd.DataFrame:
        """
        Prepare features for ML model training/prediction.
        
        Args:
            dataframe: OHLCV dataframe with technical indicators
            regime: Current market regime
            include_regime_features: Whether to include regime-specific features
            
        Returns:
            Feature dataframe ready for ML
        """
        df = dataframe.copy()
        
        # Basic price features
        df['returns_1'] = df['close'].pct_change(1)
        df['returns_5'] = df['close'].pct_change(5)
        df['returns_20'] = df['close'].pct_change(20)
        
        # Volatility features
        df['volatility_10'] = df['returns_1'].rolling(10).std()
        df['volatility_30'] = df['returns_1'].rolling(30).std()
        df['volatility_ratio'] = df['volatility_10'] / df['volatility_30']
        
        # Volume features
        df['volume_sma'] = df['volume'].rolling(20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma']
        df['volume_price_trend'] = df['volume_ratio'] * df['returns_1']
        
        # Technical indicators (if available)
        if 'rsi' in df.columns:
            df['rsi_normalized'] = (df['rsi'] - 50) / 50
        if 'adx' in df.columns:
            df['adx_normalized'] = df['adx'] / 100
        if 'bb_width' in df.columns:
            df['bb_width_normalized'] = df['bb_width']
        
        # Regime-specific features
        if include_regime_features:
            df['regime_trending'] = int(regime == 'trending')
            df['regime_low_vol'] = int(regime == 'low_volatility')
            df['regime_high_vol'] = int(regime == 'high_volatility')
        
        # Lagged features
        for lag in [1, 2, 3, 5]:
            df[f'returns_lag_{lag}'] = df['returns_1'].shift(lag)
            df[f'volume_ratio_lag_{lag}'] = df['volume_ratio'].shift(lag)
        
        # Target variables (for training)
        df['target_returns_1'] = df['returns_1'].shift(-1)
        df['target_returns_5'] = df['returns_5'].shift(-5)
        df['target_direction'] = (df['target_returns_1'] > 0).astype(int)
        
        # Remove rows with NaN values
        df = df.dropna()
        
        return df
    
    def create_training_data(
        self,
        dataframe: pd.DataFrame,
        regime: str,
        target_column: str = 'target_direction'
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Create training data for ML model.
        
        Args:
            dataframe: Prepared feature dataframe
            regime: Market regime
            target_column: Target column name
            
        Returns:
            Tuple of (features, target)
        """
        # Select feature columns (exclude target and OHLCV)
        feature_columns = [
            col for col in dataframe.columns
            if col not in [
                'open', 'high', 'low', 'close', 'volume', 'date',
                'target_returns_1', 'target_returns_5', 'target_direction'
            ]
        ]
        
        X = dataframe[feature_columns].copy()
        y = dataframe[target_column].copy()
        
        # Remove any remaining NaN values
        mask = ~(X.isna().any(axis=1) | y.isna())
        X = X[mask]
        y = y[mask]
        
        logger.info(f"Created training data: {len(X)} samples, {len(feature_columns)} features")
        
        return X, y
    
    def train_model(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        regime: str,
        model_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Train a model for specific regime.
        
        Args:
            X: Feature matrix
            y: Target vector
            regime: Market regime
            model_name: Optional model name
            
        Returns:
            Training results dictionary
        """
        if len(X) < self.min_training_samples:
            raise ValueError(f"Insufficient training data: {len(X)} < {self.min_training_samples}")
        
        # Generate model name if not provided
        if model_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_name = f"{regime}_{timestamp}"
        
        # Import here to avoid dependency issues
        try:
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.model_selection import train_test_split, cross_val_score
            from sklearn.metrics import accuracy_score, classification_report
            from xgboost import XGBClassifier
        except ImportError as e:
            logger.error(f"Required ML libraries not available: {e}")
            raise
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.training_config['test_size'],
            random_state=self.training_config['random_state'],
            stratify=y
        )
        
        # Train model based on configuration
        if self.training_config['model_type'] == 'XGBoost':
            model = XGBClassifier(
                random_state=self.training_config['random_state'],
                eval_metric='logloss'
            )
        else:
            model = RandomForestClassifier(
                random_state=self.training_config['random_state'],
                n_estimators=100
            )
        
        # Train model
        model.fit(X_train, y_train)
        
        # Evaluate model
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        # Cross-validation score
        cv_scores = cross_val_score(
            model, X_train, y_train,
            cv=self.training_config['cv_folds'],
            scoring='accuracy'
        )
        
        # Save model
        model_path = self.model_path / f"{model_name}.joblib"
        joblib.dump(model, model_path)
        
        # Update tracking
        self.models[regime] = {
            'model': model,
            'model_path': model_path,
            'model_name': model_name,
            'features': list(X.columns),
            'trained_at': datetime.now(),
            'accuracy': accuracy,
            'cv_score': cv_scores.mean(),
            'cv_std': cv_scores.std()
        }
        
        self.last_training[regime] = datetime.now()
        
        # Performance tracking
        self.model_performance[model_name] = {
            'regime': regime,
            'accuracy': accuracy,
            'cv_score': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'training_samples': len(X_train),
            'test_samples': len(X_test)
        }
        
        logger.info(
            f"Trained {self.training_config['model_type']} model for {regime}: "
            f"accuracy={accuracy:.3f}, cv_score={cv_scores.mean():.3f}±{cv_scores.std():.3f}"
        )
        
        return {
            'model_name': model_name,
            'regime': regime,
            'accuracy': accuracy,
            'cv_score': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'training_samples': len(X_train),
            'test_samples': len(X_test),
            'model_path': str(model_path)
        }
    
    def predict(
        self,
        X: pd.DataFrame,
        regime: str,
        return_probability: bool = True
    ) -> np.ndarray:
        """
        Make predictions using trained model.
        
        Args:
            X: Feature matrix
            regime: Market regime
            return_probability: Whether to return probabilities
            
        Returns:
            Predictions or probabilities
        """
        if regime not in self.models:
            logger.warning(f"No model found for regime: {regime}")
            return np.zeros(len(X))
        
        model_info = self.models[regime]
        model = model_info['model']
        
        # Ensure feature alignment
        feature_columns = model_info['features']
        X_aligned = X[feature_columns].copy()
        
        # Handle missing features
        missing_features = set(feature_columns) - set(X.columns)
        if missing_features:
            logger.warning(f"Missing features: {missing_features}")
            for feature in missing_features:
                X_aligned[feature] = 0
        
        # Fill NaN values
        X_aligned = X_aligned.fillna(0)
        
        if return_probability:
            if hasattr(model, 'predict_proba'):
                return model.predict_proba(X_aligned)[:, 1]  # Probability of positive class
            else:
                return model.predict(X_aligned)
        else:
            return model.predict(X_aligned)
    
    def should_retrain(self, regime: str) -> bool:
        """
        Check if model should be retrained.
        
        Args:
            regime: Market regime
            
        Returns:
            True if retraining is needed
        """
        if regime not in self.last_training:
            return True
        
        days_since_training = (datetime.now() - self.last_training[regime]).days
        return days_since_training >= self.retrain_frequency_days
    
    def get_model_info(self, regime: str) -> Optional[Dict[str, Any]]:
        """
        Get information about trained model.
        
        Args:
            regime: Market regime
            
        Returns:
            Model information dictionary or None
        """
        return self.models.get(regime)
    
    def get_all_models(self) -> Dict[str, Any]:
        """Get information about all trained models."""
        return {
            regime: {
                'model_name': info['model_name'],
                'trained_at': info['trained_at'],
                'accuracy': info['accuracy'],
                'cv_score': info['cv_score'],
                'cv_std': info['cv_std']
            }
            for regime, info in self.models.items()
        }
    
    def cleanup_old_models(self, regime: str) -> None:
        """
        Clean up old models for a regime.
        
        Args:
            regime: Market regime
        """
        if regime not in self.models:
            return
        
        # Get all model files for this regime
        regime_models = list(self.model_path.glob(f"{regime}_*.joblib"))
        
        if len(regime_models) > self.max_models_per_regime:
            # Sort by modification time (oldest first)
            regime_models.sort(key=lambda x: x.stat().st_mtime)
            
            # Remove oldest models
            models_to_remove = regime_models[:-self.max_models_per_regime]
            for model_path in models_to_remove:
                model_path.unlink()
                logger.info(f"Removed old model: {model_path.name}")
    
    def load_model(self, regime: str, model_name: str) -> bool:
        """
        Load a specific model from disk.
        
        Args:
            regime: Market regime
            model_name: Model name
            
        Returns:
            True if loaded successfully
        """
        model_path = self.model_path / f"{model_name}.joblib"
        
        if not model_path.exists():
            logger.error(f"Model file not found: {model_path}")
            return False
        
        try:
            model = joblib.load(model_path)
            
            # Get model info from performance tracking
            if model_name in self.model_performance:
                perf = self.model_performance[model_name]
                
                self.models[regime] = {
                    'model': model,
                    'model_path': model_path,
                    'model_name': model_name,
                    'features': [],  # Will be updated when used
                    'trained_at': datetime.fromtimestamp(model_path.stat().st_mtime),
                    'accuracy': perf.get('accuracy', 0.0),
                    'cv_score': perf.get('cv_score', 0.0),
                    'cv_std': perf.get('cv_std', 0.0)
                }
                
                logger.info(f"Loaded model: {model_name} for regime: {regime}")
                return True
            else:
                logger.warning(f"No performance data found for model: {model_name}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {e}")
            return False
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """
        Get performance summary of all models.
        
        Returns:
            Performance summary dictionary
        """
        if not self.model_performance:
            return {'total_models': 0, 'regimes': []}
        
        regimes = list(set(perf['regime'] for perf in self.model_performance.values()))
        
        summary = {
            'total_models': len(self.model_performance),
            'regimes': regimes,
            'regime_models': {}
        }
        
        for regime in regimes:
            regime_models = [
                perf for perf in self.model_performance.values()
                if perf['regime'] == regime
            ]
            
            if regime_models:
                summary['regime_models'][regime] = {
                    'count': len(regime_models),
                    'avg_accuracy': np.mean([m['accuracy'] for m in regime_models]),
                    'avg_cv_score': np.mean([m['cv_score'] for m in regime_models]),
                    'latest_model': max(regime_models, key=lambda x: x.get('trained_at', datetime.min))
                }
        
        return summary
    
    def __repr__(self) -> str:
        """String representation."""
        return f"FreqAIHelper(models={len(self.models)}, path={self.model_path})"
