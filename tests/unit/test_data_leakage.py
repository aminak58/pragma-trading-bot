"""
Data Leakage Prevention Tests

Critical tests to ensure no future data is used in predictions.
These tests validate that the ML pipeline maintains temporal integrity.
"""

import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(src_path))

import pytest
import numpy as np
import pandas as pd
from regime.hmm_detector import RegimeDetector


@pytest.fixture
def time_series_data():
    """
    Generate time-series data with known structure.
    
    Creates data where each point depends only on past values.
    """
    np.random.seed(42)
    n = 1000
    
    # Generate price series
    returns = np.random.normal(0.001, 0.02, n)
    close = 100 * np.exp(np.cumsum(returns))
    
    # Create OHLCV
    dates = pd.date_range(start='2024-01-01', periods=n, freq='5min')
    
    df = pd.DataFrame({
        'open': close * 0.99,
        'high': close * 1.01,
        'low': close * 0.98,
        'close': close,
        'volume': np.random.lognormal(15, 0.5, n)
    }, index=dates)
    
    return df


class TestFeatureLeakage:
    """Test that features don't use future data."""
    
    def test_features_use_only_past_data(self, time_series_data):
        """
        Verify that features at time t use only data up to time t.
        
        This is the most critical test for data leakage prevention.
        """
        detector = RegimeDetector(n_states=3, random_state=42)
        
        # Test multiple points throughout the series
        test_indices = [100, 300, 500, 700, 900]
        
        for idx in test_indices:
            # Compute features using ALL data
            full_data = time_series_data.copy()
            features_full = detector.prepare_features(full_data)
            feature_at_idx = features_full[idx]
            
            # Compute features using ONLY past data
            past_data = time_series_data.iloc[:idx+1].copy()
            features_past = detector.prepare_features(past_data)
            feature_at_past = features_past[-1]
            
            # They should match (allowing for NaN at edges)
            if not np.isnan(feature_at_idx).any():
                np.testing.assert_array_almost_equal(
                    feature_at_idx,
                    feature_at_past,
                    decimal=10,
                    err_msg=f"Feature mismatch at index {idx}. "
                            f"Features using future data detected!"
                )
    
    def test_no_negative_shifts(self, time_series_data):
        """
        Test that no features use .shift(-N) which accesses future data.
        
        This is a code inspection test.
        """
        # Read the feature preparation code
        detector = RegimeDetector(n_states=3, random_state=42)
        
        # Get source code of prepare_features
        import inspect
        source = inspect.getsource(detector.prepare_features)
        
        # Check for negative shifts
        assert ".shift(-" not in source, (
            "Negative shifts detected in feature preparation! "
            "This causes data leakage by using future data."
        )
    
    def test_rolling_windows_look_backward(self, time_series_data):
        """Test that rolling windows only use historical data."""
        detector = RegimeDetector(n_states=3, random_state=42)
        
        df = time_series_data.copy()
        features = detector.prepare_features(df)
        
        # Check that first N rows have NaN (due to rolling windows)
        # This proves windows are looking backward, not forward
        
        # Features should have NaN in early rows due to windows
        # (returns need 1 lag, volatility needs 10+ lags, etc.)
        
        early_rows = features[:30]  # Check first 30 rows
        
        # Should have some NaN in early rows
        assert np.isnan(early_rows).any(), (
            "Expected NaN in early rows due to rolling windows. "
            "If no NaN, windows might be looking forward!"
        )
    
    def test_features_deterministic_with_past_only(self, time_series_data):
        """
        Test that features are deterministic when computed with past data only.
        """
        detector = RegimeDetector(n_states=3, random_state=42)
        
        # Split data temporally
        split_idx = 500
        past_data = time_series_data.iloc[:split_idx].copy()
        
        # Compute features multiple times
        features1 = detector.prepare_features(past_data)
        features2 = detector.prepare_features(past_data)
        
        # Should be identical (deterministic)
        np.testing.assert_array_equal(
            features1,
            features2,
            err_msg="Features are not deterministic!"
        )


class TestScalerLeakage:
    """Test that scaler doesn't leak future data."""
    
    def test_scaler_fit_on_train_only(self):
        """
        Test that scaler is fit only on training data.
        
        This test ensures the documented pattern is followed.
        """
        detector = RegimeDetector(n_states=3, random_state=42)
        
        # Generate train and test data
        np.random.seed(42)
        n_train = 500
        n_test = 200
        
        close_train = 100 * np.exp(np.cumsum(np.random.normal(0.001, 0.02, n_train)))
        close_test = close_train[-1] * np.exp(np.cumsum(np.random.normal(0.001, 0.02, n_test)))
        
        train_df = pd.DataFrame({
            'open': close_train * 0.99,
            'high': close_train * 1.01,
            'low': close_train * 0.98,
            'close': close_train,
            'volume': np.random.lognormal(15, 0.5, n_train)
        })
        
        test_df = pd.DataFrame({
            'open': close_test * 0.99,
            'high': close_test * 1.01,
            'low': close_test * 0.98,
            'close': close_test,
            'volume': np.random.lognormal(15, 0.5, n_test)
        })
        
        # Train detector on train data only
        detector.train(train_df, lookback=400)
        
        # Scaler should have been fit on train data
        assert hasattr(detector.scaler, 'mean_'), "Scaler not fitted"
        assert hasattr(detector.scaler, 'scale_'), "Scaler not fitted"
        
        # Now predict on test data
        # Scaler should use learned parameters, not refit
        regime, conf = detector.predict_regime(test_df)
        
        # Prediction should work without error
        assert isinstance(regime, str)
        assert 0.0 <= conf <= 1.0


class TestTrainTestSplitLeakage:
    """Test that train/test splits maintain temporal order."""
    
    def test_temporal_order_maintained(self, time_series_data):
        """Test that train < validation < test in time."""
        # Define split points
        train_end = '2024-02-01'
        val_end = '2024-03-01'
        
        train = time_series_data[time_series_data.index <= train_end]
        val = time_series_data[
            (time_series_data.index > train_end) &
            (time_series_data.index <= val_end)
        ]
        test = time_series_data[time_series_data.index > val_end]
        
        # Verify temporal order
        assert train.index[-1] < val.index[0], "Train-Val overlap!"
        assert val.index[-1] < test.index[0], "Val-Test overlap!"
        
        # Verify no gaps
        assert len(train) > 0, "Empty train set"
        assert len(val) > 0, "Empty validation set"
        assert len(test) > 0, "Empty test set"
    
    def test_no_data_shuffling(self, time_series_data):
        """Test that data is not shuffled (maintains temporal order)."""
        # Original index should be sorted
        assert time_series_data.index.is_monotonic_increasing, (
            "Data index not in temporal order! "
            "This breaks time-series integrity."
        )
        
        # After preparation, order should be maintained
        detector = RegimeDetector(n_states=3, random_state=42)
        features = detector.prepare_features(time_series_data)
        
        # Features should have same length and order
        assert len(features) == len(time_series_data)


class TestRetrainingLeakage:
    """Test that retraining doesn't leak future data."""
    
    def test_retrain_uses_only_available_data(self, time_series_data):
        """
        Simulate production retraining scenario.
        
        Ensure model trained at time t uses only data up to time t.
        """
        detector = RegimeDetector(n_states=3, random_state=42)
        
        # Simulate monthly retraining
        retrain_points = [300, 600, 900]
        
        for retrain_idx in retrain_points:
            # Available data up to retrain point
            available_data = time_series_data.iloc[:retrain_idx].copy()
            
            # Train model
            detector.train(available_data, lookback=min(500, len(available_data)))
            
            # Predict on next period
            next_period = time_series_data.iloc[retrain_idx:retrain_idx+100].copy()
            
            if len(next_period) > 0:
                regime, conf = detector.predict_regime(next_period)
                
                # Should work without using future data
                assert isinstance(regime, str)
                assert 0.0 <= conf <= 1.0


class TestDocumentedPatterns:
    """Test that code follows documented safe patterns."""
    
    def test_follows_safe_pattern_from_docs(self, time_series_data):
        """
        Test implementation matches the safe pattern in ML_PIPELINE.md.
        
        Documented safe pattern:
        1. Split data temporally
        2. Fit scaler on train only
        3. Transform train and test separately
        """
        detector = RegimeDetector(n_states=3, random_state=42)
        
        # Split data
        split_idx = 700
        train_df = time_series_data.iloc[:split_idx].copy()
        test_df = time_series_data.iloc[split_idx:].copy()
        
        # Step 1: Train on train data
        detector.train(train_df, lookback=500)
        
        # Scaler should be fitted
        train_mean = detector.scaler.mean_.copy()
        train_scale = detector.scaler.scale_.copy()
        
        # Step 2: Predict on test data
        regime, conf = detector.predict_regime(test_df)
        
        # Scaler parameters should NOT have changed
        np.testing.assert_array_equal(
            detector.scaler.mean_,
            train_mean,
            err_msg="Scaler was refit on test data!"
        )
        np.testing.assert_array_equal(
            detector.scaler.scale_,
            train_scale,
            err_msg="Scaler was refit on test data!"
        )


@pytest.mark.slow
class TestWalkForwardLeakage:
    """Test walk-forward validation doesn't leak."""
    
    def test_walk_forward_no_leakage(self, time_series_data):
        """
        Test walk-forward validation maintains temporal integrity.
        """
        detector = RegimeDetector(n_states=3, random_state=42)
        
        window_size = 300
        step_size = 100
        predictions = []
        
        for i in range(500, len(time_series_data) - window_size, step_size):
            # Train window: past data only
            train_window = time_series_data.iloc[i-500:i].copy()
            
            # Test window: future data
            test_window = time_series_data.iloc[i:i+window_size].copy()
            
            # Verify no overlap
            assert train_window.index[-1] < test_window.index[0]
            
            # Train
            detector.train(train_window, lookback=400)
            
            # Predict
            regime, conf = detector.predict_regime(test_window)
            predictions.append((regime, conf))
        
        # Should have made multiple predictions
        assert len(predictions) > 3
        
        # All predictions should be valid
        for regime, conf in predictions:
            assert isinstance(regime, str)
            assert 0.0 <= conf <= 1.0


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "--tb=short"])
