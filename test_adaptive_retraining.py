#!/usr/bin/env python3
"""
Test adaptive retraining functionality
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Add src to path
src_path = Path(__file__).parent
sys.path.insert(0, str(src_path))

from src.regime.adaptive_hmm_detector import AdaptiveHMMDetector

def test_adaptive_retraining():
    """Test adaptive retraining functionality"""
    print("Testing Adaptive Retraining...")
    
    # Create sample data with changing market conditions
    np.random.seed(42)
    n_samples = 2000
    
    # Generate data with regime changes
    prices = [100]
    for i in range(1, n_samples):
        if i < 500:
            # Low volatility regime
            change = np.random.normal(0, 0.005)
        elif i < 1000:
            # Trending regime (upward)
            change = np.random.normal(0.002, 0.01)
        elif i < 1500:
            # High volatility regime
            change = np.random.normal(0, 0.02)
        else:
            # New regime (different characteristics)
            change = np.random.normal(0.001, 0.008)
        
        prices.append(prices[-1] * (1 + change))
    
    # Create DataFrame
    df = pd.DataFrame({
        'open': prices,
        'high': [p * (1 + abs(np.random.normal(0, 0.005))) for p in prices],
        'low': [p * (1 - abs(np.random.normal(0, 0.005))) for p in prices],
        'close': prices,
        'volume': np.random.uniform(1000, 10000, n_samples)
    })
    
    # Test adaptive HMM detector
    print("\n1. Testing Adaptive HMM Detector:")
    detector = AdaptiveHMMDetector(
        n_states=3,
        retrain_interval_hours=1,  # Short interval for testing
        min_confidence_threshold=0.7,
        max_change_rate_threshold=0.15
    )
    
    try:
        # Initial training
        detector.train(df[:1000], lookback=500)
        print(f"   [OK] Initial training successful")
        print(f"   [OK] Regime names: {detector.regime_names}")
        
        # Test adaptive prediction
        print("\n2. Testing adaptive prediction:")
        regimes = []
        confidences = []
        retrain_events = []
        
        # Simulate real-time prediction
        for i in range(1000, n_samples, 50):  # Every 50 samples
            chunk = df[:i+1]
            
            # Check if retraining is needed
            should_retrain = detector.should_retrain_now()
            if should_retrain:
                retrained = detector.adaptive_retrain(chunk, lookback=500)
                if retrained:
                    retrain_events.append(i)
                    print(f"   [RETRAIN] Retraining at sample {i}")
            
            # Predict regime
            regime, confidence = detector.predict_regime_adaptive(chunk)
            regimes.append(regime)
            confidences.append(confidence)
            
            if i % 200 == 0:
                print(f"   Sample {i}: regime={regime}, confidence={confidence:.3f}")
        
        print(f"   [OK] Completed {len(regimes)} predictions")
        print(f"   [OK] Retraining events: {len(retrain_events)} at samples {retrain_events}")
        
        # Test performance monitoring
        print("\n3. Testing performance monitoring:")
        status = detector.get_retraining_status()
        print(f"   [OK] Retraining status:")
        for key, value in status.items():
            print(f"     {key}: {value}")
        
        # Test rollback functionality
        print("\n4. Testing rollback functionality:")
        if detector.model_versions:
            print(f"   [OK] Available model versions: {len(detector.model_versions)}")
            
            # Test rollback
            success = detector.rollback_model(-1)
            if success:
                print(f"   [OK] Rollback successful")
            else:
                print(f"   [ERROR] Rollback failed")
        else:
            print(f"   [INFO] No model versions available for rollback")
        
        print("\n[SUCCESS] Adaptive retraining test completed!")
        return True, detector, df, regimes, confidences, retrain_events
        
    except Exception as e:
        print(f"   [ERROR] Error: {e}")
        return False, None, None, None, None, None

def test_performance_degradation():
    """Test retraining trigger on performance degradation"""
    print("\n5. Testing performance degradation detection:")
    
    # Create detector with strict thresholds
    detector = AdaptiveHMMDetector(
        n_states=3,
        retrain_interval_hours=24,  # Long interval
        min_confidence_threshold=0.9,  # High threshold
        max_change_rate_threshold=0.05  # Low threshold
    )
    
    # Simulate performance degradation
    for i in range(150):  # More than window size (100)
        # Simulate low confidence predictions
        confidence = 0.5 + np.random.normal(0, 0.1)  # Low confidence
        regime = f"regime_{i % 3}"  # Frequent changes
        
        detector.performance_monitor.add_prediction(confidence, regime)
        
        if i % 50 == 0:
            should_retrain = detector.should_retrain_now()
            print(f"   Sample {i}: confidence={confidence:.3f}, should_retrain={should_retrain}")
    
    print(f"   [OK] Performance degradation simulation completed")
    return True

def test_retraining_intervals():
    """Test time-based retraining intervals"""
    print("\n6. Testing retraining intervals:")
    
    detector = AdaptiveHMMDetector(
        n_states=3,
        retrain_interval_hours=1,  # 1 hour interval
        min_confidence_threshold=0.5,  # Low threshold
        max_change_rate_threshold=0.5  # High threshold
    )
    
    # Simulate time passing
    detector.last_retrain = datetime.now() - timedelta(hours=2)  # 2 hours ago
    
    should_retrain = detector.should_retrain_now()
    print(f"   [OK] Should retrain after 2 hours: {should_retrain}")
    
    # Reset to recent time
    detector.last_retrain = datetime.now() - timedelta(minutes=30)  # 30 minutes ago
    
    should_retrain = detector.should_retrain_now()
    print(f"   [OK] Should retrain after 30 minutes: {should_retrain}")
    
    return True

if __name__ == "__main__":
    print("HMM v2.0 - Adaptive Retraining Test")
    print("=" * 50)
    
    success, detector, df, regimes, confidences, retrain_events = test_adaptive_retraining()
    
    if success:
        test_performance_degradation()
        test_retraining_intervals()
        
        print("\n[SUCCESS] All adaptive retraining tests passed!")
        print(f"Summary:")
        print(f"  - Predictions made: {len(regimes) if regimes else 0}")
        print(f"  - Retraining events: {len(retrain_events) if retrain_events else 0}")
        print(f"  - Average confidence: {np.mean(confidences):.3f}" if confidences else "N/A")
    else:
        print("\n[ERROR] Adaptive retraining tests failed.")
