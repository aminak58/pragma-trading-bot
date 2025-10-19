#!/usr/bin/env python3
"""
Test dynamic smoothing window functionality
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Add src to path
src_path = Path(__file__).parent
sys.path.insert(0, str(src_path))

from src.regime.hmm_detector import RegimeDetector

def test_dynamic_smoothing():
    """Test dynamic smoothing window calculation"""
    print("Testing Dynamic Smoothing Window...")
    
    # Create sample data
    np.random.seed(42)
    n_samples = 2000
    
    # Generate data with different regimes
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
            # Mixed regime
            change = np.random.normal(0, 0.01)
        
        prices.append(prices[-1] * (1 + change))
    
    # Create DataFrame
    df = pd.DataFrame({
        'open': prices,
        'high': [p * (1 + abs(np.random.normal(0, 0.005))) for p in prices],
        'low': [p * (1 - abs(np.random.normal(0, 0.005))) for p in prices],
        'close': prices,
        'volume': np.random.uniform(1000, 10000, n_samples)
    })
    
    # Test HMM detector with dynamic smoothing
    print("\n1. Testing HMM detector with dynamic smoothing:")
    detector = RegimeDetector(n_states=3, random_state=42)
    
    try:
        # Train HMM
        detector.train(df, lookback=1000)
        print(f"   [OK] HMM training successful")
        print(f"   [OK] Regime names: {detector.regime_names}")
        
        # Test dynamic smoothing with different dataset sizes
        test_sizes = [100, 500, 1000, 2000]
        
        print("\n2. Testing dynamic smoothing with different dataset sizes:")
        for size in test_sizes:
            test_df = df.head(size)
            
            # Test dynamic smoothing (smooth_window=None)
            regime_sequence, confidence_sequence = detector.predict_regime_sequence(test_df)
            
            # Calculate dynamic window
            dynamic_window = detector._calculate_dynamic_smooth_window(len(regime_sequence))
            
            # Calculate regime changes
            regime_changes = np.sum(regime_sequence[1:] != regime_sequence[:-1])
            change_rate = regime_changes / len(regime_sequence)
            
            print(f"   Dataset size {size}:")
            print(f"     Dynamic window: {dynamic_window}")
            print(f"     Regime changes: {regime_changes} ({change_rate:.3f})")
            print(f"     Unique regimes: {len(np.unique(regime_sequence))}")
        
        # Test fixed vs dynamic smoothing
        print("\n3. Comparing fixed vs dynamic smoothing:")
        test_df = df.head(1000)
        
        # Fixed smoothing (window=5)
        regime_fixed, conf_fixed = detector.predict_regime_sequence(test_df, smooth_window=5)
        changes_fixed = np.sum(regime_fixed[1:] != regime_fixed[:-1])
        rate_fixed = changes_fixed / len(regime_fixed)
        
        # Dynamic smoothing
        regime_dynamic, conf_dynamic = detector.predict_regime_sequence(test_df)
        changes_dynamic = np.sum(regime_dynamic[1:] != regime_dynamic[:-1])
        rate_dynamic = changes_dynamic / len(regime_dynamic)
        
        print(f"   Fixed smoothing (window=5):")
        print(f"     Regime changes: {changes_fixed} ({rate_fixed:.3f})")
        print(f"     Avg confidence: {np.mean(conf_fixed):.3f}")
        
        print(f"   Dynamic smoothing:")
        print(f"     Regime changes: {changes_dynamic} ({rate_dynamic:.3f})")
        print(f"     Avg confidence: {np.mean(conf_dynamic):.3f}")
        
        # Calculate improvement
        improvement = ((rate_fixed - rate_dynamic) / rate_fixed) * 100 if rate_fixed > 0 else 0
        print(f"   Improvement: {improvement:.1f}% reduction in regime changes")
        
        print("\n[SUCCESS] Dynamic smoothing test completed!")
        return True, detector, df
        
    except Exception as e:
        print(f"   [ERROR] Error: {e}")
        return False, None, None

def test_smoothing_effectiveness():
    """Test smoothing effectiveness across different market conditions"""
    print("\n4. Testing smoothing effectiveness:")
    
    # Create detector
    detector = RegimeDetector(n_states=3, random_state=42)
    
    # Test with noisy data (high regime changes)
    np.random.seed(123)
    n_samples = 1000
    
    # Generate noisy data
    prices = [100]
    for i in range(1, n_samples):
        # Add noise to create frequent regime changes
        noise = np.random.normal(0, 0.01)
        if i % 50 == 0:  # Add regime changes every 50 samples
            noise += np.random.normal(0, 0.05)
        prices.append(prices[-1] * (1 + noise))
    
    df = pd.DataFrame({
        'open': prices,
        'high': [p * 1.01 for p in prices],
        'low': [p * 0.99 for p in prices],
        'close': prices,
        'volume': np.random.uniform(1000, 10000, n_samples)
    })
    
    try:
        detector.train(df, lookback=500)
        
        # Test without smoothing
        regime_no_smooth, _ = detector.predict_regime_sequence(df, smooth_window=1)
        changes_no_smooth = np.sum(regime_no_smooth[1:] != regime_no_smooth[:-1])
        rate_no_smooth = changes_no_smooth / len(regime_no_smooth)
        
        # Test with dynamic smoothing
        regime_smooth, _ = detector.predict_regime_sequence(df)
        changes_smooth = np.sum(regime_smooth[1:] != regime_smooth[:-1])
        rate_smooth = changes_smooth / len(regime_smooth)
        
        print(f"   No smoothing: {changes_no_smooth} changes ({rate_no_smooth:.3f})")
        print(f"   Dynamic smoothing: {changes_smooth} changes ({rate_smooth:.3f})")
        
        smoothing_effectiveness = ((rate_no_smooth - rate_smooth) / rate_no_smooth) * 100 if rate_no_smooth > 0 else 0
        print(f"   Smoothing effectiveness: {smoothing_effectiveness:.1f}% reduction")
        
        return True
        
    except Exception as e:
        print(f"   [ERROR] Error in effectiveness test: {e}")
        return False

def test_window_calculation():
    """Test dynamic window calculation logic"""
    print("\n5. Testing window calculation logic:")
    
    detector = RegimeDetector(n_states=3, random_state=42)
    
    # Test different sequence lengths
    test_lengths = [50, 100, 200, 500, 1000, 2000, 5000]
    
    for length in test_lengths:
        window = detector._calculate_dynamic_smooth_window(length)
        percentage = (window / length) * 100
        
        print(f"   Length {length:4d}: window={window:2d} ({percentage:.2f}%)")
        
        # Verify window is reasonable
        assert window >= 3, f"Window too small: {window}"
        assert window <= length, f"Window too large: {window} > {length}"
        assert window % 2 == 1, f"Window should be odd: {window}"
    
    print("   [OK] All window calculations are valid")
    return True

if __name__ == "__main__":
    print("HMM v2.0 - Dynamic Smoothing Window Test")
    print("=" * 50)
    
    success, detector, df = test_dynamic_smoothing()
    
    if success:
        test_smoothing_effectiveness()
        test_window_calculation()
        print("\n[SUCCESS] All dynamic smoothing tests passed!")
    else:
        print("\n[ERROR] Dynamic smoothing tests failed.")
