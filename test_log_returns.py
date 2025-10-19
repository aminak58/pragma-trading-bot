#!/usr/bin/env python3
"""
Test log returns implementation in HMM detector
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

def test_log_returns():
    """Test log returns vs pct_change"""
    print("Testing Log Returns Implementation...")
    
    # Create sample data
    np.random.seed(42)
    n_samples = 1000
    
    # Generate realistic price data with trend and volatility
    trend = np.linspace(100, 150, n_samples)
    noise = np.random.normal(0, 2, n_samples)
    prices = trend + noise
    
    # Create DataFrame
    df = pd.DataFrame({
        'open': prices,
        'high': prices + np.random.uniform(0, 1, n_samples),
        'low': prices - np.random.uniform(0, 1, n_samples),
        'close': prices,
        'volume': np.random.uniform(1000, 10000, n_samples)
    })
    
    # Test both methods
    print("\n1. Testing pct_change (old method):")
    pct_returns_1 = df['close'].pct_change(1).dropna()
    pct_returns_5 = df['close'].pct_change(5).dropna()
    pct_returns_20 = df['close'].pct_change(20).dropna()
    
    print(f"   Returns_1: mean={pct_returns_1.mean():.6f}, std={pct_returns_1.std():.6f}")
    print(f"   Returns_5: mean={pct_returns_5.mean():.6f}, std={pct_returns_5.std():.6f}")
    print(f"   Returns_20: mean={pct_returns_20.mean():.6f}, std={pct_returns_20.std():.6f}")
    
    print("\n2. Testing log returns (new method):")
    log_returns_1 = np.log(df['close'] / df['close'].shift(1)).dropna()
    log_returns_5 = np.log(df['close'] / df['close'].shift(5)).dropna()
    log_returns_20 = np.log(df['close'] / df['close'].shift(20)).dropna()
    
    print(f"   Returns_1: mean={log_returns_1.mean():.6f}, std={log_returns_1.std():.6f}")
    print(f"   Returns_5: mean={log_returns_5.mean():.6f}, std={log_returns_5.std():.6f}")
    print(f"   Returns_20: mean={log_returns_20.mean():.6f}, std={log_returns_20.std():.6f}")
    
    # Test HMM detector with new implementation
    print("\n3. Testing HMM detector with log returns:")
    detector = RegimeDetector(n_states=3, random_state=42)
    
    try:
        # Test feature preparation
        features = detector.prepare_features(df)
        print(f"   [OK] Features shape: {features.shape}")
        print(f"   [OK] Features mean: {np.mean(features, axis=0)}")
        print(f"   [OK] Features std: {np.std(features, axis=0)}")
        
        # Test training
        detector.train(df, lookback=500)
        print(f"   [OK] HMM training successful")
        print(f"   [OK] Regime names: {detector.regime_names}")
        
        # Test prediction
        regime, confidence = detector.predict_regime(df)
        print(f"   [OK] Current regime: {regime} (confidence: {confidence:.3f})")
        
        print("\n[SUCCESS] Log returns implementation successful!")
        return True
        
    except Exception as e:
        print(f"   [ERROR] Error: {e}")
        return False

def compare_methods():
    """Compare pct_change vs log returns"""
    print("\n" + "="*60)
    print("COMPARISON: pct_change vs log returns")
    print("="*60)
    
    # Create sample data with extreme values (common in crypto)
    np.random.seed(42)
    n_samples = 1000
    
    # Generate data with some extreme moves (like crypto)
    prices = [100]
    for i in range(1, n_samples):
        # Add some extreme moves occasionally
        if i % 100 == 0:
            change = np.random.choice([-0.1, 0.1])  # 10% moves
        else:
            change = np.random.normal(0, 0.01)  # 1% normal moves
        prices.append(prices[-1] * (1 + change))
    
    df = pd.DataFrame({
        'open': prices,
        'high': [p * 1.01 for p in prices],
        'low': [p * 0.99 for p in prices],
        'close': prices,
        'volume': np.random.uniform(1000, 10000, n_samples)
    })
    
    # Compare methods
    pct_returns = df['close'].pct_change(1).dropna()
    log_returns = np.log(df['close'] / df['close'].shift(1)).dropna()
    
    print(f"Data points: {len(pct_returns)}")
    print(f"Price range: {df['close'].min():.2f} - {df['close'].max():.2f}")
    print(f"Max price change: {df['close'].pct_change().max():.3f}")
    print()
    
    print("pct_change method:")
    print(f"  Mean: {pct_returns.mean():.6f}")
    print(f"  Std:  {pct_returns.std():.6f}")
    print(f"  Min:  {pct_returns.min():.6f}")
    print(f"  Max:  {pct_returns.max():.6f}")
    print(f"  Skew: {pct_returns.skew():.6f}")
    print()
    
    print("log returns method:")
    print(f"  Mean: {log_returns.mean():.6f}")
    print(f"  Std:  {log_returns.std():.6f}")
    print(f"  Min:  {log_returns.min():.6f}")
    print(f"  Max:  {log_returns.max():.6f}")
    print(f"  Skew: {log_returns.skew():.6f}")
    print()
    
    # Calculate stability metrics
    pct_stability = 1 / (1 + pct_returns.std())
    log_stability = 1 / (1 + log_returns.std())
    
    print("Stability comparison:")
    print(f"  pct_change stability: {pct_stability:.6f}")
    print(f"  log returns stability: {log_stability:.6f}")
    print(f"  Improvement: {((log_stability - pct_stability) / pct_stability * 100):.2f}%")

if __name__ == "__main__":
    print("HMM v2.0 - Log Returns Implementation Test")
    print("=" * 50)
    
    success = test_log_returns()
    compare_methods()
    
    if success:
        print("\n[SUCCESS] All tests passed! Log returns implementation is working correctly.")
    else:
        print("\n[ERROR] Tests failed. Please check the implementation.")
