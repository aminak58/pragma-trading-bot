"""
Basic usage example of HMM Regime Detector

This script demonstrates how to use the RegimeDetector class
to detect market regimes from OHLCV data.
"""

import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

import pandas as pd
import numpy as np
from regime.hmm_detector import RegimeDetector


def generate_sample_data(n_samples: int = 1000) -> pd.DataFrame:
    """
    Generate sample OHLCV data for testing.
    
    Args:
        n_samples: Number of samples to generate
        
    Returns:
        DataFrame with OHLCV data
    """
    np.random.seed(42)
    
    # Generate price data with random walk
    returns = np.random.normal(0.0005, 0.02, n_samples)
    close = 100 * np.exp(np.cumsum(returns))
    
    # Generate OHLC from close
    high = close * (1 + np.abs(np.random.normal(0, 0.01, n_samples)))
    low = close * (1 - np.abs(np.random.normal(0, 0.01, n_samples)))
    open_price = close * (1 + np.random.normal(0, 0.005, n_samples))
    
    # Generate volume
    volume = np.random.lognormal(15, 0.5, n_samples)
    
    df = pd.DataFrame({
        'open': open_price,
        'high': high,
        'low': low,
        'close': close,
        'volume': volume
    })
    
    return df


def main():
    """Main example function."""
    print("=" * 60)
    print("HMM Regime Detector - Basic Usage Example")
    print("=" * 60)
    
    # Step 1: Generate sample data
    print("\n1. Generating sample OHLCV data...")
    df = generate_sample_data(n_samples=1000)
    print(f"   Generated {len(df)} candles")
    print(f"   Price range: ${df['close'].min():.2f} - ${df['close'].max():.2f}")
    
    # Step 2: Initialize detector
    print("\n2. Initializing RegimeDetector...")
    detector = RegimeDetector(n_states=3, random_state=42)
    print(f"   {detector}")
    
    # Step 3: Train the model
    print("\n3. Training HMM on data...")
    detector.train(df, lookback=500)
    print(f"   {detector}")
    print(f"   Regime names: {detector.regime_names}")
    
    # Step 4: Predict current regime
    print("\n4. Predicting current market regime...")
    regime, confidence = detector.predict_regime(df)
    print(f"   Current regime: {regime}")
    print(f"   Confidence: {confidence:.2%}")
    
    # Step 5: Get probability distribution
    print("\n5. Getting regime probability distribution...")
    probs = detector.get_regime_probabilities(df)
    for regime_name, prob in sorted(probs.items(), key=lambda x: x[1], reverse=True):
        print(f"   {regime_name:20s}: {prob:.2%}")
    
    # Step 6: Examine transition matrix
    print("\n6. State transition probabilities...")
    trans_matrix = detector.get_transition_matrix()
    print("   Transition Matrix:")
    for i in range(detector.n_states):
        regime_i = detector.regime_names[i]
        print(f"   {regime_i:20s} ->", end="")
        for j in range(detector.n_states):
            regime_j = detector.regime_names[j]
            print(f" {regime_j[:10]:10s}:{trans_matrix[i,j]:.2%}", end="")
        print()
    
    print("\n" + "=" * 60)
    print("✅ Example completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
