#!/usr/bin/env python3
"""
Test improved HMM implementation
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import logging

# Add src to path
sys.path.insert(0, 'src')

from regime.hmm_detector import RegimeDetector

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_data():
    """Load BTC/USDT data"""
    try:
        data_path = "user_data/data/binance/BTC_USDT-5m.feather"
        if Path(data_path).exists():
            df = pd.read_feather(data_path)
            df['date'] = pd.to_datetime(df['date'], unit='ms')
            df.set_index('date', inplace=True)
            logger.info(f"Loaded {len(df)} candles from {data_path}")
            return df
        else:
            logger.error(f"Data file not found: {data_path}")
            return None
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        return None

def test_improved_hmm(df):
    """Test improved HMM implementation"""
    logger.info("Testing improved HMM...")
    detector = RegimeDetector(n_states=3, random_state=42)
    
    if len(df) >= 1000:
        logger.info("Training HMM with 1000 candles...")
        try:
            detector.train(df, lookback=1000)
            
            # Get regime predictions for entire sequence
            regime_sequence, confidence_sequence = detector.predict_regime_sequence(df, smooth_window=5)
            
            logger.info("✓ HMM trained successfully")
            
            # Calculate regime distribution
            unique_regimes, counts = np.unique(regime_sequence, return_counts=True)
            regime_dist = dict(zip(unique_regimes, counts))
            
            logger.info(f"Regime distribution: {regime_dist}")
            logger.info(f"Avg confidence: {np.mean(confidence_sequence):.3f}")
            logger.info(f"Confidence range: {np.min(confidence_sequence):.3f} - {np.max(confidence_sequence):.3f}")
            
            # Check for stability (regime changes)
            regime_changes = np.sum(regime_sequence[1:] != regime_sequence[:-1])
            total_periods = len(regime_sequence)
            change_rate = regime_changes / total_periods
            
            logger.info(f"Regime changes: {regime_changes} out of {total_periods} ({change_rate:.3f})")
            
            if change_rate < 0.1:  # Less than 10% changes
                logger.info("✅ HMM is stable!")
                return True
            else:
                logger.warning(f"⚠️ HMM may be too noisy (change rate: {change_rate:.3f})")
                return False
            
        except Exception as e:
            logger.error(f"✗ HMM training failed: {e}")
            return False
    else:
        logger.error("Insufficient data for training")
        return False

def main():
    """Main test function"""
    logger.info("Starting improved HMM test...")
    
    df = load_data()
    if df is not None:
        success = test_improved_hmm(df)
        if success:
            print("✅ Improved HMM test successful!")
        else:
            print("❌ Improved HMM test failed!")
    else:
        print("❌ Could not load data!")

if __name__ == "__main__":
    main()
