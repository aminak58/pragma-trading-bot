"""
Example: Using RegimeAdaptiveStrategy with Freqtrade

This example demonstrates how to:
1. Copy the strategy to Freqtrade user_data directory
2. Run backtests with the regime-adaptive strategy
3. Analyze regime-based performance

Requirements:
- Freqtrade installed
- Historical data downloaded
- Strategy copied to user_data/strategies/
"""

import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

import pandas as pd
from regime.hmm_detector import RegimeDetector


def demonstrate_regime_detection():
    """
    Demonstrate regime detection on sample data.
    
    This shows what the strategy does internally.
    """
    print("=" * 60)
    print("HMM Regime Detection - Integration Example")
    print("=" * 60)
    
    # Load sample data (in real usage, Freqtrade provides this)
    print("\n1. Loading sample OHLCV data...")
    
    # For demonstration, create sample data
    import numpy as np
    np.random.seed(42)
    
    n = 1000
    returns = np.random.normal(0.0005, 0.02, n)
    close = 100 * np.exp(np.cumsum(returns))
    
    df = pd.DataFrame({
        'open': close * 0.99,
        'high': close * 1.01,
        'low': close * 0.98,
        'close': close,
        'volume': np.random.lognormal(15, 0.5, n)
    })
    
    print(f"   Loaded {len(df)} candles")
    print(f"   Price range: ${df['close'].min():.2f} - ${df['close'].max():.2f}")
    
    # Initialize detector
    print("\n2. Initializing HMM Regime Detector...")
    detector = RegimeDetector(n_states=3, random_state=42)
    
    # Train
    print("\n3. Training HMM on historical data...")
    detector.train(df, lookback=500)
    print(f"   Regime names: {list(detector.regime_names.values())}")
    
    # Predict
    print("\n4. Predicting current regime...")
    regime, confidence = detector.predict_regime(df)
    print(f"   Current regime: {regime}")
    print(f"   Confidence: {confidence:.2%}")
    
    # Get probabilities
    print("\n5. Regime probability distribution:")
    probs = detector.get_regime_probabilities(df)
    for regime_name, prob in sorted(probs.items(), key=lambda x: x[1], reverse=True):
        print(f"   {regime_name:20s}: {prob:.2%}")
    
    print("\n" + "=" * 60)
    print("This is what the strategy does automatically!")
    print("=" * 60)


def print_strategy_usage():
    """Print instructions for using the strategy with Freqtrade."""
    print("\n" + "=" * 60)
    print("Using RegimeAdaptiveStrategy with Freqtrade")
    print("=" * 60)
    
    print("\n📋 Step 1: Copy Strategy to Freqtrade")
    print("-" * 60)
    print("""
# Option A: Direct copy
cp src/strategies/regime_adaptive_strategy.py \\
   /path/to/freqtrade/user_data/strategies/

# Option B: Symlink (for development)
ln -s $(pwd)/src/strategies/regime_adaptive_strategy.py \\
      /path/to/freqtrade/user_data/strategies/
    """)
    
    print("\n📋 Step 2: Download Data")
    print("-" * 60)
    print("""
cd /path/to/freqtrade

freqtrade download-data \\
  --exchange binance \\
  --pairs BTC/USDT ETH/USDT BNB/USDT \\
  --timeframes 5m \\
  --days 180
    """)
    
    print("\n📋 Step 3: Run Backtest")
    print("-" * 60)
    print("""
freqtrade backtesting \\
  --strategy RegimeAdaptiveStrategy \\
  --timerange 20240701-20251010 \\
  --timeframe 5m
    """)
    
    print("\n📋 Step 4: Analyze Results")
    print("-" * 60)
    print("""
# View results
freqtrade backtesting-analysis

# Plot results
freqtrade plot-dataframe \\
  --strategy RegimeAdaptiveStrategy \\
  --pairs BTC/USDT \\
  --indicators1 ema_short,ema_long,bb_upper,bb_lower \\
  --indicators2 rsi,adx
    """)
    
    print("\n📋 Step 5: Hyperopt (Optional)")
    print("-" * 60)
    print("""
freqtrade hyperopt \\
  --hyperopt-loss SharpeHyperOptLoss \\
  --strategy RegimeAdaptiveStrategy \\
  --epochs 100 \\
  --spaces buy sell
    """)
    
    print("\n" + "=" * 60)


def print_strategy_features():
    """Print key features of the regime-adaptive strategy."""
    print("\n" + "=" * 60)
    print("RegimeAdaptiveStrategy Features")
    print("=" * 60)
    
    features = {
        "🧠 Regime Detection": [
            "3-state HMM (trending, low_volatility, high_volatility)",
            "Automatic regime labeling based on market characteristics",
            "Confidence scoring for each prediction",
            "Probability distribution over all regimes"
        ],
        "📊 Technical Indicators": [
            "EMA (12/26 periods)",
            "RSI (14 period)",
            "ADX (14 period)",
            "Bollinger Bands (20 period, 2 std)",
            "ATR for volatility",
            "MACD for momentum"
        ],
        "🎯 Entry Logic": [
            "Trending: Momentum following (ADX > threshold, EMA crossover)",
            "Low Volatility: Mean reversion (price near BB lower, low RSI)",
            "High Volatility: Conservative (high volume confirmation)"
        ],
        "🚪 Exit Logic": [
            "Regime change to high volatility",
            "Overbought RSI",
            "Trend weakening (ADX decline)",
            "Mean reversion completion (price > BB upper)",
            "MACD bearish crossover"
        ],
        "🛡️ Risk Management": [
            "Regime-adaptive stoploss (wider for trends, tighter for volatility)",
            "Trailing stop (1% profit, 2% offset)",
            "Dynamic leverage (3x trending, 2x low vol, 1x high vol)",
            "Minimum confidence threshold for entries",
            "Trade confirmation with regime check"
        ],
        "📈 Performance Tracking": [
            "Regime occurrence statistics",
            "Trades per regime",
            "Detailed logging",
            "Entry/exit reasons tagged"
        ]
    }
    
    for category, items in features.items():
        print(f"\n{category}")
        print("-" * 60)
        for item in items:
            print(f"  • {item}")
    
    print("\n" + "=" * 60)


def main():
    """Run all examples."""
    # Demonstrate regime detection
    demonstrate_regime_detection()
    
    # Print strategy features
    print_strategy_features()
    
    # Print usage instructions
    print_strategy_usage()
    
    print("\n✅ Integration example complete!")
    print("\nNext steps:")
    print("  1. Copy strategy to Freqtrade")
    print("  2. Download data")
    print("  3. Run backtest")
    print("  4. Analyze regime-based performance")


if __name__ == "__main__":
    main()
