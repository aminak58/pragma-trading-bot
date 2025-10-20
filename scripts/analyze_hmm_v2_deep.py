"""
HMM v2.0 Deep Analysis Script
- Analyze regime probabilities and Trend Phase Score
- Design optimal Entry Logic based on HMM insights
- Target: WinRate > 40%, Sharpe > 1.0, MDD < 1%
"""
from pathlib import Path
import sys

# Add project root and src to path
root_path = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(root_path))
sys.path.insert(0, str(root_path / 'src'))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from src.regime.hmm_detector import RegimeDetector

def analyze_hmm_v2():
    """Deep analysis of HMM v2.0 capabilities"""
    
    print("HMM v2.0 Deep Analysis Starting...")
    
    # Load data
    data_path = Path.cwd() / 'user_data' / 'data' / 'binance' / 'BTC_USDT-5m.feather'
    if not data_path.exists():
        print(f"Data file not found: {data_path}")
        return
    
    # Load and prepare data
    df = pd.read_feather(data_path)
    
    # Use recent data for analysis
    df = df.tail(5000)  # Last 5000 candles
    
    print(f"Data loaded: {len(df)} candles from {df.index[0]} to {df.index[-1]}")
    
    # Initialize HMM detector
    detector = RegimeDetector()
    
    # Train HMM
    print("Training HMM...")
    detector.train(df)
    
    # Get regime predictions
    regimes, confidences = detector.predict_regime_sequence(df, smooth_window=5)
    
    # Compute Trend Phase Score
    print("Computing Trend Phase Score...")
    trend_scores = detector.compute_trend_phase_score(df)
    
    # Align data
    min_len = min(len(df), len(regimes), len(confidences), len(trend_scores))
    df = df.tail(min_len)
    regimes = regimes[-min_len:]
    confidences = confidences[-min_len:]
    trend_scores = trend_scores.tail(min_len)
    
    # Create analysis dataframe
    analysis_df = df.copy()
    analysis_df['regime'] = regimes
    analysis_df['regime_conf'] = confidences
    analysis_df['trend_score'] = trend_scores
    
    # Calculate forward returns for validation
    analysis_df['fwd_ret_5'] = analysis_df['close'].pct_change(5).shift(-5)
    analysis_df['fwd_ret_10'] = analysis_df['close'].pct_change(10).shift(-10)
    analysis_df['fwd_ret_20'] = analysis_df['close'].pct_change(20).shift(-20)
    
    print("\nHMM v2.0 Analysis Results:")
    print("=" * 50)
    
    # 1. Regime Distribution
    print("\n1. Regime Distribution:")
    regime_counts = analysis_df['regime'].value_counts()
    for regime, count in regime_counts.items():
        pct = count / len(analysis_df) * 100
        print(f"   {regime}: {count} ({pct:.1f}%)")
    
    # 2. Confidence Analysis
    print("\n2. Confidence Analysis:")
    conf_stats = analysis_df['regime_conf'].describe()
    print(f"   Mean: {conf_stats['mean']:.3f}")
    print(f"   Std: {conf_stats['std']:.3f}")
    print(f"   Min: {conf_stats['min']:.3f}")
    print(f"   Max: {conf_stats['max']:.3f}")
    
    # High confidence periods
    high_conf = analysis_df[analysis_df['regime_conf'] > 0.8]
    print(f"   High confidence (>0.8): {len(high_conf)} ({len(high_conf)/len(analysis_df)*100:.1f}%)")
    
    # 3. Trend Phase Score Analysis
    print("\n3. Trend Phase Score Analysis:")
    score_stats = analysis_df['trend_score'].describe()
    print(f"   Mean: {score_stats['mean']:.3f}")
    print(f"   Std: {score_stats['std']:.3f}")
    print(f"   Min: {score_stats['min']:.3f}")
    print(f"   Max: {score_stats['max']:.3f}")
    
    # Percentiles
    percentiles = [10, 25, 50, 75, 90]
    print("   Percentiles:")
    for p in percentiles:
        val = analysis_df['trend_score'].quantile(p/100)
        print(f"     P{p}: {val:.3f}")
    
    # 4. Regime-Specific Analysis
    print("\n4. Regime-Specific Analysis:")
    for regime in analysis_df['regime'].unique():
        regime_data = analysis_df[analysis_df['regime'] == regime]
        print(f"\n   {regime.upper()}:")
        print(f"      Count: {len(regime_data)}")
        print(f"      Avg Confidence: {regime_data['regime_conf'].mean():.3f}")
        print(f"      Avg Trend Score: {regime_data['trend_score'].mean():.3f}")
        print(f"      Avg 5-candle Return: {regime_data['fwd_ret_5'].mean():.4f}")
        print(f"      Avg 20-candle Return: {regime_data['fwd_ret_20'].mean():.4f}")
    
    # 5. Correlation Analysis
    print("\n5. Correlation Analysis:")
    correlations = analysis_df[['regime_conf', 'trend_score', 'fwd_ret_5', 'fwd_ret_10', 'fwd_ret_20']].corr()
    print("   Trend Score vs Forward Returns:")
    print(f"     vs 5-candle: {correlations.loc['trend_score', 'fwd_ret_5']:.3f}")
    print(f"     vs 10-candle: {correlations.loc['trend_score', 'fwd_ret_10']:.3f}")
    print(f"     vs 20-candle: {correlations.loc['trend_score', 'fwd_ret_20']:.3f}")
    
    # 6. Optimal Entry Conditions Analysis
    print("\n6. Optimal Entry Conditions Analysis:")
    
    # High confidence + high trend score
    high_conf_high_score = analysis_df[
        (analysis_df['regime_conf'] > 0.7) & 
        (analysis_df['trend_score'] > analysis_df['trend_score'].quantile(0.8))
    ]
    if len(high_conf_high_score) > 0:
        avg_return = high_conf_high_score['fwd_ret_20'].mean()
        print(f"   High Conf + High Score: {len(high_conf_high_score)} periods")
        print(f"   Avg 20-candle return: {avg_return:.4f}")
    
    # Trending regime + high trend score
    trending_high_score = analysis_df[
        (analysis_df['regime'] == 'trending') & 
        (analysis_df['trend_score'] > analysis_df['trend_score'].quantile(0.75))
    ]
    if len(trending_high_score) > 0:
        avg_return = trending_high_score['fwd_ret_20'].mean()
        print(f"   Trending + High Score: {len(trending_high_score)} periods")
        print(f"   Avg 20-candle return: {avg_return:.4f}")
    
    # 7. Entry Logic Recommendations
    print("\n7. Entry Logic Recommendations:")
    print("   Based on analysis, optimal entry conditions:")
    
    # Calculate optimal thresholds
    trend_p75 = analysis_df['trend_score'].quantile(0.75)
    trend_p80 = analysis_df['trend_score'].quantile(0.80)
    conf_threshold = 0.7
    
    print(f"   Trend Score Threshold: >{trend_p75:.3f} (P75)")
    print(f"   Confidence Threshold: >{conf_threshold:.3f}")
    print(f"   Preferred Regimes: trending, high_volatility")
    
    # Test these conditions
    optimal_entries = analysis_df[
        (analysis_df['trend_score'] > trend_p75) &
        (analysis_df['regime_conf'] > conf_threshold) &
        (analysis_df['regime'].isin(['trending', 'high_volatility']))
    ]
    
    if len(optimal_entries) > 0:
        avg_return = optimal_entries['fwd_ret_20'].mean()
        win_rate = (optimal_entries['fwd_ret_20'] > 0).mean()
        print(f"   Optimal Entries: {len(optimal_entries)} periods")
        print(f"   Avg 20-candle return: {avg_return:.4f}")
        print(f"   Win rate: {win_rate:.1%}")
    
    # 8. Risk Management Recommendations
    print("\n8. Risk Management Recommendations:")
    
    # Calculate volatility by regime
    for regime in analysis_df['regime'].unique():
        regime_data = analysis_df[analysis_df['regime'] == regime]
        volatility = regime_data['fwd_ret_20'].std()
        print(f"   {regime}: Volatility = {volatility:.4f}")
    
    print("\nHMM v2.0 Analysis Complete!")
    return analysis_df

if __name__ == "__main__":
    analysis_df = analyze_hmm_v2()
