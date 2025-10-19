"""
HMM Analysis Script
Analyze HMM training and regime detection with visualizations
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import logging

# Try to import seaborn, if not available use matplotlib
try:
    import seaborn as sns
    plt.style.use('seaborn-v0_8')
except ImportError:
    print("Seaborn not available, using matplotlib default style")
    plt.style.use('default')

# Add src to path
src_path = Path(__file__).parent.parent
sys.path.insert(0, str(src_path))

from src.regime.hmm_detector import RegimeDetector

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_data():
    """Load BTC/USDT data from Freqtrade format"""
    try:
        # Try to load from user_data (feather format)
        data_path = src_path / "user_data" / "data" / "binance" / "BTC_USDT-5m.feather"
        if data_path.exists():
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

def analyze_hmm_training(df):
    """Analyze HMM training process"""
    logger.info("Starting HMM analysis...")
    
    # Initialize HMM detector
    detector = RegimeDetector(n_states=3, random_state=42)
    
    # Train on different data sizes
    sizes = [500, 1000, 1500, 2000]
    results = {}
    
    for size in sizes:
        if len(df) >= size:
            logger.info(f"Training HMM with {size} candles...")
            try:
                detector.train(df, lookback=size)
                
                # Get regime predictions for entire sequence
                regime_sequence, confidence_sequence = detector.predict_regime_sequence(df)
                probs = detector.get_regime_probabilities(df)
                
                results[size] = {
                    'regime': regime_sequence,
                    'confidence': confidence_sequence,
                    'probabilities': probs,
                    'trained': True
                }
                logger.info(f"✓ HMM trained successfully with {size} candles")
                
            except Exception as e:
                logger.error(f"✗ HMM training failed with {size} candles: {e}")
                results[size] = {'trained': False, 'error': str(e)}
        else:
            logger.warning(f"Not enough data for {size} candles (have {len(df)})")
            results[size] = {'trained': False, 'error': 'Insufficient data'}
    
    return results

def create_visualizations(df, results):
    """Create comprehensive HMM visualizations"""
    logger.info("Creating visualizations...")
    
    # Set up the plot style
    fig = plt.figure(figsize=(20, 16))
    
    # 1. Price and Volume Chart
    ax1 = plt.subplot(4, 2, 1)
    ax1.plot(df.index, df['close'], label='Close Price', linewidth=1)
    ax1.set_title('BTC/USDT Price Chart (5m)', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Price (USDT)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Volume Chart
    ax2 = plt.subplot(4, 2, 2)
    ax2.bar(df.index, df['volume'], alpha=0.7, width=0.8)
    ax2.set_title('Volume', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Volume')
    ax2.grid(True, alpha=0.3)
    
    # 3. Returns Distribution
    ax3 = plt.subplot(4, 2, 3)
    returns = df['close'].pct_change().dropna()
    ax3.hist(returns, bins=50, alpha=0.7, density=True)
    ax3.set_title('Returns Distribution', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Returns')
    ax3.set_ylabel('Density')
    ax3.grid(True, alpha=0.3)
    
    # 4. Volatility (Rolling 20-period)
    ax4 = plt.subplot(4, 2, 4)
    volatility = returns.rolling(20).std()
    # Align indices properly
    vol_data = volatility.dropna()
    ax4.plot(vol_data.index, vol_data.values, linewidth=1)
    ax4.set_title('Rolling Volatility (20 periods)', fontsize=14, fontweight='bold')
    ax4.set_ylabel('Volatility')
    ax4.grid(True, alpha=0.3)
    
    # 5. HMM Regime Detection Results
    ax5 = plt.subplot(4, 2, 5)
    if any(r.get('trained', False) for r in results.values()):
        # Use the largest successful training
        best_size = max([s for s, r in results.items() if r.get('trained', False)])
        regime_data = results[best_size]['regime']
        confidence_data = results[best_size]['confidence']
        
        # Ensure regime_data is a list/array
        if isinstance(regime_data, str):
            regime_data = [regime_data] * len(df)
        if isinstance(confidence_data, (int, float)):
            confidence_data = [confidence_data] * len(df)
        
        # Plot regime as background color
        regime_colors = {'trending': 'green', 'low_volatility': 'blue', 'high_volatility': 'red'}
        for i, (regime, conf) in enumerate(zip(regime_data, confidence_data)):
            if regime in regime_colors:
                ax5.axvspan(df.index[i], df.index[i+1] if i+1 < len(df) else df.index[i], 
                           alpha=0.3, color=regime_colors[regime])
        
        ax5.plot(df.index, df['close'], linewidth=1, color='black')
        ax5.set_title(f'HMM Regime Detection (Trained on {best_size} candles)', 
                     fontsize=14, fontweight='bold')
        ax5.set_ylabel('Price (USDT)')
        ax5.grid(True, alpha=0.3)
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor=color, alpha=0.3, label=regime) 
                          for regime, color in regime_colors.items()]
        ax5.legend(handles=legend_elements)
    
    # 6. Confidence Levels
    ax6 = plt.subplot(4, 2, 6)
    if any(r.get('trained', False) for r in results.values()):
        best_size = max([s for s, r in results.items() if r.get('trained', False)])
        confidence_data = results[best_size]['confidence']
        
        # Ensure confidence_data is a list/array
        if isinstance(confidence_data, (int, float)):
            confidence_data = [confidence_data] * len(df)
        
        ax6.plot(df.index, confidence_data, linewidth=1, color='purple')
        ax6.set_title('HMM Confidence Levels', fontsize=14, fontweight='bold')
        ax6.set_ylabel('Confidence')
        ax6.set_ylim(0, 1)
        ax6.grid(True, alpha=0.3)
    
    # 7. Regime Probabilities
    ax7 = plt.subplot(4, 2, 7)
    if any(r.get('trained', False) for r in results.values()):
        best_size = max([s for s, r in results.items() if r.get('trained', False)])
        probs = results[best_size]['probabilities']
        
        for regime, prob_values in probs.items():
            # Ensure prob_values is a list/array
            if isinstance(prob_values, (int, float)):
                prob_values = [prob_values] * len(df)
            elif len(prob_values) != len(df):
                # Pad or truncate to match df length
                if len(prob_values) < len(df):
                    prob_values = list(prob_values) + [prob_values[-1]] * (len(df) - len(prob_values))
                else:
                    prob_values = prob_values[:len(df)]
            
            ax7.plot(df.index, prob_values, label=f'P({regime})', linewidth=1)
        
        ax7.set_title('Regime Probabilities', fontsize=14, fontweight='bold')
        ax7.set_ylabel('Probability')
        ax7.set_ylim(0, 1)
        ax7.legend()
        ax7.grid(True, alpha=0.3)
    
    # 8. Training Results Summary
    ax8 = plt.subplot(4, 2, 8)
    ax8.axis('off')
    
    # Create summary text
    summary_text = "HMM Training Results Summary:\n\n"
    for size, result in results.items():
        if result.get('trained', False):
            regime = result['regime']
            confidence = result['confidence']
            unique_regimes = np.unique(regime)
            avg_confidence = np.mean(confidence)
            
            summary_text += f"Size {size}: ✓ Success\n"
            summary_text += f"  - Regimes detected: {len(unique_regimes)}\n"
            summary_text += f"  - Unique regimes: {unique_regimes}\n"
            summary_text += f"  - Avg confidence: {avg_confidence:.3f}\n"
            summary_text += f"  - Min confidence: {np.min(confidence):.3f}\n"
            summary_text += f"  - Max confidence: {np.max(confidence):.3f}\n\n"
        else:
            summary_text += f"Size {size}: ✗ Failed\n"
            summary_text += f"  - Error: {result.get('error', 'Unknown')}\n\n"
    
    ax8.text(0.05, 0.95, summary_text, transform=ax8.transAxes, 
             fontsize=10, verticalalignment='top', fontfamily='monospace')
    
    plt.tight_layout()
    # Save to output directory
    output_path = '/output/hmm_analysis.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"Visualization saved as '{output_path}'")
    
    return fig

def main():
    """Main analysis function"""
    logger.info("Starting HMM Analysis...")
    
    # Load data
    df = load_data()
    if df is None:
        logger.error("Failed to load data. Exiting.")
        return
    
    logger.info(f"Data loaded: {len(df)} candles from {df.index[0]} to {df.index[-1]}")
    
    # Analyze HMM training
    results = analyze_hmm_training(df)
    
    # Create visualizations
    fig = create_visualizations(df, results)
    
    # Print summary
    logger.info("\n" + "="*60)
    logger.info("HMM ANALYSIS SUMMARY")
    logger.info("="*60)
    
    for size, result in results.items():
        if result.get('trained', False):
            regime = result['regime']
            confidence = result['confidence']
            unique_regimes = np.unique(regime)
            avg_confidence = np.mean(confidence)
            
            logger.info(f"\nTraining Size: {size}")
            logger.info(f"  ✓ Success")
            logger.info(f"  - Regimes detected: {len(unique_regimes)}")
            logger.info(f"  - Unique regimes: {unique_regimes}")
            logger.info(f"  - Avg confidence: {avg_confidence:.3f}")
            logger.info(f"  - Confidence range: {np.min(confidence):.3f} - {np.max(confidence):.3f}")
            
            # Regime distribution
            regime_counts = {reg: np.sum(regime == reg) for reg in unique_regimes}
            logger.info(f"  - Regime distribution: {regime_counts}")
        else:
            logger.info(f"\nTraining Size: {size}")
            logger.info(f"  ✗ Failed: {result.get('error', 'Unknown')}")
    
    logger.info("\n" + "="*60)
    logger.info("Analysis complete! Check 'hmm_analysis.png' for visualizations.")
    logger.info("="*60)

if __name__ == "__main__":
    main()
