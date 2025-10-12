# HMM Regime Detection Module

Market regime detection using Hidden Markov Models (HMM) for adaptive trading strategies.

## Overview

This module provides a 3-state Gaussian HMM that classifies market conditions into:

- **Low Volatility**: Calm, ranging markets (low risk)
- **High Volatility**: Choppy, volatile markets (high risk)
- **Trending**: Strong directional movements (momentum opportunity)

## Features

### Multi-Feature Analysis
The detector analyzes 7 key market features:

1. **Returns** (3 timeframes)
   - Short-term (1 period)
   - Medium-term (5 periods)
   - Long-term (20 periods)

2. **Volatility** (2 windows)
   - 10-period rolling standard deviation
   - 30-period rolling standard deviation

3. **Volume Ratio**
   - Current volume vs 20-period average

4. **Trend Strength** (ADX)
   - Average Directional Index approximation

### Automatic Regime Labeling
The model automatically assigns meaningful names to detected states based on their statistical characteristics.

## Usage

### Basic Example

```python
from regime.hmm_detector import RegimeDetector
import pandas as pd

# Initialize detector
detector = RegimeDetector(n_states=3, random_state=42)

# Train on historical data
detector.train(dataframe, lookback=500)

# Predict current regime
regime, confidence = detector.predict_regime(dataframe)
print(f"Current regime: {regime} (confidence: {confidence:.2%})")

# Get all regime probabilities
probs = detector.get_regime_probabilities(dataframe)
```

### Integration with Freqtrade Strategy

```python
class AdaptiveStrategy(IStrategy):
    def __init__(self, config: dict):
        super().__init__(config)
        self.regime_detector = RegimeDetector(n_states=3)
        
    def populate_indicators(self, dataframe, metadata):
        # Train/update regime detector periodically
        if len(dataframe) >= 500:
            self.regime_detector.train(dataframe, lookback=500)
            
            # Add regime predictions
            regime, confidence = self.regime_detector.predict_regime(dataframe)
            dataframe['regime'] = regime
            dataframe['regime_confidence'] = confidence
            
        return dataframe
        
    def populate_entry_trend(self, dataframe, metadata):
        # Adapt entry logic based on regime
        dataframe.loc[
            (
                (dataframe['regime'] == 'trending') &
                (dataframe['regime_confidence'] > 0.7) &
                # Add your trending indicators
            ),
            'enter_long'
        ] = 1
        return dataframe
```

## API Reference

### `RegimeDetector`

**Constructor:**
```python
RegimeDetector(n_states=3, random_state=42)
```

**Methods:**

- `train(dataframe, lookback=500)`: Train the model on historical data
- `predict_regime(dataframe)`: Get current regime and confidence
- `get_regime_probabilities(dataframe)`: Get probability distribution
- `get_transition_matrix()`: Get state transition probabilities

**Attributes:**

- `is_trained`: Whether model has been trained
- `regime_names`: Mapping of states to regime names
- `n_states`: Number of hidden states

## Technical Details

### Model Architecture
- **Type**: Gaussian Hidden Markov Model
- **States**: 3 (configurable)
- **Covariance**: Full
- **Iterations**: 100
- **Features**: 7 (standardized)

### Feature Engineering
All features are:
1. Calculated from OHLCV data
2. Normalized using StandardScaler
3. NaN-handled with forward/backward fill

### State Assignment Logic
States are labeled based on:
- **Volatility metrics**: Mean of volatility features
- **Trend strength**: ADX values
- **Comparative analysis**: Highest volatility → high_volatility, highest trend → trending

## Performance Considerations

### Training Time
- ~100-500ms for 500 candles on typical hardware
- Linear scaling with data size

### Memory Usage
- Minimal: ~1-5MB for typical models
- Scales with feature dimensionality and state count

### Recommendations
- **Minimum data**: 100 candles (200+ recommended)
- **Training frequency**: Every 100-500 new candles
- **Lookback window**: 500 candles (balance between responsiveness and stability)

## Validation

The implementation includes:
- Input validation for required columns
- Error handling for insufficient data
- NaN handling in feature preparation
- Deterministic predictions (with fixed random_state)

## Examples

See `examples/hmm_basic_usage.py` for a complete working example.

## Testing

Run unit tests:
```bash
pytest tests/unit/test_hmm_detector.py -v
```

## References

- Hidden Markov Models: [hmmlearn documentation](https://hmmlearn.readthedocs.io/)
- Regime Detection in Finance: Academic research on market state detection
- Freqtrade: [Strategy customization guide](https://www.freqtrade.io/en/stable/)
