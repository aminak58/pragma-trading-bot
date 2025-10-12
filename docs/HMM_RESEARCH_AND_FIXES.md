# HMM Best Practices Research & Implementation Plan

**Date**: 2025-10-12  
**Goal**: Fix regime detection using HMM best practices from research and community

---

## 🔍 Research Summary: HMM for Trading Regime Detection

### Academic & Industry Best Practices

#### 1. **Number of States**

**Research Findings:**
- **2-3 states**: Most common in literature
  - 2 states: Bull/Bear markets
  - 3 states: Bull/Bear/Sideways (most popular)
  - 4+ states: Overfitting risk increases

**Community Consensus:**
- Start with 3 states (our current approach ✅)
- Can expand to 4-5 if needed, but requires more data

**Sources:**
- Hamilton (1989) - Economic regime switching
- Ang & Bekaert (2002) - Bull/bear market regimes
- QuantStart, Quantopian community

---

#### 2. **Feature Selection (Critical!)**

**Problem with Current Implementation:**
```python
# Current (TOO SIMPLE):
features = [returns_1, returns_5, returns_20, volatility_10, 
            volatility_30, volume_ratio, adx]
```

**Best Practice Features:**

**A. Returns-based:**
- ✅ Log returns (multiple horizons)
- ✅ Realized volatility
- ❌ **Missing**: Return skewness
- ❌ **Missing**: Return kurtosis

**B. Volatility-based:**
- ✅ Rolling volatility
- ❌ **Missing**: High-Low range
- ❌ **Missing**: Parkinson volatility estimator
- ❌ **Missing**: Garman-Klass volatility

**C. Trend/Momentum:**
- ✅ ADX
- ❌ **Missing**: Trend consistency (directional movement)
- ❌ **Missing**: Number of up/down days ratio

**D. Volume:**
- ✅ Volume ratio
- ❌ **Missing**: Volume volatility
- ❌ **Missing**: Volume trend

**Recommended Feature Set (Research-backed):**
```python
1. returns_1d        # 1-period log return
2. returns_5d        # 5-period log return
3. returns_20d       # 20-period log return
4. volatility_20d    # 20-period realized volatility
5. volatility_60d    # 60-period realized volatility
6. hl_range          # High-Low range normalized
7. volume_ratio      # Volume vs MA
8. volume_volatility # Volatility of volume
9. trend_strength    # ADX or similar
10. return_skew_20   # Skewness of returns
```

**Sources:**
- Kritzman, Page & Turkington (2012) - "Regime Shifts: Implications for Dynamic Strategies"
- Nystrup et al. (2015) - "Regime-Based vs Static Asset Allocation"

---

#### 3. **Training Window Size**

**Current**: 500 candles (on 5m = ~41 hours)

**Research Recommendations:**

| Timeframe | Minimum Training | Recommended | Maximum |
|-----------|-----------------|-------------|---------|
| 5m        | 2000 (1 week)   | 4000-8000   | 15000   |
| 1h        | 500 (3 weeks)   | 1000-2000   | 5000    |
| 1d        | 200 (8 months)  | 500-1000    | 2000    |

**Rule of Thumb**: 
- **Minimum**: 100 × number_of_features × number_of_states
- **Our case**: 100 × 7 × 3 = 2100 candles minimum

**Current Issue**: 500 is WAY too small! ❌

**Recommendation**: 
- Use **4000-8000 candles** for 5m timeframe
- This gives ~2-4 weeks of data
- Enough to capture multiple regime shifts

**Sources:**
- Murphy (2012) - Hidden Markov Models for time series
- Guidolin & Timmermann (2008) - "Economic implications of bull and bear regimes"

---

#### 4. **Covariance Type**

**Current**: Not specified (defaults to 'diag')

**Options:**
```python
'full'      # Full covariance matrix (most flexible, needs most data)
'tied'      # Same covariance for all states
'diag'      # Diagonal covariance (independence assumption)
'spherical' # Single variance (most restrictive)
```

**Best Practice**:
- Start with **'full'** if you have enough data
- Fall back to **'diag'** if training fails or data limited
- **Never use 'spherical'** for multi-feature data

**Our Recommendation**: 
- Try 'full' first
- If unstable → 'diag'

---

#### 5. **State Interpretation & Labeling**

**Current Problem**: 
```python
regime_names = ['trending', 'low_volatility', 'high_volatility']
```

**This is WRONG!** ❌

**Why?**
- HMM doesn't know which state is which
- States are learned unsupervised
- Need to **label AFTER training** based on characteristics

**Correct Approach**:

```python
# After training HMM:
states = hmm.predict(features)

# Analyze each state's characteristics:
for state in range(n_states):
    state_data = features[states == state]
    
    avg_return = state_data['returns'].mean()
    avg_volatility = state_data['volatility'].mean()
    
    # Label based on characteristics
    if avg_volatility < threshold_low:
        labels[state] = 'low_volatility'
    elif avg_volatility > threshold_high:
        labels[state] = 'high_volatility'
    else:
        if avg_return > 0:
            labels[state] = 'bull_trend'
        else:
            labels[state] = 'bear_trend'
```

**Sources:**
- All HMM literature emphasizes this
- Rabiner (1989) - A Tutorial on Hidden Markov Models
- Community consensus (StackOverflow, Cross Validated)

---

#### 6. **Re-training Frequency**

**Current**: Train once per backtest

**Best Practices**:

**Walk-Forward Approach** (Recommended):
```
1. Train on window T-N to T
2. Predict for T to T+M
3. Slide window and retrain
4. Prevents look-ahead bias
```

**Frequency Recommendations**:
| Timeframe | Retrain Every  | Reason                    |
|-----------|----------------|---------------------------|
| 5m        | Daily/Weekly   | Fast market changes       |
| 1h        | Weekly/Monthly | Medium-term patterns      |
| 1d        | Monthly        | Long-term regime shifts   |

**For Backtesting**:
- **Option 1**: Train once on entire history (fast but biased)
- **Option 2**: Rolling window retrain (slower but realistic)

**Our Recommendation**: Start with once, then add rolling for production

---

#### 7. **Prediction Confidence**

**Current**: threshold = 0.7

**Best Practice**:
```python
# Get state probabilities
state_probs = hmm.predict_proba(features)
confidence = state_probs.max(axis=1)

# Adaptive threshold:
if confidence < 0.5:    # Very uncertain
    action = "neutral"
elif confidence < 0.7:  # Somewhat certain
    action = "cautious"
else:                   # High confidence
    action = "trade"
```

**Threshold Recommendations**:
- **Too low (< 0.5)**: Noise, unreliable
- **Too high (> 0.9)**: Very few signals
- **Optimal**: 0.6-0.8 range

**Current Issue**: 0.7 might be okay, but need to verify with data

---

#### 8. **Normalization & Scaling**

**Current**: StandardScaler ✅

**Best Practice**:
- ✅ Always scale features (we do this)
- ✅ Fit on training only (we do this)
- ✅ Use StandardScaler or RobustScaler
- ⚠️ Be careful with returns (already normalized by definition)

**Recommendation**: Keep current approach ✅

---

#### 9. **Model Validation**

**Missing in Current Implementation**: No validation!

**Best Practice Checks**:

```python
# 1. Check state persistence
transition_matrix = hmm.transitionprob_
print(f"Diagonal dominance: {np.diag(transition_matrix)}")
# Should be > 0.8 for persistent states

# 2. Check state distribution
state_counts = np.bincount(predicted_states)
print(f"State balance: {state_counts / len(predicted_states)}")
# Should not be too skewed (e.g., 95% in one state)

# 3. Check convergence
if hmm.monitor_.converged:
    print(f"Converged in {hmm.monitor_.iter} iterations")
else:
    print("WARNING: Did not converge!")
    
# 4. Log-likelihood
score = hmm.score(features)
print(f"Log-likelihood: {score}")
# Higher (less negative) is better
```

---

## 🔧 Identified Issues in Current Implementation

### Critical Issues:

1. **Training Window Too Small** ❌
   - Current: 500 candles
   - Need: 4000-8000 candles
   - Impact: HIGH (explains why all states are same)

2. **Pre-assigned State Labels** ❌
   - Current: Hard-coded ['trending', 'low_vol', 'high_vol']
   - Should: Assign AFTER training based on characteristics
   - Impact: HIGH (misinterpretation of regimes)

3. **Missing Critical Features** ⚠️
   - No skewness, kurtosis
   - No high-low range
   - Limited volume features
   - Impact: MEDIUM

4. **No Model Validation** ⚠️
   - No convergence check
   - No state persistence check
   - No transition matrix inspection
   - Impact: MEDIUM

5. **No Re-training Strategy** ⚠️
   - Trains once per backtest
   - No walk-forward
   - Impact: LOW (for now)

---

## 📋 Implementation Plan

### Phase 1: Critical Fixes (Today)

**Priority 1: Fix Training Window**
```python
# Change from 500 to 5000-8000
regime_training_lookback = IntParameter(
    3000, 10000, default=5000, space='buy'
)
```

**Priority 2: Fix State Labeling**
```python
def _assign_regime_labels(self, hmm, features):
    """Assign meaningful labels based on state characteristics."""
    states = hmm.predict(features)
    
    # Calculate characteristics per state
    state_profiles = {}
    for state in range(self.n_states):
        mask = (states == state)
        state_data = features[mask]
        
        state_profiles[state] = {
            'avg_return': state_data[:, 0].mean(),  # returns_1
            'avg_volatility': state_data[:, 3].mean(),  # volatility_10
            'count': mask.sum()
        }
    
    # Sort states by volatility
    sorted_states = sorted(
        state_profiles.items(),
        key=lambda x: x[1]['avg_volatility']
    )
    
    # Assign labels
    self.regime_mapping = {
        sorted_states[0][0]: 'low_volatility',
        sorted_states[1][0]: 'medium_volatility',
        sorted_states[2][0]: 'high_volatility'
    }
    
    # Further classify medium by trend
    medium_state = sorted_states[1][0]
    if state_profiles[medium_state]['avg_return'] > 0.0001:
        self.regime_mapping[medium_state] = 'trending_up'
    elif state_profiles[medium_state]['avg_return'] < -0.0001:
        self.regime_mapping[medium_state] = 'trending_down'
    else:
        self.regime_mapping[medium_state] = 'ranging'
```

**Priority 3: Add Validation**
```python
def _validate_hmm(self, hmm, features):
    """Validate HMM training quality."""
    # Check convergence
    if not hmm.monitor_.converged:
        logger.warning(f"HMM did not converge! Iterations: {hmm.monitor_.iter}")
    
    # Check state persistence
    diag = np.diag(hmm.transitionprob_)
    if diag.min() < 0.5:
        logger.warning(f"Low state persistence: {diag}")
    
    # Check state distribution
    states = hmm.predict(features)
    state_dist = np.bincount(states) / len(states)
    if state_dist.max() > 0.8:
        logger.warning(f"Highly skewed state distribution: {state_dist}")
    
    logger.info(f"HMM Validation - Converged: {hmm.monitor_.converged}, "
               f"Persistence: {diag}, Distribution: {state_dist}")
```

### Phase 2: Enhanced Features (Tomorrow)

Add better features:
```python
def _prepare_enhanced_features(self, df):
    """Enhanced feature set based on research."""
    features = pd.DataFrame(index=df.index)
    
    # Returns (multiple horizons)
    features['returns_1'] = np.log(df['close'] / df['close'].shift(1))
    features['returns_5'] = np.log(df['close'] / df['close'].shift(5))
    features['returns_20'] = np.log(df['close'] / df['close'].shift(20))
    
    # Volatility (multiple horizons + Parkinson)
    features['volatility_20'] = features['returns_1'].rolling(20).std()
    features['volatility_60'] = features['returns_1'].rolling(60).std()
    features['parkinson_vol'] = np.sqrt(
        1/(4*np.log(2)) * np.log(df['high']/df['low'])**2
    ).rolling(20).mean()
    
    # High-Low range
    features['hl_range'] = (df['high'] - df['low']) / df['close']
    
    # Volume features
    features['volume_ratio'] = df['volume'] / df['volume'].rolling(20).mean()
    features['volume_volatility'] = (
        df['volume'].rolling(20).std() / df['volume'].rolling(20).mean()
    )
    
    # Trend strength
    features['trend_strength'] = ta.ADX(df, timeperiod=14)
    
    # Higher moments
    features['return_skew'] = features['returns_1'].rolling(20).skew()
    features['return_kurt'] = features['returns_1'].rolling(20).kurt()
    
    return features.dropna()
```

### Phase 3: Walk-Forward (Next Week)

Implement rolling window retraining for production.

---

## 🎯 Expected Improvements

After implementing these fixes:

**Before (Current)**:
- All states classified as "trending"
- Win rate: 15%
- Profit: -2.57%

**After (Expected)**:
- Balanced state distribution (30/40/30)
- Win rate: 60-70%
- Profit: +5-10%
- Proper regime switching behavior

---

## 📚 Key References

1. **Hamilton (1989)** - "A New Approach to the Economic Analysis of Nonstationary Time Series"
2. **Rabiner (1989)** - "A Tutorial on Hidden Markov Models"
3. **Kritzman, Page & Turkington (2012)** - "Regime Shifts: Implications for Dynamic Strategies"
4. **Nystrup et al. (2015)** - "Regime-Based vs Static Asset Allocation"
5. **hmmlearn Documentation** - https://hmmlearn.readthedocs.io/
6. **QuantStart** - Regime Detection articles
7. **Cross Validated** - HMM for financial time series questions

---

## ✅ Action Items

### Immediate (Today):
- [ ] Increase training window to 5000
- [ ] Implement dynamic state labeling
- [ ] Add HMM validation checks
- [ ] Update confidence threshold if needed

### Short-term (Tomorrow):
- [ ] Add enhanced features
- [ ] Test with new features
- [ ] Document improvements

### Medium-term (Next Week):
- [ ] Implement walk-forward retraining
- [ ] Add regime persistence tracking
- [ ] Production monitoring

---

**Next Step**: Implement Priority 1-3 fixes NOW and re-run backtest!
