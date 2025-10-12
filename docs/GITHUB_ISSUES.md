# GitHub Issues for Pragma Trading Bot

**Repository:** https://github.com/aminak58/pragma-trading-bot

---

## 📋 Sprint 1 Issues (Week 1-2)

### Issue #1: Project Setup & Documentation ✅
**Status:** Completed  
**Label:** setup, documentation  
**Milestone:** Foundation

**Description:**
Initial project setup with complete documentation structure.

**Completed:**
- [x] Create repository structure
- [x] Write PROJECT_CHARTER.md
- [x] Write ARCHITECTURE.md
- [x] Write ROADMAP.md
- [x] Setup .gitignore
- [x] Create requirements.txt
- [x] Setup Git branches (main + develop)
- [x] Push to GitHub

**Closed:** 2025-10-12

---

### Issue #2: Environment Setup & Validation
**Priority:** 🔴 Critical  
**Label:** setup, environment  
**Milestone:** Foundation  
**Estimate:** 3 hours

**Description:**
Setup development environment and validate Freqtrade installation.

**Tasks:**
- [ ] Install dependencies from requirements.txt
- [ ] Verify Freqtrade 2024.x installation
- [ ] Install hmmlearn for regime detection
- [ ] Setup Python 3.11 virtual environment
- [ ] Download test data (180 days)
- [ ] Run baseline backtest with sample strategy
- [ ] Document environment setup in README

**Validation Command:**
```bash
freqtrade download-data --exchange binance \
  --pairs BTC/USDT ETH/USDT BNB/USDT \
  --timeframes 5m 15m 1h --days 180

freqtrade backtesting \
  --strategy SampleStrategy \
  --timerange 20250701-20251010
```

**Acceptance Criteria:**
- ✅ All dependencies installed without errors
- ✅ Data downloads successfully
- ✅ Baseline backtest runs and completes
- ✅ Environment documented

---

### Issue #3: HMM Regime Detector Implementation
**Priority:** 🔴 Critical  
**Label:** feature, hmm, regime-detection  
**Milestone:** Foundation  
**Estimate:** 8 hours  
**Depends on:** #2

**Description:**
Implement HMM-based market regime detection using hmmlearn.

**Technical Spec:**
- **Library:** hmmlearn
- **Model:** Gaussian HMM
- **States:** 3 (high_volatility, low_volatility, trending)
- **Features:** Returns, Volatility, Volume Ratio, ADX
- **Covariance:** full
- **Training:** 500 candles minimum

**Tasks:**
- [ ] Create `src/regime/hmm_detector.py`
- [ ] Implement Gaussian HMM (3-state)
- [ ] Add feature preparation method
- [ ] Add training method
- [ ] Add prediction method with confidence scoring
- [ ] Add state transition tracking

**Code Structure:**
```python
class RegimeDetector:
    def __init__(n_states: int = 3)
    def prepare_features(dataframe: DataFrame) -> np.ndarray
    def train(dataframe: DataFrame, lookback: int = 500) -> Self
    def predict_regime(dataframe: DataFrame) -> Tuple[str, float]
    def get_transition_matrix() -> np.ndarray
```

**Acceptance Criteria:**
- ✅ HMM trains without errors
- ✅ Predictions are deterministic (stable)
- ✅ Regime labels make sense for test data
- ✅ Confidence scores between 0-1
- ✅ Code documented with docstrings

---

### Issue #4: HMM Unit Tests
**Priority:** 🟡 High  
**Label:** testing, hmm  
**Milestone:** Foundation  
**Estimate:** 4 hours  
**Depends on:** #3

**Description:**
Comprehensive unit tests for HMM regime detector.

**Tasks:**
- [ ] Create `tests/unit/test_hmm_detector.py`
- [ ] Test training functionality
- [ ] Test prediction stability
- [ ] Test feature preparation
- [ ] Test error handling
- [ ] Test edge cases (insufficient data, NaN values)
- [ ] Achieve >80% code coverage

**Test Cases:**
1. **Training Test:** Model trains successfully
2. **Prediction Test:** Returns valid regime and confidence
3. **Stability Test:** Same input = same output
4. **Edge Cases:** Handles missing data gracefully
5. **Performance Test:** Processes 1000 candles < 1 second

**Acceptance Criteria:**
- ✅ All tests pass
- ✅ Coverage > 80%
- ✅ No warnings or errors
- ✅ Tests documented

---

### Issue #5: HMM Integration with Freqtrade
**Priority:** 🟡 High  
**Label:** integration, hmm  
**Milestone:** Foundation  
**Estimate:** 4 hours  
**Depends on:** #4

**Description:**
Integrate HMM regime detector with Freqtrade dataframe pipeline.

**Tasks:**
- [ ] Add regime detection to indicator pipeline
- [ ] Create regime feature columns
- [ ] Test integration with sample strategy
- [ ] Validate regime changes on historical data
- [ ] Performance benchmarking
- [ ] Document integration pattern

**Integration Pattern:**
```python
def populate_indicators(self, dataframe, metadata):
    # Train/update HMM
    self.regime_detector.train(dataframe)
    
    # Add regime features
    regime, confidence = self.regime_detector.predict_regime(dataframe)
    dataframe['regime'] = regime
    dataframe['regime_confidence'] = confidence
    
    return dataframe
```

**Acceptance Criteria:**
- ✅ Regime column added to dataframe
- ✅ No performance degradation (<5% slower)
- ✅ Regime changes make sense visually
- ✅ Integration documented

---

## 📊 Issue Summary

| Issue | Priority | Status | Estimate |
|-------|----------|--------|----------|
| #1: Project Setup | 🔴 | ✅ Done | - |
| #2: Environment | 🔴 | ⏳ Todo | 3h |
| #3: HMM Implementation | 🔴 | ⏳ Todo | 8h |
| #4: HMM Tests | 🟡 | ⏳ Todo | 4h |
| #5: HMM Integration | 🟡 | ⏳ Todo | 4h |

**Total Sprint 1:** 19 hours (2-3 days work)

---

## 🎯 How to Create Issues in GitHub

**Option 1: Manual (Recommended for now):**
1. Go to: https://github.com/aminak58/pragma-trading-bot/issues/new
2. Copy title and description from above
3. Add labels and milestone
4. Submit

**Option 2: GitHub CLI:**
```bash
gh issue create --title "Issue title" --body "Description" --label "label1,label2"
```

---

**Last Updated:** 2025-10-12 00:37
