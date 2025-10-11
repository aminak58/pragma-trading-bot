"""
Script to create GitHub Issues for Pragma Trading Bot
Uses GitHub API to automate issue creation
"""

import requests
import json
import os

# Configuration
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN", "")  # Set via environment variable
if not GITHUB_TOKEN:
    print("❌ Error: GITHUB_TOKEN environment variable not set")
    print("💡 Usage: set GITHUB_TOKEN=your_token && python scripts/create_issues.py")
    exit(1)

OWNER = "aminak58"
REPO = "pragma-trading-bot"
API_BASE = "https://api.github.com"

headers = {
    "Authorization": f"token {GITHUB_TOKEN}",
    "Accept": "application/vnd.github.v3+json"
}

def create_milestone():
    """Create Foundation milestone"""
    url = f"{API_BASE}/repos/{OWNER}/{REPO}/milestones"
    data = {
        "title": "Foundation",
        "state": "open",
        "description": "HMM + FreqAI infrastructure - Sprint 1",
        "due_on": "2025-10-25T23:59:59Z"
    }
    
    response = requests.post(url, headers=headers, json=data)
    if response.status_code == 201:
        print("✅ Milestone 'Foundation' created")
        return response.json()['number']
    else:
        print(f"⚠️ Milestone creation failed: {response.json()}")
        return None

def create_labels():
    """Create necessary labels"""
    url = f"{API_BASE}/repos/{OWNER}/{REPO}/labels"
    
    labels = [
        {"name": "setup", "color": "d4c5f9", "description": "Project setup tasks"},
        {"name": "environment", "color": "0052cc", "description": "Environment configuration"},
        {"name": "feature", "color": "0e8a16", "description": "New features"},
        {"name": "hmm", "color": "fbca04", "description": "HMM regime detection"},
        {"name": "testing", "color": "c2e0c6", "description": "Test implementation"},
        {"name": "integration", "color": "5319e7", "description": "Integration work"},
        {"name": "regime-detection", "color": "e99695", "description": "Market regime detection"},
    ]
    
    for label in labels:
        response = requests.post(url, headers=headers, json=label)
        if response.status_code == 201:
            print(f"✅ Label '{label['name']}' created")
        elif response.status_code == 422:
            print(f"ℹ️ Label '{label['name']}' already exists")
        else:
            print(f"⚠️ Label '{label['name']}' failed: {response.status_code}")

def create_issue(title, body, labels, milestone=None):
    """Create a GitHub issue"""
    url = f"{API_BASE}/repos/{OWNER}/{REPO}/issues"
    
    data = {
        "title": title,
        "body": body,
        "labels": labels
    }
    
    if milestone:
        data["milestone"] = milestone
    
    response = requests.post(url, headers=headers, json=data)
    if response.status_code == 201:
        issue_number = response.json()['number']
        print(f"✅ Issue #{issue_number}: {title}")
        return issue_number
    else:
        print(f"❌ Issue creation failed: {response.json()}")
        return None

def main():
    print("🚀 Creating GitHub Issues for Pragma Trading Bot\n")
    
    # Step 1: Create milestone
    print("📅 Creating Milestone...")
    milestone_number = create_milestone()
    print()
    
    # Step 2: Create labels
    print("🏷️ Creating Labels...")
    create_labels()
    print()
    
    # Step 3: Create issues
    print("📋 Creating Issues...")
    
    # Issue #2: Environment Setup
    create_issue(
        title="Environment Setup & Validation",
        body="""## 🎯 Description
Setup development environment and validate Freqtrade installation.

## 📋 Tasks
- [ ] Install dependencies from requirements.txt
- [ ] Verify Freqtrade 2024.x installation
- [ ] Install hmmlearn for regime detection
- [ ] Setup Python 3.11 virtual environment
- [ ] Download test data (180 days)
- [ ] Run baseline backtest with sample strategy
- [ ] Document environment setup in README

## 🧪 Validation Commands
```bash
# Download test data
freqtrade download-data --exchange binance \\
  --pairs BTC/USDT ETH/USDT BNB/USDT \\
  --timeframes 5m 15m 1h --days 180

# Run baseline backtest
freqtrade backtesting \\
  --strategy SampleStrategy \\
  --timerange 20250701-20251010
```

## ✅ Acceptance Criteria
- ✅ All dependencies installed without errors
- ✅ Data downloads successfully
- ✅ Baseline backtest runs and completes
- ✅ Environment documented in README

## 📊 Metadata
- **Priority:** 🔴 Critical
- **Estimate:** 3 hours

## 🔗 Dependencies
None""",
        labels=["setup", "environment"],
        milestone=milestone_number
    )
    
    # Issue #3: HMM Implementation
    create_issue(
        title="HMM Regime Detector Implementation",
        body="""## 🎯 Description
Implement HMM-based market regime detection using hmmlearn.

## 📋 Technical Spec
- **Library:** hmmlearn
- **Model:** Gaussian HMM
- **States:** 3 (high_volatility, low_volatility, trending)
- **Features:** Returns, Volatility, Volume Ratio, ADX
- **Covariance:** full
- **Training:** 500 candles minimum

## 📋 Tasks
- [ ] Create `src/regime/hmm_detector.py`
- [ ] Implement Gaussian HMM (3-state)
- [ ] Add feature preparation method
- [ ] Add training method
- [ ] Add prediction method with confidence scoring
- [ ] Add state transition tracking

## 💻 Code Structure
```python
class RegimeDetector:
    def __init__(n_states: int = 3)
    def prepare_features(dataframe: DataFrame) -> np.ndarray
    def train(dataframe: DataFrame, lookback: int = 500) -> Self
    def predict_regime(dataframe: DataFrame) -> Tuple[str, float]
    def get_transition_matrix() -> np.ndarray
```

## ✅ Acceptance Criteria
- ✅ HMM trains without errors
- ✅ Predictions are deterministic (stable)
- ✅ Regime labels make sense for test data
- ✅ Confidence scores between 0-1
- ✅ Code documented with docstrings

## 📊 Metadata
- **Priority:** 🔴 Critical
- **Estimate:** 8 hours
- **Depends on:** #2""",
        labels=["feature", "hmm", "regime-detection"],
        milestone=milestone_number
    )
    
    # Issue #4: HMM Tests
    create_issue(
        title="HMM Unit Tests",
        body="""## 🎯 Description
Comprehensive unit tests for HMM regime detector.

## 📋 Tasks
- [ ] Create `tests/unit/test_hmm_detector.py`
- [ ] Test training functionality
- [ ] Test prediction stability
- [ ] Test feature preparation
- [ ] Test error handling
- [ ] Test edge cases (insufficient data, NaN values)
- [ ] Achieve >80% code coverage

## 🧪 Test Cases
1. **Training Test:** Model trains successfully
2. **Prediction Test:** Returns valid regime and confidence
3. **Stability Test:** Same input = same output
4. **Edge Cases:** Handles missing data gracefully
5. **Performance Test:** Processes 1000 candles < 1 second

## ✅ Acceptance Criteria
- ✅ All tests pass
- ✅ Coverage > 80%
- ✅ No warnings or errors
- ✅ Tests documented

## 📊 Metadata
- **Priority:** 🟡 High
- **Estimate:** 4 hours
- **Depends on:** #3""",
        labels=["testing", "hmm"],
        milestone=milestone_number
    )
    
    # Issue #5: HMM Integration
    create_issue(
        title="HMM Integration with Freqtrade",
        body="""## 🎯 Description
Integrate HMM regime detector with Freqtrade dataframe pipeline.

## 📋 Tasks
- [ ] Add regime detection to indicator pipeline
- [ ] Create regime feature columns
- [ ] Test integration with sample strategy
- [ ] Validate regime changes on historical data
- [ ] Performance benchmarking
- [ ] Document integration pattern

## 💻 Integration Pattern
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

## ✅ Acceptance Criteria
- ✅ Regime column added to dataframe
- ✅ No performance degradation (<5% slower)
- ✅ Regime changes make sense visually
- ✅ Integration documented

## 📊 Metadata
- **Priority:** 🟡 High
- **Estimate:** 4 hours
- **Depends on:** #4""",
        labels=["integration", "hmm"],
        milestone=milestone_number
    )
    
    print("\n✅ All done! Check: https://github.com/aminak58/pragma-trading-bot/issues")

if __name__ == "__main__":
    main()
